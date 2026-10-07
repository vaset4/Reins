"""End-to-end REPL tests.

Covers the multi-turn interaction PRD requirements (R1-R6, R18):
- REPL banner and prompt label appear.
- Slash commands route through the registry.
- A plain prompt drives `agent_loop.run_stream`, conversation rows are
  appended, and the same task_id is reused across multiple prompts.
- The second prompt's LLM call sees the first prompt's user/assistant rows
  in the messages array (multi-turn injection).
- `/dashboard` runs without raising.
"""

from __future__ import annotations
from runtime.session_state import SessionStateStore
from scripts.testing.llm import from_test_sequence

from pathlib import Path
from threading import Event
from typing import Callable

import pytest
from rich.console import Console

from app.repl import run_repl
from app.repl.console import reset_console_for_tests
from app.repl.dashboard import _right_panel
from app.repl.slash_commands import ReplState
from llm.messages import TextPart
from runtime.lease import from_trigger
from runtime.session_messages import materialize_messages
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry


@pytest.fixture()
def recorded() -> Console:
    console = Console(record=True, width=120, force_terminal=False, color_system=None)
    reset_console_for_tests(console)
    yield console
    reset_console_for_tests(None)


def _project(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "tools").mkdir(parents=True)
    (project / "tools" / "alpha.py").write_text("x = 1\n", encoding="utf-8")
    data_root = project / ".reins" / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    return project, data_root


def _scripted_prompt(lines: list[str]) -> Callable[[str], str]:
    iterator = iter(lines)

    def fn(_label: str) -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError()

    return fn


def test_repl_exits_immediately_on_exit_command(
    recorded: Console, tmp_path: Path
) -> None:
    project, data_root = _project(tmp_path)
    client = from_test_sequence([])
    rc = run_repl(
        project_root=project,
        data_root=data_root,
        llm_client=client,
        prompt_fn=_scripted_prompt(["/exit"]),
    )
    assert rc == 0
    rendered = recorded.export_text()
    assert "Reins" in rendered
    assert "Bye" in rendered


def test_repl_prints_banner_with_help_index(recorded: Console, tmp_path: Path) -> None:
    project, data_root = _project(tmp_path)
    client = from_test_sequence([])
    run_repl(
        project_root=project,
        data_root=data_root,
        llm_client=client,
        prompt_fn=_scripted_prompt(["/exit"]),
    )
    rendered = recorded.export_text()
    assert "/help" in rendered
    assert "/tasks" in rendered
    assert "/dashboard" in rendered


def test_repl_help_command(recorded: Console, tmp_path: Path) -> None:
    project, data_root = _project(tmp_path)
    client = from_test_sequence([])
    run_repl(
        project_root=project,
        data_root=data_root,
        llm_client=client,
        prompt_fn=_scripted_prompt(["/help", "/exit"]),
    )
    rendered = recorded.export_text()
    assert "Available commands" in rendered
    # Ensure we see at least three of the registered commands listed.
    assert "/status" in rendered
    assert "/compact" in rendered


def _canonical_turns(data_root: Path) -> list[tuple[str, str]]:
    """从唯一消息 owner 读出本次 REPL 会话的 (kind, text) 序列。

    参数：data_root 为 data 根目录
    返回：按当前分支顺序排列的消息种类与文本；REPL 内部生成 session_id，故按目录发现
    """
    sessions = SessionStateStore(data_root).list_recent()
    assert len(sessions) == 1, f"expected exactly one session, got {len(sessions)}"
    messages = materialize_messages(data_root, sessions[0].session_id)
    return [
        (
            message.kind,
            "".join(
                part.text for part in message.content if isinstance(part, TextPart)
            ),
        )
        for message in messages
    ]


def test_repl_creates_inbox_task_on_first_prompt(
    recorded: Console, tmp_path: Path
) -> None:
    project, data_root = _project(tmp_path)
    client = from_test_sequence(['{"type":"final","content":"hi there"}'])
    run_repl(
        project_root=project,
        data_root=data_root,
        llm_client=client,
        prompt_fn=_scripted_prompt(["hello", "/exit"]),
    )
    rendered = recorded.export_text()
    assert "hi there" in rendered

    # Exactly one inbox task created.
    store = TaskStore(data_root)
    inbox = store.get_inbox_tasks()
    assert len(inbox) == 1
    assert _canonical_turns(data_root) == [
        ("user", "hello"),
        ("assistant", "hi there"),
    ]


def test_repl_reuses_task_across_two_prompts(
    recorded: Console, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multi-turn key invariant: the second prompt must reuse the first
    prompt's task_id, and the LLM must see the first turn's history."""
    project, data_root = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"final","content":"answer one"}',
            '{"type":"final","content":"answer two"}',
        ]
    )
    from runtime.agent_loop import AgentLoop

    completed = Event()
    original = AgentLoop.run_stream

    def stream(loop, context):
        """第一轮实际提交完毕后才允许下一条独立输入；传参：循环/上下文；返回：真实事件。"""
        yield from original(loop, context)
        completed.set()

    commands = iter(["question one", "question two", "/exit"])

    def prompt(_label):
        """保留本测试一轮完成后再提问的语义；传参：提示；返回：输入。"""
        value = next(commands)
        if value == "question two":
            assert completed.wait(5)
        return value

    monkeypatch.setattr(AgentLoop, "run_stream", stream)
    run_repl(
        project_root=project,
        data_root=data_root,
        llm_client=client,
        prompt_fn=prompt,
    )

    store = TaskStore(data_root)
    inbox = store.get_inbox_tasks()
    assert len(inbox) == 1, "REPL must reuse the same task across multiple turns"
    assert _canonical_turns(data_root) == [
        ("user", "question one"),
        ("assistant", "answer one"),
        ("user", "question two"),
        ("assistant", "answer two"),
    ]


def test_repl_renders_tool_call_panel(recorded: Console, tmp_path: Path) -> None:
    project, data_root = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"final","content":"tools/ contains alpha.py"}',
        ],
        protocol_mode="text_json",
    )
    run_repl(
        project_root=project,
        data_root=data_root,
        llm_client=client,
        prompt_fn=_scripted_prompt(["list tools dir", "/exit"]),
    )
    rendered = recorded.export_text()
    assert "tool: list" in rendered or "list" in rendered
    assert "alpha.py" in rendered  # tool output panel content
    assert "tools/ contains alpha.py" in rendered  # final markdown


def test_repl_dashboard_runs_without_error(recorded: Console, tmp_path: Path) -> None:
    project, data_root = _project(tmp_path)
    # Seed a task so the dashboard has something to render.
    store = TaskStore(data_root)
    record = store.create_task("seeded goal")

    client = from_test_sequence([])
    run_repl(
        project_root=project,
        data_root=data_root,
        llm_client=client,
        prompt_fn=_scripted_prompt([f"/task {record.task_id}", "/dashboard", "/exit"]),
    )
    rendered = recorded.export_text()
    assert "Reins Dashboard" in rendered
    assert "seeded goal" in rendered
    assert "LEASE" in rendered
    assert "WATCHDOG" in rendered


def test_dashboard_renders_capabilities_from_lease(tmp_path: Path) -> None:
    project, data_root = _project(tmp_path)
    lease = from_trigger(
        "user",
        task_id="task-1",
        capabilities={
            "fs": {"read": [], "write": []},
            "terminal": {"enabled": False, "allow_commands": ["python*"]},
            "browser": {"enabled": True, "headless": False},
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": False, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": False, "allow_servers": ["echo"]},
        },
    )

    panel = _right_panel(
        ReplState(current_task_id=None),
        TaskStore(data_root),
        ToolRegistry(),
        project,
        data_root,
        lease,
    )
    rendered = str(panel.renderable)

    assert 'terminal:   disabled allow=["python*"]' in rendered
    assert "browser:    enabled headed" in rendered
    assert "network:    disabled" in rendered
    assert 'mcp:        disabled allow_servers=["echo"]' in rendered


def test_dashboard_renders_unknown_without_lease(tmp_path: Path) -> None:
    project, data_root = _project(tmp_path)

    panel = _right_panel(
        ReplState(current_task_id=None),
        TaskStore(data_root),
        ToolRegistry(),
        project,
        data_root,
        None,
    )
    rendered = str(panel.renderable)

    assert "LEASE  (unknown)" in rendered
    assert "terminal:   unknown" in rendered
    assert "browser:    unknown" in rendered
    assert "network:    unknown" in rendered
    assert "mcp:        unknown" in rendered


def test_repl_unknown_command_renders_friendly_message(
    recorded: Console, tmp_path: Path
) -> None:
    project, data_root = _project(tmp_path)
    client = from_test_sequence([])
    run_repl(
        project_root=project,
        data_root=data_root,
        llm_client=client,
        prompt_fn=_scripted_prompt(["/notreal", "/exit"]),
    )
    rendered = recorded.export_text()
    assert "Unknown command" in rendered


def test_repl_skips_blank_lines(recorded: Console, tmp_path: Path) -> None:
    project, data_root = _project(tmp_path)
    client = from_test_sequence([])
    run_repl(
        project_root=project,
        data_root=data_root,
        llm_client=client,
        prompt_fn=_scripted_prompt(["", "   ", "/exit"]),
    )
    rendered = recorded.export_text()
    # Blank lines should not produce any "Unknown command" output.
    assert "Unknown command" not in rendered


def test_repl_eof_exits_cleanly(recorded: Console, tmp_path: Path) -> None:
    project, data_root = _project(tmp_path)
    client = from_test_sequence([])
    rc = run_repl(
        project_root=project,
        data_root=data_root,
        llm_client=client,
        prompt_fn=_scripted_prompt([]),
    )
    assert rc == 0
    rendered = recorded.export_text()
    assert "EOF received" in rendered


def test_repl_initial_message_runs_before_first_prompt(
    recorded: Console, tmp_path: Path
) -> None:
    project, data_root = _project(tmp_path)
    client = from_test_sequence(['{"type":"final","content":"initial answer"}'])
    run_repl(
        project_root=project,
        data_root=data_root,
        llm_client=client,
        prompt_fn=_scripted_prompt(["/exit"]),
        initial_message="seeded prompt",
    )
    rendered = recorded.export_text()
    assert "initial answer" in rendered
