from __future__ import annotations

import threading
from pathlib import Path

import approval
import pytest
from approval.batch_types import ApprovalBatch
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput

from app.repl.slash_commands import ReplState
from app.run_task import run_task
from llm.messages import TextPart, ToolCallPart, UserMessage
from scripts.testing.llm import from_test_native_tool_then_final, from_test_sequence
from frontends.tui.fullscreen import _build_application
from frontends.tui.event_adapter import TuiEventAdapter
from frontends.tui.session import FullscreenTui, parse_approval_decision
from frontends.tui.transcript import TranscriptBuffer, truncate_text
from runtime.stream_events import (
    AssistantReasoningDelta,
    AssistantTextDelta,
    AssistantTurnComplete,
    LifecycleChanged,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from runtime.session_message_store import SessionMessageStore
from runtime.workspaces import WorkspaceStore
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk


class DummyClient:
    pass


def test_transcript_appends_assistant_delta() -> None:
    transcript = TranscriptBuffer()
    adapter = TuiEventAdapter(transcript)

    adapter.render(AssistantTextDelta(text="hello "))
    adapter.render(AssistantTextDelta(text="world"))
    adapter.render(
        AssistantTurnComplete(
            content="hello world",
            usage={"input_tokens": 1, "output_tokens": 2},
            stop_reason="end_turn",
        )
    )

    items = transcript.snapshot()
    assert items[0].role == "assistant"
    assert items[0].body == "hello world"
    assert any("tokens=1/2" in item.body for item in items)


def test_transcript_merges_reasoning_deltas_into_one_row() -> None:
    """多片思考增量并进同一条，不是一片一行。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：无
    返回：无

    流式一轮思考链会来几十上百片，一片一行会把 transcript 刷爆。
    """
    transcript = TranscriptBuffer()
    adapter = TuiEventAdapter(transcript)

    adapter.render(AssistantReasoningDelta(text="先看清"))
    adapter.render(AssistantReasoningDelta(text="用户要什么"))

    reasoning = [item for item in transcript.snapshot() if item.title == "reasoning"]
    assert len(reasoning) == 1
    assert reasoning[0].body == "先看清用户要什么"


def test_transcript_does_not_merge_reasoning_into_other_status_rows() -> None:
    """思考链不能并进上一条同为 status 角色的行里。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：无
    返回：无

    lifecycle / lease / state 都用 status 角色，只按角色合并会把思考链塞进它们里面。
    """
    transcript = TranscriptBuffer()
    adapter = TuiEventAdapter(transcript)

    adapter.render(
        LifecycleChanged(
            lifecycle="running", reason="", checkpoint_id=None, segment_id="seg-1"
        )
    )
    adapter.render(AssistantReasoningDelta(text="先看清"))

    items = transcript.snapshot()
    assert [item.title for item in items] == ["lifecycle", "reasoning"]


def test_tool_events_render_as_transcript_blocks() -> None:
    transcript = TranscriptBuffer()
    adapter = TuiEventAdapter(transcript)

    adapter.render(
        ToolExecutionStarted(
            tool_name="list",
            args={"path": "tools"},
            call_id="call-1",
            risk="safe",
        )
    )
    adapter.render(
        ToolExecutionCompleted(
            tool_name="list",
            output="tools/",
            call_id="call-1",
        )
    )

    items = transcript.snapshot()
    assert [item.role for item in items] == ["tool", "tool"]
    assert items[0].title == "list"
    assert items[0].state == "pending"
    assert items[1].state == "success"
    assert "tools/" in items[1].body


def test_transcript_uses_pi_style_fragments() -> None:
    transcript = TranscriptBuffer()
    transcript.append("user", "you", "inspect this")
    transcript.append("assistant", "assistant", "working on it")
    transcript.append("tool", "list", "args: {}", state="pending")

    fragments = transcript.fragments(width=40)
    styles = [style for style, _text in fragments]
    rendered = "".join(text for _style, text in fragments)

    assert "class:user.block" in styles
    assert "class:assistant.body" in styles
    assert "class:tool.pending.border" in styles
    assert "list  running" in rendered
    assert "-" in rendered


def test_slash_help_renders_into_transcript(tmp_path: Path) -> None:
    session = FullscreenTui(
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
        llm_client=DummyClient(),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
    )

    session.submit("/help")

    rendered = "\n".join(item.body for item in session.transcript.snapshot())
    assert "Available commands" in rendered
    assert "/status" in rendered


def test_slash_command_is_echoed_as_user_message(tmp_path: Path) -> None:
    session = FullscreenTui(
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
        llm_client=DummyClient(),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
    )

    session.submit("/help")

    first = session.transcript.snapshot()[0]
    assert first.role == "user"
    assert first.body == "/help"


def test_agent_turn_uses_thread_local_task_store(tmp_path: Path) -> None:
    session = FullscreenTui(
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
        llm_client=from_test_sequence(["已完成真实执行链"]),
        tool_registry=ToolRegistry(),
    )
    errors: list[Exception] = []

    def run_turn() -> None:
        try:
            session._drive_agent_turn("123456")
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run_turn)
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert errors == []
    assert any(
        item.body == "已完成真实执行链" for item in session.transcript.snapshot()
    )


def test_run_turn_renders_started_and_finished(tmp_path: Path) -> None:
    session = FullscreenTui(
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
        llm_client=from_test_sequence(["已完成真实执行链"]),
        tool_registry=ToolRegistry(),
    )

    session._run_turn("123456")

    titles = [item.title for item in session.transcript.snapshot()]
    assert "run started" in titles
    assert "run finished" in titles
    assert any(
        item.body == "已完成真实执行链" for item in session.transcript.snapshot()
    )


def test_new_fullscreen_session_binds_workspace_before_accepting_input(
    tmp_path: Path,
) -> None:
    """新全屏会话在首条输入前保存原目录；传参：隔离目录；返回：无。"""
    data_root = tmp_path / "data"
    session = FullscreenTui(
        project_root=tmp_path,
        data_root=data_root,
        llm_client=from_test_sequence(["收到"]),
        tool_registry=ToolRegistry(),
    )
    try:
        assert (
            WorkspaceStore(data_root).for_session(session.state.session_id).project_root
            == tmp_path
        )
        assert not SessionMessageStore(data_root).read_entries(session.state.session_id)
    finally:
        session.close()


def test_fullscreen_resume_from_another_directory_uses_original_workspace(
    tmp_path: Path,
) -> None:
    """从B目录恢复A会话时，真实文件工具与权限仍归A；传参：隔离目录；返回：无。"""
    original, launch, data_root = (
        tmp_path / "original",
        tmp_path / "launch",
        tmp_path / "data",
    )
    original.mkdir()
    launch.mkdir()
    # 【全屏会话】【恢复归属】1. 通过生产入口建立可由/resume恢复的原始会话
    run_task(
        "建立会话",
        original,
        data_root=data_root,
        session_id="session-original",
        llm_client=from_test_sequence(["原会话已建立"]),
        tool_registry=ToolRegistry(),
    )
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "write-original",
                "file_write",
                {"path": "result.txt", "content": "原目录成果"},
            ),
        ],
        "写入完成",
    )
    session = FullscreenTui(project_root=launch, data_root=data_root, llm_client=client)
    try:
        # 【全屏会话】【恢复归属】2. 换目录启动后沿实际恢复命令继续执行
        session.submit("/resume session-original")
        assert session.state.session_id == "session-original"
        session._drive_agent_turn("写入原目录")
        assert (original / "result.txt").read_text(encoding="utf-8") == "原目录成果"
        assert not (launch / "result.txt").exists()
        assert session.project_root == original
        assert (
            WorkspaceStore(data_root).for_session("session-original").project_root
            == original
        )
    finally:
        session.close()


@pytest.mark.parametrize("missing_binding", [False, True])
def test_fullscreen_rejects_unavailable_original_workspace_before_input(
    tmp_path: Path,
    missing_binding: bool,
) -> None:
    """原目录消失或旧会话缺少归属时，不接纳新输入或改绑；传参：目录及缺失类型；返回：无。"""
    original, launch, data_root = (
        tmp_path / "original",
        tmp_path / "launch",
        tmp_path / "data",
    )
    launch.mkdir()
    messages = SessionMessageStore(data_root)
    workspaces = WorkspaceStore(data_root)
    if not missing_binding:
        original.mkdir()
        workspaces.bind_session("session-original", original)
        original.rmdir()
    messages.append_message(
        "session-original", UserMessage("old", (TextPart("保留历史"),))
    )
    previous = messages.read_entries("session-original")
    session = FullscreenTui(
        project_root=launch,
        data_root=data_root,
        llm_client=from_test_sequence(["不应执行"]),
        tool_registry=ToolRegistry(),
    )
    session.state = ReplState(session_id="session-original")
    try:
        session._run_turn("继续执行")
        items = session.transcript.snapshot()
        expected = (
            "session workspace is missing" if missing_binding else "原工作区不可用"
        )
        assert any(expected in item.body for item in items)
        assert items[-1].title == "run failed"
        assert all(
            item.title != "run started" and item.role != "assistant" for item in items
        )
        assert messages.read_entries("session-original") == previous
        saved = workspaces.find_for_session("session-original")
        if missing_binding:
            assert saved is None
        else:
            assert saved is not None and saved.project_root == original
    finally:
        session.close()


@pytest.mark.parametrize(
    ("choice", "expected_effects"), [("once", ["执行动作"]), ("deny", [])]
)
def test_fullscreen_bound_session_preserves_batch_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
    expected_effects: list[str],
) -> None:
    """全屏工作区绑定后仍只执行批次明确授权项；传参：目录、替换器、选择及效果；返回：无。"""
    effects: list[str] = []

    def execute(_args: dict[str, object]) -> str:
        """记录审批通过后的真实效果；传参：工具参数；返回：执行结果。"""
        effects.append("执行动作")
        return "已执行"

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "confirm_action",
            "需要授权的动作",
            {},
            "agent",
            ToolRisk.CONFIRM,
            False,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.NO,
            executor=execute,
        )
    )
    client = from_test_native_tool_then_final(
        [ToolCallPart("confirm", "confirm_action", {})], "本轮结束"
    )
    session = FullscreenTui(
        project_root=tmp_path,
        data_root=tmp_path / "data",
        llm_client=client,
        tool_registry=registry,
    )
    reached = threading.Event()
    original_present = session.batch_approvals._present_batch

    def present(identity: str, batch: ApprovalBatch) -> None:
        """等待真实批次完成展示后再提交选择；传参：审批身份和批次；返回：无。"""
        original_present(identity, batch)
        reached.set()

    monkeypatch.setattr(session.batch_approvals, "_present_batch", present)
    monkeypatch.setattr(
        "approval.batch._batch_backend", session._batch_approval_backend
    )
    monkeypatch.setattr("approval._config_path", lambda: tmp_path / "config.yaml")
    try:
        session.submit("执行动作")
        assert reached.wait(10) and effects == []
        session.submit(f"/approve {session._batch_identity} 1={choice}")
        assert session._worker is not None
        session._worker.join(timeout=10)
        assert not session._worker.is_alive()
        assert effects == expected_effects
        items = session.transcript.snapshot()
        assert any(item.title == "run finished" for item in items)
        assert all(item.role != "error" for item in items)
    finally:
        session.close()


def test_session_scrolls_transcript_viewport(tmp_path: Path) -> None:
    session = FullscreenTui(
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
        llm_client=DummyClient(),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
    )
    for index in range(120):
        session.transcript.append("status", f"line-{index}", "")

    bottom = "".join(text for _style, text in session.transcript_fragments())
    session.scroll_up(10)
    upper = "".join(text for _style, text in session.transcript_fragments())

    assert "line-119" in bottom
    assert "line-119" not in upper


def test_busy_transcript_renders_working_spinner(tmp_path: Path) -> None:
    session = FullscreenTui(
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
        llm_client=DummyClient(),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
    )
    session._busy = True

    rendered = "".join(text for _style, text in session.transcript_fragments())

    assert "Working..." in rendered


def test_fullscreen_application_can_be_built_without_real_console(
    tmp_path: Path,
) -> None:
    session = FullscreenTui(
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
        llm_client=DummyClient(),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
    )

    app = _build_application(
        session,
        input_obj=DummyInput(),
        output_obj=DummyOutput(),
    )

    assert app.full_screen is True


def test_footer_renders_two_line_pi_style_status(tmp_path: Path) -> None:
    session = FullscreenTui(
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
        llm_client=DummyClient(),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
    )

    rendered = "".join(text for _style, text in session.footer_fragments())

    assert "\n" in rendered
    assert "ready" in rendered
    assert "scroll Wheel/PgUp/PgDn" in rendered
    assert "/help /status /exit" in rendered


def test_footer_renders_running_run_status(tmp_path: Path) -> None:
    session = FullscreenTui(
        project_root=tmp_path,
        data_root=tmp_path / ".reins" / "data",
        llm_client=DummyClient(),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
    )
    session.state.current_run_id = "run-1234567890abcdef"
    session._busy = True

    rendered = "".join(text for _style, text in session.footer_fragments())

    assert "running run 567890abcdef" in rendered


def test_approval_decision_parser() -> None:
    assert parse_approval_decision("/approve") is approval.ApprovalDecision.ONCE
    assert parse_approval_decision("/approve task") is approval.ApprovalDecision.TASK
    assert parse_approval_decision("/deny") is approval.ApprovalDecision.DENY
    assert parse_approval_decision("/status") is None


def test_truncate_text_marks_omitted_tail() -> None:
    assert truncate_text("abcdef", 5) == "a ..."
