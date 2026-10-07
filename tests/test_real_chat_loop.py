from __future__ import annotations
from scripts.testing.llm import from_test_sequence

from pathlib import Path

from app.run_task import run_task
from llm.messages import AgentMessage, TextPart
from runtime.agent_loop import AgentLoop, State
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_messages import append_user_message, materialize_messages
from runtime.types import RunContext, Trigger
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry


def test_chat_loop_executes_tool_registry_then_returns_final(tmp_path: Path) -> None:
    project = _project_with_tools(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"final","content":"tools contains alpha.py and nested."}',
        ],
        protocol_mode="text_json",
    )

    response = run_task(
        "帮我总结当前项目 tools 目录的结构",
        project,
        llm_client=client,
        data_root=project / ".reins" / "data",
        session_id="session-test-real-chat",
        run_id="run-test-real-chat",
    )

    assert response.status == "done"
    assert response.output == "tools contains alpha.py and nested."

    data_root = project / ".reins" / "data"
    # 对话落在 session 维度的唯一消息 owner 上，tool_result 保留工具原始输出
    messages = materialize_messages(data_root, "session-test-real-chat")
    assert [message.kind for message in messages] == [
        "user",
        "assistant",
        "tool_result",
        "assistant",
    ]
    assert "alpha.py" in _message_text(messages[2])

    facts = RunFactStore(project / ".reins" / "data").read_run("run-test-real-chat")
    assert any(
        row.get("event") == "tool:request" and _fact_tool_name(row) == "list"
        for row in facts
    )
    assert any(
        row.get("event") == "tool:response"
        and _fact_tool_name(row) == "list"
        and _fact_tool_status(row) == "ok"
        for row in facts
    )
    assert TaskStore(data_root).read_summary(response.task_id) == response.output


def test_chat_loop_runs_past_segment_step_limit(tmp_path: Path) -> None:
    """步数上限取消后循环不再暂停，脚本耗尽时以 provider 失败收尾；传参：临时目录；返回：无。"""
    project = _project_with_tools(tmp_path)
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task("inspect tools twice")
    append_user_message(data_root, "session-test-budget", record.goal)
    lease = from_trigger(
        "user",
        task_id=record.task_id,
        capabilities={
            "fs": {
                "project_root": str(project),
                "read": [
                    str(project),
                    str(data_root),
                    str(project / ".reins" / "workspace"),
                ],
                "write": [str(data_root), str(project / ".reins" / "workspace")],
                "deny_read": ["*.pem", "*.key", ".env"],
            },
            "terminal": {"enabled": True, "allow_commands": []},
            "browser": {
                "enabled": True,
                "profile": "default",
                "deny_domains": [],
                "headless": True,
            },
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
        },
        max_steps=1,
    )
    context = RunContext(
        session_id="session-test-budget",
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": record.goal},
        capability_lease=lease,
        segment_id="user-test-budget",
    )
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
        ],
        protocol_mode="text_json",
    )
    loop = AgentLoop(
        data_root,
        llm_client=client,
        tool_registry=build_tool_registry(repo_root=project, data_root=data_root),
    )

    assert loop.run(context) is State.FAILED
    assert loop.last_output.startswith("MODEL_PROVIDER_ERROR")
    #  Run 收尾不代写 Task 终态，任务仍留在办
    assert store.require_task(record.task_id).status == "active"
    facts = RunFactStore(data_root).read_run(context.run_id)
    terminal = [fact for fact in facts if fact.get("event") == "run:lifecycle"][-1]
    assert terminal["lifecycle"] == "failed"
    assert terminal["reason"] == "provider_error"


def _project_with_tools(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    tools = project / "tools"
    (tools / "nested").mkdir(parents=True)
    (tools / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")
    (tools / "nested" / "beta.py").write_text("print('beta')\n", encoding="utf-8")
    return project


def _message_text(message: AgentMessage) -> str:
    """拼接一条消息里全部文本块，用于断言工具输出是否原样保留。"""
    return "".join(part.text for part in message.content if isinstance(part, TextPart))


def _fact_tool_name(row: dict[str, object]) -> str:
    tool = row.get("tool")
    if isinstance(tool, dict):
        return str(tool.get("name", ""))
    return str(row.get("tool_name", ""))


def _fact_tool_status(row: dict[str, object]) -> str:
    tool = row.get("tool")
    if isinstance(tool, dict):
        return str(tool.get("status", ""))
    return str(row.get("status", ""))
