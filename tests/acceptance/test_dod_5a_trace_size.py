from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final

from contextlib import closing
from pathlib import Path

import pytest

from approval import ApprovalDecision
from llm.messages import ToolCallPart, ToolResultMessage
from runtime.agent_loop import AgentLoop, State
from runtime.session_messages import (
    materialize_messages,
    append_user_message,
    read_history_rows,
)
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tests.acceptance.helpers.sandbox_fixtures import (
    SandboxProject,
    create_sandbox_project,
    lease_for,
)
from tests.support.approval import install_approval
from tools import builtin_tools
from tools.memory_tools import MemoryToolExecutor
from tools.readonly_file_tools import ReadOnlyFileToolExecutor
from tools.readonly_inspection import ReadOnlyInspectionExecutor
from tools.readonly_web_tools import ReadOnlyWebToolExecutor
from tools.tool_registry import ToolRegistry
from tools.write_file_tools import WriteFileToolExecutor

_SESSION_ID = "session-dod-5a-trace-size"
_MAX_TRACE_BYTES = 10 * 1024 * 1024


def test_dod_5a_trace_size_50_steps_under_10mb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """五十次真实工具执行保存完整关联，记录体积受控；传参：临时根、依赖替换；返回：无。"""
    sandbox = create_sandbox_project(tmp_path, monkeypatch, task_id="dod5a-50step")
    _configure_builtins(monkeypatch, sandbox)
    install_approval(monkeypatch, lambda _req: ApprovalDecision.ONCE)
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)
    _expose_loaded(registry, "todo", "memory_note", "web_fetch")
    steps = _build_step_calls(sandbox)
    assert len(steps) == 50
    calls = [
        ToolCallPart(f"dod5a-{index:03d}", tool, args)
        for index, (tool, args) in enumerate(steps)
    ]
    client = from_test_native_tool_then_final(calls, "已核对五十项执行结果")
    context = RunContext(
        task_id=sandbox.task_id,
        trigger=Trigger.USER,
        session_id=_SESSION_ID,
        payload={"message": "读取、写入、网页和记忆操作"},
        capability_lease=lease_for(sandbox, max_steps=100),
    )
    WorkspaceStore(sandbox.data_root).bind_session(_SESSION_ID, sandbox.project_root)
    append_user_message(sandbox.data_root, _SESSION_ID, context.payload["message"])
    loop = AgentLoop(sandbox.data_root, llm_client=client, tool_registry=registry)
    assert loop.run(context) is State.DONE
    results = [
        message
        for message in materialize_messages(sandbox.data_root, _SESSION_ID)
        if isinstance(message, ToolResultMessage)
    ]
    assert len(results) == 50
    assert all(result.status == "success" for result in results), [
        (result.tool_name, result.content)
        for result in results
        if result.status != "success"
    ]
    with closing(TaskStore(sandbox.data_root)) as store:
        store.update_summary(
            sandbox.task_id, "# Summary\n\nDoD 5a trace-size summary from live store."
        )
        assert "trace-size summary from live store" in store.read_summary(
            sandbox.task_id
        )
    size_bytes = sum(
        path.stat().st_size for path in sandbox.data_root.rglob("*") if path.is_file()
    )
    assert size_bytes < _MAX_TRACE_BYTES
    task_dir = sandbox.data_root / "tasks" / sandbox.task_id
    assert not (task_dir / "trajectory.jsonl").exists()
    assert len(read_history_rows(sandbox.data_root, _SESSION_ID)) >= 50
    assert 1 <= len(read_history_rows(sandbox.data_root, _SESSION_ID, limit=20)) <= 20


def _expose_loaded(registry: ToolRegistry, *names: str) -> None:
    """把按需工具直接放进本轮目录；传参：目录与工具名；返回：无。

    本用例检验的是五十次真实执行下的记录体积，不是发现流程；按需工具要先经
    capabilities 加载才进模型可见目录，这里直接按已加载发布。
    """
    definitions = []
    for name in names:
        definition = registry.get(name)
        assert definition is not None, name
        definition.deferred = False
        definitions.append(definition)
    registry.publish(definitions, replace_names=names)


def _configure_builtins(
    monkeypatch: pytest.MonkeyPatch,
    sandbox: SandboxProject,
) -> None:
    inspection = ReadOnlyInspectionExecutor(sandbox.project_root, 50, 4000, 50)
    monkeypatch.setattr(builtin_tools, "_INSPECTION", inspection)
    monkeypatch.setattr(
        builtin_tools, "_FILE_TOOLS", ReadOnlyFileToolExecutor(inspection)
    )
    monkeypatch.setattr(
        builtin_tools, "_WRITE_FILE_TOOLS", WriteFileToolExecutor(sandbox.project_root)
    )
    monkeypatch.setattr(
        builtin_tools, "_WEB_TOOLS", ReadOnlyWebToolExecutor(fetcher=_mock_fetcher)
    )
    monkeypatch.setattr(
        builtin_tools, "_MEMORY_TOOLS", MemoryToolExecutor(sandbox.data_root)
    )


def _mock_fetcher(url: str) -> str:
    return (
        "<html><head><title>DoD5A</title></head>"
        f"<body><h1>Fetched</h1><p>{url}</p></body></html>"
    )


def _build_step_calls(sandbox: SandboxProject) -> list[tuple[str, dict[str, object]]]:
    """构造五十项实际文件、网页、待办和记忆操作；参数：隔离项目；返回：当前工具调用。"""
    steps: list[tuple[str, dict[str, object]]] = []
    workspace = sandbox.workspace_scratch
    files = [workspace / f"dod5a_{idx}.txt" for idx in range(5)]

    for idx, target in enumerate(files):
        steps.append(
            ("file_write", {"path": str(target), "content": f"step-{idx} from dod5a"})
        )

    for idx in range(25):
        target = files[idx % len(files)]
        phase = idx % 3
        if phase == 0:
            steps.append(("file_read", {"path": str(target)}))
        elif phase == 1:
            steps.append(("list", {"path": str(workspace)}))
        else:
            steps.append(("grep", {"path": str(workspace), "query": "step-"}))

    for idx in range(10):
        steps.append(("web_fetch", {"url": f"https://example.com/dod5a/{idx}"}))

    steps.extend(
        [
            ("todo", {"action": "add", "content": "DoD5A todo item 1"}),
            ("todo", {"action": "add", "content": "DoD5A todo item 2"}),
            ("todo", {"action": "list"}),
            ("todo", {"action": "update", "idx": 0, "status": "in_progress"}),
            (
                "memory_note",
                {"note": "DoD5A memory note one", "scope": "working_memory"},
            ),
            ("memory_query", {"action": "search", "query": "DoD5A"}),
            ("todo", {"action": "update", "idx": 1, "status": "done"}),
            ("todo", {"action": "list", "filter": "pending"}),
            (
                "memory_note",
                {"note": "DoD5A memory note two", "scope": "working_memory"},
            ),
            ("memory_query", {"action": "search", "query": "note"}),
        ]
    )
    return steps
