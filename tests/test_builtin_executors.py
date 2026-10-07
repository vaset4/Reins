from __future__ import annotations

from pathlib import Path

from runtime.lease import from_trigger
from runtime.watchdog import Watchdog
from tools import builtin_tools
from tools.tool_registry import ToolRegistry
from tools.types import ToolError, ToolErrorCategory
from tools.write_file_tools import WriteFileToolExecutor


def test_model_visible_builtin_tools_have_executors() -> None:
    registry = ToolRegistry()

    builtin_tools.register_tools(registry)

    missing = [
        item.name
        for item in registry.list_definitions(model_visible_only=True)
        if item.executor is None and not item.runtime_action
    ]
    assert missing == []


def test_file_write_executor_happy_and_invalid_path(
    monkeypatch, tmp_path: Path
) -> None:
    from approval import ApprovalDecision

    registry = ToolRegistry()
    monkeypatch.setattr(
        builtin_tools, "_WRITE_FILE_TOOLS", WriteFileToolExecutor(tmp_path)
    )
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.TASK,
    )
    builtin_tools.register_tools(registry)
    lease = from_trigger(
        "user",
        task_id="task",
        capabilities={
            "fs": {"project_root": str(tmp_path), "read": [], "write": []},
            "terminal": {"enabled": True, "allow_commands": []},
            "browser": {"enabled": True, "profile": "default"},
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
        },
    )
    target = tmp_path / "out.txt"

    result = registry.execute_tool(
        "file_write",
        {"path": str(target), "content": "hello"},
        lease,
        watchdog=Watchdog(lease, data_root=tmp_path / "data"),
    )

    assert not isinstance(result, ToolError)
    assert target.read_text(encoding="utf-8") == "hello"
    denied = registry.execute_tool(
        "file_write",
        {"path": str(tmp_path.parent / "out.txt"), "content": "no"},
        lease,
        watchdog=Watchdog(lease, data_root=tmp_path / "data"),
    )
    assert isinstance(denied, ToolError)
    assert denied.category is ToolErrorCategory.PERMISSION


def test_agent_and_todo_executors(monkeypatch, tmp_path: Path) -> None:
    """正式注册表拒绝脱离运行的提问并保存统一待办动作；参数：替换器和临时根；返回：无。"""
    from approval import ApprovalDecision
    from contextlib import closing
    from tasks.store import TaskStore

    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.TASK,
    )
    registry = ToolRegistry()
    builtin_tools.register_tools(registry, repo_root=tmp_path, data_root=tmp_path)
    with closing(TaskStore(tmp_path)) as tasks:
        tasks.create_task("检查待办执行", task_id="task-1")
    lease = from_trigger("user", task_id="task-1")
    watchdog = Watchdog(lease, data_root=tmp_path)

    ask = registry.execute_tool(
        "ask_user",
        {"question": "Continue?"},
        lease,
        watchdog=watchdog,
    )
    added = registry.execute_tool(
        "todo",
        {"action": "add", "content": "first"},
        lease,
        watchdog=watchdog,
    )
    items = registry.execute_tool("todo", {"action": "list"}, lease, watchdog=watchdog)

    assert isinstance(ask, ToolError) and "active session runtime" in ask.message
    assert added == {"idx": 0, "content": "first", "status": "pending"}
    assert items == [{"idx": 0, "content": "first", "status": "pending"}]
    registry.close()


def test_delegate_requires_an_active_session_runtime(
    monkeypatch, tmp_path: Path
) -> None:
    from approval import ApprovalDecision
    from runtime.watchdog import Watchdog

    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: ApprovalDecision.ONCE,
    )
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)

    result = registry.execute_tool(
        "delegate",
        {"task": "inspect repository state", "name": "inspector"},
        from_trigger("user", task_id="task-1"),
        watchdog=Watchdog(from_trigger("user", task_id="task-1"), data_root=tmp_path),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.INVALID_INPUT
    assert "active session runtime" in result.message
