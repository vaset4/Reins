from __future__ import annotations

from pathlib import Path

from runtime.lease import from_trigger
from runtime.watchdog import WatchdogDecision
from tools import builtin_tools
from tools.tool_registry import ToolRegistry
from tools.types import ToolError, ToolErrorCategory


# S1-60: terminal/code_execution 经唯一闸门 tool_registry.execute_tool 时，
# 其 fs 目标（terminal cwd / code_execution 落盘 cwd）必须真正过 path_security
# 的 deny 列表 + workspace 限定，而非仅靠确认弹窗（弹窗在 cron/permanent_grant
# 下会被跳过，且从不阻止 .ssh/.env 读取）。这些测试钉住边界 fail-closed。


class _Watchdog:
    data_root = None
    tool_timeout_seconds = 2.0

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def run_tool_with_timeout(
        self, operation, *, cancellation=None, on_late=None
    ) -> object:
        self.events.append("executor")
        return operation()  # type: ignore[operator]

    def reserve_tool_step(self, *, operation_id: str = "") -> WatchdogDecision:
        """记录一次工具派发；传参：操作身份；返回：放行决定，步数不再拦派发。"""
        return WatchdogDecision(False)

    def record_tool_failure(self, _tool: str, _args: dict[str, object]) -> None:
        return None


def _workspace_lease(project: Path, task_id: str = "task-1"):
    workspace = project / ".reins" / "workspace"
    data = project / ".reins" / "data"
    return from_trigger(
        "user",
        task_id=task_id,
        capabilities={
            "fs": {
                "project_root": str(project),
                "read": [str(project), str(data), str(workspace)],
                "write": [str(data), str(workspace)],
                "deny_read": [
                    "%USERPROFILE%\\.ssh\\",
                    "%USERPROFILE%\\.aws\\",
                    "*.pem",
                    "*.key",
                    ".env",
                ],
            },
            "terminal": {"enabled": True, "allow_commands": []},
            "browser": {"enabled": True, "profile": "default", "deny_domains": []},
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
        },
    )


def test_terminal_cwd_denied_outside_workspace(monkeypatch, tmp_path: Path) -> None:
    # cwd 指向工作区外的目录（lease workspace 限定下应 DENY），且确认弹窗不该被触达。
    events: list[str] = []
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: events.append("approval"),
    )
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)
    project = tmp_path / "project"
    (project / ".reins" / "workspace" / "task-1").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()

    result = registry.execute_tool(
        "terminal_tool",
        {"command": "echo hi", "cwd": str(outside)},
        _workspace_lease(project),
        watchdog=_Watchdog(events),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "path_security_deny"
    assert "approval" not in events
    assert "executor" not in events


def test_terminal_cwd_denied_for_ssh_dir(monkeypatch, tmp_path: Path) -> None:
    # cwd 指向 ~/.ssh（敏感 deny 目录）→ DENY，不进确认弹窗兜底。
    events: list[str] = []
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: events.append("approval"),
    )
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    ssh_dir = home / ".ssh"
    ssh_dir.mkdir(parents=True)
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)
    project = tmp_path / "project"
    (project / ".reins" / "workspace" / "task-1").mkdir(parents=True)

    result = registry.execute_tool(
        "terminal_tool",
        {"command": "echo hi", "cwd": str(ssh_dir)},
        _workspace_lease(project),
        watchdog=_Watchdog(events),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "path_security_deny"
    assert "approval" not in events


def test_code_execution_takes_no_caller_supplied_working_directory(
    monkeypatch, tmp_path: Path
) -> None:
    # S1-60 原来担心「脚本落盘的 cwd 能越界」。现在这个担心结构上不成立：
    # 工具不声明任何路径参数，cwd 一律由 lease 的工作区推出来，
    # 模型没有任何入口把执行目录指到工作区外，所以这里钉住的是「入口不存在」本身。
    events: list[str] = []
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: events.append("approval"),
    )
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)
    project = tmp_path / "project"
    (project / ".reins" / "workspace" / "task-1").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()

    definition = registry.get("code_execution_tool")
    assert definition is not None
    properties = definition.parameters.get("properties")
    assert isinstance(properties, dict)
    assert set(properties) == {"code", "timeout_seconds", "scope"}

    result = registry.execute_tool(
        "code_execution_tool",
        {"code": "print('x')", "cwd": str(outside)},
        _workspace_lease(project),
        watchdog=_Watchdog(events),
    )

    # 额外塞一个外部目录只会被判成非法参数，既跑不出去，也进不了审批
    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.INVALID_INPUT
    assert "cwd" in result.message
    assert "approval" not in events
    assert "executor" not in events


def test_code_execution_without_a_lease_workspace_refuses_to_run(
    tmp_path: Path,
) -> None:
    # 另一半：cwd 是从 lease 推的，lease 里没有工作区时明确拒绝，
    # 不能回落到进程当前目录去执行模型给的脚本
    events: list[str] = []
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)
    lease = from_trigger("user", task_id="task-1", capabilities={})

    result = registry.execute_tool(
        "code_execution_tool",
        {"code": "print('x')"},
        lease,
        watchdog=_Watchdog(events),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "code_execution_no_workspace"


def test_exec_boundary_denies_before_cron_permanent_grant(
    monkeypatch, tmp_path: Path
) -> None:
    # cron + permanent_grant 路径仍须先过 exec_boundary DENY（不被 grant 跳过）。
    events: list[str] = []
    monkeypatch.setattr(
        "tools.tool_registry._has_permanent_grant",
        lambda _lease, _tool, _args: events.append("grant") or True,
    )
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: events.append("approval"),
    )
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)
    project = tmp_path / "project"
    (project / ".reins" / "workspace" / "task-1").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    lease = from_trigger(
        "cron",
        task_id="task-1",
        capabilities={
            "fs": {
                "project_root": str(project),
                "read": [str(project)],
                "write": [str(project / ".reins" / "workspace")],
            },
            "terminal": {"enabled": True, "allow_commands": []},
            "browser": {"enabled": True, "profile": "default", "deny_domains": []},
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
        },
    )

    result = registry.execute_tool(
        "terminal_tool",
        {"command": "echo hi", "cwd": str(outside)},
        lease,
        watchdog=_Watchdog(events),
    )

    assert isinstance(result, ToolError)
    assert result.message == "path_security_deny"
    assert "grant" not in events
    assert "approval" not in events
