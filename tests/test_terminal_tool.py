from __future__ import annotations

from pathlib import Path

from runtime.lease import from_trigger
from runtime.watchdog import WatchdogDecision
from tools import builtin_tools
from tools.terminal_tool import executor
from tools.tool_registry import ToolRegistry
from tools.types import ToolError, ToolErrorCategory


class _Watchdog:
    data_root = None
    tool_timeout_seconds = 2.0

    def run_tool_with_timeout(
        self, operation, *, cancellation=None, on_late=None
    ) -> object:
        return operation()  # type: ignore[operator]

    def reserve_tool_step(self, *, operation_id: str = "") -> WatchdogDecision:
        """记录一次工具派发；传参：操作身份；返回：放行决定，步数不再拦派发。"""
        return WatchdogDecision(False)

    def record_tool_failure(self, _tool: str, _args: dict[str, object]) -> None:
        return None


def _workspace_lease(project: Path, *, allow_commands: list[str] | None = None):
    workspace = project / ".reins" / "workspace"
    (workspace / "task").mkdir(parents=True, exist_ok=True)
    return from_trigger(
        "user",
        task_id="task",
        capabilities={
            "fs": {
                "project_root": str(project),
                "read": [str(project), str(workspace)],
                "write": [str(workspace)],
            },
            "terminal": {
                "enabled": True,
                "allow_commands": allow_commands or [],
            },
            "browser": {"enabled": True, "profile": "default", "deny_domains": []},
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
        },
    )


def test_terminal_denies_default_blacklist() -> None:
    lease = from_trigger("user", task_id="task")

    result = executor({"command": r"del /f /s /q C:\\", "__lease__": lease})

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "terminal_command_denied"


def test_terminal_runs_via_exec_channel_in_workspace(tmp_path: Path) -> None:
    project = tmp_path / "project"
    lease = _workspace_lease(project)

    result = executor(
        {
            "command": "cd",
            "__lease__": lease,
            "__timeout_seconds__": 5,
        }
    )

    assert isinstance(result, dict)
    assert result["exit_code"] == 0
    reported = Path(str(result["stdout"]).strip()).resolve()
    expected = (project / ".reins" / "workspace" / "task").resolve()
    assert reported == expected


def test_terminal_uses_requested_project_directory(tmp_path: Path) -> None:
    """已有cwd参数必须选中实际目录而非静默忽略；参数：隔离根；返回：无。"""
    project = tmp_path / "project"
    lease = _workspace_lease(project)
    requested = project / "source files"
    requested.mkdir()
    result = executor({"command": "cd", "cwd": "source files", "__lease__": lease})
    assert isinstance(result, dict)
    assert result["exit_code"] == 0
    assert Path(str(result["stdout"]).strip()).resolve() == requested.resolve()


def test_terminal_disabled_capability_denied(tmp_path: Path) -> None:
    project = tmp_path / "project"
    lease = _workspace_lease(project)
    object.__setattr__(
        lease,
        "capabilities",
        {**lease.capabilities, "terminal": {"enabled": False}},
    )

    result = executor({"command": "echo hi", "__lease__": lease})

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION


def test_terminal_denies_each_combined_command_segment(tmp_path: Path) -> None:
    project = tmp_path / "project"
    lease = _workspace_lease(project)
    object.__setattr__(
        lease,
        "capabilities",
        {
            **lease.capabilities,
            "terminal": {"enabled": True, "deny_commands": ["whoami"]},
        },
    )

    result = executor({"command": "echo ok & whoami", "__lease__": lease})

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "terminal_command_denied"


def test_terminal_denies_grouped_command_segment(tmp_path: Path) -> None:
    project = tmp_path / "project"
    lease = _workspace_lease(project)
    object.__setattr__(
        lease,
        "capabilities",
        {
            **lease.capabilities,
            "terminal": {"enabled": True, "deny_commands": ["whoami"]},
        },
    )

    result = executor({"command": "(whoami)", "__lease__": lease})

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "terminal_command_denied"


def test_terminal_allow_list_applies_to_each_combined_command_segment(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    lease = _workspace_lease(project, allow_commands=["echo*"])

    result = executor({"command": "echo ok & whoami", "__lease__": lease})

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "terminal_command_not_allowed"


def test_terminal_missing_lease_denied() -> None:
    result = executor({"command": "echo hi"})

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION


def test_terminal_confirm_tool_downgrades_in_cron() -> None:
    registry = ToolRegistry()
    builtin_tools.register_tools(registry)

    result = registry.execute_tool(
        "terminal_tool",
        {"command": "git log"},
        from_trigger("cron", task_id="task"),
        watchdog=_Watchdog(),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "requires_permanent_grant_in_cron_mode"
