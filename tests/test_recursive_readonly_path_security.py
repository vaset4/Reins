from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from runtime.lease import from_trigger
from runtime.watchdog import WatchdogDecision
from tools.builtin_tools import build_tool_registry
from tools.types import ToolError, ToolErrorCategory

BLOCKED_FILE = "blocked_secret.txt"
BLOCKED_TEXT = "SENTINEL_BLOCKED_TEXT"


class _Watchdog:
    tool_timeout_seconds = 30.0
    data_root = None

    def run_tool_with_timeout(
        self, operation: Callable[[], object], *, cancellation=None, on_late=None
    ) -> object:
        return operation()

    def reserve_tool_step(self, *, operation_id: str = "") -> WatchdogDecision:
        """记录一次工具派发；传参：操作身份；返回：放行决定，步数不再拦派发。"""
        return WatchdogDecision(False)

    def record_tool_failure(self, tool: str, args: dict[str, object]) -> None:
        del tool, args


def test_recursive_readonly_tools_fail_closed_on_denied_child_path(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    allowed = project / "allowed"
    data_root = project / ".reins" / "data"
    allowed.mkdir(parents=True)
    data_root.mkdir(parents=True)
    (allowed / "visible.txt").write_text("visible text", encoding="utf-8")
    (allowed / BLOCKED_FILE).write_text(BLOCKED_TEXT, encoding="utf-8")

    registry = build_tool_registry(repo_root=project, data_root=data_root)
    lease = from_trigger(
        "user",
        task_id="task",
        capabilities={
            "fs": {
                "project_root": str(project),
                "read": [str(allowed), str(data_root)],
                "write": [str(data_root)],
                "deny_read": [BLOCKED_FILE],
            }
        },
    )

    file_result = registry.execute_tool(
        "file_read",
        {"path": f"allowed/{BLOCKED_FILE}"},
        lease,
        watchdog=_Watchdog(),
    )
    grep_result = registry.execute_tool(
        "grep",
        {"path": "allowed", "query": BLOCKED_TEXT},
        lease,
        watchdog=_Watchdog(),
    )
    find_result = registry.execute_tool(
        "find_path",
        {"path": "allowed", "query": BLOCKED_FILE},
        lease,
        watchdog=_Watchdog(),
    )

    for result in (file_result, grep_result, find_result):
        assert isinstance(result, ToolError)
        assert result.category is ToolErrorCategory.PERMISSION

    assert BLOCKED_TEXT not in str(grep_result)
    assert BLOCKED_FILE not in str(find_result)


def test_recursive_readonly_tools_still_return_allowed_children(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    allowed = project / "allowed"
    data_root = project / ".reins" / "data"
    allowed.mkdir(parents=True)
    data_root.mkdir(parents=True)
    (allowed / "visible.txt").write_text("VISIBLE_SENTINEL", encoding="utf-8")

    registry = build_tool_registry(repo_root=project, data_root=data_root)
    lease = from_trigger(
        "user",
        task_id="task",
        capabilities={
            "fs": {
                "project_root": str(project),
                "read": [str(allowed), str(data_root)],
                "write": [str(data_root)],
                "deny_read": [BLOCKED_FILE],
            }
        },
    )

    grep_result = registry.execute_tool(
        "grep",
        {"path": "allowed", "query": "VISIBLE_SENTINEL"},
        lease,
        watchdog=_Watchdog(),
    )
    find_result = registry.execute_tool(
        "find_path",
        {"path": "allowed", "query": "visible.txt"},
        lease,
        watchdog=_Watchdog(),
    )

    assert isinstance(grep_result, dict)
    assert "VISIBLE_SENTINEL" in str(grep_result["content"])
    assert isinstance(find_result, dict)
    assert "visible.txt" in str(find_result["content"])
