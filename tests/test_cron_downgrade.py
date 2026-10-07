from __future__ import annotations

from pathlib import Path

from runtime.lease import from_trigger
from runtime.watchdog import Watchdog
from tools.types import ToolError, ToolErrorCategory
from tools.tool_registry import (
    Idempotent,
    TARGET_SCOPE_LOGICAL,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_AGENT,
    ToolDefinition,
    ToolRegistry,
    ToolRisk,
)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="confirm_tool",
            description="confirm tool",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level=ToolRisk.CONFIRM,
            readonly=False,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source=TOOL_SOURCE_BUILTIN,
            idempotent=Idempotent.CONDITIONAL,
            executor=lambda _args: "ok",
        )
    )
    return registry


def test_cron_confirm_without_permanent_grant_returns_permission(monkeypatch) -> None:
    called = False

    def approval_backend(_req: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("cron should not wait for approval")

    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval", approval_backend
    )

    result = _registry().execute_tool(
        "confirm_tool",
        {},
        from_trigger("cron", task_id="task"),
        watchdog=Watchdog(from_trigger("cron", task_id="task")),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "requires_permanent_grant_in_cron_mode"
    assert called is False


def test_cron_confirm_with_required_permanent_grant_executes(tmp_path: Path) -> None:
    lease = from_trigger(
        "cron",
        task_id="task",
        capabilities={
            "fs": {"project_root": str(tmp_path), "read": [], "write": []},
            "terminal": {"enabled": True, "allow_commands": []},
            "browser": {"enabled": True, "profile": "default"},
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
            "schedule": {
                "required_permanent_grants": [
                    {"tool": "confirm_tool", "args": {}},
                ]
            },
        },
    )

    result = _registry().execute_tool(
        "confirm_tool", {}, lease, watchdog=Watchdog(lease, data_root=tmp_path / "data")
    )
    assert result["content"] == "ok"
    assert result["meta"]["restore_point_ids"]
    assert result["meta"]["restore_record_errors"] == []


def test_user_confirm_goes_through_approval(monkeypatch, tmp_path: Path) -> None:
    from approval import ApprovalDecision

    calls: list[str] = []

    def approval_backend(_req: object) -> ApprovalDecision:
        calls.append("approval")
        return ApprovalDecision.TASK

    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval", approval_backend
    )

    lease = from_trigger(
        "user", task_id="task", capabilities={"fs": {"project_root": str(tmp_path)}}
    )
    result = _registry().execute_tool(
        "confirm_tool",
        {},
        lease,
        watchdog=Watchdog(lease, data_root=tmp_path / "data"),
    )

    assert result["content"] == "ok"
    assert result["meta"]["restore_point_ids"]
    assert result["meta"]["restore_record_errors"] == []
    assert calls == ["approval"]
