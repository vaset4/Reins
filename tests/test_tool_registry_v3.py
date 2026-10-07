from __future__ import annotations

from pathlib import Path

import pytest

from runtime.lease import from_trigger
from runtime.watchdog import WatchdogDecision
from tools.builtin_tools import build_tool_registry
from tools.types import ToolError, ToolErrorCategory
from tools.tool_registry import (
    Idempotent,
    TARGET_SCOPE_LOGICAL,
    TARGET_SCOPE_PATH,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_AGENT,
    ToolDefinition,
    ToolRegistry,
    ToolRisk,
    get_default_tool_registry,
)


class _Watchdog:
    def __init__(self, events: list[str], data_root: Path | None = None) -> None:
        self.events = events
        self.data_root = data_root

    def run_tool_with_timeout(
        self, operation: object, *, cancellation=None, on_late=None
    ) -> object:
        self.events.append("watchdog")
        return operation()  # type: ignore[operator]

    def reserve_tool_step(self) -> WatchdogDecision:
        """测试桩提供真实派发所需预算接口；传参：无；返回：允许执行。"""
        return WatchdogDecision(False)

    def record_tool_failure(self, tool: str, args: dict[str, object]) -> None:
        self.events.append(f"failure:{tool}:{args}")


def _definition(
    *,
    name: str = "tool",
    risk: ToolRisk = ToolRisk.SAFE,
    idempotent: Idempotent | None = Idempotent.YES,
    target_scope_rule: str = TARGET_SCOPE_LOGICAL,
    readonly: bool = True,
    executor: object = None,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="tool",
        parameters={"path": {"type": "string"}},
        toolset=TOOLSET_AGENT,
        risk_level=risk,
        readonly=readonly,
        target_scope_rule=target_scope_rule,
        source=TOOL_SOURCE_BUILTIN,
        idempotent=idempotent,
        executor=executor,  # type: ignore[arg-type]
    )


def test_register_requires_idempotent() -> None:
    with pytest.raises(ValueError, match="idempotent"):
        ToolRegistry().register(_definition(idempotent=None))


def test_risk_and_idempotent_enums_register_cleanly() -> None:
    registry = ToolRegistry()
    registry.register(_definition(name="safe_tool", risk=ToolRisk.SAFE))
    registry.register(
        _definition(
            name="confirm_tool",
            risk=ToolRisk.CONFIRM,
            idempotent=Idempotent.CONDITIONAL,
        )
    )
    registry.register(
        _definition(name="deny_tool", risk=ToolRisk.DENY, idempotent=Idempotent.NO)
    )

    assert registry.get("safe_tool").risk is ToolRisk.SAFE
    assert registry.get("confirm_tool").idempotent is Idempotent.CONDITIONAL
    assert registry.get("deny_tool").risk is ToolRisk.DENY


def test_legacy_risk_values_warn_and_normalize() -> None:
    registry = ToolRegistry()

    with pytest.warns(DeprecationWarning):
        registry.register(_definition(name="legacy", risk="readonly"))  # type: ignore[arg-type]

    assert registry.get("legacy").risk is ToolRisk.SAFE


def test_default_tools_are_migrated_to_risk_and_idempotent() -> None:
    registry = get_default_tool_registry()
    definitions = registry.list_definitions()

    assert len([item for item in definitions if item.risk is not None]) >= 10
    assert registry.get("file_read").risk is ToolRisk.SAFE
    assert registry.get("file_read").idempotent is Idempotent.YES
    assert registry.get("file_write").risk is ToolRisk.CONFIRM
    assert registry.get("file_write").idempotent is Idempotent.CONDITIONAL


def test_execute_tool_denies_path_before_approval(monkeypatch, tmp_path: Path) -> None:
    from path_security import Decision

    events: list[str] = []
    registry = ToolRegistry()
    registry.register(
        _definition(
            name="file_write",
            risk=ToolRisk.CONFIRM,
            idempotent=Idempotent.CONDITIONAL,
            target_scope_rule=TARGET_SCOPE_PATH,
            readonly=False,
            executor=lambda _args: events.append("executor"),
        )
    )
    monkeypatch.setattr(
        "tools.tool_registry.path_security.check_write",
        lambda _path, _lease, *, filtered=False: events.append("path") or Decision.DENY,
    )
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _req: events.append("approval"),
    )

    result = registry.execute_tool(
        "file_write",
        {"path": str(tmp_path / "x.txt")},
        from_trigger("user", task_id="task"),
        watchdog=_Watchdog(events, data_root=tmp_path),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert events == ["path"]


def test_execute_tool_main_path_order_and_failure_record(
    monkeypatch, tmp_path: Path
) -> None:
    from approval import ApprovalDecision
    from path_security import Decision

    events: list[str] = []
    approval_roots: list[Path] = []
    registry = ToolRegistry()
    registry.register(
        _definition(
            name="file_write",
            risk=ToolRisk.CONFIRM,
            idempotent=Idempotent.CONDITIONAL,
            target_scope_rule=TARGET_SCOPE_PATH,
            readonly=False,
            executor=lambda _args: events.append("executor") or "ok",
        )
    )
    monkeypatch.setattr(
        "tools.tool_registry.path_security.check_write",
        lambda _path, _lease, *, filtered=False: (
            events.append("path") or Decision.CONFIRM
        ),
    )
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda req: (
            approval_roots.append(req.data_root)
            or events.append("approval")
            or ApprovalDecision.TASK
        ),
    )

    result = registry.execute_tool(
        "file_write",
        {"path": str(tmp_path / "x.txt")},
        from_trigger(
            "user", task_id="task", capabilities={"fs": {"project_root": str(tmp_path)}}
        ),
        watchdog=_Watchdog(events, data_root=tmp_path),
    )

    assert result == "ok"
    assert events == ["path", "approval", "watchdog", "executor"]
    assert approval_roots == [tmp_path.resolve()]

    events.clear()
    registry = ToolRegistry()
    registry.register(
        _definition(
            name="file_write",
            risk=ToolRisk.CONFIRM,
            idempotent=Idempotent.CONDITIONAL,
            target_scope_rule=TARGET_SCOPE_PATH,
            readonly=False,
            executor=lambda _args: ToolError(ToolErrorCategory.UNKNOWN, "failed"),
        )
    )

    result = registry.execute_tool(
        "file_write",
        {"path": str(tmp_path / "x.txt")},
        from_trigger(
            "user", task_id="task", capabilities={"fs": {"project_root": str(tmp_path)}}
        ),
        watchdog=_Watchdog(events, data_root=tmp_path),
    )

    assert isinstance(result, ToolError)
    assert events[:3] == ["path", "approval", "watchdog"]
    assert len(events) == 4
    assert events[3].startswith("failure:file_write:")


def test_prepared_execution_separates_approval_from_single_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 prepare 完成审批但不执行，prepared 只能消费一次

    作者：LKX
    时间：2026-08-16 00:00:00
    传参：monkeypatch 替换安全与审批边界；tmp_path 提供授权路径
    返回：无；断言 approval/executor 顺序和一次性消费
    """
    from approval import ApprovalDecision
    from path_security import Decision

    events: list[str] = []
    registry = ToolRegistry()
    registry.register(
        _definition(
            risk=ToolRisk.CONFIRM,
            target_scope_rule=TARGET_SCOPE_PATH,
            readonly=False,
            executor=lambda _args: events.append("executor") or "ok",
        )
    )
    monkeypatch.setattr(
        "tools.tool_registry.path_security.check_write",
        lambda _path, _lease, *, filtered=False: (
            events.append("path") or Decision.CONFIRM
        ),
    )
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _request: events.append("approval") or ApprovalDecision.ONCE,
    )
    watchdog = _Watchdog(events, data_root=tmp_path / "data")

    prepared = registry.prepare_tool_execution(
        "tool",
        {"path": str(tmp_path / "x.txt")},
        from_trigger(
            "user",
            task_id="task-1",
            capabilities={"fs": {"project_root": str(tmp_path)}},
        ),
        watchdog=watchdog,
    )

    assert not isinstance(prepared, ToolError)
    assert events == ["path", "approval"]
    result = registry.execute_prepared_tool(prepared)
    assert isinstance(result, dict) and result["content"] == "ok"
    assert result["meta"]["restore_point_ids"]
    assert events == ["path", "approval", "watchdog", "executor"]
    with pytest.raises(RuntimeError, match="already consumed"):
        registry.execute_prepared_tool(prepared)


def test_retry_only_when_retryable_and_idempotent() -> None:
    registry = ToolRegistry()
    calls = 0

    def flaky(_args: dict[str, object]) -> object:
        nonlocal calls
        calls += 1
        if calls < 4:
            return ToolError(ToolErrorCategory.TRANSPORT, "try again")
        return "ok"

    registry.register(
        _definition(
            name="file_write",
            risk=ToolRisk.SAFE,
            idempotent=Idempotent.YES,
            executor=flaky,
        )
    )
    assert (
        registry.execute_tool(
            "file_write",
            {},
            from_trigger("user", task_id="task"),
            watchdog=_Watchdog([]),
        )
        == "ok"
    )
    assert calls == 4

    registry = ToolRegistry()
    calls = 0
    registry.register(
        _definition(
            name="terminal",
            risk=ToolRisk.SAFE,
            idempotent=Idempotent.NO,
            executor=flaky,
        )
    )

    result = registry.execute_tool(
        "terminal",
        {},
        from_trigger("user", task_id="task"),
        watchdog=_Watchdog([]),
    )

    assert isinstance(result, ToolError)
    assert calls == 1


def test_conditional_side_effect_failure_does_not_retry() -> None:
    """条件幂等标签不能证明副作用不存在；传参：无；返回：无，实际执行只发生一次。"""
    calls: list[bool] = []
    registry = ToolRegistry()
    registry.register(
        _definition(
            name="conditional",
            risk=ToolRisk.SAFE,
            idempotent=Idempotent.CONDITIONAL,
            executor=lambda _args: (
                calls.append(True)
                or ToolError(ToolErrorCategory.TRANSPORT, "effect unknown")
            ),
        )
    )
    result = registry.execute_tool(
        "conditional", {}, from_trigger("user", task_id="task"), watchdog=_Watchdog([])
    )
    assert isinstance(result, ToolError)
    assert calls == [True]


def test_confirm_tool_requires_active_data_root_before_approval(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    registry = ToolRegistry()
    registry.register(
        _definition(
            name="confirm_tool",
            risk=ToolRisk.CONFIRM,
            idempotent=Idempotent.CONDITIONAL,
            executor=lambda _args: events.append("executor") or "ok",
        )
    )
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _request: events.append("approval"),
    )

    result = registry.execute_tool(
        "confirm_tool",
        {},
        from_trigger("user", task_id="task"),
        watchdog=_Watchdog(events),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "approval_data_root_required"
    assert events == []

    invalid_root = tmp_path / "data-root-file"
    invalid_root.write_text("not a directory", encoding="utf-8")
    result = registry.execute_tool(
        "confirm_tool",
        {},
        from_trigger("user", task_id="task"),
        watchdog=_Watchdog(events, data_root=invalid_root),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "approval_data_root_required"
    assert events == []


def test_file_read_missing_file_is_invalid_input(tmp_path: Path) -> None:
    data_root = tmp_path / ".reins" / "data"
    registry = build_tool_registry(repo_root=tmp_path, data_root=data_root)
    lease = from_trigger(
        "user",
        task_id="task",
        capabilities={
            "fs": {
                "project_root": str(tmp_path),
                "read": [str(tmp_path), str(data_root)],
                "write": [str(data_root)],
            }
        },
    )

    result = registry.execute_tool(
        "file_read",
        {"path": "missing.txt"},
        lease,
        watchdog=_Watchdog([]),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.INVALID_INPUT
    assert "path not found" in result.message
    assert "missing.txt" in result.message


def _run_approval_outcome(
    monkeypatch, tmp_path: Path, outcome: object
) -> tuple[object, list[str]]:
    """把审批闸出口固定成给定结果后执行一次工具；传参：补丁器、临时根与出口；返回：结果与副作用记录。"""
    executed: list[str] = []

    def _decide(_req: object) -> object:
        """返回固定出口；传参：审批请求；返回：注入的决定，或抛出注入的异常。"""
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr("tools.tool_registry.approval.request_approval", _decide)
    registry = ToolRegistry()
    registry.register(
        _definition(
            name="file_write",
            risk=ToolRisk.CONFIRM,
            idempotent=Idempotent.CONDITIONAL,
            readonly=False,
            executor=lambda _args: executed.append("executor") or "ok",
        )
    )
    result = registry.execute_tool(
        "file_write",
        {"path": str(tmp_path / "x.txt")},
        from_trigger("user", task_id="task"),
        watchdog=_Watchdog([], data_root=tmp_path),
    )
    return result, executed


def test_approval_facility_failure_is_not_reported_as_unknown(
    monkeypatch, tmp_path: Path
) -> None:
    """审批设施故障报成设施故障而非未知，且不回传原始异常正文；传参：补丁器与临时根；返回：无。"""
    from approval import ApprovalUnavailable

    result, executed = _run_approval_outcome(
        monkeypatch,
        tmp_path,
        ApprovalUnavailable("approval backend failed: backend down"),
    )

    assert isinstance(result, ToolError)
    assert result.category is not ToolErrorCategory.UNKNOWN
    assert result.category is ToolErrorCategory.TRANSPORT
    assert result.details["approval_state"] == "unavailable"
    assert "backend down" not in result.message
    assert executed == []


def test_approval_outcomes_carry_distinct_states(monkeypatch, tmp_path: Path) -> None:
    """拒绝、中断、故障三种出口标识互不相同且都不执行工具；传参：补丁器与临时根；返回：无。"""
    from approval import ApprovalDecision, ApprovalUnavailable

    outcomes = {
        "denied": ApprovalDecision.DENY,
        "cancelled": ApprovalDecision.CANCELLED,
        "unavailable": ApprovalUnavailable("approval backend failed: down"),
    }
    categories: dict[str, ToolErrorCategory] = {}
    states: dict[str, object] = {}
    for name, outcome in outcomes.items():
        result, executed = _run_approval_outcome(monkeypatch, tmp_path, outcome)
        assert isinstance(result, ToolError)
        assert executed == [], f"{name} 不应执行工具"
        categories[name] = result.category
        states[name] = result.details.get("approval_state")

    assert all(isinstance(state, str) and state for state in states.values())
    assert len(set(states.values())) == 3
    assert categories["denied"] is ToolErrorCategory.PERMISSION
    assert categories["cancelled"] is ToolErrorCategory.CANCELLED


def test_denied_approval_stays_non_retryable_permission(
    monkeypatch, tmp_path: Path
) -> None:
    """用户拒绝仍是不可自行重试的权限拒绝且正文不变；传参：补丁器与临时根；返回：无。"""
    from approval import ApprovalDecision

    result, _executed = _run_approval_outcome(
        monkeypatch, tmp_path, ApprovalDecision.DENY
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.retryable is False
    assert result.message == "approval_denied"
