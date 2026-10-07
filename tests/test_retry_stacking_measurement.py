from __future__ import annotations

from pathlib import Path

from runtime.tool_results import _to_run_tools_result
from runtime.agent_loop import AgentLoop
from runtime.lease import from_trigger
from runtime.watchdog import WatchdogDecision
from runtime.types import RunContext, RunToolsRequest, Trigger
from tasks.store import TaskStore
from tools.tool_registry import (
    Idempotent,
    TARGET_SCOPE_LOGICAL,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_AGENT,
    ToolDefinition,
    ToolRegistry,
    ToolRisk,
)
from tools.types import ToolError, ToolErrorCategory


class _Watchdog:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def run_tool_with_timeout(
        self, operation, *, cancellation=None, on_late=None
    ) -> object:
        return operation()  # type: ignore[operator]

    def reserve_tool_step(self, *, operation_id: str = "") -> WatchdogDecision:
        """记录一次工具派发；传参：操作身份；返回：放行决定，步数不再拦派发。"""
        return WatchdogDecision(False)

    def record_tool_failure(self, tool: str, args: dict[str, object]) -> None:
        del args
        self.failures.append(tool)


def _retryable_tool(call_count: list[int]) -> ToolDefinition:
    def executor(_args: dict[str, object]) -> object:
        call_count.append(1)
        return ToolError(ToolErrorCategory.TRANSPORT, "transient transport")

    return ToolDefinition(
        name="always_transport_error",
        description="measurement fake",
        parameters={},
        toolset=TOOLSET_AGENT,
        risk_level=ToolRisk.SAFE,
        readonly=True,
        target_scope_rule=TARGET_SCOPE_LOGICAL,
        source=TOOL_SOURCE_BUILTIN,
        idempotent=Idempotent.YES,
        executor=executor,
    )


def test_retry_stacking_attempt_count_baseline(tmp_path: Path) -> None:
    registry = ToolRegistry()
    tool_calls: list[int] = []
    registry.register(_retryable_tool(tool_calls))

    raw_result = registry.execute_tool(
        "always_transport_error",
        {},
        from_trigger("user", task_id="task"),
        watchdog=_Watchdog(),
    )

    recovery_counts = _measure_recovery_counts(raw_result, tmp_path)

    assert len(tool_calls) == 4
    assert recovery_counts == [1, 2, 3, 4, 5, 6]
    assert recovery_counts[-1] > AgentLoop().recovery_policy.recoverable_error_repeats
    assert len(tool_calls) * recovery_counts[-1] == 24


def _measure_recovery_counts(raw_result: object, tmp_path: Path) -> list[int]:
    loop = AgentLoop(tmp_path / "data")
    request = RunToolsRequest(
        action="always_transport_error",
        tool_name="always_transport_error",
    )
    result = _to_run_tools_result(request, raw_result)
    context = _run_context(tmp_path)

    counts: list[int] = []
    for _ in range(6):
        observation = loop._handle_recoverable_tool_error(context, result)
        counts.append(int(observation.meta["budget_count"]))
    return counts


def _run_context(tmp_path: Path) -> RunContext:
    store = TaskStore(tmp_path / "data")
    record = store.create_task("measure retry stacking")
    return RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": "measure retry stacking"},
        capability_lease=from_trigger("user", task_id=record.task_id),
    )
