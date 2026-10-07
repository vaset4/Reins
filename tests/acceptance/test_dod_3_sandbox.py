from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final

from collections.abc import Callable
from pathlib import Path

import pytest
from tasks.store import TaskStore

from approval import ApprovalDecision, ApprovalRequest, register_approval_backend
from llm.messages import ToolCallPart
from llm.types import TokenUsage
from runtime.agent_loop import AgentLoop, State
from runtime.run_facts import RunFactStore
from runtime.types import RunContext, Trigger
from runtime.watchdog import Watchdog
from tests.acceptance.helpers.sandbox_fixtures import (
    ApprovalRecorder,
    create_sandbox_project,
    lease_for,
    sandbox_registry,
    SandboxProject,
)
from tools.types import ToolError, ToolErrorCategory


@pytest.mark.parametrize(
    "path_factory",
    [
        lambda sandbox: Path("C:/Windows/System32/reins-test.txt"),
        lambda sandbox: sandbox.home / ".ssh" / "test",
        lambda sandbox: sandbox.home / "reins-test.pem",
    ],
)
def test_case_1_out_of_bounds_writes_are_denied_without_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_factory: Callable[[SandboxProject], Path],
) -> None:
    sandbox = create_sandbox_project(tmp_path, monkeypatch)
    calls: list[object] = []
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        _record_task_approval(calls),
    )

    target = path_factory(sandbox)
    result = sandbox_registry().execute_tool(
        "file_write",
        {"path": str(target), "content": "blocked"},
        lease_for(sandbox),
    )

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert "deny" in result.message
    assert calls == []


def test_case_2_workspace_write_is_allowed_without_approval_and_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = create_sandbox_project(tmp_path, monkeypatch)
    calls: list[object] = []
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        _record_task_approval(calls),
    )
    target = sandbox.workspace_scratch / "test.txt"
    registry = sandbox_registry()

    write_result = registry.execute_tool(
        "file_write",
        {"path": str(target), "content": "workspace ok"},
        lease_for(sandbox),
        watchdog=Watchdog(lease_for(sandbox), data_root=sandbox.data_root),
    )
    read_result = registry.execute_tool(
        "file_read", {"path": str(target)}, lease_for(sandbox)
    )

    assert write_result == "wrote"
    assert read_result == "workspace ok"
    assert calls == []


def test_case_3_project_source_write_confirms_once_then_uses_task_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = create_sandbox_project(tmp_path, monkeypatch)
    recorder = ApprovalRecorder(ApprovalDecision.TASK)
    register_approval_backend(recorder)
    args: dict[str, object] = {
        "path": str(sandbox.source_file),
        "content": "print('after')\n",
    }

    try:
        registry = sandbox_registry()
        lease = lease_for(sandbox)
        watchdog = Watchdog(lease, data_root=sandbox.data_root)
        assert (
            registry.execute_tool("file_write", args, lease, watchdog=watchdog)
            == "wrote"
        )
        assert (
            registry.execute_tool("file_write", args, lease, watchdog=watchdog)
            == "wrote"
        )
    finally:
        register_approval_backend(None)

    task_data = TaskStore(sandbox.data_root).load_task_payload(sandbox.task_id)
    assert len(recorder.calls) == 1
    assert task_data["grants"][-1]["tool"] == "file_write"
    assert task_data["grants"][-1]["scope"] == "task"


def test_case_4_cron_confirm_without_grant_fails_without_approval_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = create_sandbox_project(tmp_path, monkeypatch)
    calls: list[object] = []
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        _record_task_approval(calls),
    )
    registry = sandbox_registry()
    args: dict[str, object] = {
        "path": str(sandbox.source_file),
        "content": "cron write\n",
    }
    cron_lease = lease_for(sandbox, trigger="cron")

    result = registry.execute_tool("file_write", args, cron_lease)

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.PERMISSION
    assert result.message == "requires_permanent_grant_in_cron_mode"
    assert calls == []

    loop = AgentLoop(
        sandbox.data_root,
        llm_client=from_test_native_tool_then_final(
            [ToolCallPart("cron-write", "file_write", args)], "缺少授权，文件未改"
        ),
        tool_registry=registry,
    )
    context = RunContext(
        task_id=sandbox.task_id,
        trigger=Trigger.CRON,
        payload={"message": "执行已安排的文件修改"},
        capability_lease=cron_lease,
    )
    assert loop.run(context) is State.DONE
    facts = RunFactStore(sandbox.data_root).read_run(context.run_id)
    result_fact = next(row for row in facts if row["event"] == "tool:response")
    assert result_fact["tool"]["status"] == "error"
    assert result_fact["tool"]["meta"]["tool_error_category"] == "permission"
    assert not any(row.get("lifecycle") == "waiting_approval" for row in facts)
    assert sandbox.source_file.read_text(encoding="utf-8") == "print('before')\n"

    granted = lease_for(
        sandbox,
        trigger="cron",
        required_permanent_grants=[{"tool": "file_write", "args": args}],
    )
    assert (
        registry.execute_tool(
            "file_write",
            args,
            granted,
            watchdog=Watchdog(granted, data_root=sandbox.data_root),
        )
        == "wrote"
    )


def test_case_5_step_and_token_counters_record_without_pausing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mock_notification: object,
) -> None:
    sandbox = create_sandbox_project(tmp_path, monkeypatch)
    step_watchdog = Watchdog(
        lease_for(sandbox, max_steps=3),
        task_id=sandbox.task_id,
        data_root=sandbox.data_root,
        segment_id="user-step",
    )

    assert not step_watchdog.tick(steps_taken=3).paused
    assert not step_watchdog.tick(steps_taken=4).paused
    assert step_watchdog.steps_taken == 4

    token_watchdog = Watchdog(
        lease_for(sandbox, max_tokens=1000),
        task_id=sandbox.task_id,
        data_root=sandbox.data_root,
        segment_id="user-token",
    )
    result = token_watchdog.tick(
        token_usage=TokenUsage(input_tokens=750, output_tokens=750)
    )

    assert not result.paused
    assert token_watchdog.tokens_used == 1500
    # 用量只计数不暂停，因此不再产生暂停通知
    assert len(mock_notification.calls) == 0  # type: ignore[attr-defined]


def test_case_6_failure_count_does_not_replace_actual_dispatch_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = create_sandbox_project(tmp_path, monkeypatch)
    watchdog = Watchdog(
        lease_for(sandbox, max_steps=1),
        task_id=sandbox.task_id,
        data_root=sandbox.data_root,
        segment_id="user-failure",
    )

    decisions = [
        watchdog.record_tool_failure("file_read", {"path": "/tmp/missing"})
        for _index in range(3)
    ]
    assert not any(decision.paused for decision in decisions)

    final_decision = None
    for _index in range(7):
        final_decision = watchdog.record_tool_failure(
            "file_read", {"path": "/tmp/missing"}
        )
    assert final_decision is not None
    assert not final_decision.paused
    assert sum(watchdog.tool_failures.values()) == 10
    assert not watchdog.reserve_tool_step().paused
    assert not watchdog.reserve_tool_step().paused
    assert watchdog.steps_taken == 2


def _record_task_approval(
    calls: list[object],
) -> Callable[[ApprovalRequest], ApprovalDecision]:
    def approve(request: ApprovalRequest) -> ApprovalDecision:
        calls.append(request)
        return ApprovalDecision.TASK

    return approve
