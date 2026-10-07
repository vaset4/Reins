"""独立模型调用边界的身份、证据和用量合同。

作者：xxx
时间：2026-09-28 18:30:00
"""

from collections.abc import Generator
from contextlib import closing
from pathlib import Path
from typing import TypeVar

from approval.batch import BatchAuthorizer
from approval.session import ApprovalSession
from context.production_builder import ProductionContextBuilder
from runtime.context_preparation import system_prompt_estimate
from runtime.cancellation import CancellationToken
from runtime.extension_execution import ExtensionExecution
from runtime.execution_context import TurnExecutionContext
from runtime.extensions import RuntimeExtensions
from runtime.lease import from_trigger
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.model_evidence import ModelEvidenceWriter
from runtime.model_execution import ModelRequestRunner
from runtime.run_evidence import RunEvidenceStore
from runtime.run_facts import RunFactStore
from runtime.progress import ProgressGuard
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_user_message
from runtime.session_state import SessionStateStore
from runtime.shared_budget import BudgetOwner
from runtime.stream_events import AssistantTextDelta, StreamEvent
from runtime.types import RunContext, Trigger
from runtime.types import RunToolsRequest
from runtime.tool_executor import ToolBatchExecutor
from runtime.tool_operations import ToolOperationStore
from runtime.tool_policy import RuntimeToolPolicy
from runtime.watchdog import Watchdog
from scripts.testing.llm import from_test_turns
from tasks.store import TaskStore
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk
from tools.types import ToolError, ToolErrorCategory

T = TypeVar("T")


def _consume(stream: Generator[StreamEvent, None, T]) -> tuple[list[StreamEvent], T]:
    """保留公开事件和返回的调用身份；传参：执行流；返回：事件与完整结果。"""
    events: list[StreamEvent] = []
    while True:
        try:
            events.append(next(stream))
        except StopIteration as finished:
            return events, finished.value


def test_interleaved_main_and_auxiliary_keep_local_identity_and_usage(
    tmp_path: Path,
) -> None:
    """主、辅助、主共用真实客户端但各自保留身份；传参：隔离目录；返回：无。"""
    with closing(TaskStore(tmp_path)) as tasks:
        task = tasks.create_task("比较本地方案")
    context = RunContext(
        task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": task.goal},
        capability_lease=from_trigger("user", task_id=task.task_id),
    )
    append_user_message(tmp_path, context.session_id, task.goal)
    registry, token = ToolRegistry(), CancellationToken()
    facts, evidence = RunFactStore(tmp_path), RunEvidenceStore(tmp_path)
    client = from_test_turns(["主请求一", "内部摘要", "主请求二"])
    runner = ModelRequestRunner(
        client,
        registry=registry,
        cancellation=token,
        evidence=ModelEvidenceWriter(evidence, facts),
        facts=facts,
        run_evidence=evidence,
        states=SessionStateStore(tmp_path),
        ledger=LedgerWriter(LedgerStore(tmp_path), source="test"),
        extensions=ExtensionExecution(RuntimeExtensions(), token, facts),
    )
    watchdog = Watchdog(
        context.capability_lease,
        task_id=task.task_id,
        data_root=tmp_path,
        cancellation=token,
    )
    bundle = ProductionContextBuilder(
        tmp_path, system_prompt_provider=system_prompt_estimate
    ).build(
        task=task.goal,
        context=context,
        tool_registry=registry.snapshot(),
        toolset_policy={},
    )
    first_stream = runner.invoke(bundle, context, watchdog)
    dispatched_events = []
    for event in first_stream:
        dispatched_events.append(event)
        if isinstance(event, AssistantTextDelta):
            break
    assert isinstance(dispatched_events[-1], AssistantTextDelta)
    auxiliary = runner.invoke_auxiliary(bundle, context=context, watchdog=watchdog)
    first_events, first = _consume(first_stream)
    first_events[:0] = dispatched_events
    last_events, last = _consume(runner.invoke(bundle, context, watchdog))
    calls = (first, auxiliary, last)
    assert [call.request_index for call in calls] == [1, 2, 3]
    assert len({call.request_id for call in calls}) == 3
    assert [call.plan.final_output for call in calls] == [
        "主请求一",
        "内部摘要",
        "主请求二",
    ]
    assert all(call.plan.request_id == call.request_id for call in calls)
    assert "内部摘要" not in str(first_events + last_events)
    rows = facts.read_run(context.run_id)
    requests = [row for row in rows if row.get("event") == "llm:request"]
    responses = [row for row in rows if row.get("event") == "llm:response"]
    attempts = [row for row in rows if row.get("event") == "llm:attempt"]
    assert [(row["request_id"], row["request_index"]) for row in requests] == [
        (call.request_id, call.request_index) for call in calls
    ]
    assert [(row["request_id"], row["request_index"]) for row in responses] == [
        (call.request_id, call.request_index) for call in (auxiliary, first, last)
    ]
    assert {row["request_id"] for row in attempts} == {
        call.request_id for call in calls
    }
    assert len(attempts) == 3
    assert not any(row.get("event") == "input:handled" for row in rows)


def test_tool_executor_commits_local_failure_and_success_without_loop(
    tmp_path: Path,
) -> None:
    """独立执行器提交同批失败与成功，真实下一请求保留两项结果；传参：隔离目录；返回：无。"""
    effects: list[str] = []

    def execute(arguments: dict[str, object]) -> object:
        """产生可核对的业务结果；传参：实际参数；返回：局部失败或实际成功。"""
        value = str(arguments["value"])
        effects.append(value)
        return (
            ToolError(
                ToolErrorCategory.INVALID_INPUT, "missing-source", retryable=False
            )
            if value == "bad"
            else "independent-success"
        )

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "probe",
            "检查资料",
            {"value": {"type": "string", "required": True}},
            "agent",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=execute,
            parallel_safe=True,
        )
    )
    with closing(TaskStore(tmp_path)) as tasks:
        task = tasks.create_task("检查两份资料")
        context = RunContext(
            task_id=task.task_id,
            trigger=Trigger.USER,
            payload={"message": task.goal},
            capability_lease=from_trigger("user", task_id=task.task_id),
        )
        append_user_message(tmp_path, context.session_id, task.goal)
        facts, evidence = RunFactStore(tmp_path), RunEvidenceStore(tmp_path)
        states, token = SessionStateStore(tmp_path), CancellationToken()
        executor = ToolBatchExecutor(
            tmp_path,
            registry=registry,
            authorizer=BatchAuthorizer(
                ApprovalSession(), SessionMessageStore(tmp_path), LedgerStore(tmp_path)
            ),
            cancellation=token,
            extension_execution=ExtensionExecution(RuntimeExtensions(), token, facts),
            policy=RuntimeToolPolicy(registry, states, {}),
            operations=ToolOperationStore(tmp_path),
            facts=facts,
            evidence=evidence,
            states=states,
            progress=ProgressGuard({}),
            history=[],
            client=None,
            collaboration=None,
        )
        owner = BudgetOwner(
            context.session_id, context.run_id, context.capability_lease
        )
        watchdog = Watchdog(
            context.capability_lease,
            task_id=task.task_id,
            data_root=tmp_path,
            cancellation=token,
            budget_owner=owner,
        )
        core = TurnExecutionContext(
            tasks.task_dir(task.task_id),
            context,
            tasks,
            watchdog,
            task.task_id,
            task.goal,
        )
        calls = tuple(
            executor.build_call(
                RunToolsRequest(
                    action="probe", tool_name="probe", arguments={"value": value}
                ),
                context=context,
                request_id="request-pair",
            )
            for value in ("bad", "good")
        )
        _events, results = _consume(
            executor.execute(core, calls, registry=registry.snapshot())
        )
        assert [result.status for result in results] == ["error", "ok"]
        assert sorted(effects) == ["bad", "good"]
        operations = ToolOperationStore(tmp_path).for_session(context.session_id)
        assert {row["operation_id"] for row in operations} == {
            call.operation_id for call in calls
        }
        for operation in operations:
            assert (operation["session_id"], operation["run_id"]) == (
                context.session_id,
                context.run_id,
            )
            decision = LedgerStore(tmp_path).read_event(
                operation["authorization"]["event_id"]
            )
            assert decision is not None and decision.event == "approval.decided"
            assert (decision.session_id, decision.run_id) == (
                context.session_id,
                context.run_id,
            )
            assert decision.payload["operation_id"] == operation["operation_id"]
            assert decision.payload["decision"] == "once"
        bundle = ProductionContextBuilder(
            tmp_path, system_prompt_provider=system_prompt_estimate
        ).build(
            task=task.goal,
            context=context,
            tool_registry=registry.snapshot(),
            toolset_policy={},
        )
        client = from_test_turns(["根据两项结果继续"])
        plan = client.continue_from_run_tools(
            bundle.model_task, results[-1], context=bundle.model_context
        )
        assert plan.model_error is None
        assert "missing-source" in str(plan.raw_model_request)
        assert "independent-success" in str(plan.raw_model_request)
