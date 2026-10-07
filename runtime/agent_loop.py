from __future__ import annotations

import json
import logging
import traceback as _tb
from collections.abc import Generator, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, cast

from approval.batch import BatchAuthorizer
from runtime.progress import ProgressGuard
from approval.session import ApprovalSession

from context.production_builder import (
    ContextReadError,
    HistorySelection,
    ProductionContextBuilder,
)
from llm.base import LLMClient
from llm.messages import UserMessage, model_visible_text, thaw_json_value
from llm.model_request import ModelActionCapability
from llm.types import (
    LLMPlan,
    ModelError,
)
from runtime.checkpoint import (
    Checkpoint,
    Idempotent as CheckpointIdempotent,
    checkpoint_to_ledger_state,
    load_latest_checkpoint,
    normalize_idempotency,
    save_checkpoint,
    summarize_checkpoint,
)
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.run_evidence import RunEvidenceStore
from runtime.model_evidence import ModelEvidenceWriter
from runtime.native_actions import link_question_answers
from runtime.run_facts import RunFactStore
from runtime.recovery_policy import RecoveryPolicy, recovery_policy_from_config
from runtime.session_messages import (
    append_assistant_message,
    materialize_messages,
)
from runtime.session_state import SessionStateStore
from runtime.persistence import RuntimeStore
from runtime.session_message_store import SessionMessageStore
from runtime.context_preparation import prepare_model_turn, system_prompt_estimate
from runtime.stream_events import (
    AssistantReasoningDelta,
    AssistantTurnComplete,
    LeaseSnapshot,
    LifecycleChanged,
    SegmentPaused,
    StreamEvent,
)
from runtime.types import RunContext, Trigger
from runtime.tool_operations import ToolOperation, ToolOperationStore, operation_payload
from runtime.model_execution import ModelCallResult, ModelRequestRunner
from runtime.runtime_errors import record_runtime_error
from runtime.tool_executor import ToolBatchExecutor
from runtime.execution_context import TurnExecutionContext
from runtime.extension_execution import ExtensionExecution
from runtime.tool_policy import RuntimeToolPolicy
from runtime.tool_results import _jsonable, _extract_error_category
from runtime.cancellation import (
    CancellationToken,
    ExecutionCancelled,
    RunBudgetExceeded,
)
from runtime.extensions import (
    ActionRequest,
    RuntimeExtensions,
    RuntimeObservation,
    run_hook,
)
from runtime.types import RunToolsRequest, RunToolsResult
from runtime.lease import is_expired
from runtime.watchdog import Watchdog, WatchdogDecision
from runtime.shared_budget import BudgetOwner, SharedRunBudget
from runtime.child_execution import execute_child
from runtime.collaboration import CollaborationRuntime
from tasks.ids import new_ulid, utc_now
from tasks.store import TaskStore
from tools.tool_registry import (
    Idempotent,
    ToolRegistry,
    get_default_tool_registry,
)

_LOG = logging.getLogger(__name__)

USER_INPUT_PAUSE_TOOLS = frozenset({"ask_user"})
USER_INPUT_PAUSE_REASON = "awaiting user input"
_RESUME_PENDING_CONSUMED = "_resume_pending_consumed"
# 旧中断记录只提供恢复证据，实际恢复经统一原生工具选择
_RESUME_CHOICE_PENDING = "_resume_choice_pending_evidence"
# 每轮 model turn 前把当前 segment 的 Watchdog 护栏余量快照进 payload，供 builder 回喂模型
_RUNTIME_BUDGET_EVIDENCE = "_runtime_budget_evidence"
# F5: replayed conversation history is sized by a token budget derived from the
# model context window, not a fixed row count. The ratio stays below the pre-trim
# threshold (0.6 x window in llm/client.py) so pre-trim remains the safety net.
WALL_CLOCK_EXPIRY_TRIGGERS = frozenset({"user", "idle"})
RUNTIME_PAUSE_EVENT_REASONS = {
    "llm_failure_budget": "segment llm failure budget hit",
    "tool_failure_budget": "segment tool failure budget hit",
    "manual_pause": "manual pause requested",
    "lease_expired": "lease expired",
    "cancelled": "user stop requested",
}


class State(str, Enum):
    # Legacy return enum. Current lifecycle detail is stored in
    # LifecycleChanged / run:lifecycle, not expanded back into State.
    DONE = "DONE"
    PAUSED = "PAUSED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class _ModelTurnResult:
    plan: LLMPlan | None = None
    last_tool_result: RunToolsResult | None = None
    continue_loop: bool = False
    terminal: bool = False


@dataclass(frozen=True, slots=True)
class _ToolTurnResult:
    last_tool_result: RunToolsResult | None = None
    terminal: bool = False


class AgentLoop:
    def __init__(
        self,
        data_root: Path | str | None = None,
        *,
        state: State | None = None,
        llm_client: LLMClient | None = None,
        tool_registry: ToolRegistry | None = None,
        runtime_config: dict[str, object] | None = None,
        cancellation: CancellationToken | None = None,
        extensions: RuntimeExtensions | None = None,
        shared_budget: SharedRunBudget | None = None,
        collaboration: CollaborationRuntime | None = None,
        approval_session: ApprovalSession | None = None,
    ) -> None:
        self.data_root = (
            Path(data_root)
            if data_root is not None
            else Path.home() / ".reins" / "data"
        )
        self.state: State | None = state
        self.llm_client = llm_client
        self.tool_registry = tool_registry or get_default_tool_registry()
        self.runtime_config = dict(runtime_config or {})
        self.approval_session = approval_session or ApprovalSession()
        self.authorizer = BatchAuthorizer(
            self.approval_session,
            SessionMessageStore(self.data_root),
            LedgerStore(self.data_root),
        )
        self.cancellation = cancellation or CancellationToken()
        self.extensions = extensions or RuntimeExtensions()
        self.shared_budget = shared_budget
        self.collaboration = collaboration
        self._owns_collaboration = collaboration is None
        self.recovery_policy: RecoveryPolicy = recovery_policy_from_config(
            self.runtime_config
        )
        self.run_facts = RunFactStore(self.data_root)
        self.run_evidence = RunEvidenceStore(self.data_root)
        self.operations = ToolOperationStore(self.data_root)
        self.model_evidence = ModelEvidenceWriter(self.run_evidence, self.run_facts)
        self.session_states = SessionStateStore(self.data_root)
        self.extension_execution = ExtensionExecution(
            self.extensions, self.cancellation, self.run_facts
        )
        self.tool_policy = RuntimeToolPolicy(
            self.tool_registry,
            self.session_states,
            self.runtime_config,
            approval_session=self.approval_session,
        )
        self.last_output = ""
        self._last_output_is_error = False
        self.tool_history: list[dict[str, object]] = []
        self._failure_budgets: dict[str, int] = {}
        self.progress = ProgressGuard(self.runtime_config)
        self.last_resume_action: str | None = None
        self.model_runner: ModelRequestRunner | None = None
        self._observed_requests: dict[str, set[str]] = {}

    @classmethod
    def resume(cls, task_id: str, data_root: Path | str) -> "AgentLoop":
        checkpoint = load_latest_checkpoint(task_id, data_root=data_root)
        if checkpoint is None:
            raise FileNotFoundError(task_id)
        loop = cls(data_root, state=_terminal_state_from_checkpoint(checkpoint.state))
        if checkpoint.pending_tool_call is not None:
            loop.last_resume_action = "inspect"
        return loop

    def _idempotency_for_tool(self, tool_name: str) -> CheckpointIdempotent | None:
        definition = self.tool_registry.get(tool_name)
        if definition is None:
            return None
        return normalize_idempotency(definition.idempotent)

    def run(self, context: RunContext) -> State:
        """消费同一条模型与工具执行流；传参：运行上下文；返回：真实终态，没有执行边界时抛错。"""
        for event in self.run_stream(context):
            del event
        if self.state is None:
            raise RuntimeError("run exited without a lifecycle boundary")
        return self.state

    def transition(
        self,
        to_state: State,
        context: RunContext,
        *,
        terminal_reason: str | None = None,
        close_reason: str | None = None,
    ) -> None:
        del to_state, context, terminal_reason, close_reason
        raise RuntimeError("AgentLoop.transition is retired; use run lifecycle")

    def _emit_lifecycle_changed(
        self,
        context: RunContext,
        *,
        state: State,
        lifecycle: str,
        reason: str,
        checkpoint_state: str | None = None,
        pending_tool_call: dict[str, object] | None = None,
        resumable: bool | None = None,
    ) -> Iterator[StreamEvent]:
        checkpoint = self._record_lifecycle_boundary(
            context,
            state=state,
            lifecycle=lifecycle,
            reason=reason,
            checkpoint_state=checkpoint_state,
            pending_tool_call=pending_tool_call,
            resumable=resumable,
        )
        yield LifecycleChanged(
            lifecycle=lifecycle,
            reason=reason,
            segment_id=context.segment_id,
            checkpoint_id=checkpoint.checkpoint_id,
        )

    def _record_lifecycle_boundary(
        self,
        context: RunContext,
        *,
        state: State,
        lifecycle: str,
        reason: str,
        checkpoint_state: str | None = None,
        pending_tool_call: dict[str, object] | None = None,
        resumable: bool | None = None,
    ) -> Checkpoint:
        if self.state is not None:
            raise RuntimeError(f"run already ended as {self.state.value}")
        pending_tool_call = self._resolve_lifecycle_pending(context, pending_tool_call)
        self.state = state
        checkpoint = save_checkpoint(
            Checkpoint(
                task_id=context.storage_task_id,
                segment_id=context.segment_id,
                state=checkpoint_state or lifecycle,
                session_id=context.session_id,
                run_id=context.run_id,
                focus_task_id=context.focus_task_id,
                terminal_focus_policy=context.terminal_focus_policy,
                compatibility_task_id=context.compatibility_task_id,
                working_memory_snapshot={"payload": dict(context.payload)},
                pending_tool_call=pending_tool_call,
                lease_snapshot=asdict(context.capability_lease),
                reason=reason,
            )
        )
        self._persist_lifecycle_boundary(
            context,
            checkpoint,
            state=state,
            lifecycle=lifecycle,
            reason=reason,
            resumable=resumable,
        )
        return checkpoint

    def _resolve_lifecycle_pending(
        self,
        context: RunContext,
        pending_tool_call: dict[str, object] | None,
    ) -> dict[str, object] | None:
        """继承未消费 pending，并拒绝调用方提供冲突身份
        传参：context 为当前运行上下文；pending_tool_call 为调用方显式 carrier
        返回：可写入 checkpoint 的 pending carrier，冲突时抛 RuntimeError
        """
        active_pending = _pending_resume_tool(context)
        if pending_tool_call is None:
            return active_pending
        if active_pending is not None and not _pending_calls_match(
            pending_tool_call, active_pending
        ):
            raise RuntimeError(
                "lifecycle pending tool conflicts with active resume pending"
            )
        return pending_tool_call

    def _persist_lifecycle_boundary(
        self,
        context: RunContext,
        checkpoint: Checkpoint,
        *,
        state: State,
        lifecycle: str,
        reason: str,
        resumable: bool | None,
    ) -> None:
        """持久化 lifecycle checkpoint 的 facts、session 与任务收尾副作用
        传参：context/checkpoint 为当前边界；state/lifecycle/reason/resumable 为边界元数据
        返回：无；所有 durable facts 与 session 状态写入完成
        """
        self._record_checkpoint_fact(context, checkpoint)
        self.run_facts.append_lifecycle(
            lifecycle=lifecycle,
            reason=reason,
            session_id=context.session_id,
            run_id=context.run_id,
            segment_id=context.segment_id,
            task_id=context.task_id,
            focus_task_id=context.focus_task_id,
            compatibility_task_id=context.compatibility_task_id,
            checkpoint_id=checkpoint.checkpoint_id,
            resumable=resumable,
        )
        self._ledger_writer().record_run_lifecycle_changed(
            lifecycle,
            task_id=context.storage_task_id,
            session_id=context.session_id,
            run_id=context.run_id,
        )
        summary = "" if self._last_output_is_error else self.last_output
        self.session_states.record_checkpoint(
            summarize_checkpoint(checkpoint),
            summary=summary,
        )
        self._record_terminal_task_writeback(context, state.value.lower())
        self.session_states.record_run_terminal(
            context,
            state.value.lower(),
            last_event="run:lifecycle",
            summary=summary,
            preserve_summary=self._last_output_is_error,
        )

    def run_stream(self, context: RunContext) -> Iterator[StreamEvent]:
        """运行唯一的模型与工具主循环；传参：持久会话上下文；返回：可流式展示的实际执行事件。"""
        with RuntimeStore(self.data_root).connection_scope():
            if self.extensions.observing:
                raise RuntimeError(
                    "observers cannot execute actions; submit an ActionRequest"
                )
            retired = {"needs_approval", "approval_granted"}.intersection(
                context.payload
            )
            if retired:
                raise ValueError(
                    f"retired legacy approval payload fields: {', '.join(sorted(retired))}"
                )
            if not isinstance(context.payload.get("message"), str):
                raise ValueError("run_stream requires payload['message'] as text")
            if self.llm_client is None:
                raise RuntimeError("llm_client is required for run_stream")
            core = self._prepare_turn_core(context)
            yield self._lease_snapshot_event(core)
            self._enter_turn_core(core)
            try:
                from contextlib import nullcontext
                from runtime.knowledge_maintenance import KnowledgeMaintenance
                from runtime.model_dispatch import foreground_turn

                foreground = context.trigger == Trigger.USER
                activity = (
                    foreground_turn(self.data_root, context.run_id)
                    if foreground
                    else nullcontext()
                )
                with activity:
                    capability = context.capability_lease.capabilities.get(
                        "background_run"
                    )
                    if (
                        foreground
                        and isinstance(capability, Mapping)
                        and capability.get("enabled") is True
                    ):
                        KnowledgeMaintenance(self.data_root).initialize(context)
                    yield from self._run_stream_loop(core)
            except StopIteration:
                yield from self._handle_stream_stop_iteration(core)
            else:
                yield from self._finish_extensions(core)
                if self.state == State.DONE:
                    # 1. 【知识维护】【轮次汇集】整轮用户交互完成才接纳来源，中间模型请求和工具回执只积累证据
                    KnowledgeMaintenance(self.data_root).observe(
                        context,
                        client=self.llm_client,
                        approval_session=self.approval_session,
                    )
            finally:
                if self._owns_collaboration and self.collaboration is not None:
                    self.collaboration.close()
                core.store.close()

    def _prepare_turn_core(self, context: RunContext) -> TurnExecutionContext:
        self.tool_registry.refresh_availability_cache(context.session_id)
        if not context.segment_id:
            context.segment_id = f"{context.trigger.value}-{new_ulid()}"
        storage_task_id = context.storage_task_id
        task_dir = self._task_dir(storage_task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        self._write_segment_start(task_dir, context)
        lease = context.capability_lease
        owner = BudgetOwner(context.session_id, context.run_id, lease)
        if self.shared_budget is None:
            self.shared_budget = SharedRunBudget(owner, self.run_facts)
        context.budget_run_id = self.shared_budget.owner.run_id
        if self.collaboration is None:
            self.collaboration = CollaborationRuntime(
                context,
                messages=SessionMessageStore(self.data_root),
                facts=self.run_facts,
                operations=self.operations,
                budget=self.shared_budget,
                cancellation=self.cancellation,
                run_child=partial(
                    execute_child,
                    data_root=self.data_root,
                    client=self.llm_client,
                    registry=self.tool_registry,
                    runtime_config=self.runtime_config,
                    extensions=self.extensions,
                    approval_session=self.approval_session,
                ),
            )
        self.extension_execution = ExtensionExecution(
            self.extensions, self.cancellation, self.run_facts
        )
        self.tool_policy = RuntimeToolPolicy(
            self.tool_registry,
            self.session_states,
            self.runtime_config,
            approval_session=self.approval_session,
        )
        self.model_runner = (
            ModelRequestRunner(
                self.llm_client,
                registry=self.tool_registry,
                cancellation=self.cancellation,
                evidence=self.model_evidence,
                facts=self.run_facts,
                run_evidence=self.run_evidence,
                states=self.session_states,
                ledger=self._ledger_writer(),
                extensions=self.extension_execution,
            )
            if self.llm_client
            else None
        )
        self.tool_executor = ToolBatchExecutor(
            self.data_root,
            registry=self.tool_registry,
            authorizer=self.authorizer,
            cancellation=self.cancellation,
            extension_execution=self.extension_execution,
            policy=self.tool_policy,
            operations=self.operations,
            facts=self.run_facts,
            evidence=self.run_evidence,
            states=self.session_states,
            progress=self.progress,
            history=self.tool_history,
            client=self.llm_client,
            collaboration=self.collaboration,
        )
        return TurnExecutionContext(
            task_dir=task_dir,
            context=context,
            store=TaskStore(self.data_root),
            watchdog=Watchdog(
                lease,
                task_id=storage_task_id,
                data_root=self.data_root,
                segment_id=context.segment_id,
                cancellation=self.cancellation,
                shared_budget=self.shared_budget,
                budget_owner=owner,
            ),
            storage_task_id=storage_task_id,
            task=str(context.payload.get("message", "")).strip(),
        )

    def _lease_snapshot_event(self, core: TurnExecutionContext) -> LeaseSnapshot:
        lease = core.context.capability_lease
        return LeaseSnapshot(
            trigger=lease.trigger,
            task_id=core.storage_task_id,
            segment_id=core.context.segment_id,
            max_steps=lease.max_steps,
            max_tokens=lease.max_tokens,
            expires_at=lease.expires_at,
            capabilities_summary=_summarize_capabilities(lease.capabilities),
        )

    def _enter_turn_core(self, core: TurnExecutionContext) -> None:
        self._observed_requests.clear()
        self.tool_executor.recover_pending(core)
        self.progress.start_run(core.context.run_id)

    def _handle_stream_stop_iteration(
        self, core: TurnExecutionContext
    ) -> Iterator[StreamEvent]:
        record_runtime_error(
            self.run_evidence,
            core.context,
            category="generator_stop_iteration",
            message="StopIteration leaked into run_stream generator",
            stage="stream_loop",
            meta={"traceback": _tb.format_exc()},
        )
        output = (
            "INTERNAL_ERROR: generator raised StopIteration — "
            "run saved; use /resume to continue."
        )
        entry_id = self._finish_failed(core.store, core.context, output)
        yield AssistantTurnComplete(
            content=output, usage={}, stop_reason="stop_iteration", entry_id=entry_id
        )
        yield from self._emit_lifecycle_changed(
            core.context,
            state=State.FAILED,
            lifecycle="failed",
            reason="generator_stop_iteration",
            checkpoint_state="failed",
        )

    def _run_stream_loop(self, core: TurnExecutionContext) -> Iterator[StreamEvent]:
        last_tool_result: RunToolsResult | None = None

        while self.state is None:
            self._deliver_pending_inputs(core.context)
            self._emit_context_built(core)
            resume_tool = yield from self._resume_pending_tool_turn(core)
            if resume_tool.terminal:
                return
            if resume_tool.last_tool_result is not None:
                last_tool_result = resume_tool.last_tool_result
                continue
            decision = self._runtime_budget_decision(core)
            if decision.paused:
                yield from self._runtime_budget_pause_turn(core, decision)
                return
            model = yield from self._model_turn(core, last_tool_result)
            if model.terminal:
                return
            if model.continue_loop:
                last_tool_result = model.last_tool_result
                continue
            if model.plan is None:
                raise RuntimeError("model turn produced no plan")
            tool = yield from self._tool_turn(core, model.plan)
            if tool.terminal:
                return
            last_tool_result = tool.last_tool_result

    def _emit_context_built(self, core: TurnExecutionContext) -> None:
        context = core.context
        self._append_trajectory(
            {
                "type": "event",
                "event": "context:built",
                "ts": utc_now(),
                "session_id": context.session_id,
                "run_id": context.run_id,
                "task_id": context.task_id,
                "focus_task_id": context.focus_task_id,
                "compatibility_task_id": context.compatibility_task_id,
                "segment_id": context.segment_id,
                "tool_history_count": len(self.tool_history),
            },
        )

    def _resume_pending_tool_turn(
        self, core: TurnExecutionContext
    ) -> Generator[StreamEvent, None, _ToolTurnResult]:
        """恢复只提交既有操作证据，旧幂等标签不触发自动重放；传参：运行；返回：继续模型决策。"""
        pending = _pending_resume_tool(core.context)
        if pending is not None:
            core.context.payload[_RESUME_CHOICE_PENDING] = {
                **_pending_resume_evidence(
                    pending,
                    self._idempotency_for_tool(str(pending.get("tool_name", ""))),
                ),
                "requested_resolution": core.context.payload.get(
                    "resume_action", "inspect"
                ),
            }
        yield from ()
        return _ToolTurnResult()

    def _has_pending_inputs(self, context: RunContext) -> bool:
        """查询已接纳但尚未交付的新要求；传参：运行；返回：是否需要模型重新判断。"""
        if self.collaboration is not None:
            self.collaboration.sync_inputs()
        return bool(
            SessionMessageStore(self.data_root).pending_inputs(context.session_id)
        )

    def _deliver_pending_inputs(self, context: RunContext) -> None:
        """合法消息边界投影已持久接纳的输入；传参：运行；返回：无。"""
        if self.collaboration is not None:
            self.collaboration.sync_inputs()
        ids = SessionMessageStore(self.data_root).deliver_inputs(
            context.session_id,
            run_id=context.run_id,
            task_id=context.material_task_id,
        )
        if ids:
            self.run_facts.append(
                {
                    "event": "input:delivered",
                    "session_id": context.session_id,
                    "run_id": context.run_id,
                    "input_ids": list(ids),
                    "task_id": context.material_task_id,
                }
            )
            entries = SessionMessageStore(self.data_root).pending_inputs(
                context.session_id, include_delivered=True
            )
            if any(
                entry.entry_id in ids
                and entry.input_source == "user"
                and entry.input_kind is None
                for entry in entries
            ):
                self.progress.reset()
            latest = next(entry for entry in entries if entry.entry_id == ids[-1])
            assert isinstance(latest.message, UserMessage)
            context.payload["input_message_id"] = ids[-1]
            context.payload["message"] = model_visible_text(latest.message)
        if SessionMessageStore(self.data_root).exists(context.session_id):
            link_question_answers(
                SessionMessageStore(self.data_root),
                self.operations,
                run=context,
                facts=self.run_facts,
            )

    def _mark_inputs_handled(self, context: RunContext) -> None:
        """回答或等待问题已提交后才确认实际观察过的输入；传参：运行；返回：无，失败保留待处理身份。"""
        for request_id, input_ids in self._observed_requests.items():
            if input_ids:
                self.run_facts.append(
                    {
                        "event": "input:handled",
                        "session_id": context.session_id,
                        "run_id": context.run_id,
                        "request_id": request_id,
                        "input_ids": sorted(input_ids),
                    }
                )

    def _runtime_budget_decision(self, core: TurnExecutionContext) -> WatchdogDecision:
        self._request_external_pause_if_needed(core)
        if self.cancellation.cancelled:
            if self.cancellation.reason == "manual_pause":
                return WatchdogDecision(True, "manual_pause", "pause requested")
            return WatchdogDecision(True, "cancelled", "user stop requested")
        lease = core.context.capability_lease
        if lease.trigger in WALL_CLOCK_EXPIRY_TRIGGERS and is_expired(lease):
            return WatchdogDecision(True, "lease_expired", "lease expired")
        return core.watchdog.tick(steps_taken=core.watchdog.steps_taken)

    def _request_external_pause_if_needed(self, core: TurnExecutionContext) -> None:
        checker = self.runtime_config.get("pause_requested")
        if checker is None:
            return
        if not callable(checker):
            raise TypeError("runtime_config['pause_requested'] must be callable")
        if checker():
            core.watchdog.request_pause()

    def _snapshot_runtime_budget(self, core: TurnExecutionContext) -> None:
        """把当前 segment 的 Watchdog 护栏余量快照进 payload，供 builder 每轮回喂模型。

        数据源取实时 watchdog 实例（steps_taken/tokens_used）与 lease 上限，纯运行态不落库。
        已知零用量、已知下界和用量缺失分别保留，首轮没有采集时不伪造用量。

        :param core: 当前轮上下文，持有 watchdog 与 context.payload
        :return: 无返回值，运行态直接写进 core.context.payload
        """
        core.context.payload[_RUNTIME_BUDGET_EVIDENCE] = core.watchdog.budget_evidence()

    def _model_turn(
        self,
        core: TurnExecutionContext,
        last_tool_result: RunToolsResult | None,
        *,
        reminder: str | None = None,
    ) -> Generator[StreamEvent, None, _ModelTurnResult]:
        """执行模型请求并记录证据与预算；传参：运行和前次结果；返回：本次决策处理结果。"""
        self._snapshot_runtime_budget(core)
        # 【运行循环】【会话读取】发模型请求前核对已投递输入身份，读失败必须在发请求前停住
        try:
            projected_ids = {
                message.message_id
                for message in materialize_messages(
                    self.data_root, core.context.session_id
                )
                if isinstance(message, UserMessage)
            }
        except Exception as exc:
            error = ContextReadError(
                "conversation_history_read_failed",
                core.context.session_id or core.storage_task_id,
                exc,
            )
            yield from self._context_read_failure_turn(core, error)
            return _ModelTurnResult(terminal=True)
        try:
            call = yield from self._call_model(
                str(core.context.payload.get("message", core.task)),
                core.context,
                last_tool_result,
                system_reminder=reminder,
                watchdog=core.watchdog,
            )
        except ContextReadError as error:
            yield from self._context_read_failure_turn(core, error)
            return _ModelTurnResult(terminal=True)
        except ExecutionCancelled:
            yield from self._runtime_budget_pause_turn(
                core, WatchdogDecision(True, "cancelled", "user stop requested")
            )
            return _ModelTurnResult(terminal=True)
        except RunBudgetExceeded as exc:
            # 租约额度已取消，此处只剩语义摘要模型自身的额度耗尽，仍按暂停交出控制权
            yield from self._runtime_budget_pause_turn(
                core, WatchdogDecision(True, "budget_exhausted", str(exc))
            )
            return _ModelTurnResult(terminal=True)
        if isinstance(call, LLMPlan):
            return (yield from self._model_error_turn(core, call, last_tool_result))
        plan = call.plan
        if plan.model_error is None:
            observed = (
                set(plan.model_attempts[-1].input_ids)
                if plan.model_attempts
                else projected_ids
            )
            self._observed_requests[call.request_id] = observed
        decision = self._runtime_budget_decision(core)
        if decision.paused:
            assert self.model_runner is not None
            yield self.model_runner.close_stream(
                core.context, plan.request_id, decision.reason or "paused"
            )
            yield from self._runtime_budget_pause_turn(core, decision)
            return _ModelTurnResult(terminal=True)
        decision = self._record_model_health(core.watchdog, plan)
        if decision.paused:
            assert self.model_runner is not None
            yield self.model_runner.close_stream(
                core.context, plan.request_id, decision.reason or "paused"
            )
            yield from self._runtime_budget_pause_turn(core, decision)
            return _ModelTurnResult(terminal=True)
        # 走了流式的客户端已经边生成边把思考链发出去了，这里再补一条就是重复显示；
        # 只实现同步 plan() 的客户端拿不到增量，仍按老样子在轮末补发这一条
        if not call.used_stream_path and plan.reasoning_content.strip():
            yield AssistantReasoningDelta(
                text=plan.reasoning_content.strip(), message_id=plan.request_id
            )
        return (
            yield from self._dispatch_model_plan(
                core, plan, last_tool_result=last_tool_result
            )
        )

    def _dispatch_model_plan(
        self,
        core: TurnExecutionContext,
        plan: LLMPlan,
        *,
        last_tool_result: RunToolsResult | None,
    ) -> Generator[StreamEvent, None, _ModelTurnResult]:
        """校验并兑现模型选定动作；传参：运行、计划与前次结果；返回：继续或终止边界。"""
        if self._has_pending_inputs(core.context) and plan.run_tools_request is None:
            assert self.model_runner is not None
            yield self.model_runner.close_stream(
                core.context, plan.request_id, "superseded"
            )
            return _ModelTurnResult(
                continue_loop=True, last_tool_result=last_tool_result
            )
        capability_gate = yield from self._model_action_capability_gate_turn(core, plan)
        if capability_gate is not None:
            return capability_gate
        if plan.model_error is not None:
            return (yield from self._model_error_turn(core, plan, last_tool_result))
        if plan.final_output is not None:
            yield from self._final_output_turn(core, plan)
            return _ModelTurnResult(terminal=True)
        if plan.run_tools_request is None:
            yield from self._invalid_model_protocol_turn(
                core, message_id=plan.request_id
            )
            return _ModelTurnResult(terminal=True)
        return _ModelTurnResult(plan=plan)

    def _model_action_capability_gate_turn(
        self, core: TurnExecutionContext, plan: LLMPlan
    ) -> Generator[StreamEvent, None, _ModelTurnResult | None]:
        """依据同一 ComposedRequest capability 拒绝越权动作。

        作者：xxx
        时间：2026-08-17 00:00:00
        传参：core 为当前运行核心；plan 为模型解析结果
        返回：拒绝时返回继续循环结果，否则返回 None
        """
        if plan.model_error is not None:
            return None
        raw_capability = plan.prompt_context.get("model_action_capability")
        if raw_capability is None:
            return None
        capability = ModelActionCapability.from_mapping(raw_capability)
        action = _model_plan_action(plan)
        if capability.allows(action):
            return None
        output = (
            f"MODEL_ACTION_NOT_ALLOWED: action={action}; "
            f"allowed={','.join(sorted(capability.allowed_actions))}"
        )
        self.last_output = output
        self._last_output_is_error = False
        core.store.update_summary(core.context.material_task_id, output)
        self._append_trajectory(
            {
                "type": "event",
                "event": "model:action_denied",
                "ts": utc_now(),
                "session_id": core.context.session_id,
                "run_id": core.context.run_id,
                "segment_id": core.context.segment_id,
                "attempted_action": action,
                "allowed_actions": sorted(capability.allowed_actions),
            }
        )
        denied = RunToolsResult.denied(
            action=action,
            tool_name=action,
            error=output,
            summary="model action outside current capability",
            meta={"error_type": "action_not_allowed", "action": action},
        )
        yield AssistantTurnComplete(
            content=output,
            usage=_usage_dict(plan),
            stop_reason="action_not_allowed",
            message_id=plan.request_id,
        )
        return _ModelTurnResult(last_tool_result=denied, continue_loop=True)

    def _record_model_health(
        self, watchdog: Watchdog, plan: LLMPlan
    ) -> WatchdogDecision:
        if plan.model_error is None:
            watchdog.record_llm_success()
            return WatchdogDecision(False)
        return watchdog.record_llm_failure()

    def _context_read_failure_turn(
        self, core: TurnExecutionContext, error: ContextReadError
    ) -> Iterator[StreamEvent]:
        output, entry_id = self._finish_context_read_failed(
            core.store, core.context, error
        )
        yield AssistantTurnComplete(
            content=output,
            usage={},
            stop_reason=f"context_error:{error.category}",
            entry_id=entry_id,
        )
        yield from self._emit_lifecycle_changed(
            core.context,
            state=State.FAILED,
            lifecycle="failed",
            reason=error.category,
            checkpoint_state="failed",
        )

    def _model_error_turn(
        self,
        core: TurnExecutionContext,
        plan: LLMPlan,
        last_tool_result: RunToolsResult | None,
    ) -> Generator[StreamEvent, None, _ModelTurnResult]:
        error = plan.model_error
        if error is None:
            raise RuntimeError("model error turn requires model_error")
        recovery = self._handle_recoverable_model_error(
            core.context, plan, last_tool_result
        )
        if recovery is not None:
            return (yield from self._recoverable_model_error_turn(core, plan, recovery))
        output = plan.final_output or error.render_output()
        entry_id = self._finish_failed(
            core.store, core.context, output, message_id=plan.request_id
        )
        yield AssistantTurnComplete(
            content=output,
            usage={},
            stop_reason=f"model_error:{error.category}",
            message_id=plan.request_id,
            entry_id=entry_id,
        )
        yield from self._emit_lifecycle_changed(
            core.context,
            state=State.FAILED,
            lifecycle="failed",
            reason=error.category,
            checkpoint_state="failed",
        )
        return _ModelTurnResult(terminal=True)

    def _recoverable_model_error_turn(
        self,
        core: TurnExecutionContext,
        plan: LLMPlan,
        recovery: RunToolsResult,
    ) -> Generator[StreamEvent, None, _ModelTurnResult]:
        error = plan.model_error
        if error is None:
            raise RuntimeError("recoverable model error requires model_error")
        if recovery.status != "ok":
            self._record_tool_error(core.context, recovery)
        entry_id = None
        if recovery.meta.get("budget_exhausted"):
            entry_id = self._finish_failed(
                core.store, core.context, recovery.output, message_id=plan.request_id
            )
        yield AssistantTurnComplete(
            content=recovery.output,
            usage={},
            stop_reason=f"model_error:{error.category}",
            message_id=plan.request_id,
            entry_id=entry_id,
        )
        if recovery.meta.get("budget_exhausted"):
            yield from self._emit_lifecycle_changed(
                core.context,
                state=State.FAILED,
                lifecycle="failed",
                reason=str(recovery.meta.get("error_type", "recoverable_error")),
                checkpoint_state="failed",
            )
            return _ModelTurnResult(terminal=True)
        return _ModelTurnResult(
            last_tool_result=recovery,
            continue_loop=True,
        )

    def _final_output_turn(
        self, core: TurnExecutionContext, plan: LLMPlan
    ) -> Iterator[StreamEvent]:
        entry_id = self._finish_success(
            core.store,
            core.context,
            plan.final_output or "",
            message_id=plan.request_id,
            reasoning=plan.reasoning_content,
        )
        self._mark_inputs_handled(core.context)
        yield AssistantTurnComplete(
            content=plan.final_output,
            usage=_usage_dict(plan),
            stop_reason="end_turn",
            message_id=plan.request_id,
            entry_id=entry_id,
        )
        yield from self._emit_lifecycle_changed(
            core.context,
            state=State.DONE,
            lifecycle="done",
            reason="final_output",
            checkpoint_state="done",
        )

    def _invalid_model_protocol_turn(
        self, core: TurnExecutionContext, *, message_id: str = ""
    ) -> Iterator[StreamEvent]:
        output = "MODEL_PROTOCOL_ERROR: model returned no final output or tool request"
        record_runtime_error(
            self.run_evidence,
            core.context,
            category="invalid_model_protocol",
            message=output,
            stage="parse",
        )
        entry_id = self._finish_failed(
            core.store, core.context, output, message_id=message_id
        )
        yield AssistantTurnComplete(
            content=output,
            usage={},
            stop_reason="protocol_error",
            entry_id=entry_id,
            message_id=message_id,
        )
        yield from self._emit_lifecycle_changed(
            core.context,
            state=State.FAILED,
            lifecycle="failed",
            reason="invalid_model_protocol",
            checkpoint_state="failed",
        )

    def _tool_turn(
        self,
        core: TurnExecutionContext,
        plan: LLMPlan,
    ) -> Generator[StreamEvent, None, _ToolTurnResult]:
        """逐项执行本轮调用并把局部失败交回模型；传参：运行和计划；返回：运行边界。"""
        request = plan.run_tools_request
        if request is None:
            raise RuntimeError("tool turn requires a tool request")
        pending = plan.prompt_context.get("pending_tool_calls")
        requests = (
            cast(list[RunToolsRequest], pending)
            if isinstance(pending, list)
            else [request]
        )
        registry = plan.registry_snapshot
        if registry is None:
            raise RuntimeError("tool plan is missing its request registry snapshot")
        calls = tuple(
            self.tool_executor.build_call(
                registry.normalize_request(item),
                context=core.context,
                request_id=plan.request_id,
                registry=registry,
            )
            for item in requests
        )
        # 【目标完成】【发布文字成果】同轮正文先进入会话并完成展示，再允许工具引用 current_answer
        body = (
            model_visible_text(plan.assistant_message)
            if plan.assistant_message is not None and plan.protocol_mode != "text_json"
            else ""
        )
        entry_id = None
        if body.strip() or plan.reasoning_content.strip():
            entry_id = append_assistant_message(
                self.data_root,
                core.context.session_id,
                body,
                run_id=core.context.run_id,
                task_id=core.context.material_task_id,
                message_id=plan.request_id or None,
                reasoning=plan.reasoning_content,
            )
        yield AssistantTurnComplete(
            content=body or None,
            usage=_usage_dict(plan),
            stop_reason="tool_use",
            message_id=plan.request_id,
            entry_id=entry_id,
        )
        results = yield from self.tool_executor.execute(core, calls, registry=registry)
        if self.cancellation.cancelled:
            yield from self._runtime_budget_pause_turn(
                core, self._runtime_budget_decision(core)
            )
            return _ToolTurnResult(terminal=True)
        budget_call = next(
            (
                call
                for call, result in zip(calls, results)
                if result.meta.get("budget_exhausted")
            ),
            None,
        )
        if budget_call is not None:
            yield from self._runtime_budget_pause_turn(
                core,
                self._runtime_budget_decision(core),
                pending_tool_call=_pending_payload_for_call(budget_call),
            )
            return _ToolTurnResult(terminal=True)
        waiting = next(
            (
                result
                for result in results
                if result.status == "ok" and _is_user_input_pause_tool(result.tool_name)
            ),
            None,
        )
        if waiting is not None:
            return (yield from self._terminal_tool_decision(core, waiting))
        last = results[-1]
        if last.status != "ok":
            return (yield from self._recoverable_tool_error_turn(core, last))
        return (yield from self._terminal_tool_decision(core, last))

    def submit_action(self, context: RunContext, request: ActionRequest) -> str:
        """持久提交后处理意图，调用本身不执行工具；传参：运行/动作请求；返回：唯一操作编号。"""
        operation_id = f"op-action-{context.run_id}-{request.request_id}"
        identity = {
            "session_id": context.session_id,
            "run_id": context.run_id,
            "operation_id": operation_id,
        }
        branch_entry_id = (
            SessionMessageStore(self.data_root).materialize(context.session_id).leaf_id
        )
        if branch_entry_id is None:
            raise ValueError("action request requires an existing session branch")
        arguments = cast(dict[str, object], thaw_json_value(request.arguments))
        call = ToolOperation(
            RunToolsRequest(
                action=request.tool, tool_name=request.tool, arguments=arguments
            ),
            f"call-action-{context.run_id}-{request.request_id}",
            request.tool,
            arguments,
            task_id=context.material_task_id,
            operation_id=operation_id,
            origin="extension",
        )
        existing = self.operations.create(
            identity,
            {
                **operation_payload(call, state="not_started"),
                "branch_entry_id": branch_entry_id,
            },
        )
        if any(
            existing["call"][key] != asdict(call)[key]
            for key in ("tool_name", "args", "task_id", "origin")
        ):
            raise ValueError("action request_id already refers to a different action")
        return operation_id

    def _finish_extensions(self, core: TurnExecutionContext) -> Iterator[StreamEvent]:
        """运行结束后收集动作意图，再经共同执行器处理；传参：原运行及预算；返回：各项实际执行事件。"""
        self.extension_execution.observe(
            core.context,
            "run_finished",
            {"state": self.state.value if self.state else None},
            watchdog=core.watchdog,
        )
        observation = RuntimeObservation(
            "run_finished",
            {
                "session_id": core.context.session_id,
                "run_id": core.context.run_id,
                "state": self.state.value if self.state else None,
            },
        )
        if not self.cancellation.cancelled:
            for hook in self.extensions.after_run:
                try:
                    requests = run_hook(
                        partial(hook, observation), core.watchdog, self.cancellation
                    )
                    if not isinstance(requests, tuple) or not all(
                        isinstance(item, ActionRequest) for item in requests
                    ):
                        raise TypeError(
                            "after-run hook must return a tuple of ActionRequest"
                        )
                except Exception as exc:
                    self.extension_execution.record_error(
                        core.context, "after_run", str(exc)
                    )
                    continue
                for request in requests:
                    self.submit_action(core.context, request)
        calls: list[ToolOperation] = []
        for row in self.operations.for_session(core.context.session_id):
            if (
                row["run_id"] != core.context.run_id
                or row.get("state") != "not_started"
                or row.get("result") is not None
                or row["call"].get("origin") != "extension"
            ):
                continue
            payload = {
                **row["call"],
                "request": RunToolsRequest(**row["call"]["request"]),
            }
            calls.append(ToolOperation(**payload))
        if calls:
            yield from self.tool_executor.execute(core, tuple(calls))

    def _recoverable_tool_error_turn(
        self, core: TurnExecutionContext, result: RunToolsResult
    ) -> Generator[StreamEvent, None, _ToolTurnResult]:
        recovery = self._handle_recoverable_tool_error(core.context, result)
        self._record_tool_error(core.context, recovery)
        # 【执行器】【局部失败】不可原样重试只限制当前动作，模型仍可查询或换方法
        yield from ()
        return _ToolTurnResult(last_tool_result=recovery)

    def _terminal_tool_decision(
        self, core: TurnExecutionContext, result: RunToolsResult
    ) -> Generator[StreamEvent, None, _ToolTurnResult]:
        if _is_user_input_pause_tool(result.tool_name):
            if self._has_pending_inputs(core.context):
                return _ToolTurnResult(last_tool_result=result)
            yield from self._user_input_pause_turn(core, result)
            return _ToolTurnResult(terminal=True)
        return _ToolTurnResult(last_tool_result=result)

    def _user_input_pause_turn(
        self, core: TurnExecutionContext, result: RunToolsResult
    ) -> Iterator[StreamEvent]:
        self._record_user_input_pause(core.store, core.context.material_task_id, result)
        self._mark_inputs_handled(core.context)
        yield SegmentPaused(
            reason=USER_INPUT_PAUSE_REASON,
            task_id=core.storage_task_id,
            segment_id=core.context.segment_id,
            resumable=True,
        )
        yield from self._emit_lifecycle_changed(
            core.context,
            state=State.PAUSED,
            lifecycle="waiting_user",
            reason=USER_INPUT_PAUSE_REASON,
            checkpoint_state="waiting_user",
            resumable=True,
        )

    def _runtime_budget_pause_turn(
        self,
        core: TurnExecutionContext,
        decision: WatchdogDecision,
        *,
        pending_tool_call: dict[str, object] | None = None,
    ) -> Iterator[StreamEvent]:
        self._snapshot_runtime_budget(core)
        event_reason = _runtime_pause_event_reason(decision)
        summary = _runtime_pause_summary(event_reason)
        self.last_output = summary
        self._last_output_is_error = False
        core.store.update_summary(core.context.material_task_id, summary)
        yield SegmentPaused(
            reason=event_reason,
            task_id=core.storage_task_id,
            segment_id=core.context.segment_id,
            resumable=True,
        )
        yield from self._emit_lifecycle_changed(
            core.context,
            state=State.PAUSED,
            lifecycle="paused",
            reason=event_reason,
            checkpoint_state="paused",
            pending_tool_call=pending_tool_call,
            resumable=True,
        )

    def _production_context_builder(self) -> ProductionContextBuilder:
        return ProductionContextBuilder(
            self.data_root,
            system_prompt_provider=system_prompt_estimate,
        )

    def _ledger_writer(self) -> LedgerWriter:
        return LedgerWriter(LedgerStore(self.data_root), source="runtime.agent_loop")

    def _call_model(
        self,
        task: str,
        context: RunContext,
        last_tool_result: RunToolsResult | None,
        *,
        system_reminder: str | None = None,
        watchdog: Watchdog,
    ) -> Generator[StreamEvent, None, ModelCallResult | LLMPlan]:
        """协调准备、实际调用与供应商超窗后的重新拟合；传参：本轮输入与运行；返回：真实调用或策略错误。"""
        if self.model_runner is None:
            raise RuntimeError("llm_client is required for real turn pipeline")
        prepared = prepare_model_turn(
            task,
            context,
            builder=self._production_context_builder(),
            registry=self.tool_registry,
            policy=self.tool_policy,
            runner=self.model_runner,
            extensions=self.extension_execution,
            cancellation=self.cancellation,
            watchdog=watchdog,
            data_root=self.data_root,
            progress=self.progress,
            collaboration=self.collaboration,
            last_tool_result=last_tool_result,
            system_reminder=system_reminder,
        )
        if isinstance(prepared, LLMPlan):
            return prepared
        force = task == "/compact"
        while True:
            try:
                bundle = prepared.fit(force=force)
                prepared = replace(prepared, bundle=bundle)
            except (OSError, ValueError) as exc:
                raise ContextReadError(
                    "context_compaction_failed", context.storage_task_id, exc
                ) from exc
            call = yield from self.model_runner.invoke(
                bundle, context, watchdog, last_tool_result=last_tool_result
            )
            error = call.plan.model_error
            if (
                error is None
                or error.category not in {"context_overflow", "payload_too_large"}
                or prepared.compactor is None
            ):
                return call
            # 1. 【模型请求】【供应商超窗】失败尝试已由runner保存，新请求重新拟合并分配独立身份
            context.payload[_RUNTIME_BUDGET_EVIDENCE] = watchdog.budget_evidence()
            force = True

    def _finish_tool_policy_error(
        self,
        store: TaskStore,
        context: RunContext,
        error: ModelError,
    ) -> str:
        output = error.render_output()
        record_runtime_error(
            self.run_evidence,
            context,
            category=error.category,
            message=output,
            stage=error.stage or "execute",
            meta={"raw_summary": error.raw_summary},
        )
        self._finish_failed(store, context, output)
        return output

    def _recall_for_context(self, context: RunContext) -> str:
        """渲染同一次材料召回的完整正文；传参：当前运行；返回：供现有内部读取者使用的文本。"""
        from context.materials import render_materials

        history = self._read_conversation_history(context.session_id)
        notices, materials = self._production_context_builder().recall_for_context(
            context,
            history=history,
        )
        return "\n\n".join(
            part for part in (notices, render_materials(materials, {})) if part
        )

    def _task_summary_context(self, task_id: str) -> dict[str, str]:
        return self._production_context_builder().task_summary_context(task_id)

    def _read_conversation_history(
        self, session_id: str, *, limit: int | None = None
    ) -> HistorySelection:
        return self._production_context_builder().read_conversation_history(
            session_id,
            limit=limit,
        )

    def _finish_failed(
        self,
        store: TaskStore,
        context: RunContext,
        output: str,
        *,
        message_id: str = "",
    ) -> str:
        """保存失败回答并返回流式身份；传参：存储、上下文、正文和消息编号；返回：条目编号。"""
        storage_task_id = context.material_task_id
        self.last_output = output
        self._last_output_is_error = True
        entry_id = append_assistant_message(
            self.data_root,
            context.session_id,
            output,
            run_id=context.run_id,
            task_id=storage_task_id,
            message_id=message_id or None,
        )
        store.append_journal(storage_task_id, f"Failed:\n\n{output}")

        return entry_id

    def _finish_context_read_failed(
        self, store: TaskStore, context: RunContext, error: ContextReadError
    ) -> tuple[str, str]:
        """记录上下文读取失败的正文和身份；传参：存储、上下文与错误；返回：正文及条目编号。"""
        output = error.render_output()
        record_runtime_error(
            self.run_evidence,
            context,
            category=error.category,
            message=error.message,
            stage="context",
            meta=error.meta(),
        )
        entry_id = self._finish_failed(store, context, output)
        return output, entry_id

    def _finish_success(
        self,
        store: TaskStore,
        context: RunContext,
        output: str,
        *,
        message_id: str = "",
        reasoning: str = "",
    ) -> str:
        """保存最终回答并返回流式关联身份；传参：存储、上下文、正文及消息身份/思考；返回：条目编号。"""
        storage_task_id = context.material_task_id
        self.last_output = output
        self._last_output_is_error = False
        entry_id = append_assistant_message(
            self.data_root,
            context.session_id,
            output,
            run_id=context.run_id,
            task_id=storage_task_id,
            message_id=message_id or None,
            reasoning=reasoning,
        )
        store.update_summary(storage_task_id, output)
        store.append_journal(storage_task_id, f"Final answer:\n\n{output}")
        return entry_id

    def _record_user_input_pause(
        self,
        store: TaskStore,
        storage_task_id: str,
        result: RunToolsResult,
    ) -> None:
        output = _pause_summary_for_tool_result(result)
        self.last_output = output
        self._last_output_is_error = False
        store.update_summary(storage_task_id, output)
        store.append_journal(storage_task_id, f"Paused for user input:\n\n{output}")

    def _record_terminal_task_writeback(self, context: RunContext, status: str) -> None:
        """记录兼容收件箱的运行状态，正式目标状态只由目标操作修改。

        传参：context 为当前运行；status 为本次运行终态；返回：无
        """
        task_id = context.compatibility_task_id
        if task_id is None:
            return
        store = TaskStore(self.data_root)
        try:
            record = store.load_task(task_id)
            if record is not None and record.is_inbox:
                store.update_task_status(task_id, status)
        finally:
            store.close()

    def _record_tool_error(self, context: RunContext, result: RunToolsResult) -> str:
        return record_runtime_error(
            self.run_evidence,
            context,
            category=_extract_error_category(result) or "tool_execution_error",
            message=result.error or result.output,
            stage="tool",
            meta=result.meta,
        )

    def _handle_recoverable_model_error(
        self,
        context: RunContext,
        plan: LLMPlan,
        last_tool_result: RunToolsResult | None,
    ) -> RunToolsResult | None:
        error = plan.model_error
        if error is None:
            return None
        if error.category == "invalid_tool_arguments":
            tool_name = _tool_name_from_model_error(error.raw_summary)
            if last_tool_result is not None:
                tool_name = last_tool_result.tool_name or last_tool_result.action
            result = self._build_error_observation_result(
                action=tool_name,
                tool_name=tool_name,
                error_type="invalid_tool_arguments",
                message=error.summary,
                retryable=True,
                args_summary={"raw_summary": error.raw_summary},
                source="model_parse",
            )
            return self._apply_recovery_budget(context, result)
        if error.category == "invalid_model_protocol":
            result = self._build_error_observation_result(
                action="model",
                tool_name="model",
                error_type="invalid_model_protocol",
                message=error.summary,
                retryable=True,
                args_summary={"stage": error.stage or "parse"},
                source="model_parse",
            )
            return self._apply_recovery_budget(
                context,
                result,
                max_repeats=self.recovery_policy.protocol_error_repeats,
            )
        if error.category == "empty_response":
            result = self._build_error_observation_result(
                action="model",
                tool_name="model",
                error_type="empty_response",
                message=error.summary,
                retryable=True,
                args_summary={"stage": error.stage or "transport"},
                source="model",
            )
            return self._apply_recovery_budget(
                context,
                result,
                max_repeats=self.recovery_policy.empty_response_repeats,
            )
        return None

    def _handle_recoverable_tool_error(
        self, context: RunContext, result: RunToolsResult
    ) -> RunToolsResult:
        category = _extract_error_category(result) or "tool_execution_error"
        error_type = (
            "invalid_tool_arguments"
            if category == "invalid_input"
            else "tool_execution_error"
        )
        retryable = _tool_error_recovery_retryable(
            self.tool_registry,
            result,
            category,
        )
        replan_required = (
            category == "invalid_input"
            and not _is_missing_required_tool_argument(result)
        )
        contract = _recovery_contract(result)
        wrong_tool_misuse = contract is not None
        message = result.error or result.output
        if replan_required:
            if wrong_tool_misuse:
                suggested = ", ".join(_WORKSPACE_FILE_TOOLS)
                message = (
                    f"{message}\n"
                    "Tool likely misused (not just bad args). For an ordinary "
                    f"workspace file, use one of: {suggested} "
                    "(find_path locates the path from a bare filename, then "
                    "file_read/inspect reads it). Do not retry the same tool."
                )
            else:
                message = (
                    f"{message}\n"
                    "Tool likely misused (not just bad args). Re-plan: do you "
                    "actually need this tool here, or pick a different one?"
                )
            retryable = True
        observation = self._build_error_observation_result(
            action=result.action,
            tool_name=result.tool_name,
            error_type=error_type,
            message=message,
            retryable=retryable,
            args_summary=self._args_summary_for_last_tool_result(result),
            source="tool",
            partial_state=str(result.meta.get("partial_state", "")),
        )
        if replan_required:
            observation.meta["replan_required"] = True
        if contract is not None:
            observation.meta.update(contract)
        return self._apply_recovery_budget(context, observation)

    def _build_error_observation_result(
        self,
        *,
        action: str,
        tool_name: str,
        error_type: str,
        message: str,
        retryable: bool,
        args_summary: object,
        source: str,
        partial_state: str = "",
    ) -> RunToolsResult:
        return RunToolsResult.error_result(
            action=action or tool_name,
            tool_name=tool_name,
            error=f"{error_type}: {message}",
            summary="recoverable error observation",
            meta={
                "recoverable_observation": True,
                "source": source,
                "error_type": error_type,
                "error_message": message,
                "retryable": retryable,
                "args_summary": _jsonable(args_summary),
                "partial_state": partial_state,
            },
        )

    def _apply_recovery_budget(
        self,
        context: RunContext,
        result: RunToolsResult,
        *,
        max_repeats: int | None = None,
    ) -> RunToolsResult:
        if max_repeats is None:
            max_repeats = self.recovery_policy.recoverable_error_repeats
        key = _recovery_budget_key(result)
        scoped_key = _scoped_recovery_budget_key(context, key)
        count = self._failure_budgets.get(scoped_key, 0) + 1
        self._failure_budgets[scoped_key] = count
        result.meta["budget_scope"] = _recovery_budget_scope(context)
        budget_remaining = max(max_repeats - count, 0)
        result.meta["budget_key"] = key
        result.meta["budget_count"] = count
        result.meta["budget_remaining"] = budget_remaining
        exhausted = count > max_repeats
        if exhausted:
            result.meta["budget_exhausted"] = True
            result.meta["retryable"] = False
            result.error = (
                f"{result.meta.get('error_type', 'recoverable_error')}: "
                "retry budget exhausted; change strategy or ask the user instead."
            )
            result.output = result.error

        error_type = str(result.meta.get("error_type", ""))
        retryable = bool(result.meta.get("retryable", False))
        can_retry = retryable and not exhausted
        recovery_action = "retry" if can_retry else "fail"
        if error_type == "invalid_model_protocol" and can_retry:
            recovery_action = "correct"
        if result.meta.get("replan_required") and can_retry:
            recovery_action = "replan"
        final_outcome = _recovery_final_outcome(
            exhausted=exhausted,
            retryable=retryable,
        )
        record_runtime_error(
            self.run_evidence,
            context,
            category=error_type or "recoverable_error",
            message=str(result.meta.get("error_message", result.error or "")),
            stage=str(result.meta.get("source", "tool")),
            recovery_action=recovery_action,
            attempt_count=count,
            budget_total=max_repeats,
            budget_remaining=budget_remaining,
            final_outcome=final_outcome,
        )

        self.tool_history.append(
            {
                "action": result.action,
                "tool_name": result.tool_name,
                "status": result.status,
                "output": result.output,
                "error": result.error,
                "observation": {
                    "tool_name": result.tool_name,
                    "args_summary": result.meta.get("args_summary", {}),
                    "error_type": error_type,
                    "error_message": result.meta.get("error_message"),
                    "retryable": result.meta.get("retryable", False),
                    "replan_required": result.meta.get("replan_required", False),
                    "budget_remaining": budget_remaining,
                    "budget_exhausted": exhausted,
                },
            }
        )
        self._append_trajectory(
            {
                "type": "event",
                "event": "tool:error_observation",
                "ts": utc_now(),
                "session_id": context.session_id,
                "run_id": context.run_id,
                "task_id": context.task_id,
                "focus_task_id": context.focus_task_id,
                "compatibility_task_id": context.compatibility_task_id,
                "segment_id": context.segment_id,
                "tool_name": result.tool_name,
                "error_category": error_type,
                "error": result.error,
                "retryable": result.meta.get("retryable", False),
                "replan_required": result.meta.get("replan_required", False),
                "budget_count": count,
                "budget_remaining": budget_remaining,
                "budget_exhausted": exhausted,
                "recovery_action": recovery_action,
                "args_summary": result.meta.get("args_summary", {}),
            },
        )
        return result

    def _args_summary_for_last_tool_result(self, result: RunToolsResult) -> object:
        for item in reversed(self.tool_history):
            if (
                item.get("tool_name") == result.tool_name
                and item.get("status") == result.status
                and item.get("error") == result.error
            ):
                return item.get("args_summary", {})
        return {}

    def _write_segment_start(self, task_dir: Path, context: RunContext) -> None:
        del task_dir
        self._append_trajectory(
            {
                "type": "lease_snapshot",
                "ts": utc_now(),
                "session_id": context.session_id,
                "run_id": context.run_id,
                "segment_id": context.segment_id,
                "parent_segment_id": context.parent_segment_id,
                "trigger": context.trigger.value,
                "task_id": context.task_id,
                "focus_task_id": context.focus_task_id,
                "focus_task": context.focus_task,
                "compatibility_task_id": context.compatibility_task_id,
                "lease_snapshot": asdict(context.capability_lease),
            },
        )

    def _append_trajectory(self, row: dict[str, Any]) -> None:
        self.run_facts.append_from_trajectory(row)

    def _task_dir(self, task_id: str) -> Path:
        return TaskStore(self.data_root).task_dir(task_id)

    def _record_checkpoint_fact(
        self, context: RunContext, checkpoint: Checkpoint
    ) -> None:
        self._ledger_writer().record_checkpoint_saved(
            checkpoint.checkpoint_id,
            checkpoint_to_ledger_state(checkpoint),
            checkpoint.reason,
            task_id=context.storage_task_id,
            session_id=context.session_id,
            run_id=context.run_id,
        )
        self.run_facts.append_checkpoint_ref(
            session_id=context.session_id,
            run_id=context.run_id,
            task_id=context.task_id,
            focus_task_id=context.focus_task_id,
            compatibility_task_id=context.compatibility_task_id,
            segment_id=context.segment_id,
            checkpoint=checkpoint,
        )


def _int_meta_value(value: object, default: int) -> int:
    """把 meta 数值字段收窄为 int。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：value 为 meta 原始字段，default 为缺省值
    返回：整数形式的 meta 字段
    """
    if value is None:
        return default
    if not isinstance(value, (str, bytes, bytearray, int)):
        raise ValueError("runtime meta value must be an integer")
    return int(value)


def _terminal_state_from_checkpoint(state: str) -> State | None:
    text = state.strip().lower()
    if text == "done":
        return State.DONE
    if text == "failed":
        return State.FAILED
    if text in {"paused", "waiting_user", "waiting_approval"}:
        return State.PAUSED
    return None


def _summarize_capabilities(capabilities: dict[str, object]) -> dict[str, Any]:
    """Compact view of `lease.capabilities` for the LeaseSnapshot event.
    Drops verbose path lists so the renderer's dim summary line stays
    one-line; the dashboard view fetches the full lease separately."""
    summary: dict[str, Any] = {}
    for key, value in capabilities.items():
        if isinstance(value, dict):
            enabled = value.get("enabled")
            if enabled is None:
                # fs/network capabilities rely on path lists, not an enabled
                # flag; surface a count instead of dumping every path.
                read_count = (
                    len(value["read"]) if isinstance(value.get("read"), list) else None
                )
                write_count = (
                    len(value["write"])
                    if isinstance(value.get("write"), list)
                    else None
                )
                if read_count is not None or write_count is not None:
                    summary[key] = {
                        "read_paths": read_count or 0,
                        "write_paths": write_count or 0,
                    }
                else:
                    summary[key] = "configured"
            else:
                summary[key] = bool(enabled)
        else:
            summary[key] = value
    return summary


def _usage_dict(plan: LLMPlan) -> dict[str, int]:
    """Pull token usage out of an LLMPlan.observation for the
    AssistantTurnComplete event. Returns an empty dict when usage is missing
    (provider may omit per `llm-guidelines.md`)."""
    observation = plan.observation
    if observation is None:
        return {}
    usage: dict[str, int] = {}
    for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(observation, attr, None)
        if isinstance(value, int):
            usage[attr] = value
    return usage


def _runtime_pause_event_reason(decision: WatchdogDecision) -> str:
    return RUNTIME_PAUSE_EVENT_REASONS.get(
        decision.reason,
        decision.message or decision.reason or "runtime budget paused",
    )


def _runtime_pause_summary(event_reason: str) -> str:
    return f"SEGMENT_PAUSED: {event_reason}; resume this task to continue."


# 当模型把普通工作区文件名/路径误送给 artifact-only 工具时，
# 这些基础发现/读取原语就是可执行的替代动作。find_path 先按名定位路径。
_WORKSPACE_FILE_TOOLS = ("find_path", "file_read", "inspect")

# artifact-only 工具只认 artifact_id，被喂普通文件名/路径就是 wrong_tool。
# 锚定工具身份而非 invalid_input category：terminal 缺 command、web_fetch
# 坏 URL 同样是 invalid_input，但那是「对的工具、错的参数」，不是用错工具，
# 不能推文件工具建议（否则就是本卡要消灭的方向性错误引导的镜像）。
_ARTIFACT_ONLY_TOOLS = frozenset({"read_artifact"})


def _is_wrong_tool_misuse(result: RunToolsResult) -> bool:
    if _extract_error_category(result) != "invalid_input":
        return False
    if _is_missing_required_tool_argument(result):
        return False
    return result.tool_name in _ARTIFACT_ONLY_TOOLS


def _recovery_contract(result: RunToolsResult) -> dict[str, object] | None:
    """Structured recovery contract for an artifact-only tool misused on an
    ordinary workspace file: category / recovery_possible / suggested_next_tools.
    Returns None for any other failure (missing argument, wrong-parameter on the
    right tool, etc.) so the contract never mislabels them as wrong_tool."""
    if not _is_wrong_tool_misuse(result):
        return None
    return {
        "category": "wrong_tool",
        "recovery_possible": True,
        "suggested_next_tools": list(_WORKSPACE_FILE_TOOLS),
    }


def _pending_resume_tool(context: RunContext) -> dict[str, object] | None:
    """读取当前待恢复调用并保留原始关联字段；传参：运行；返回：待决记录或无。"""
    if context.trigger is not Trigger.RESUME:
        return None
    if context.payload.get(_RESUME_PENDING_CONSUMED):
        return None
    pending = context.payload.get("pending_tool_call")
    if not isinstance(pending, dict):
        return None
    tool_name = str(pending.get("tool_name", "")).strip()
    if not tool_name:
        return None
    args = pending.get("args")
    return {
        **pending,
        "tool_name": tool_name,
        "args": dict(args) if isinstance(args, dict) else {},
        "call_id": str(pending.get("call_id", "")).strip(),
    }


def _pending_calls_match(
    left: Mapping[str, object], right: Mapping[str, object]
) -> bool:
    """比较 lifecycle 显式 pending 与 active resume pending 的核心身份"""
    return _pending_call_identity(left) == _pending_call_identity(right)


def _pending_call_identity(value: Mapping[str, object]) -> dict[str, object]:
    """比较待决记录的调用与发起归属；传参：记录；返回：身份投影。"""
    args = value.get("args")
    return {
        "tool_name": str(value.get("tool_name", "")).strip(),
        "args": dict(args) if isinstance(args, dict) else {},
        "call_id": str(value.get("call_id", "")).strip(),
        "operation_task_id": value.get("operation_task_id"),
        "request_id": value.get("request_id", ""),
    }


def _model_plan_action(plan: LLMPlan) -> str:
    if plan.model_error is not None:
        return "protocol_error"
    if plan.final_output is not None:
        return "final"
    if plan.run_tools_request is not None:
        return "run_tools"
    return "invalid"


def _pending_resume_evidence(
    pending: Mapping[str, object], idempotent: CheckpointIdempotent | None
) -> dict[str, object]:
    """组装叫醒模型前注入的 pending evidence（FW-B03 最小充分信息）。
    作者：LKX
    时间：2026-07-05 00:00:00
    传参：pending 为 pending_tool_call 映射；idempotent 为该工具的幂等性
    返回：含 tool_name + 幂等性 + 风险/partial_state 的 evidence 字典"""
    tool_name = str(pending.get("tool_name", ""))
    args = pending.get("args")
    return {
        "tool_name": tool_name,
        "idempotency": idempotent or "unknown",
        "args": dict(args) if isinstance(args, dict) else {},
        "call_id": str(pending.get("call_id", "")).strip(),
    }


def _pending_payload_for_call(call: ToolOperation) -> dict[str, object]:
    """保存未启动调用的已知身份；传参：调用；返回：旧记录不补造缺失归属的载荷。"""
    payload: dict[str, object] = {
        "tool_name": call.tool_name,
        "args": dict(call.args),
        "call_id": call.call_id,
        "operation_id": call.operation_id,
    }
    if call.task_id is not None:
        payload["operation_task_id"] = call.task_id
    if call.request_id:
        payload["request_id"] = call.request_id
    return payload


def _tool_can_be_retried_safely(registry: ToolRegistry, tool_name: str) -> bool:
    definition = registry.get(tool_name)
    if definition is None:
        return False
    if definition.risk.value != "safe":
        return False
    return (
        definition.idempotent is Idempotent.YES or str(definition.idempotent) == "yes"
    )


def _tool_error_recovery_retryable(
    registry: ToolRegistry,
    result: RunToolsResult,
    category: str,
) -> bool:
    if not _tool_can_be_retried_safely(registry, result.tool_name):
        return False
    if bool(result.meta.get("retryable", False)):
        return True
    return category == "invalid_input" and _is_missing_required_tool_argument(result)


def _is_missing_required_tool_argument(result: RunToolsResult) -> bool:
    text = (result.error or result.output or "").strip()
    if text.casefold().startswith("invalid_input:"):
        text = text.split(":", maxsplit=1)[1].strip()
    return text.casefold().startswith("missing required parameters for tool:")


def _recovery_budget_key(result: RunToolsResult) -> str:
    error_type = result.meta.get("error_type", "")
    global_budget_types = {"rate_limited", "invalid_model_protocol", "empty_response"}
    semantic_key = str(result.meta.get("semantic_key", "")).strip()
    if semantic_key:
        payload = {"error_type": error_type, "semantic_key": semantic_key}
    elif result.meta.get("category") == "wrong_tool":
        payload = {
            "error_type": error_type,
            "category": "wrong_tool",
            "args_summary": result.meta.get("args_summary", {}),
        }
    elif error_type in global_budget_types:
        payload = {"error_type": error_type}
    else:
        payload = {
            "tool_name": result.tool_name,
            "error_type": error_type,
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _recovery_budget_scope(context: RunContext) -> str:
    return context.segment_id or context.run_id


def _scoped_recovery_budget_key(context: RunContext, budget_key: str) -> str:
    payload = {
        "scope": _recovery_budget_scope(context),
        "budget_key": budget_key,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _tool_name_from_model_error(raw_summary: str) -> str:
    prefix = "tool="
    if raw_summary.startswith(prefix):
        return raw_summary.removeprefix(prefix).split(";", maxsplit=1)[0] or "unknown"
    return "unknown"


def _is_user_input_pause_tool(tool_name: str) -> bool:
    return tool_name in USER_INPUT_PAUSE_TOOLS


def _pause_summary_for_tool_result(result: RunToolsResult) -> str:
    text = result.output.strip()
    if not text:
        text = result.content.strip() if result.content else ""
    return text or f"SEGMENT_PAUSED: {USER_INPUT_PAUSE_REASON}"


def _recovery_final_outcome(*, exhausted: bool, retryable: bool) -> str:
    if exhausted:
        return "exhausted"
    if not retryable:
        return "failed"
    return ""


__all__ = ["AgentLoop", "State"]
