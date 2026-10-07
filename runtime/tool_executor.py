"""拥有完整工具批次的准备、授权、派发、结果提交和恢复投影。

作者：xxx
时间：2026-09-13 22:00:00
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from runtime.tool_operations import ToolOperation
from tools.tool_registry import PreparedToolExecution, ToolDefinition, ToolRegistry
from tools.file_resources import FILE_READERS, FILE_WRITERS, resources_conflict


import json
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import cast

from approval import ApprovalUnavailable
from approval.batch import BatchAuthorizer
from tools.file_resources import describe_resource
from runtime.progress import ProgressGuard, observe

from context.artifact_ref import store_large_output
from llm.base import LLMClient
from llm.messages import AssistantMessage, JsonValue, ToolCallPart, thaw_json_value
from llm.toolset_policy import (
    policy_to_mapping,
)
from llm.types import (
    LLMPlan,
)
from runtime.checkpoint import (
    Checkpoint,
    checkpoint_to_ledger_state,
    save_post_tool_checkpoint,
    save_pre_tool_checkpoint,
)
from runtime.ledger import LedgerStore
from runtime.ledger_writer import LedgerWriter
from runtime.run_evidence import RunEvidenceStore
from runtime.native_actions import NativeActionContext, NativeActions, ResumeTarget
from runtime.memory_actions import MEMORY_ACTIONS, MemoryActions
from runtime.skill_actions import SKILL_ACTIONS, SkillActions
from runtime.knowledge_jobs import KNOWLEDGE_ACTIONS, KnowledgeJobs
from runtime.context_actions import CONTEXT_ACTIONS, ContextActions
from runtime.scheduled_actions import SCHEDULED_ACTIONS, ScheduledActions
from runtime.run_facts import RunFactStore
from runtime.session_messages import (
    ToolExchange,
    append_tool_exchange,
    append_tool_calls,
    materialize_messages,
)
from runtime.session_state import SessionStateStore
from runtime.session_message_store import SessionMessageStore
from runtime.stream_events import (
    StreamEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from runtime.types import RunContext
from runtime.tool_operations import (
    ToolOperationStore,
    new_operation_id,
    operation_payload,
)
from runtime.cancellation import CancellationToken
from runtime.extensions import ResultView, RuntimeObservation, ToolProposal, run_hook
from runtime.types import RunToolsRequest, RunToolsResult
from runtime.lease import Lease
from runtime.collaboration import CollaborationRuntime
from tasks.ids import new_ulid, utc_now
from tools.types import ToolError, ToolErrorCategory
from tools.agent_tools import AGENT_ACTIONS


from runtime.execution_context import OperationOrigin, TurnExecutionContext
from runtime.extension_execution import ExtensionExecution
from runtime.tool_policy import RuntimeToolPolicy
from runtime.tool_results import (
    MAX_TOOL_OUTPUT_CHARS,
    _to_run_tools_result,
    _render_tool_conversation,
    _jsonable,
    _extract_error_category,
    _tool_prompt_truncated,
    execution_display_details,
)

_RESUME_PENDING_CONSUMED = "_resume_pending_consumed"


def execution_groups(
    calls: Sequence[ToolOperation],
    registry: ToolRegistry,
    *,
    prepared: Mapping[str, object] | None = None,
) -> Iterator[tuple[ToolOperation, ...]]:
    """允许明确独立工具和无冲突文件写并行，其余操作形成串行边界。

    传参：按模型顺序的调用、定义快照；返回：保持先后约束的可并行调用组
    """
    independent: list[ToolOperation] = []
    for call in calls:
        execution = prepared.get(call.operation_id) if prepared is not None else None
        definition = (
            execution.definition
            if isinstance(execution, PreparedToolExecution)
            else registry.get(call.tool_name)
        )
        resource = call.resource or {}
        parallel = allows_parallel_execution(definition)
        if (
            definition is not None
            and definition.name in FILE_WRITERS | FILE_READERS
            and not resource.get("known")
        ):
            parallel = False
        conflict = resource.get("known") and any(
            item.resource
            and item.resource.get("known")
            and resources_conflict(resource, item.resource)
            for item in independent
        )
        if independent and (not parallel or conflict):
            yield tuple(independent)
            independent.clear()
        if parallel:
            independent.append(call)
            continue
        yield (call,)
    if independent:
        yield tuple(independent)


def allows_parallel_execution(definition: ToolDefinition | None) -> bool:
    """仅对明确没有共享写状态的后端允许并行；传参：最终定义；返回：是否可并行。"""
    return (
        definition is not None
        and definition.parallel_safe
        and not definition.runtime_action
        and (
            definition.readonly
            or (definition.source == "builtin" and definition.name in FILE_WRITERS)
        )
    )


class ToolBatchExecutor:
    """拥有整批授权、后端派发、有序提交与恢复投影，不拥有模型反馈循环。"""

    def __init__(
        self,
        data_root: Path,
        *,
        registry: ToolRegistry,
        authorizer: BatchAuthorizer,
        cancellation: CancellationToken,
        extension_execution: ExtensionExecution,
        policy: RuntimeToolPolicy,
        operations: ToolOperationStore,
        facts: RunFactStore,
        evidence: RunEvidenceStore,
        states: SessionStateStore,
        progress: ProgressGuard,
        history: list[dict[str, object]],
        client: LLMClient | None,
        collaboration: CollaborationRuntime | None,
    ) -> None:
        """接收既有执行依赖与共享进展；返回：不持有主循环的执行器。"""
        self.data_root = data_root
        self.tool_registry = registry
        self.authorizer = authorizer
        self.cancellation = cancellation
        self.extension_execution = extension_execution
        self.extensions = extension_execution.extensions
        self.policy = policy
        self.operations = operations
        self.run_facts = facts
        self.run_evidence = evidence
        self.session_states = states
        self.progress = progress
        self.tool_history = history
        self.llm_client = client
        self.collaboration = collaboration
        self._prepared_resumes: dict[str, PreparedToolExecution] = {}
        self._resume_sources: dict[tuple[str, str], str] = {}

    def build_call(
        self,
        request: RunToolsRequest,
        *,
        context: RunContext,
        request_id: str,
        registry: ToolRegistry | None = None,
    ) -> ToolOperation:
        """冻结工具发起时的目标、请求与操作身份；传参：请求与运行；返回：调用记录。"""
        snapshot = registry or self.tool_registry.snapshot()
        return ToolOperation(
            request=request,
            call_id=request.call_id or f"call-{new_ulid()}",
            tool_name=request.tool_name or request.action,
            args=dict(request.arguments),
            task_id=context.material_task_id,
            request_id=request_id,
            operation_id=new_operation_id(),
            registry_version=snapshot.version,
            definition_version=snapshot.definition_version(
                request.tool_name or request.action
            ),
            resource=describe_resource(
                snapshot.get(request.tool_name or request.action),
                request.arguments,
                context.capability_lease,
            ),
        )

    def execute(
        self,
        core: TurnExecutionContext,
        calls: tuple[ToolOperation, ...],
        *,
        registry: ToolRegistry | None = None,
    ) -> Generator[StreamEvent, None, list[RunToolsResult]]:
        """先公告整组，再按后端约束执行和提交结果；传参：运行、调用和授权；返回：原顺序结果。"""
        try:
            self._announce_tool_batch(core.context, calls)
            snapshot = registry or self.tool_registry.snapshot()
            prepared = self._authorize_tool_batch(core, calls, registry=snapshot)
            # 【文件并行】【实际资源】扩展可能调整候选，以授权后的真实工具和参数决定冲突
            calls = tuple(
                replace(
                    call,
                    resource=describe_resource(
                        item.definition, item.arguments, item.lease
                    ),
                    definition_version=item.definition_version,
                )
                if isinstance(
                    item := prepared[call.operation_id], PreparedToolExecution
                )
                else call
                for call in calls
            )
            results: list[RunToolsResult] = []
            for group in execution_groups(calls, snapshot, prepared=prepared):
                group_results = yield from self._execute_tool_group(
                    core, group, registry=snapshot, prepared_batch=prepared
                )
                self._observe_progress_batch(
                    core.context,
                    group,
                    group_results,
                    prepared=prepared,
                    registry=snapshot,
                )
                results.extend(group_results)
            # 【执行器】【提交结果】全部调用已经回填后才能清除恢复检查点
            if calls[-1].origin != "extension":
                self._record_post_tool_checkpoint(core.context, calls[-1])
            return results
        finally:
            self._prepared_resumes.clear()
            self._resume_sources.clear()

    def _observe_progress_batch(
        self,
        context: RunContext,
        calls: tuple[ToolOperation, ...],
        results: list[RunToolsResult],
        *,
        prepared: dict[str, PreparedToolExecution | RunToolsResult],
        registry: ToolRegistry,
    ) -> None:
        """整组结果提交后读取当前文件环境，事实与决策共用观察；传参：运行、操作、结果和实际准备；返回：无。"""
        for call, result in zip(calls, results):
            resource = call.resource or {}
            if (
                resource.get("known")
                and resource.get("write")
                and result.meta.get("execution_state") != "not_started"
            ):
                self.progress.tracked_paths.add(str(resource["path"]))
        observations = []
        for call, result in zip(calls, results):
            item = prepared[call.operation_id]
            definition = (
                item.definition
                if isinstance(item, PreparedToolExecution)
                else registry.get(call.tool_name)
            )
            observations.append(
                observe(
                    call, result, definition, tracked_paths=self.progress.tracked_paths
                )
            )
        for observation in self.progress.observe_batch(observations):
            signal = (
                observation.decision == "suspected"
                and observation.repeats == self.progress.remind_after
            )
            event = "progress:no_progress" if signal else "progress:observed"
            self.run_facts.append(
                {
                    "event": event,
                    "session_id": context.session_id,
                    "run_id": context.run_id,
                    "task_id": context.material_task_id,
                    "evidence": asdict(observation),
                }
            )

    def _authorize_tool_batch(
        self,
        core: TurnExecutionContext,
        calls: tuple[ToolOperation, ...],
        *,
        registry: ToolRegistry,
    ) -> dict[str, PreparedToolExecution | RunToolsResult]:
        """准备全批后集中授权，原自动放行项也等待交互结束；传参：运行、调用和目录；返回：逐项授权结果。"""
        batch_id = f"batch-{calls[0].operation_id}"
        prepared = {
            call.operation_id: self._prepare_operation(
                core, call, registry=registry, approval_batch=batch_id
            )
            for call in calls
        }
        targets = [
            item.authorization_target()
            for item in prepared.values()
            if isinstance(item, PreparedToolExecution)
        ]
        requests = tuple(
            item.approval_request
            for item in targets
            if item.approval_request is not None
        )
        try:
            decisions = self.authorizer.authorize(requests)
        except (ApprovalUnavailable, OSError, ValueError) as exc:
            error = ToolError(
                ToolErrorCategory.TRANSPORT,
                "approval facility unavailable",
                retryable=False,
                details={
                    "approval_state": "unavailable",
                    "execution_state": "not_started",
                },
                diagnostics={"approval_cause": str(exc)},
            )
            return {
                call.operation_id: _to_run_tools_result(call.request, error)
                for call in calls
            }
        for call in calls:
            item = prepared[call.operation_id]
            if isinstance(item, PreparedToolExecution):
                authorization = decisions[call.operation_id]
                item = item.with_authorization(authorization, self.authorizer)
                prepared[call.operation_id] = item
                if item.approval_target is not None:
                    self._prepared_resumes[call.operation_id] = item.approval_target
                # 【审批】【派发证据】授权记录先于 started；任一写入失败都不开始本批副作用
                self.operations.write(
                    {
                        "session_id": core.context.session_id,
                        "run_id": core.context.run_id,
                        "operation_id": call.operation_id,
                    },
                    {"authorization": authorization.evidence()},
                )
        return prepared

    def _announce_tool_batch(
        self, context: RunContext, calls: tuple[ToolOperation, ...]
    ) -> None:
        """副作用前保存操作身份和模型调用组；传参：运行和调用；返回：无，写入失败阻止派发。"""
        for call in calls:
            self._save_tool_operation(context, call, state="not_started")
            self._append_tool_request_fact(context, call)
        messages = materialize_messages(self.data_root, context.session_id)
        announced = {
            part.call_id
            for message in messages
            if isinstance(message, AssistantMessage)
            for part in message.content
            if isinstance(part, ToolCallPart)
        }
        missing = tuple(
            ToolExchange(call.call_id, call.tool_name, call.args)
            for call in calls
            if call.call_id not in announced
        )
        if missing:
            source_id = (
                calls[0].request_id
                if messages
                and isinstance(messages[-1], AssistantMessage)
                and messages[-1].message_id == calls[0].request_id
                and all(call.request_id == calls[0].request_id for call in calls)
                else None
            )
            append_tool_calls(
                self.data_root,
                context.session_id,
                missing,
                run_id=context.run_id,
                task_id=calls[0].task_id,
                source_message_id=source_id,
            )

    def _execute_tool_group(
        self,
        core: TurnExecutionContext,
        calls: tuple[ToolOperation, ...],
        *,
        registry: ToolRegistry,
        prepared_batch: dict[str, PreparedToolExecution | RunToolsResult],
    ) -> Generator[StreamEvent, None, list[RunToolsResult]]:
        """只把后端执行交给线程，授权和持久化由主循环负责；传参：运行、调用及授权；返回：结果组。"""
        results: list[RunToolsResult] = []
        with ThreadPoolExecutor(max_workers=len(calls)) as executor:
            pending: list[tuple[ToolOperation, Future[object] | RunToolsResult]] = []
            for call in calls:
                prepared = prepared_batch[call.operation_id]
                if isinstance(prepared, RunToolsResult):
                    pending.append((call, prepared))
                    continue
                parallel = allows_parallel_execution(prepared.definition)
                if not parallel:
                    results.extend(
                        (yield from self._commit_pending_tools(core, pending))
                    )
                    pending.clear()
                invalid = registry.validate_prepared_tool(prepared)
                if invalid is not None:
                    pending.append((call, _to_run_tools_result(call.request, invalid)))
                    continue
                denied = self._reserve_tool_dispatch(core, call)
                if denied is not None:
                    pending.append((call, denied))
                    continue
                if call.origin != "extension":
                    self._record_pre_tool_checkpoint(core.context, call)
                call = replace(
                    call,
                    execution_request={
                        "tool": prepared.definition.name,
                        "arguments": dict(prepared.arguments),
                        "registry_version": prepared.registry_version,
                        "definition_version": prepared.definition_version,
                        "source": prepared.definition.source,
                    },
                )
                prepared = replace(
                    prepared,
                    on_late=partial(
                        self._record_late_tool_result,
                        OperationOrigin.capture(core.context),
                        call,
                    ),
                )
                self._save_tool_operation(core.context, call, state="started")
                yield ToolExecutionStarted(
                    tool_name=call.tool_name,
                    args=dict(call.args),
                    call_id=call.call_id,
                    risk=self._risk_for_tool(call.tool_name),
                )
                if prepared.definition.runtime_action:
                    outcome = self.tool_registry.execute_prepared_tool(
                        prepared,
                        runtime_executor=partial(
                            self._execute_native_action, core, call
                        ),
                    )
                    pending.append((call, _to_run_tools_result(call.request, outcome)))
                else:
                    pending.append(
                        (
                            call,
                            executor.submit(
                                self.tool_registry.execute_prepared_tool, prepared
                            ),
                        )
                    )
                if not parallel:
                    results.extend(
                        (yield from self._commit_pending_tools(core, pending))
                    )
                    pending.clear()
            results.extend((yield from self._commit_pending_tools(core, pending)))
        return results

    def _commit_pending_tools(
        self,
        core: TurnExecutionContext,
        pending: list[tuple[ToolOperation, Future[object] | RunToolsResult]],
    ) -> Generator[StreamEvent, None, list[RunToolsResult]]:
        """等待一组已派发结果并按公告顺序提交；传参：运行/待收结果；返回：实际结果列表。"""
        results: list[RunToolsResult] = []
        for call, outcome in pending:
            raw = outcome.result() if isinstance(outcome, Future) else outcome
            result = self._commit_tool_result(
                core, call, _to_run_tools_result(call.request, raw)
            )
            results.append(result)
            yield ToolExecutionCompleted(
                tool_name=call.tool_name,
                output=result.output,
                call_id=call.call_id,
                is_error=result.status != "ok",
                error_category=_extract_error_category(result),
                execution=execution_display_details(result.meta),
            )
        return results

    def _execute_native_action(
        self,
        core: TurnExecutionContext,
        call: ToolOperation,
        prepared: PreparedToolExecution,
    ) -> RunToolsResult:
        """在运行所有者线程执行原生动作，复用实际校验后的参数；传参：运行/操作/准备结果；返回：工具结果。"""
        if prepared.definition.name in SCHEDULED_ACTIONS:
            assert self.llm_client is not None
            actions = ScheduledActions(
                core.context, data_root=self.data_root, client=self.llm_client
            )
            return actions.execute(
                replace(
                    call,
                    tool_name=prepared.definition.name,
                    args=dict(prepared.arguments),
                )
            )
        if prepared.definition.name in AGENT_ACTIONS:
            if self.collaboration is None:
                raise RuntimeError("collaboration runtime is missing")
            policy = self.policy.resolve(core.context)
            if isinstance(policy, LLMPlan):
                raise ValueError("collaboration requires a valid inherited tool policy")
            return self.collaboration.execute(
                core.context,
                replace(
                    call,
                    tool_name=prepared.definition.name,
                    args=dict(prepared.arguments),
                ),
                toolset_policy=policy_to_mapping(policy),
            )
        # 【协作】【目标归属】子执行者可以完成自己的事项；共同父目标由整合者依据整体成果确认
        target_goal = prepared.arguments.get("goal_ref", core.context.focus_task_id)
        if (
            prepared.definition.name == "goal"
            and prepared.arguments.get("action") == "complete"
            and core.context.parent_run_id is not None
            and target_goal == core.context.task_id
        ):
            return RunToolsResult.error_result(
                action="goal",
                error="parent goal completion belongs to its integrating agent; send your results to parent",
            )
        native = self._native_actions(core)
        if prepared.definition.name in CONTEXT_ACTIONS:
            return ContextActions(native.context, data_root=self.data_root).execute(
                replace(
                    call, tool_name=prepared.definition.name, args=prepared.arguments
                )
            )
        if prepared.definition.name in MEMORY_ACTIONS:
            return MemoryActions(native.context, data_root=self.data_root).execute(
                replace(
                    call, tool_name=prepared.definition.name, args=prepared.arguments
                )
            )
        if prepared.definition.name in SKILL_ACTIONS:
            return SkillActions(native.context, data_root=self.data_root).execute(
                replace(
                    call, tool_name=prepared.definition.name, args=prepared.arguments
                )
            )
        if prepared.definition.name in KNOWLEDGE_ACTIONS:
            assert self.llm_client is not None
            return KnowledgeJobs(
                native.context, data_root=self.data_root, client=self.llm_client
            ).execute(
                replace(
                    call, tool_name=prepared.definition.name, args=prepared.arguments
                )
            )
        outcome = native.execute(
            replace(call, tool_name=prepared.definition.name, args=prepared.arguments)
        )
        if isinstance(outcome, ResumeTarget):
            return self._retry_native_operation(core, call, outcome)
        return outcome

    def _native_actions(self, core: TurnExecutionContext) -> NativeActions:
        """为候选读取与真实派发注入同一组原生动作依赖；传参：运行；返回：动作服务。"""
        return NativeActions(
            NativeActionContext(
                core.context,
                core.store,
                SessionMessageStore(self.data_root),
                self.session_states,
                self.operations,
                self.run_facts,
                self.tool_registry,
                self.policy.config(),
                ledger=LedgerStore(self.data_root),
                policy=self.policy,
            )
        )

    def _retry_native_operation(
        self, core: TurnExecutionContext, call: ToolOperation, resumed: ResumeTarget
    ) -> RunToolsResult:
        """恢复目标再次经过参数、路径、审批和预算边界；传参：当前运行/恢复调用/原请求；返回：实际结果。"""
        # 1. 【操作恢复】【认领派发】原生解析已重读当前分支和效果，认领者才能派发原操作
        original = {
            "session_id": resumed.session_id,
            "run_id": resumed.run_id,
            "operation_id": resumed.operation_id,
        }
        authorized_source = self._resume_sources[
            (call.operation_id, resumed.operation_id)
        ]
        try:
            if resumed.source != authorized_source:
                raise ValueError(
                    "operation changed after recovery authorization; inspect current effects"
                )
            previous = self.operations.claim_retry(
                original, call.operation_id, expected_source=authorized_source
            )
        except ValueError as exc:
            return RunToolsResult.error_result(
                action=call.tool_name,
                error=str(exc),
                meta={"execution_state": "not_started"},
            )
        if previous != call.operation_id:
            return RunToolsResult.ok(
                action=call.tool_name,
                content="该操作已有恢复尝试，请查询该尝试的实际结果",
                meta={"retry_operation_id": previous},
            )
        request = resumed.request
        target = replace(
            call,
            request=request,
            tool_name=request.tool_name,
            args=dict(request.arguments),
            task_id=resumed.task_id,
        )
        prepared = self._prepared_resumes.pop(call.operation_id)
        if prepared.approval_target is not None:
            self._prepared_resumes[call.operation_id] = prepared.approval_target
        invalid = self.tool_registry.validate_prepared_tool(prepared)
        if invalid is not None:
            return _to_run_tools_result(call.request, invalid)
        denied = self._reserve_tool_dispatch(core, call)
        if denied is not None:
            return denied
        execution: dict[str, object] = {
            "tool": prepared.definition.name,
            "arguments": dict(prepared.arguments),
            "task_id": resumed.task_id,
        }
        owner = replace(call, execution_request=execution)
        self._save_tool_operation(core.context, owner, state="started")
        prepared = replace(
            prepared,
            on_late=partial(
                self._record_late_tool_result,
                OperationOrigin.capture(core.context),
                owner,
            ),
        )
        raw = self.tool_registry.execute_prepared_tool(
            prepared,
            runtime_executor=partial(self._execute_native_action, core, target),
        )
        result = _to_run_tools_result(request, raw)
        return replace(
            result,
            action=call.tool_name,
            tool_name=call.tool_name,
            meta={
                **result.meta,
                "resumed_operation_id": call.args["operation_id"],
                "resumed_execution_request": execution,
            },
        )

    def _prepare_operation(
        self,
        core: TurnExecutionContext,
        call: ToolOperation,
        *,
        registry: ToolRegistry | None = None,
        approval_batch: str | None = None,
    ) -> PreparedToolExecution | RunToolsResult:
        """在真实派发前核对参数、权限和预算；传参：运行、调用和授权；返回：可执行对象或未执行结果。"""
        error = None
        if self.cancellation.cancelled:
            return RunToolsResult.error_result(
                action=call.tool_name,
                error="cancelled before dispatch",
                meta={
                    "execution_state": "not_started",
                    "tool_error_category": "cancelled",
                },
            )
        if self.has_pending_inputs(core.context):
            return RunToolsResult.error_result(
                action=call.tool_name,
                error="new input arrived before dispatch; reconsider this action",
                meta={
                    "execution_state": "not_started",
                    "tool_error_category": "superseded_by_input",
                },
            )
        if call.request.validation_error:
            error = ToolError(
                ToolErrorCategory.INVALID_INPUT,
                call.request.validation_error,
                retryable=False,
            )
        proposal = (
            self._prepare_extension(core, call) if error is None else call.request
        )
        if isinstance(proposal, ToolError):
            return _to_run_tools_result(call.request, proposal)
        policy = self.policy.validate(core.context, proposal)
        if policy is not None:
            error = ToolError(
                ToolErrorCategory.PERMISSION, policy.render_output(), retryable=False
            )
        if error is None:
            lease = _lease_for_operation(core.context.capability_lease, call)
            prepared = (
                registry or self.tool_registry.snapshot()
            ).prepare_tool_execution(
                proposal.tool_name,
                proposal.arguments,
                lease,
                watchdog=core.watchdog,
                cancellation=CancellationToken(self.cancellation),
                operation_id=call.operation_id,
                on_late=partial(
                    self._record_late_tool_result,
                    OperationOrigin.capture(core.context),
                    call,
                ),
                superseded=partial(self.has_pending_inputs, core.context),
                approval_batch=approval_batch,
            )
            if not isinstance(prepared, ToolError):
                prepared.capture_identity = {
                    "session_id": core.context.session_id,
                    "run_id": core.context.run_id,
                    "input_id": str(core.context.payload.get("input_id", "")),
                }
                if (
                    prepared.definition.name == "resume_operation"
                    and prepared.arguments.get("action") == "retry"
                ):
                    try:
                        resumed = self._native_actions(core).prepare_resume(
                            replace(call, args=prepared.arguments)
                        )
                    except ValueError as exc:
                        return _to_run_tools_result(
                            call.request,
                            ToolError(
                                ToolErrorCategory.INVALID_INPUT,
                                str(exc),
                                retryable=False,
                            ),
                        )
                    if isinstance(resumed, RunToolsResult):
                        return resumed
                    target_call = replace(
                        call,
                        request=resumed.request,
                        tool_name=resumed.request.tool_name,
                        args=dict(resumed.request.arguments),
                        task_id=resumed.task_id,
                    )
                    target = self._prepare_operation(
                        core,
                        target_call,
                        registry=registry,
                        approval_batch=approval_batch,
                    )
                    if isinstance(target, RunToolsResult):
                        return target
                    self._resume_sources[(call.operation_id, resumed.operation_id)] = (
                        resumed.source
                    )
                    return replace(prepared, approval_target=target)
                return prepared
            error = prepared
        result = _to_run_tools_result(call.request, error)
        return replace(result, meta={**result.meta, "execution_state": "not_started"})

    def _reserve_tool_dispatch(
        self, core: TurnExecutionContext, call: ToolOperation
    ) -> RunToolsResult | None:
        """最终排定执行后检查新输入并记下本次步数；传参：运行/操作；返回：被新输入顶掉时的拒绝结果，否则放行。"""
        if self.cancellation.cancelled or self.has_pending_inputs(core.context):
            return RunToolsResult.error_result(
                action=call.tool_name,
                error="dispatch cancelled or superseded by new input",
                meta={"execution_state": "not_started"},
            )
        core.watchdog.reserve_tool_step(operation_id=call.operation_id)
        return None

    def _prepare_extension(
        self, core: TurnExecutionContext, call: ToolOperation
    ) -> RunToolsRequest | ToolError:
        """等待扩展提出候选，不让它直接获得执行或授权能力；传参：运行/调用；返回：待重新校验的请求。"""
        if not self.extensions.before_tool:
            return call.request
        proposal = ToolProposal(
            call.tool_name, cast(Mapping[str, JsonValue], call.args), call.operation_id
        )
        token = CancellationToken(self.cancellation)
        try:
            value = core.watchdog.run_tool_with_timeout(
                partial(self.extensions.prepare, proposal, token), cancellation=token
            )
        except Exception as exc:
            return ToolError(
                ToolErrorCategory.UNKNOWN,
                f"before-tool extension failed: {exc}",
                retryable=False,
                details={"execution_state": "not_started"},
            )
        if isinstance(value, ToolError):
            return replace(
                value, details={**value.details, "execution_state": "not_started"}
            )
        if value.operation_id != call.operation_id:
            return ToolError(
                ToolErrorCategory.INVALID_INPUT,
                "extension cannot change operation identity",
                retryable=False,
                details={"execution_state": "not_started"},
            )
        return RunToolsRequest(
            action=value.tool,
            tool_name=value.tool,
            arguments=cast(dict[str, object], thaw_json_value(value.arguments)),
        )

    def _commit_tool_result(
        self, core: TurnExecutionContext, call: ToolOperation, result: RunToolsResult
    ) -> RunToolsResult:
        """先保存真实结果再提交会话投影；传参：运行、调用与结果；返回：带证据引用的结果。"""
        result = self._retain_tool_content(core.context, call, result)
        self._save_tool_operation(
            core.context, call, state=str(result.meta["execution_state"]), result=result
        )
        self._append_tool_response_fact(core.context, call, result)
        result = self._project_extension_result(core, call, result)
        self._save_tool_operation(
            core.context, call, state=str(result.meta["execution_state"]), result=result
        )
        self._persist_tool_exchange(core, call, result)
        self._append_tool_history(call, result)
        self.extension_execution.observe(
            core.context,
            "tool_committed",
            {
                "operation_id": call.operation_id,
                "status": result.status,
                "output": result.output,
                "error": result.error,
            },
            watchdog=core.watchdog,
        )
        return result

    def _save_tool_operation(
        self,
        context: RunContext | OperationOrigin,
        call: ToolOperation,
        *,
        state: str,
        result: RunToolsResult | None = None,
    ) -> None:
        """操作记录保留模型视图，长原文与诊断分别用引用；传参：运行、调用、状态和结果；返回：无。"""
        if result is not None:
            raw_view = json.loads(
                _render_tool_conversation(replace(result, model_view=None))
            )
            result = replace(
                result,
                output=raw_view["output"],
                content=raw_view["output"],
                error=raw_view["error"],
                diagnostics={},
            )
        self.operations.write(
            {
                "session_id": context.session_id,
                "run_id": context.run_id,
                "operation_id": call.operation_id,
            },
            operation_payload(call, state=state, result=result),
        )

    def _record_late_tool_result(
        self, context: OperationOrigin, call: ToolOperation, raw: object
    ) -> None:
        """运行结束后仍按原操作身份保存迟到证据；传参：原运行、调用及结果；返回：无。"""
        result = self._retain_tool_content(
            context, call, _to_run_tools_result(call.request, raw)
        )
        self._save_tool_operation(context, call, state="late_completed", result=result)
        self._append_tool_response_fact(context, call, result, late=True)

    def _project_extension_result(
        self, core: TurnExecutionContext, call: ToolOperation, result: RunToolsResult
    ) -> RunToolsResult:
        """原结果提交后派生模型视图，失败只增加独立注释；传参：运行/操作/原结果；返回：带可追溯视图的结果。"""
        observation = RuntimeObservation(
            "tool_result",
            {
                "operation_id": call.operation_id,
                "status": result.status,
                "output": result.output,
                "error": result.error,
            },
        )
        for hook in self.extensions.result_views:
            try:
                view = run_hook(
                    partial(hook, observation), core.watchdog, self.cancellation
                )
                if not isinstance(view, ResultView):
                    raise TypeError("result-view hook must return ResultView")
                result = replace(
                    result,
                    model_view=view.content,
                    annotations=(*result.annotations, *view.annotations),
                )
            except Exception as exc:
                self.extension_execution.record_error(
                    core.context, "result_view", str(exc)
                )
                result = replace(
                    result,
                    annotations=(*result.annotations, f"result view failed: {exc}"),
                )
        return result

    def _persist_tool_exchange(
        self,
        core: TurnExecutionContext,
        call: ToolOperation,
        result: RunToolsResult,
    ) -> None:
        """把一次工具交换提交为一对 canonical Session Entry。

        作者：xxx
        时间：2026-08-27 20:00:00
        传参：core 为本轮上下文；call 为工具调用记录；result 为工具执行结果
        返回：无；Ledger 事件保留为运行证据，不再作为下一轮消息重建来源
        """
        rendered = _render_tool_conversation(result)
        self._ledger_writer().record_tool_requested(
            call.tool_name,
            call.call_id,
            dict(call.args),
            task_id=call.task_id,
            session_id=core.context.session_id,
            run_id=core.context.run_id,
        )
        self._ledger_writer().record_tool_completed(
            call.tool_name,
            call.call_id,
            result.status,
            content=rendered,
            task_id=call.task_id,
            session_id=core.context.session_id,
            run_id=core.context.run_id,
        )
        # 1. 消息事实只提交一次：assistant tool_call 与 tool_result 由 call_id 关联
        append_tool_exchange(
            self.data_root,
            core.context.session_id,
            ToolExchange(
                call_id=call.call_id,
                tool_name=call.tool_name,
                args=dict(call.args),
                rendered=rendered,
                status=result.status,
                error=result.error,
                artifact_refs=tuple(
                    cast(list[str], result.meta.get("artifact_refs", []))
                ),
            ),
            run_id=core.context.run_id,
            task_id=call.task_id,
        )

    def _retain_tool_content(
        self,
        context: RunContext | OperationOrigin,
        call: ToolOperation,
        result: RunToolsResult,
    ) -> RunToolsResult:
        """保存超长原文和独立诊断，再生成模型引用。

        传参：运行、发起调用与真实结果；返回：带原文引用的结果，保存失败直接暴露
        """
        result = replace(
            result,
            meta={
                "execution_state": "completed",
                **result.meta,
                "operation_id": call.operation_id,
                "call_id": call.call_id,
                "execution_request": call.execution_request,
            },
        )
        if result.diagnostics:
            self.run_evidence.write_record(
                session_id=context.session_id,
                run_id=context.run_id,
                kind="tool_diagnostic",
                source_id=call.operation_id,
                payload={
                    "call_id": call.call_id,
                    "request_id": call.request_id,
                    "diagnostics": result.diagnostics,
                },
            )
        if len(result.output) <= MAX_TOOL_OUTPUT_CHARS:
            return result
        ref = store_large_output(
            self.data_root,
            call.task_id or context.material_task_id,
            result.output,
            threshold=MAX_TOOL_OUTPUT_CHARS,
            summary=result.summary or f"{call.tool_name} result",
        )
        assert ref is not None
        refs = [*cast(list[str], result.meta.get("artifact_refs", [])), ref.artifact_id]
        return replace(
            result,
            meta={
                **result.meta,
                "artifact_refs": refs,
                "result_artifact_id": ref.artifact_id,
                "read_full_result": {
                    "tool": "read_artifact",
                    "artifact_id": ref.artifact_id,
                    "mode": "full",
                    "offset": 0,
                },
            },
        )

    def _risk_for_tool(self, tool_name: str) -> str:
        """读取已声明风险供执行事件展示；传参：工具名；返回：风险或unknown。"""
        definition = self.tool_registry.get(tool_name)
        if definition is None:
            return "unknown"
        return definition.risk.value

    def _record_pre_tool_checkpoint(
        self, context: RunContext, call: ToolOperation
    ) -> None:
        """副作用前保留原操作与租约恢复依据；传参：运行、调用；返回：无。"""
        checkpoint = save_pre_tool_checkpoint(
            context.segment_id,
            call.tool_name,
            call.args,
            call.call_id,
            {"state": "before", "request": asdict(call.request)},
            operation_task_id=call.task_id,
            request_id=call.request_id,
            operation_id=call.operation_id,
            task_id=context.storage_task_id,
            session_id=context.session_id,
            run_id=context.run_id,
            focus_task_id=context.focus_task_id,
            terminal_focus_policy=context.terminal_focus_policy,
            compatibility_task_id=context.compatibility_task_id,
            lease_snapshot=asdict(context.capability_lease),
        )
        self._record_checkpoint_fact(context, checkpoint)

    def _record_post_tool_checkpoint(
        self, context: RunContext, call: ToolOperation
    ) -> None:
        """全批结果提交后关闭待恢复状态；传参：运行、调用；返回：无。"""
        post_checkpoint = save_post_tool_checkpoint(
            context.segment_id,
            {"state": "after"},
            task_id=context.storage_task_id,
            session_id=context.session_id,
            run_id=context.run_id,
            focus_task_id=context.focus_task_id,
            terminal_focus_policy=context.terminal_focus_policy,
            compatibility_task_id=context.compatibility_task_id,
            lease_snapshot=asdict(context.capability_lease),
        )
        self._record_checkpoint_fact(context, post_checkpoint)

    def _append_tool_request_fact(
        self, context: RunContext, call: ToolOperation
    ) -> None:
        """记录冻结操作身份及原始参数；传参：运行、调用；返回：无。"""
        self.run_facts.append_from_trajectory(
            {
                "type": "event",
                "event": "tool:request",
                "ts": utc_now(),
                "session_id": context.session_id,
                "run_id": context.run_id,
                "task_id": context.task_id,
                "focus_task_id": context.focus_task_id,
                "compatibility_task_id": context.compatibility_task_id,
                "segment_id": context.segment_id,
                "tool_call_id": call.call_id,
                "operation_task_id": call.task_id,
                "operation_id": call.operation_id,
                "request_id": call.request_id,
                "registry_version": call.registry_version,
                "definition_version": call.definition_version,
                "tool_name": call.tool_name,
                "risk": self._risk_for_tool(call.tool_name),
                "args": call.args,
            },
        )

    def _append_tool_response_fact(
        self,
        context: RunContext | OperationOrigin,
        call: ToolOperation,
        result: RunToolsResult,
        *,
        late: bool = False,
    ) -> None:
        """记录真实结果和原操作归属，迟到响应独立标记；传参：运行、调用、结果；返回：无。"""
        self.run_facts.append_from_trajectory(
            {
                "type": "event",
                "event": "tool:late_response" if late else "tool:response",
                "ts": utc_now(),
                "session_id": context.session_id,
                "run_id": context.run_id,
                "task_id": context.task_id,
                "focus_task_id": context.focus_task_id,
                "compatibility_task_id": context.compatibility_task_id,
                "segment_id": context.segment_id,
                "tool_call_id": call.call_id,
                "operation_task_id": call.task_id,
                "operation_id": call.operation_id,
                "request_id": call.request_id,
                "tool_name": call.tool_name,
                "status": result.status,
                "error_category": _extract_error_category(result),
                "output": result.output,
                "error": result.error,
                "summary": result.summary,
                "prompt_truncated": _tool_prompt_truncated(result),
                "meta": _jsonable(result.meta),
            },
        )

    def _append_tool_history(self, call: ToolOperation, result: RunToolsResult) -> None:
        """保存本运行结果供错误恢复观察；传参：调用、结果；返回：无。"""
        self.tool_history.append(
            {
                "action": result.action,
                "tool_name": result.tool_name,
                "status": result.status,
                "output": result.output,
                "error": result.error,
                "args_summary": _jsonable(call.args),
            }
        )

    def recover_pending(self, core: TurnExecutionContext) -> None:
        """崩溃后补回结果投影，副作用状态缺失时只报未知；传参：恢复运行；返回：无，不执行工具。"""
        store = SessionMessageStore(self.data_root)
        if not store.exists(core.context.session_id):
            return
        current = store.materialize(core.context.session_id)
        if not current.pending_tool_calls:
            return
        records = {
            row["call"]["call_id"]: row
            for row in self.operations.for_session(core.context.session_id)
        }
        calls = {
            part.call_id: part
            for message in current.messages
            if isinstance(message, AssistantMessage)
            for part in message.content
            if isinstance(part, ToolCallPart)
        }
        for call_id in current.pending_tool_calls:
            part, row = calls[call_id], records.get(call_id)
            if row is not None and isinstance(row.get("result"), dict):
                result = RunToolsResult(**row["result"])
            else:
                state = (
                    "not_started"
                    if row is not None and row.get("state") == "not_started"
                    else "unknown"
                )
                result = RunToolsResult.error_result(
                    action=part.tool_name,
                    error=f"interrupted operation: execution={state}; inspect actual effects before retry",
                    meta={
                        "execution_state": state,
                        "operation_id": row["operation_id"] if row else None,
                    },
                )
                if row is not None:
                    self.operations.write(
                        {
                            key: row[key]
                            for key in ("session_id", "run_id", "operation_id")
                        },
                        {**row, "state": state, "result": asdict(result)},
                    )
            append_tool_exchange(
                self.data_root,
                core.context.session_id,
                ToolExchange(
                    call_id,
                    part.tool_name,
                    cast(dict[str, object], thaw_json_value(part.arguments)),
                    _render_tool_conversation(result),
                    result.status,
                    result.error,
                    tuple(cast(list[str], result.meta.get("artifact_refs", []))),
                ),
                run_id=row["run_id"] if row else core.context.run_id,
                task_id=row["call"].get("task_id") if row else None,
            )
        core.context.payload[_RESUME_PENDING_CONSUMED] = True

    def _ledger_writer(self) -> LedgerWriter:
        """复用既有Ledger保存执行证据；传参：无；返回：写者。"""
        return LedgerWriter(LedgerStore(self.data_root), source="runtime.agent_loop")

    def _record_checkpoint_fact(
        self, context: RunContext, checkpoint: Checkpoint
    ) -> None:
        """将同一检查点关联到Ledger与运行事实；传参：运行、检查点；返回：无。"""
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

    def has_pending_inputs(self, context: RunContext) -> bool:
        """查询已接纳但尚未交付的新要求；传参：运行；返回：是否需要模型重新判断。"""
        if self.collaboration is not None:
            self.collaboration.sync_inputs()
        return bool(
            SessionMessageStore(self.data_root).pending_inputs(context.session_id)
        )


def _lease_for_operation(lease: Lease, call: ToolOperation) -> Lease:
    """让工具与授权记录跟随发起目标，跨目标时不沿用旧目标的授权快照。

    传参：lease 为运行授权边界；call 为固定归属的调用；返回：同权限和预算的执行快照
    """
    if call.task_id is None or call.task_id == lease.task_id:
        return lease
    capabilities = {
        key: value for key, value in lease.capabilities.items() if key != "task"
    }
    return replace(lease, task_id=call.task_id, capabilities=capabilities)
