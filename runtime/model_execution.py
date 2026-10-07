"""模型调用、尝试用量和请求证据的单一执行边界；作者：xxx；时间：2026-09-28 18:00:00。"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import cast


from context.production_builder import (
    ContextSegmentEvidence,
    ProductionContextBundle,
)
from llm.base import LLMClient
from llm.model_request import ComposedRequest
from llm.messages import TextPart
from llm.types import (
    LLMPlan,
    ModelObservation,
    ModelAttemptEvent,
    ModelRetryNotice,
    ModelAttemptStarted,
    ModelStreamOutput,
)
from runtime.ledger_writer import LedgerWriter
from runtime.run_evidence import RunEvidenceStore
from runtime.model_evidence import ModelEvidenceWriter
from runtime.run_facts import RunFactStore
from runtime.session_state import SessionState, SessionStateStore
from runtime.stream_events import (
    AssistantReasoningDelta,
    AssistantTextDelta,
    AssistantStreamClosed,
    ModelRetryScheduled,
    ModelRequestStarted,
    StreamEvent,
)
from runtime.types import RunContext
from runtime.extension_execution import ExtensionExecution
from runtime.cancellation import (
    CancellationToken,
    ExecutionCancelled,
    RunBudgetExceeded,
)
from runtime.types import RunToolsResult
from runtime.watchdog import Watchdog
from tasks.ids import new_ulid, utc_now
from tools.tool_registry import (
    ToolRegistry,
)


from runtime.runtime_errors import record_runtime_error

_RUNTIME_BUDGET_EVIDENCE = "_runtime_budget_evidence"


def _validate_current_knowledge_source(context: RunContext, data_root: Path) -> None:
    """【知识维护】【派发边界】切支、取消或来源损坏在下一次收费调用前暴露；参数：运行/存储；返回：无。"""
    from runtime.knowledge_maintenance import KnowledgeMaintenance, automatic_origin
    from runtime.knowledge_sources import frozen_knowledge_source
    from runtime.session_message_store import SessionMessageStore

    origin = automatic_origin(context)
    if origin is None:
        return
    current = KnowledgeMaintenance(data_root).load(origin["work_id"])
    if current.get("cancel_requested") or current["state"] in {
        "cancelled",
        "cancelling",
    }:
        raise ExecutionCancelled("knowledge work was cancelled before model dispatch")
    frozen_knowledge_source(SessionMessageStore(data_root), origin)


def _contains_provided_source(prepared: object, source_text: str) -> bool:
    """核对实际采用请求中的完整材料，摘要等其他调用不冒领覆盖；参数：准备结果/来源原文；返回：是否实际携带。"""
    if not isinstance(prepared, ComposedRequest):
        return False
    return any(
        isinstance(part, TextPart) and source_text in part.text
        for message in prepared.request.messages
        for part in message.content
    )


@dataclass(frozen=True, slots=True)
class ModelCallResult:
    """一次模型调用的结果，附带本次走的是哪条路。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：plan 为解析后的计划；request_id/request_index 固定本次请求身份与序号；
          used_stream_path 记录增量是否已经边生成边发过
    返回：携带本次身份的不可变结果，后续辅助调用不会覆盖它

    used_stream_path 决定轮末要不要补发那条整块思考链：流式已经发过就不能再补。
    """

    plan: LLMPlan
    request_id: str
    request_index: int
    used_stream_path: bool = False


class ModelRequestRunner:
    """每次调用局部分配身份，主请求和辅助请求共享证据及响应保护。"""

    def __init__(
        self,
        client: LLMClient,
        *,
        registry: ToolRegistry,
        cancellation: CancellationToken,
        evidence: ModelEvidenceWriter,
        facts: RunFactStore,
        run_evidence: RunEvidenceStore,
        states: SessionStateStore,
        ledger: LedgerWriter,
        extensions: ExtensionExecution,
    ) -> None:
        """接收模型与现有证据、取消、扩展依赖；返回：独立编号的调用所有者。"""
        self.llm_client = client
        self.tool_registry = registry
        self.cancellation = cancellation
        self.model_evidence = evidence
        self.run_facts = facts
        self.run_evidence = run_evidence
        self.session_states = states
        self.ledger = ledger
        self.extension_execution = extensions
        self._request_index = 0

    def invoke(
        self,
        bundle: ProductionContextBundle,
        context: RunContext,
        watchdog: Watchdog,
        *,
        last_tool_result: RunToolsResult | None = None,
    ) -> Generator[StreamEvent, None, ModelCallResult]:
        """通过同一边界执行主请求或摘要请求；传参：上下文、运行和预算；返回：真实模型结果。"""
        context = replace(
            context, payload=dict(context.payload), focus_task=dict(context.focus_task)
        )
        llm_client = self.llm_client
        if llm_client is None:
            raise RuntimeError("llm_client is required for real model execution")
        # 【模型】【请求身份】每次主请求及辅助摘要分别记录，沿用同一运行预算和停止信号
        self._request_index += 1
        request_index = self._request_index
        request_id = f"request-{new_ulid()}"
        bundle = replace(
            bundle,
            model_context={
                **bundle.model_context,
                "model_request_id": request_id,
                "model_attempt_recorder": partial(
                    self._record_model_attempt, context, watchdog
                ),
                "reserve_model_attempt": watchdog.reserve_model_attempt,
                "cancellation": self.cancellation,
            },
        )
        self.extension_execution.observe(
            context,
            "context_built",
            {
                "request_id": request_id,
                "segments": [segment.name for segment in bundle.segments],
            },
            watchdog=watchdog,
        )
        self.model_evidence.record_request(
            context,
            request_id=request_id,
            request_index=request_index,
        )
        self._emit_context_segments(context, bundle.segments)
        prepared = bundle.model_context.get("prepared_request")
        if isinstance(prepared, ComposedRequest):
            self.run_facts.append(
                {
                    "event": "context:request_allocation",
                    "session_id": context.session_id,
                    "run_id": context.run_id,
                    "task_id": context.storage_task_id,
                    "request_id": request_id,
                    "budget": dict(prepared.token_estimate),
                    "selection": dict(prepared.trim_delta or {}),
                }
            )
        self.ledger.record_model_requested(
            _model_provider_name(llm_client),
            _model_name(llm_client),
            [segment.name for segment in bundle.segments],
            task_id=context.storage_task_id,
            session_id=context.session_id,
            run_id=context.run_id,
        )
        try:
            from runtime.knowledge_maintenance import automatic_origin
            from runtime.model_dispatch import maintenance_call

            admission = (
                maintenance_call(
                    self.session_states.database.data_root, self.cancellation
                )
                if automatic_origin(context) is not None
                else nullcontext()
            )
            with admission:
                _validate_current_knowledge_source(
                    context, self.session_states.database.data_root
                )
                plan, used_stream_path = yield from _model_call(
                    llm_client, bundle, last_tool_result
                )
            call = ModelCallResult(plan, request_id, request_index, used_stream_path)
            from llm.protected_output import protect_plan

            call = replace(
                call,
                plan=protect_plan(
                    call.plan,
                    files=self.tool_registry.redacted_files,
                    session_id=context.session_id,
                    request_id=request_id,
                    lease=context.capability_lease,
                ),
            )
            if call.plan.registry_snapshot is None:
                registry = bundle.model_context.get("tool_registry")
                if not isinstance(registry, ToolRegistry):
                    raise RuntimeError("model request is missing its tool registry")
                call = replace(
                    call, plan=replace(call.plan, registry_snapshot=registry.snapshot())
                )
            call = replace(
                call,
                plan=replace(call.plan, request_id=request_id),
                request_id=request_id,
                request_index=request_index,
            )
            self._record_prompt_snapshot_state(context, call.plan)
            self._record_model_observation(context, call)
            self._settle_observation(watchdog, call.plan)
            self._record_provided_sources(context, call.plan, request_id, bundle=bundle)
            return call
        except Exception as exc:
            reason = (
                "cancelled"
                if isinstance(exc, ExecutionCancelled)
                else "model_request_failed"
            )
            if isinstance(exc, RunBudgetExceeded):
                reason = "budget_exhausted"
            yield self.close_stream(context, request_id, reason)
            raise

    def _record_provided_sources(
        self,
        context: RunContext,
        plan: LLMPlan,
        request_id: str,
        *,
        bundle: ProductionContextBundle,
    ) -> None:
        """【知识维护】【材料交付】响应成功后登记完整原文覆盖；参数：运行、真实结果和请求身份；返回：无。"""
        from runtime.knowledge_maintenance import KnowledgeMaintenance, automatic_origin

        origin = automatic_origin(context)
        provided = context.payload.get("provided_source_message_ids")
        source_text = context.payload.get("provided_source_text")
        prepared = bundle.model_context.get("prepared_request")
        if (
            origin is None
            or not provided
            or not isinstance(source_text, str)
            or plan.model_error is not None
        ):
            return
        if not isinstance(provided, list) or any(
            not isinstance(identity, str) for identity in provided
        ):
            raise ValueError(
                "provided knowledge source identities must be a list of strings"
            )
        if not _contains_provided_source(prepared, source_text):
            return
        manager = KnowledgeMaintenance(self.session_states.database.data_root)
        if not set(provided).issubset(
            manager.load(origin["work_id"])["read_message_ids"]
        ):
            manager.update(
                origin["work_id"],
                read_message_ids=provided,
                source_delivery_request_id=request_id,
            )

    def close_stream(
        self, context: RunContext, request_id: str, reason: str
    ) -> AssistantStreamClosed:
        """先提交未完成输出的关闭事实，不把局部正文写成回答；参数：运行、请求和原因；返回：关闭事件。"""
        self.run_facts.append(
            {
                "event": "llm:stream_closed",
                "session_id": context.session_id,
                "run_id": context.run_id,
                "task_id": context.storage_task_id,
                "request_id": request_id,
                "reason": reason,
                "committed": False,
            }
        )
        return AssistantStreamClosed(request_id, reason)

    def invoke_auxiliary(
        self,
        bundle: ProductionContextBundle,
        *,
        context: RunContext,
        watchdog: Watchdog,
    ) -> ModelCallResult:
        """保存辅助调用证据并更新运行用量投影；传参：请求、运行、记录器；返回：带独立身份的调用结果。"""
        stream = self.invoke(bundle, context, watchdog)
        while True:
            try:
                next(stream)
            except StopIteration as finished:
                call = cast(ModelCallResult, finished.value)
                break
        # 1. 【模型调用】【用量投影】保留既有RunContext写者合同，摘要发布后重建材料读取最新用量；不增加预算上限
        context.payload[_RUNTIME_BUDGET_EVIDENCE] = watchdog.budget_evidence()
        return call

    def _settle_observation(self, watchdog: Watchdog, plan: LLMPlan) -> None:
        """旧同步客户端没有尝试事件时记录其已报用量；传参：共享记录器、真实计划；返回：无。"""
        if plan.model_attempts or not isinstance(plan.observation, ModelObservation):
            return
        token_total = _model_observation_token_total(plan.observation)
        if token_total is not None:
            watchdog.tick(
                steps_taken=watchdog.steps_taken,
                tokens_used=watchdog.tokens_used + token_total,
            )

    def _record_model_attempt(
        self, context: RunContext, watchdog: Watchdog, attempt: ModelAttemptEvent
    ) -> None:
        """必要证据提交后按尝试身份结算，不因取消丢掉已知消耗；传参：运行、预算与尝试；返回：无。"""
        self.model_evidence.record_attempt(context, attempt)
        watchdog.settle_model_attempt(attempt)

    def _record_prompt_snapshot_state(self, context: RunContext, plan: LLMPlan) -> None:
        """保存本次真实请求的稳定提示指纹；传参：运行、计划；返回：无。"""
        if not isinstance(plan, LLMPlan):
            return
        snapshot = plan.request_bundle_evidence.get("stable_prompt_snapshot")
        if not isinstance(snapshot, dict) or not context.session_id:
            return
        prompt_hash = str(snapshot.get("hash", "")).strip()
        if not prompt_hash:
            return
        state = self.session_states.load(context.session_id) or SessionState(
            session_id=context.session_id
        )
        state.stable_prompt_hash = prompt_hash
        state.stable_prompt_updated_at = utc_now()
        state.updated_at = state.stable_prompt_updated_at
        self.session_states.save(state)

    def _emit_context_segments(
        self,
        context: RunContext,
        segments: tuple[ContextSegmentEvidence, ...],
    ) -> None:
        """保存模型实际获得的材料分层来源；传参：运行、材料片段；返回：无。"""
        if not context.session_id or not context.run_id:
            return
        segment_rows = [item.to_run_fact_segment() for item in segments]
        self.ledger.record_context_segments(
            segment_rows,
            task_id=context.storage_task_id,
            session_id=context.session_id,
            run_id=context.run_id,
        )
        self.run_facts.append(
            {
                "event": "context:segments",
                "session_id": context.session_id,
                "run_id": context.run_id,
                "task_id": context.storage_task_id,
                "segments": segment_rows,
            }
        )

    def _record_model_observation(
        self, context: RunContext, call: ModelCallResult
    ) -> None:
        """按结果自带的身份写响应和用量证据；传参：运行、调用结果；返回：无。"""
        plan, request_index = call.plan, call.request_index
        evidence_paths = self._write_model_evidence(context, call)
        row: dict[str, object] = {
            "type": "event",
            "event": "llm:response",
            "ts": utc_now(),
            "session_id": context.session_id,
            "run_id": context.run_id,
            "task_id": context.task_id,
            "focus_task_id": context.focus_task_id,
            "compatibility_task_id": context.compatibility_task_id,
            "segment_id": context.segment_id,
            "schema_version": 2,
            "request_id": plan.request_id,
            "request_index": request_index,
            "has_final": plan.final_output is not None,
            "has_run_tools": plan.run_tools_request is not None,
            "evidence": evidence_paths,
        }
        if plan.model_error is not None:
            row["error"] = {
                "category": plan.model_error.category,
                "summary": plan.model_error.summary,
            }
        if isinstance(plan.observation, ModelObservation):
            row["observation"] = asdict(plan.observation)
        self.run_facts.append_from_trajectory(row)
        self.model_evidence.record_usage(context, plan)

    def _write_model_evidence(
        self, context: RunContext, call: ModelCallResult
    ) -> dict[str, object]:
        """保存本次请求视图，并让协议失败指向同一次模型证据。

        传参：context/plan 为当前运行和模型结果；返回：可回取证据路径
        """
        plan = call.plan
        paths = self.model_evidence.write_plan(
            context, plan, request_index=call.request_index
        )
        if plan.model_error is not None:
            obs_fields = _extract_observation_fields(plan.observation)
            errors_path = record_runtime_error(
                self.run_evidence,
                context,
                category=plan.model_error.category,
                message=plan.model_error.summary,
                stage=plan.model_error.stage or "parse",
                evidence_path=str(paths["parsed_plan"]),
                raw_response_path=str(paths["model_response"]),
                observation_fields=obs_fields,
            )
            paths["errors"] = errors_path
        return paths


def _model_call(
    llm_client: LLMClient,
    bundle: ProductionContextBundle,
    last_tool_result: RunToolsResult | None,
) -> Generator[StreamEvent, None, tuple[LLMPlan, bool]]:
    """按客户端能力决定走流式还是同步，两条路都交出同一形状的结果。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：llm_client 为模型客户端；bundle 为已组装的本轮输入；
          last_tool_result 为上一轮工具结果，None 表示首轮
    返回：生成器；流式路径 yield 增量事件，收尾返回计划和流路径标记

    用能力探测而不是判客户端类型：plan_stream / continue_stream 不在 LLMClient
    协议里，仅提供同步 plan() 的客户端仍调用同一模型协议，并在返回后发布结果。
    """
    if last_tool_result is None:
        stream_fn = getattr(llm_client, "plan_stream", None)
        if stream_fn is None:
            plan = llm_client.plan(bundle.model_task, context=bundle.model_context)
            return plan, False
        stream = stream_fn(bundle.model_task, context=bundle.model_context)
    else:
        stream_fn = getattr(llm_client, "continue_stream", None)
        if stream_fn is None:
            plan = llm_client.continue_from_run_tools(
                bundle.model_task,
                last_tool_result,
                context=bundle.model_context,
            )
            return plan, False
        stream = stream_fn(
            bundle.model_task,
            last_tool_result,
            context=bundle.model_context,
        )
    plan = yield from _forward_output_deltas(
        stream, message_id=str(bundle.model_context.get("model_request_id", ""))
    )
    return plan, True


def _forward_output_deltas(
    stream: Generator[ModelStreamOutput, None, LLMPlan],
    *,
    message_id: str = "",
) -> Generator[StreamEvent, None, LLMPlan]:
    """把 client 的 provider 中立增量转成界面事件，边收边发。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：stream 为 client 的增量生成器
    返回：生成器；交付正文、思考或独立重试状态，收尾返回LLMPlan

    一次请求的多次尝试分别保留请求、响应和用量证据；重试清理失败尝试的局部显示。
    """
    while True:
        try:
            delta = next(stream)
        except StopIteration as stop:
            plan: LLMPlan = stop.value
            return plan
        # 1. 【模型调用】【网络重试】等待状态单独显示，不写入模型回答或重复触发工具
        if isinstance(delta, ModelAttemptStarted):
            yield ModelRequestStarted(
                message_id, delta.attempt_index, delta.max_attempts
            )
        elif isinstance(delta, ModelRetryNotice):
            # 【模型调用】【重试交接】下一尝试从空白输出开始，失败尝试的正文不拼接到新回答
            yield AssistantStreamClosed(message_id, "retry", retrying=True)
            yield ModelRetryScheduled(
                attempt_index=delta.attempt_index,
                max_attempts=delta.max_attempts,
                wait_seconds=delta.wait_seconds,
                error_category=delta.error_category,
            )
        elif delta.channel == "thinking":
            yield AssistantReasoningDelta(text=delta.text, message_id=message_id)
        else:
            yield AssistantTextDelta(text=delta.text, message_id=message_id)


def _model_provider_name(llm_client: object) -> str:
    """提取模型供应商名称。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：llm_client 为当前运行使用的模型客户端
    返回：用于 Ledger 证据的非空供应商名称
    """
    target = getattr(llm_client, "resolved_target", None)
    provider = getattr(target, "provider", None)
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    provider = getattr(llm_client, "provider", None)
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    return type(llm_client).__name__


def _model_name(llm_client: object) -> str:
    """提取模型名称。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：llm_client 为当前运行使用的模型客户端
    返回：用于 Ledger 证据的非空模型名称
    """
    target = getattr(llm_client, "resolved_target", None)
    model = getattr(target, "model", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    config = getattr(llm_client, "_config", None)
    model = getattr(config, "model", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    return type(llm_client).__name__


def _extract_observation_fields(
    observation: ModelObservation | None,
) -> dict[str, object]:
    """提取已有供应商观测用于故障证据；传参：模型观测；返回：可序列化字段。"""
    if observation is None:
        return {}
    fields: dict[str, object] = {}
    if observation.provider:
        fields["provider"] = observation.provider
    if observation.model:
        fields["model"] = observation.model
    if observation.config_source:
        fields["config_source"] = observation.config_source
    if observation.credential_source:
        fields["credential_source"] = observation.credential_source
    if observation.base_url_host:
        fields["base_url_host"] = observation.base_url_host
    if observation.profile_name:
        fields["profile_name"] = observation.profile_name
    if observation.credential_name:
        fields["credential_name"] = observation.credential_name
    if observation.should_fallback:
        fields["should_fallback"] = observation.should_fallback
    if observation.fallback_reason:
        fields["fallback_reason"] = observation.fallback_reason
    return fields


def _model_observation_token_total(observation: ModelObservation) -> int | None:
    """只汇总客户端实际报告的用量；传参：模型观测；返回：已知总量或None。"""
    if observation.total_tokens is not None:
        return observation.total_tokens
    total = (observation.prompt_tokens or 0) + (observation.completion_tokens or 0)
    return total if total > 0 else None
