from __future__ import annotations
from runtime.cancellation import (
    CancellationToken,
    ExecutionCancelled,
    RunBudgetExceeded,
)
from runtime.shared_budget import ModelReservation
from llm.stream_control import interruptible_events

from functools import partial
from collections.abc import Callable, Collection, Generator, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from time import perf_counter, sleep
from typing import Final, cast
from uuid import uuid4

from llm.config import LLMProviderConfig
from llm.messages import (
    AgentMessage,
    AssistantMessage,
    StopReason,
    ThinkingPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    model_visible_text,
    freeze_json_object,
)
from llm.model_registry import (
    ModelRegistry,
    ModelSelectionError,
    ModelSelector,
    SelectionDecision,
)
from llm.model_request import (
    ComposedRequest,
    ModelPreference,
    PreferenceKind,
    composed_request_evidence,
    compose_model_request,
)
from llm.parser import parse_llm_response, parse_tool_call_parts
from llm.production_target import (
    production_allowed_model_keys,
    production_connection,
    production_model_registry,
)
from llm.provider_adapter import (
    AdapterRegistry,
    ProviderAdapterError,
    validate_adapter_target,
)
from llm.provider_connection import ProviderConnectionError, ResolvedConnection
from llm.provider_result import (
    ProviderCallResult,
    ProviderError,
)
from llm.provider_stream import (
    ModelStreamEvent,
    StreamAssembler,
    StreamInterruptedError,
    StreamProtocolError,
)
from llm.providers.builtin import build_builtin_adapter_registry
from llm.reasoning import combine_reasoning_text, split_think_blocks
from llm.resolved_target import ResolvedModelTarget, sanitize_base_url
from llm.response_evidence import provider_call_evidence, request_source_evidence
from llm.retry_utils import MAX_TOTAL_ATTEMPTS, compute_wait
from llm.toolset_policy import ToolsetPolicyError
from llm.types import (
    LLMPlan,
    ModelAttemptEvent,
    ModelError,
    ModelObservation,
    ModelOutputDelta,
    ModelUsage,
    ModelRetryNotice,
    ModelStreamOutput,
    ModelAttemptStarted,
)
from context.window import ContextWindowExceeded, require_request_fits
from runtime.types import RunToolsResult
from tools.tool_registry import ToolRegistry, get_default_tool_registry


# 测试客户端及未配置实例的展示窗口；生产派发仍要求有效的resolved_target
_DEFAULT_CONTEXT_WINDOW = 128000
_MISSING_TARGET_CODE = "missing_config:resolved_target"
_NO_RESULT_SUMMARY = "provider call did not return a response"
# 只有 native 模式的 text 块是真正的散文答案；text_json 模式下它是协议信封，不能逐字上屏
_NATIVE_TOOL_CALLS = "native_tool_calls"
# 溢出交由有来源的上下文管理处理，不在客户端删除历史
_OVERFLOW_CATEGORIES = frozenset({"context_overflow", "payload_too_large"})


@dataclass(frozen=True, slots=True)
class ModelInput:
    task: str
    stage: str
    conversation_history: tuple[AgentMessage, ...] = ()
    system_reminder: str = ""
    recoverable_error_notice: str = ""
    no_progress_observation: str = ""
    task_summary_layers: dict[str, object] | None = None
    artifact_output_dir: str = ""
    runtime_context: dict[str, object] = field(default_factory=dict)
    history_truncated_retained: int | None = None


@dataclass(frozen=True, slots=True)
class _StageTiming:
    """一个阶段的起始时刻与计时基准。"""

    started_at: str
    started_perf: float = 0.0

    def elapsed_ms(self) -> int:
        """返回从阶段开始到此刻的毫秒耗时。"""
        return int(round((perf_counter() - self.started_perf) * 1000))


@dataclass(frozen=True, slots=True)
class _UsageTokens:
    """一次调用的 token 用量；未知量按 None 记录，不伪造成 0。"""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class _ObservationFacts:
    """产出观测记录所需的成败、用量与编排计数。"""

    success: bool
    error_category: str | None
    attempt_count: int = 1
    total_wait: float = 0.0
    retry_after: float | None = None
    tokens: _UsageTokens = field(default_factory=_UsageTokens)


@dataclass(frozen=True, slots=True)
class _ProviderCall:
    """一次 Provider 调用的结果：助手消息或错误，附本轮请求体与选型证据。"""

    result: ProviderCallResult | None = None
    error: ModelError | None = None
    request_body: Mapping[str, object] = field(default_factory=dict)
    selection: dict[str, object] = field(default_factory=dict)
    attempt: ModelAttemptEvent | None = None


@dataclass(frozen=True, slots=True)
class _AttemptTrace:
    """传递一次逻辑请求的身份和必要证据写入器。

    传参：request_id 为逻辑请求；recorder 为调用方注入的持久化边界
    返回：不可变调用配置
    """

    request_id: str
    recorder: Callable[[ModelAttemptEvent], None] | None = None
    cancellation: CancellationToken | None = None
    reserve: Callable[[ModelReservation], None] | None = None
    protect_result: Callable[[ProviderCallResult], ProviderCallResult] | None = None


@dataclass(frozen=True, slots=True)
class _AttemptOutcome:
    """重试编排的最终状态。"""

    composed: ComposedRequest
    call: _ProviderCall
    attempts: int
    total_wait: float
    trim_delta: dict[str, object] | None
    request_id: str
    model_attempts: tuple[ModelAttemptEvent, ...] = ()

    @property
    def message(self) -> AssistantMessage | None:
        """取本轮拿到的助手消息；错误已被桥成 ModelError 时返回 None。"""
        if self.call.error is not None or self.call.result is None:
            return None
        return self.call.result.message

    def facts(self, *, success: bool, error_category: str | None) -> _ObservationFacts:
        """把编排计数与 Provider 用量合成观测事实。"""
        return _ObservationFacts(
            success=success,
            error_category=error_category,
            attempt_count=len(self.model_attempts),
            total_wait=self.total_wait,
            retry_after=self.call.error.retry_after if self.call.error else None,
            tokens=_attempt_usage_tokens(self.model_attempts),
        )


class MissingConfigurationLLMClient:
    def __init__(self, missing_fields: list[str] | None = None) -> None:
        self._missing_fields = missing_fields or ["base_url", "model"]

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        del context
        return self._error_plan("plan", task)

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: RunToolsResult,
        context: object | None = None,
    ) -> LLMPlan:
        del run_tools_result, context
        return self._error_plan("continue", task)

    def _error_plan(self, stage: str, task: str) -> LLMPlan:
        summary = f"missing provider configuration: {', '.join(self._missing_fields)}"
        error = ModelError.create(
            category="missing_config",
            summary=summary,
            raw_summary=", ".join(self._missing_fields),
            stage=stage,
        )
        model_input = _build_model_input(task, stage)
        return LLMPlan(
            final_output=error.render_output(),
            model_error=error,
            observation=ModelObservation(
                stage=stage,
                provider="missing_config",
                model="(missing)",
                started_at=_now_iso(),
                elapsed_ms=0,
                attempt_count=1,
                success=False,
                error_category=error.category,
            ),
            prompt_context=_prompt_context(model_input),
            render_text_to_model=render_model_input_text(model_input),
        )


class RealLLMClient:
    _INSTANT_FALLBACK = frozenset(
        {"auth", "billing", "model_not_found", "missing_config"}
    )
    _CONSECUTIVE_FALLBACK = frozenset({"rate_limited", "overloaded", "server_error"})
    _CONSECUTIVE_THRESHOLD = 3

    def __init__(
        self,
        config: LLMProviderConfig,
        *,
        adapter_registry: AdapterRegistry | None = None,
        model_registry: ModelRegistry | None = None,
        connection: ResolvedConnection | None = None,
        allowed_actions: list[str] | None = None,
        protocol_mode: str = "native_tool_calls",
        resolved_target: ResolvedModelTarget | None = None,
    ) -> None:
        """按 Adapter 注入面构造生产 LLM 客户端。

        作者：LKX
        时间：2026-08-30 18:40:00
        传参：config 为供应商配置；adapter_registry/model_registry/connection 为可注入的
              typed Provider 栈三件套；allowed_actions/protocol_mode/resolved_target 为
              本轮编排口径
        返回：无

        模型注册表与连接都从 resolved_target 推出，而 resolved_target 缺失时无法推出任何
        真实端点，所以这里不构造、也不用假值占位，等真正发请求时再要（见 _require_*）。
        构造期不崩是为了让不打算发请求的调用方（只读 context_window 等）能建对象。
        """
        self._config = config
        self._adapter_registry = adapter_registry or build_builtin_adapter_registry()
        self._model_registry = model_registry
        self._connection = connection
        self._protocol_mode = protocol_mode
        self._resolved_target = resolved_target
        self._consecutive_error_counts: dict[str, int] = {}
        self._context_window: int = (
            resolved_target.context_window
            if resolved_target
            else _DEFAULT_CONTEXT_WINDOW
        )
        self._allowed_actions = (
            tuple(allowed_actions) if allowed_actions is not None else None
        )

    @property
    def resolved_target(self) -> ResolvedModelTarget | None:
        return self._resolved_target

    @property
    def context_window(self) -> int:
        return self._context_window

    def _require_model_registry(self) -> ModelRegistry:
        """取本轮选型用的模型注册表；缺 resolved_target 时明确报缺配置。

        作者：LKX
        时间：2026-08-30 18:40:00
        传参：无
        返回：ModelRegistry；无注入且无 resolved_target 时抛 ProviderConnectionError
        """
        if self._model_registry is not None:
            return self._model_registry
        registry = production_model_registry(self._require_target(), self._config)
        self._model_registry = registry
        return registry

    def _require_connection(self) -> ResolvedConnection:
        """取本轮调用的连接；缺 resolved_target 时明确报缺配置。

        作者：LKX
        时间：2026-08-30 18:40:00
        传参：无
        返回：ResolvedConnection；无注入且无 resolved_target 时抛 ProviderConnectionError
        """
        if self._connection is not None:
            return self._connection
        connection = production_connection(self._require_target())
        self._connection = connection
        return connection

    def _require_target(self) -> ResolvedModelTarget:
        """要求已解析的模型目标；没有就报缺配置而不是拿默认值顶上。"""
        if self._resolved_target is None:
            raise ProviderConnectionError(_MISSING_TARGET_CODE)
        return self._resolved_target

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        return _drain(self.plan_stream(task, context))

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: RunToolsResult,
        context: object | None = None,
    ) -> LLMPlan:
        return _drain(self.continue_stream(task, run_tools_result, context))

    def plan_stream(
        self, task: str, context: object | None = None
    ) -> Generator[ModelStreamOutput, None, LLMPlan]:
        """与 plan() 同一次调用，但把模型生成过程中的增量逐片交出去。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：task 为本轮任务；context 为运行上下文
        返回：生成器；交付模型增量或重试提示，收尾 return LLMPlan

        故意不进 LLMClient Protocol：约 20 个测试假件与 sediment_reflection 只认
        plan()，Runtime 用 getattr 探测这个方法，探不到就走同步回落。
        """
        return self._execute_stage(
            "plan",
            _build_model_input(
                task,
                "plan",
                conversation_history=_extract_history(context),
                task_summary_layers=_extract_summary_layers(context),
                system_reminder=_extract_system_reminder(context),
                artifact_output_dir=_extract_artifact_output_dir(context),
                runtime_context=_extract_runtime_context(context),
                history_truncated_retained=_extract_history_truncated(context),
            ),
        )

    def continue_stream(
        self,
        task: str,
        run_tools_result: RunToolsResult,
        context: object | None = None,
    ) -> Generator[ModelStreamOutput, None, LLMPlan]:
        """continue_from_run_tools() 的流式形态，语义与它完全一致。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：task 为本轮任务；run_tools_result 为上一轮工具结果；context 为运行上下文
        返回：生成器；交付模型增量或重试提示，收尾 return LLMPlan
        """
        return self._execute_stage(
            "continue",
            _build_model_input(
                task,
                "continue",
                conversation_history=_extract_history(context),
                task_summary_layers=_extract_summary_layers(context),
                system_reminder=_extract_system_reminder(context),
                recoverable_error_notice=_extract_recoverable_error_notice(context),
                no_progress_observation=_extract_no_progress_observation(context),
                artifact_output_dir=_extract_artifact_output_dir(context),
                runtime_context=_extract_runtime_context(context),
                history_truncated_retained=_extract_history_truncated(context),
            ),
        )

    def _execute_stage(
        self, stage: str, model_input: ModelInput
    ) -> Generator[ModelStreamOutput, None, LLMPlan]:
        """跑完一个阶段的模型调用：组装、重试、解析、记证据。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：stage 为阶段名；model_input 为本轮模型输入
        返回：生成器；透传增量，收尾 return LLMPlan；失败也以 plan 形态返回，不向上抛
        """
        timing = _StageTiming(started_at=_now_iso(), started_perf=perf_counter())
        model_context = _model_context_from_input(model_input)
        trace = _attempt_trace(model_context)
        if self._allowed_actions is not None:
            model_context["allowed_actions"] = self._allowed_actions
        registry = _registry_from_context(model_context)
        try:
            composed = self._compose(model_input, model_context)
        except ToolsetPolicyError as exc:
            return self._policy_error_plan(stage, timing, str(exc))
        outcome = yield from self._run_attempts(composed, trace=trace)
        message = outcome.message
        if message is None:
            return self._failure_plan(stage, timing, outcome=outcome)
        plan = self._success_plan(stage, timing, outcome=outcome, registry=registry)
        # 正文里内联的 <think> 块只能在整段文本上解析（split_think_blocks 没有增量形态），
        # 这半边推理仍在轮末一次性交出去，不逐字流
        _, inline = split_think_blocks(model_visible_text(message))
        if inline.strip():
            yield ModelOutputDelta("thinking", inline.strip())
        return plan

    def _compose(
        self,
        model_input: ModelInput,
        model_context: dict[str, object],
    ) -> ComposedRequest:
        """按本轮上下文组装 typed 请求。"""
        prepared = model_context.get("prepared_request")
        if isinstance(prepared, ComposedRequest):
            if (
                prepared.context_window != self.context_window
                or prepared.protocol_mode != self._protocol_mode
            ):
                raise ValueError("prepared model request target changed")
            if prepared.context_baseline is not None:
                from context.selection_store import content_digest, selection_scope

                # 1. 【上下文】【发送核对】准备后的权限、分支或有效要求变化必须重新组装，不能继续采用旧正文
                baseline = prepared.context_baseline
                if baseline["scope"] != selection_scope(model_context) or baseline[
                    "effective_requirements_digest"
                ] != content_digest(model_context.get("effective_requirements")):
                    raise ValueError(
                        "prepared model request context boundary changed; prepare a new request"
                    )
            if (
                prepared.context_baseline is not None
                and prepared.context_baseline.get("model_target_identity") is not None
            ):
                descriptor = self._require_model_registry().require(
                    production_allowed_model_keys()[0]
                )
                identity = {
                    "provider": descriptor.provider,
                    "model": descriptor.model_id,
                    "api_family": descriptor.api_family,
                }
                if prepared.context_baseline["model_target_identity"] != identity:
                    raise ValueError("prepared model request target changed")
            return prepared
        return self.prepare_request(
            model_input.task, model_context, stage=model_input.stage
        )

    def prepare_request(
        self,
        task: str,
        context: Mapping[str, object],
        *,
        stage: str = "plan",
    ) -> ComposedRequest:
        """使用真实请求组装器预估并固定窗口；传参：目标、上下文及阶段；返回：尚未派发的请求。"""
        model_context = dict(context)
        if self._model_registry is not None or self._resolved_target is not None:
            descriptor = self._require_model_registry().require(
                production_allowed_model_keys()[0]
            )
            model_context["model_target_identity"] = {
                "provider": descriptor.provider,
                "model": descriptor.model_id,
                "api_family": descriptor.api_family,
            }
        if self._allowed_actions is not None:
            model_context["allowed_actions"] = self._allowed_actions
        purpose = model_context.get("context_purpose")
        if purpose in {"compaction", "compaction_confirmation"}:
            model_context["allowed_actions"] = ("final",)
            stage = str(purpose)
        return compose_model_request(
            task=task,
            stage=stage,
            protocol_mode=self._protocol_mode,
            model_context=model_context,
            registry=_registry_from_context(model_context),
            context_window=self._context_window,
            max_output_tokens=self._resolved_target.output_token_limit
            if self._resolved_target is not None
            else None,
            optional_preferences=(
                ModelPreference(
                    PreferenceKind.REASONING_LEVEL,
                    self._resolved_target.reasoning_effort,
                ),
            )
            if self._resolved_target is not None
            and self._resolved_target.reasoning_effort is not None
            else (),
        )

    def _run_attempts(
        self, composed: ComposedRequest, *, trace: _AttemptTrace
    ) -> Generator[ModelStreamOutput, None, _AttemptOutcome]:
        """按重试预算反复调用 Provider，直到成功、不可重试或用尽预算。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：composed 为本轮 typed 请求；trace 为请求身份和持久化边界
        返回：生成器；透传每次 attempt 的增量，收尾 return _AttemptOutcome

        溢出类错误交给有Session来源和预算的上下文管理，不在客户端删除历史后盲目重试。
        每次尝试在真正派发前与收尾后提交证据；重试不会覆盖前一次的请求、错误或用量。
        """
        trim_delta = (
            dict(composed.trim_delta) if composed.trim_delta is not None else None
        )
        attempts = 0
        total_wait = 0.0
        call = _ProviderCall()
        records: list[ModelAttemptEvent] = []
        while attempts < MAX_TOTAL_ATTEMPTS:
            attempts += 1
            call = yield from self._call_provider(
                composed, trace=trace, attempt_index=attempts
            )
            if call.attempt is not None:
                records.append(call.attempt)
            if call.error is None:
                break
            if (
                call.error.category in _OVERFLOW_CATEGORIES
                or not call.error.retryable
                or attempts >= MAX_TOTAL_ATTEMPTS
            ):
                break
            wait = compute_wait(attempts, retry_after=call.error.retry_after)
            # 【模型调用】【等待重试】通知独立于正文，调用方可显示等待且不会误当模型决策
            yield ModelRetryNotice(
                attempts + 1, MAX_TOTAL_ATTEMPTS, wait, call.error.category
            )
            total_wait += wait
            if trace.cancellation is None:
                sleep(wait)
            else:
                try:
                    _interruptible_backoff(wait, trace.cancellation)
                except ExecutionCancelled as exc:
                    call = replace(
                        call,
                        error=ModelError.create(
                            category="cancelled",
                            summary=str(exc),
                            stage="backoff",
                            retryable=False,
                        ),
                    )
                    break
        return _AttemptOutcome(
            composed=composed,
            call=call,
            attempts=attempts,
            total_wait=total_wait,
            trim_delta=trim_delta,
            request_id=trace.request_id,
            model_attempts=tuple(records),
        )

    def _call_provider(
        self, composed: ComposedRequest, *, trace: _AttemptTrace, attempt_index: int
    ) -> Generator[ModelStreamOutput, None, _ProviderCall]:
        """选模型、取 Adapter、跑一次流式调用并把异常桥成 ModelError。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：composed 为本轮 typed 请求
        返回：生成器；透传下层增量，四类边界异常在此转成 ModelError，不向上抛

        选型与连接配置失败、Adapter 契约失败、格式和顺序等协议错误均不可重试；
        已开始却缺少完成标记的明确断流归传输失败，错误原文仍保留在 summary 里。
        事件迭代改到 yield from 里发生，仍落在同一个 try 内，四类异常覆盖面不变；
        已交付的增量无法撤回；失败尝试的内容、错误和已采集用量一并保存在证据中。
        """
        try:
            require_request_fits(composed.request, composed.context_window)
            return (
                yield from self._stream_once(
                    composed, trace=trace, attempt_index=attempt_index
                )
            )
        except ContextWindowExceeded as exc:
            return _ProviderCall(
                error=ModelError.create(
                    category="context_overflow",
                    summary=str(exc),
                    stage="request_build",
                    retryable=False,
                )
            )
        except (ExecutionCancelled, RunBudgetExceeded) as exc:
            return _ProviderCall(
                error=ModelError.create(
                    category="cancelled"
                    if isinstance(exc, ExecutionCancelled)
                    else "budget_exhausted",
                    summary=str(exc),
                    stage="dispatch",
                    retryable=False,
                )
            )
        except ModelSelectionError as exc:
            return _ProviderCall(error=_selection_model_error(exc))
        except ProviderConnectionError as exc:
            return _ProviderCall(error=_connection_model_error(exc))
        except ProviderAdapterError as exc:
            return _ProviderCall(error=_adapter_model_error(exc))
        except StreamProtocolError as exc:
            return _ProviderCall(error=_stream_protocol_model_error(exc))

    def _stream_once(
        self, composed: ComposedRequest, *, trace: _AttemptTrace, attempt_index: int
    ) -> Generator[ModelStreamOutput, None, _ProviderCall]:
        """按 typed 栈发一次请求：Selector -> Adapter -> StreamAssembler。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：composed 为本轮 typed 请求
        返回：生成器；模型仍在生成时逐片 yield 增量，收尾 return _ProviderCall

        原先 assemble(list(...)) 会把惰性事件流整个拍平，增量到不了界面。改成逐事件
        accept + finish 收口：组装器仍独占内容顺序、tool JSON 与 terminal 校验，
        这里只是顺路把 thinking / text 分片转成 provider 中立增量交上去。
        """
        request = composed.request
        if trace.cancellation is not None and trace.cancellation.cancelled:
            raise ExecutionCancelled("model request cancelled before dispatch")
        decision = ModelSelector(self._require_model_registry()).select(
            production_allowed_model_keys(),
            request.required_capabilities,
            request.optional_preferences,
        )
        adapter = self._adapter_registry.require(decision.selected.api_family)
        validate_adapter_target(adapter, decision.selected)
        connection = self._require_connection()
        attempt_id = f"attempt-{uuid4().hex}"
        if trace.reserve is not None:
            from context.window import request_budget

            # 【模型调用】【用量记账】预留言明只记录本次估算，不再回压输出上限
            trace.reserve(
                ModelReservation(
                    attempt_id,
                    request_budget(request, composed.context_window).total,
                    request.max_output_tokens,
                    trace.request_id,
                )
            )
        body = adapter.build_request(request, model_id=decision.selected.model_id)
        cache_hints = getattr(adapter, "apply_cache_hints", None)
        if callable(cache_hints):
            body = cache_hints(
                body, request=request, model=decision.selected, connection=connection
            )
        request_body = freeze_json_object(body, path="prepared_request")
        timing = _StageTiming(started_at=_now_iso(), started_perf=perf_counter())
        started = ModelAttemptEvent(
            phase="started",
            request_id=trace.request_id,
            attempt_id=attempt_id,
            attempt_index=attempt_index,
            provider=decision.selected.provider,
            model=decision.selected.model_id,
            started_at=timing.started_at,
            request=request_body,
            input_ids=tuple(
                message.message_id
                for message in request.messages
                if isinstance(message, UserMessage)
            ),
            api_family=adapter.api_family,
            sources=request_source_evidence(composed),
        )
        if trace.recorder is not None:
            trace.recorder(started)
        yield ModelAttemptStarted(attempt_index, MAX_TOTAL_ATTEMPTS)
        from llm.context_baseline import adopt_request_context

        # 【模型调用】【尝试证据】建立响应流时同步失败，也必须闭合已提交的尝试记录
        try:
            if trace.cancellation is not None and trace.cancellation.cancelled:
                raise ExecutionCancelled("model request cancelled before dispatch")
            adopt_request_context(composed, request_id=trace.request_id)
            call = yield from _consume_provider_stream(
                adapter.stream(
                    request,
                    model=decision.selected,
                    connection=connection,
                    cancellation=trace.cancellation,
                    prepared_body=request_body,
                ),
                protocol_mode=composed.protocol_mode,
                attempt=started,
                api_family=adapter.api_family,
                cancellation=trace.cancellation,
            )
        except ExecutionCancelled as exc:
            call = _ProviderCall(
                error=ModelError.create(
                    category="cancelled",
                    summary=str(exc),
                    stage="dispatch",
                    retryable=False,
                )
            )
        except ProviderAdapterError as exc:
            call = _ProviderCall(error=_adapter_model_error(exc))
        except StreamProtocolError as exc:
            call = _ProviderCall(error=_stream_protocol_model_error(exc))
        if call.result is not None and trace.protect_result is not None:
            protected = trace.protect_result(call.result)
            error = call.error
            if error is not None and protected.error is not None:
                error = replace(
                    error,
                    summary=protected.error.summary,
                    raw_summary=protected.error.summary,
                )
            call = replace(call, result=protected, error=error)
        completed = _complete_attempt(started, call, timing=timing)
        if trace.recorder is not None:
            trace.recorder(completed)
        return replace(
            call,
            request_body=request_body,
            selection=_selection_evidence(decision),
            attempt=completed,
        )

    def _success_plan(
        self,
        stage: str,
        timing: _StageTiming,
        *,
        outcome: _AttemptOutcome,
        registry: ToolRegistry,
    ) -> LLMPlan:
        """把成功的助手消息解析成 plan 并挂上观测与证据。

        作者：LKX
        时间：2026-08-30 18:50:00
        传参：stage 为阶段名；timing 为本轮计时；outcome 为重试结果；
              registry 为工具注册表
        返回：解析后的 LLMPlan
        """
        message = outcome.message
        assert message is not None
        composed = outcome.composed
        plan = _parse_assistant_message(
            message,
            protocol_mode=composed.protocol_mode,
            allowed_tool_names=composed.tool_selection.allowed_tool_names,
            registry=composed.registry_snapshot or registry,
        )
        # 【上下文】【摘要完整性】标题齐全仍可能被输出额度截断，未完成的摘要不能替代原文
        if (
            composed.prompt_context.get("stage") == "compaction"
            and message.stop_reason is StopReason.MAX_OUTPUT_TOKENS
        ):
            plan = LLMPlan(
                model_error=ModelError.create(
                    category="output_limit_exceeded",
                    stage="compaction",
                    summary="semantic summary reached max_output_tokens before completion; original history is retained",
                )
            )
        plan.reasoning_content = _reasoning_content(message)
        plan.assistant_message = message
        plan.observation = self._observation(
            stage,
            timing,
            outcome.facts(
                success=plan.model_error is None,
                error_category=plan.model_error.category if plan.model_error else None,
            ),
        )
        _attach_request_facts(plan, outcome)
        plan.trim_delta = outcome.trim_delta
        return plan

    def _failure_plan(
        self,
        stage: str,
        timing: _StageTiming,
        *,
        outcome: _AttemptOutcome,
    ) -> LLMPlan:
        """没拿到助手消息时以错误 plan 收口，保住"总是返回 LLMPlan"的约定。

        作者：LKX
        时间：2026-08-30 18:50:00
        传参：stage 为阶段名；timing 为本轮计时；outcome 为重试结果
        返回：携带 ModelError 的 LLMPlan
        """
        error = outcome.call.error or ModelError.create(
            category="provider_error",
            summary=_NO_RESULT_SUMMARY,
            stage="transport",
        )
        plan = LLMPlan(
            final_output=error.render_output(),
            model_error=error,
            observation=self._observation(
                stage,
                timing,
                outcome.facts(success=False, error_category=error.category),
            ),
        )
        _attach_request_facts(plan, outcome)
        plan.trim_delta = outcome.trim_delta
        return plan

    def _observation(
        self,
        stage: str,
        timing: _StageTiming,
        facts: _ObservationFacts,
    ) -> ModelObservation:
        """按本轮事实产出模型观测记录。

        作者：LKX
        时间：2026-08-30 18:50:00
        传参：stage 为阶段名；timing 为本轮计时；facts 为成败、用量与编排计数
        返回：ModelObservation
        """
        should_fallback, fallback_reason = self._update_fallback_state(
            facts.success, facts.error_category
        )
        target = self._resolved_target
        tokens = facts.tokens
        return ModelObservation(
            stage=stage,
            provider=target.provider if target else "openai_compatible",
            model=self._config.model,
            started_at=timing.started_at,
            elapsed_ms=timing.elapsed_ms(),
            attempt_count=facts.attempt_count,
            success=facts.success,
            error_category=facts.error_category,
            was_retried=facts.attempt_count > 1,
            prompt_tokens=tokens.prompt_tokens,
            completion_tokens=tokens.completion_tokens,
            total_tokens=tokens.total_tokens,
            total_wait_seconds=facts.total_wait,
            retry_after_used=facts.retry_after,
            config_source=target.config_source if target else "",
            credential_source=target.credential_source if target else "",
            base_url_host=sanitize_base_url(target.base_url) if target else "",
            profile_name=target.profile_name if target else "",
            credential_name=target.credential_name if target else "",
            should_fallback=should_fallback,
            fallback_reason=fallback_reason,
            cache_read_input_tokens=tokens.cache_read_input_tokens,
            cache_creation_input_tokens=tokens.cache_creation_input_tokens,
        )

    def _policy_error_plan(
        self,
        stage: str,
        timing: _StageTiming,
        message: str,
    ) -> LLMPlan:
        """工具集策略非法时在组装阶段收口，不发任何请求。"""
        error = ModelError.create(
            category="invalid_model_protocol",
            summary=f"invalid toolset policy: {message}",
            raw_summary=message,
            stage="policy",
        )
        return LLMPlan(
            final_output=error.render_output(),
            model_error=error,
            observation=self._observation(
                stage,
                timing,
                _ObservationFacts(success=False, error_category=error.category),
            ),
            protocol_mode=self._protocol_mode,
            request_bundle_evidence={"toolset_policy_error": message},
        )

    def _update_fallback_state(
        self,
        success: bool,
        error_category: str | None,
    ) -> tuple[bool, str]:
        """Compute an advisory fallback signal from the recent error pattern.

        Returns ``(should_fallback, reason)`` where ``should_fallback`` means the
        error pattern *suggests* a secondary provider would help — it does NOT
        switch providers and no fallback is performed. The signal is recorded as
        an observation only; real fallback-to-secondary wiring is out of scope.
        """
        if success:
            self._consecutive_error_counts.clear()
            return False, ""
        category = (error_category or "").lower()
        if not category:
            return False, ""
        if category in self._INSTANT_FALLBACK:
            return True, category
        self._consecutive_error_counts[category] = (
            self._consecutive_error_counts.get(category, 0) + 1
        )
        count = self._consecutive_error_counts[category]
        if (
            category in self._CONSECUTIVE_FALLBACK
            and count >= self._CONSECUTIVE_THRESHOLD
        ):
            return True, f"{category} x{count}"
        return False, ""


def _build_model_input(
    task: str,
    stage: str,
    *,
    conversation_history: tuple[AgentMessage, ...] = (),
    task_summary_layers: dict[str, object] | None = None,
    system_reminder: str = "",
    recoverable_error_notice: str = "",
    no_progress_observation: str = "",
    artifact_output_dir: str = "",
    runtime_context: dict[str, object] | None = None,
    history_truncated_retained: int | None = None,
) -> ModelInput:
    return ModelInput(
        task=task,
        stage=stage,
        conversation_history=conversation_history,
        system_reminder=system_reminder,
        recoverable_error_notice=recoverable_error_notice,
        no_progress_observation=no_progress_observation,
        task_summary_layers=dict(task_summary_layers or {}),
        artifact_output_dir=artifact_output_dir,
        runtime_context=dict(runtime_context or {}),
        history_truncated_retained=history_truncated_retained,
    )


def _model_context_from_input(model_input: ModelInput) -> dict[str, object]:
    context: dict[str, object] = dict(model_input.runtime_context)
    context.update(
        {
            "conversation_history": model_input.conversation_history,
            "task_summary_layers": dict(model_input.task_summary_layers or {}),
        }
    )
    if model_input.system_reminder:
        context["system_reminder"] = model_input.system_reminder
    # 上一轮协议错误的原因只在本轮有效：重试时必须让模型读到它，否则模型只能重复犯同一个错
    if model_input.recoverable_error_notice:
        context["recoverable_error_notice"] = model_input.recoverable_error_notice
    if model_input.no_progress_observation:
        context["no_progress_observation"] = model_input.no_progress_observation
    if model_input.artifact_output_dir:
        context["artifact_output_dir"] = model_input.artifact_output_dir
    if model_input.history_truncated_retained is not None:
        context["history_truncated_retained"] = model_input.history_truncated_retained
    return context


def _registry_from_context(model_context: Mapping[str, object]) -> ToolRegistry:
    registry = model_context.get("tool_registry")
    if isinstance(registry, ToolRegistry):
        return registry
    return get_default_tool_registry()


def render_model_input_text(model_input: ModelInput) -> str:
    parts = [
        f"[stage]\n{model_input.stage}",
        f"[task]\n{model_input.task}",
    ]
    if model_input.system_reminder:
        parts.append(f"[system_reminder]\n{model_input.system_reminder}")
    if model_input.recoverable_error_notice:
        parts.append(
            f"[recoverable_error_notice]\n{model_input.recoverable_error_notice}"
        )
    if model_input.no_progress_observation:
        parts.append(
            f"[no_progress_observation]\n{model_input.no_progress_observation}"
        )
    return "\n".join(parts)


def _extract_history(context: object | None) -> tuple[AgentMessage, ...]:
    """从运行上下文取本轮选中的历史消息。

    历史已由 ProductionContextBuilder 选好并经 Session Store 校验，这里只做类型确认后透传，
    不重新过滤或改写内容——那会造出第二个决定"模型能看到哪些历史"的地方。
    """
    if not isinstance(context, dict):
        return ()
    raw_history = context.get("conversation_history")
    if not isinstance(raw_history, (list, tuple)):
        return ()
    # AgentMessage 是联合别名，运行时须按三个具体类型判定
    return tuple(
        item
        for item in raw_history
        if isinstance(item, (UserMessage, AssistantMessage, ToolResultMessage))
    )


def _extract_history_truncated(context: object | None) -> int | None:
    """取历史截断说明需要的保留条数；未截断时返回 None。"""
    if not isinstance(context, dict):
        return None
    value = context.get("history_truncated_retained")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _extract_summary_layers(context: object | None) -> dict[str, object]:
    if not isinstance(context, dict):
        return {}
    value = context.get("task_summary_layers")
    return dict(value) if isinstance(value, dict) else {}


def _extract_system_reminder(context: object | None) -> str:
    if not isinstance(context, dict):
        return ""
    value = context.get("system_reminder")
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _extract_recoverable_error_notice(context: object | None) -> str:
    """从模型上下文取出上一轮的可恢复错误说明。

    作者：xxx
    时间：2026-09-01 00:00:00
    传参：context 为 agent_loop 组装的模型上下文
    返回：错误原因文本；首轮或无可恢复错误时返回空串
    """
    if not isinstance(context, dict):
        return ""
    value = context.get("recoverable_error_notice")
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _extract_no_progress_observation(context: object | None) -> str:
    """读取本轮无进展观察；传参：可选模型上下文；返回：观察正文或空串。"""
    if not isinstance(context, dict):
        return ""
    value = context.get("no_progress_observation")
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _extract_artifact_output_dir(context: object | None) -> str:
    if not isinstance(context, dict):
        return ""
    value = context.get("artifact_output_dir")
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _extract_runtime_context(context: object | None) -> dict[str, object]:
    return dict(context) if isinstance(context, dict) else {}


def _format_summary_layer_context(layers: dict[str, object]) -> list[str]:
    rows: list[str] = []
    for key in ("resume_hint", "intent", "progress", "summary"):
        value = str(layers.get(key, "")).strip()
        if value:
            rows.append(f"{key}={value}")
    return rows


def _prompt_context(model_input: ModelInput) -> dict[str, object]:
    return {
        "stage": model_input.stage,
        "system_reminder": bool(model_input.system_reminder),
    }


# ProviderError 分类到 ModelError 分类的全映射。下游按字面值分支，透传新值等于静默失效：
# permission_denied 折回 auth 保住 403 的硬阻断，invalid_request 折回旧侧 4xx 兜底 format_error，
# unsupported_capability 折回 missing_config（配置边界、重试无用、硬阻断）。
# 用下标取值而非 .get 兜底，将来 ProviderErrorCategory 加成员会以 KeyError 暴露。
_MODEL_CATEGORY_BY_PROVIDER_CATEGORY: Final[Mapping[str, str]] = {
    "missing_config": "missing_config",
    "auth": "auth",
    "permission_denied": "auth",
    "billing": "billing",
    "invalid_request": "format_error",
    "model_not_found": "model_not_found",
    "context_overflow": "context_overflow",
    "payload_too_large": "payload_too_large",
    "rate_limited": "rate_limited",
    "overloaded": "overloaded",
    "server_error": "server_error",
    "timeout": "timeout",
    "transport_error": "transport_error",
    "cancelled": "cancelled",
    "invalid_provider_response": "invalid_provider_response",
    "invalid_model_protocol": "invalid_model_protocol",
    "unsupported_capability": "missing_config",
    "empty_response": "empty_response",
    "provider_error": "provider_error",
}
# 流协议错误码里只有这一条是"模型没给内容"而不是"协议坏了"：agent_loop 的空响应恢复计数、
# recovery_policy.empty_response_repeats 与全局失败预算都按 empty_response 生效。
_EMPTY_MESSAGE_CODE = "empty_assistant_message"
# Adapter 契约错误按消息前缀分类；注册与选型类配置错重试无用，一律硬阻断。
_MODEL_CATEGORY_BY_ADAPTER_CODE: Final[Mapping[str, str]] = {
    "unsupported_capability": "missing_config",
    "invalid_provider_response": "invalid_provider_response",
    "invalid_model_protocol": "invalid_model_protocol",
    "provider_state_mismatch": "invalid_model_protocol",
    "adapter_family_mismatch": "missing_config",
    "unknown_api_family": "missing_config",
    "duplicate_api_family": "missing_config",
    "blank_api_family": "missing_config",
}


def _provider_model_error(error: ProviderError) -> ModelError:
    """把 Provider 边界错误桥成 ModelError，保住 retryable 与退避秒数。

    作者：LKX
    时间：2026-08-30 18:50:00
    传参：error 为 Provider 上报的失败事实
    返回：分类已折回旧值域的 ModelError
    """
    return ModelError.create(
        category=_MODEL_CATEGORY_BY_PROVIDER_CATEGORY[error.category],
        summary=error.summary,
        raw_summary=error.summary,
        stage=error.stage,
        retryable=error.retryable,
        retry_after=error.retry_after_seconds,
    )


def _stream_protocol_model_error(exc: StreamProtocolError) -> ModelError:
    """把流组装失败桥成 ModelError：明确断流可重试，其余保持协议分类。

    作者：LKX
    时间：2026-08-30 18:50:00
    传参：exc 为 StreamAssembler 抛出的协议错误
    返回：ModelError，仅已开始且没有完成标记的断流允许重试
    """
    message = str(exc)
    if isinstance(exc, StreamInterruptedError):
        return ModelError.create(
            category="transport_error",
            summary=message,
            raw_summary=message,
            stage="stream_assemble",
            retryable=True,
        )
    code = message.split(":", 1)[0]
    category = (
        "empty_response" if code == _EMPTY_MESSAGE_CODE else "invalid_model_protocol"
    )
    return ModelError.create(
        category=category,
        summary=message,
        raw_summary=message,
        stage="stream_assemble",
    )


def _adapter_model_error(exc: ProviderAdapterError) -> ModelError:
    """把 Adapter 契约错误按码前缀桥成 ModelError。

    作者：LKX
    时间：2026-08-30 18:50:00
    传参：exc 为 Adapter 或注册表抛出的契约错误
    返回：不可重试的 ModelError；未登记的新码落 provider_error 而不是崩掉 plan
    """
    message = str(exc)
    code = message.split(":", 1)[0]
    return ModelError.create(
        category=_MODEL_CATEGORY_BY_ADAPTER_CODE.get(code, "provider_error"),
        summary=message,
        raw_summary=message,
        stage="request_build",
    )


def _selection_model_error(exc: ModelSelectionError) -> ModelError:
    """选型失败是配置与能力边界问题，重试无用，按硬阻断分类交回调用方。

    作者：LKX
    时间：2026-08-30 18:50:00
    传参：exc 为 ModelSelector 抛出的选型错误
    返回：missing_config 分类的 ModelError

    能力不满足必须走到调用方而不是被吞成 unknown，同时 plan() 仍要返回 LLMPlan，
    所以在这里转成错误 plan 而不是向上抛。
    """
    message = str(exc)
    return ModelError.create(
        category="missing_config",
        summary=message,
        raw_summary=message,
        stage="model_select",
    )


def _connection_model_error(exc: ProviderConnectionError) -> ModelError:
    """连接配置缺失或非法时按硬阻断分类交回调用方。"""
    message = str(exc)
    return ModelError.create(
        category="missing_config",
        summary=message,
        raw_summary=message,
        stage="connection",
    )


def _parse_assistant_message(
    message: AssistantMessage,
    *,
    protocol_mode: str,
    allowed_tool_names: Collection[str],
    registry: ToolRegistry,
) -> LLMPlan:
    """先核对供应商结束原因，再把完整助手消息解析成动作或答案。

    作者：LKX
    时间：2026-08-30 18:50:00
    传参：message 为助手消息；protocol_mode 为协议模式；allowed_tool_names 为放行工具名；
          registry 为工具注册表
    返回：解析后的 LLMPlan
    """
    # 【Agent运行】【停止归因】中断或结束原因缺失时保留响应证据，禁止将正文或调用当成完整计划
    interrupted = {
        StopReason.MAX_OUTPUT_TOKENS: (
            "output_limit_exceeded",
            "model response reached max_output_tokens before completion",
        ),
        StopReason.CONTENT_FILTER: (
            "content_filter",
            "provider stopped the response with content_filter",
        ),
        StopReason.CANCELLED: (
            "cancelled",
            "provider cancelled the response before completion",
        ),
        StopReason.UNKNOWN: (
            "invalid_provider_response",
            "provider did not report a recognized completion reason",
        ),
        None: (
            "invalid_provider_response",
            "provider did not report a completion reason",
        ),
    }
    if message.stop_reason in interrupted:
        category, summary = interrupted[message.stop_reason]
        return LLMPlan(
            model_error=ModelError.create(
                category=category, summary=summary, stage="response"
            )
        )
    call_parts = tuple(
        part for part in message.content if isinstance(part, ToolCallPart)
    )
    # 【Agent运行】【工具交接】工具结束信号缺少调用块属于空响应，旁白不能替代最终回答
    if call_parts or message.stop_reason is StopReason.TOOL_CALL:
        return parse_tool_call_parts(
            call_parts,
            allowed_tool_names=allowed_tool_names,
            registry=registry,
        )
    visible_text, _ = split_think_blocks(model_visible_text(message))
    return parse_llm_response(
        visible_text,
        protocol_mode=protocol_mode,
        allowed_tool_names=allowed_tool_names,
        registry=registry,
    )


def _output_delta(
    event: ModelStreamEvent,
    block_kinds: dict[str, str],
    protocol_mode: str,
) -> ModelOutputDelta | None:
    """把一个流事件读成可上屏的增量；不该上屏的返回 None。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：event 为本次流事件；block_kinds 记录已开块的种类；protocol_mode 为本轮协议模式
    返回：thinking / text 增量，或 None

    content_delta 事件本身不带 content_kind（只有 content_start 带），所以这里自己记
    一份 block_id -> kind。text_json 模式下 text 块装的是 {"type":"final",...} 协议信封，
    逐字上屏等于把协议噪声糊到用户脸上，故该模式只放思考链、不放正文。
    """
    if event.kind == "content_start":
        block_kinds[event.block_id] = str(event.content_kind)
        return None
    if event.kind != "content_delta" or not event.delta:
        return None
    kind = block_kinds.get(event.block_id)
    if kind == "thinking":
        return ModelOutputDelta("thinking", event.delta)
    if kind == "text" and protocol_mode == _NATIVE_TOOL_CALLS:
        return ModelOutputDelta("text", event.delta)
    return None


def _drain(stream: Generator[ModelStreamOutput, None, LLMPlan]) -> LLMPlan:
    """把增量生成器跑到底、丢掉增量，只取最终 plan。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：stream 为 plan_stream / continue_stream 返回的生成器
    返回：生成器收尾时的 LLMPlan

    只认同步 plan() 的调用方（约 20 个测试假件、runtime/sediment_reflection）不关心
    增量，与 runtime 里同步 run() 排空事件流的既有写法同构。
    """
    while True:
        try:
            next(stream)
        except StopIteration as stop:
            plan: LLMPlan = stop.value
            return plan


def _reasoning_content(message: AssistantMessage) -> str:
    """合并两处推理来源：独立 thinking 分片与正文里内联的 think 块。

    作者：LKX
    时间：2026-08-30 18:50:00
    传参：message 为助手消息
    返回：合并后的推理正文

    两种形状都可能出现在同一条消息里，只取一处会丢内容。
    """
    thinking = "\n\n".join(
        part.text for part in message.content if isinstance(part, ThinkingPart)
    )
    _, inline_reasoning = split_think_blocks(model_visible_text(message))
    return combine_reasoning_text(thinking, inline_reasoning)


def _attempt_trace(context: Mapping[str, object]) -> _AttemptTrace:
    """读取调用方提供的请求身份与证据写入器，独立调用生成自己的身份。

    传参：context 为本轮运行上下文；返回：请求跟踪配置，非法写入器显式报错
    """
    recorder = context.get("model_attempt_recorder")
    if recorder is not None and not callable(recorder):
        raise TypeError("model_attempt_recorder must be callable")
    from llm.protected_output import protect_provider_result

    request_id = str(context.get("model_request_id") or f"request-{uuid4().hex}")
    return _AttemptTrace(
        request_id=request_id,
        recorder=cast(Callable[[ModelAttemptEvent], None] | None, recorder),
        cancellation=cast(CancellationToken | None, context.get("cancellation")),
        reserve=cast(
            Callable[[ModelReservation], None] | None,
            context.get("reserve_model_attempt"),
        ),
        protect_result=partial(
            protect_provider_result,
            files=_registry_from_context(context).redacted_files,
            context=context,
            request_id=request_id,
        ),
    )


def _consume_provider_stream(
    events: Iterator[ModelStreamEvent],
    *,
    protocol_mode: str,
    attempt: ModelAttemptEvent,
    api_family: str,
    cancellation: CancellationToken | None = None,
) -> Generator[ModelStreamOutput, None, _ProviderCall]:
    """消费真实 Provider 流，把协议错误作为本次尝试的结果交回。

    传参：events 为实际响应流；protocol_mode 决定正文展示；返回：尝试结果与增量
    """
    assembler = StreamAssembler()
    block_kinds: dict[str, str] = {}
    try:
        for event in interruptible_events(events, cancellation):
            assembler.accept(event)
            delta = _output_delta(event, block_kinds, protocol_mode)
            if delta is not None:
                yield delta
        result = assembler.finish()
    except ProviderAdapterError as exc:
        error = _adapter_model_error(exc)
    except StreamProtocolError as exc:
        error = _stream_protocol_model_error(exc)
    except ExecutionCancelled as exc:
        error = ModelError.create(
            category="cancelled", summary=str(exc), stage="stream", retryable=False
        )
    else:
        return _ProviderCall(
            result=result,
            error=_provider_model_error(result.error) if result.error else None,
        )
    # 【模型调用】【尝试证据】协议破损不抹掉此前已确认的内容和计量
    failed = assembler.failure_result(
        ProviderError(
            category=(
                "transport_error"
                if error.category == "transport_error"
                else "cancelled"
                if error.category == "cancelled"
                else "invalid_provider_response"
            ),
            stage="stream_assemble",
            retryable=error.retryable,
            summary=error.summary,
            provider=attempt.provider,
            model=attempt.model,
            api_family=api_family,
        )
    )
    return _ProviderCall(result=failed, error=error)


def _complete_attempt(
    started: ModelAttemptEvent,
    call: _ProviderCall,
    *,
    timing: _StageTiming,
) -> ModelAttemptEvent:
    """结束一次尝试，保留失败时已上报的用量和部分响应。

    传参：started 为发送前记录；call 为实际结果；timing 为计时；返回：结束记录
    """
    message = (
        (call.result.message or call.result.partial_message) if call.result else None
    )
    return replace(
        started,
        phase="finished",
        elapsed_ms=timing.elapsed_ms(),
        response=provider_call_evidence(call.result, call.error),
        error=call.error,
        usage=(
            call.result.usage
            if call.result is not None and call.result.usage is not None
            else message.usage
            if message is not None and message.usage is not None
            else ModelUsage()
        ),
    )


def _attempt_usage_tokens(attempts: tuple[ModelAttemptEvent, ...]) -> _UsageTokens:
    """汇总请求内各尝试的已知用量，任一尝试未报告时总量保持未知。

    传参：attempts 为实际完成的尝试；返回：阶段用量，详细已知量仍保留在各次记录中
    """
    fields = {
        "prompt_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
        "total_tokens": "total_tokens",
        "cache_read_input_tokens": "cache_read_input_tokens",
        "cache_creation_input_tokens": "cache_write_input_tokens",
    }
    totals: dict[str, int | None] = {}
    for name, source in fields.items():
        values = [getattr(attempt.usage, source).value for attempt in attempts]
        known = [value for value in values if value is not None]
        totals[name] = sum(known) if known and len(known) == len(values) else None
    return _UsageTokens(**totals)


def _attach_request_facts(plan: LLMPlan, outcome: _AttemptOutcome) -> None:
    """把请求侧证据挂到 plan 上：提示上下文、真实请求体、响应投影与选型证据。"""
    composed = outcome.composed
    plan.prompt_context = {**plan.prompt_context, **composed.prompt_context}
    plan.render_text_to_model = composed.render_text_to_model
    plan.raw_model_request = dict(outcome.call.request_body)
    plan.raw_model_response = provider_call_evidence(
        outcome.call.result, outcome.call.error
    )
    plan.protocol_mode = composed.protocol_mode
    evidence = composed_request_evidence(composed)
    evidence.update(outcome.call.selection)
    plan.request_bundle_evidence = evidence
    plan.request_id = outcome.request_id
    plan.model_attempts = outcome.model_attempts
    plan.registry_snapshot = composed.registry_snapshot


def _selection_evidence(decision: SelectionDecision) -> dict[str, object]:
    """记下选型结论旁的落选原因与提示，便于事后判断为什么选了这个模型。"""
    return {
        "selected_model_key": decision.selected.model_key,
        "selection_rejections": [
            {"model_key": item.model_key, "reasons": list(item.reasons)}
            for item in decision.rejections
        ],
        "selection_notices": list(decision.notices),
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _interruptible_backoff(seconds: float, token: CancellationToken) -> None:
    """重试等待可响应停止，取消后不会再派发；传参：秒数与信号；返回：无。"""
    if token.wait(seconds):
        raise ExecutionCancelled("model request cancelled during retry backoff")
