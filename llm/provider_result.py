from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from llm.error_classifier import classify_http_status, classify_provider_exception
from llm.messages import AssistantMessage
from llm.types import ErrorCategory, MeasurementStatus, ModelUsage, UsageMeasurement


ProviderErrorCategory = Literal[
    "missing_config",
    "auth",
    "permission_denied",
    "billing",
    "invalid_request",
    "model_not_found",
    "context_overflow",
    "payload_too_large",
    "rate_limited",
    "overloaded",
    "server_error",
    "timeout",
    "transport_error",
    "cancelled",
    "invalid_provider_response",
    "invalid_model_protocol",
    "unsupported_capability",
    "empty_response",
    "provider_error",
]
ProviderErrorStage = Literal[
    "connection", "request_build", "transport", "stream_decode", "stream_assemble"
]
_ERROR_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "missing_config",
        "auth",
        "permission_denied",
        "billing",
        "invalid_request",
        "model_not_found",
        "context_overflow",
        "payload_too_large",
        "rate_limited",
        "overloaded",
        "server_error",
        "timeout",
        "transport_error",
        "cancelled",
        "invalid_provider_response",
        "invalid_model_protocol",
        "unsupported_capability",
        "empty_response",
        "provider_error",
    }
)
_ERROR_STAGES: Final[frozenset[str]] = frozenset(
    {"connection", "request_build", "transport", "stream_decode", "stream_assemble"}
)
_USAGE_FIELDS: Final[tuple[str, ...]] = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
)
# 只有这五类失败值得重试，与 llm/types.py:RETRYABLE_ERROR_CATEGORIES 保持同一口径
_RETRYABLE_ERROR_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"transport_error", "timeout", "rate_limited", "overloaded", "server_error"}
)
# 三组异常按继承链类名判定，不看报文：SDK 与 httpx 的取消/超时/连接类都在这里收口
_CANCELLED_ERROR_NAMES: Final[frozenset[str]] = frozenset(
    {"CancelledError", "CancelledException"}
)
_TIMEOUT_ERROR_NAMES: Final[frozenset[str]] = frozenset(
    {"TimeoutException", "ReadTimeout", "ConnectTimeout", "APITimeoutError", "Timeout"}
)
_TRANSPORT_ERROR_NAMES: Final[frozenset[str]] = frozenset(
    {"ConnectError", "NetworkError", "TransportError", "APIConnectionError"}
)
# 无 HTTP 状态码时，只有 Provider SDK 根类才退回报文关键字分类；其余异常属编程错误，继续抛出
_PROVIDER_SDK_ERROR_NAMES: Final[frozenset[str]] = frozenset(
    {"OpenAIError", "APIError", "AnthropicError"}
)
# 状态码到分类的结构映射；503 与 529 同为过载，413 是载荷过大而非上下文溢出
_STATUS_CATEGORIES: Final[Mapping[int, ProviderErrorCategory]] = {
    401: "auth",
    402: "billing",
    403: "permission_denied",
    404: "model_not_found",
    408: "timeout",
    409: "transport_error",
    413: "payload_too_large",
    425: "transport_error",
    429: "rate_limited",
    503: "overloaded",
    529: "overloaded",
}
# 这三类结构判定会被报文推翻：402 可能是配额重置、404 可能不是模型问题、4xx 兜底可能是上下文溢出
_MESSAGE_REFINABLE_CATEGORIES: Final[frozenset[str]] = frozenset(
    {"billing", "model_not_found", "invalid_request"}
)
# ErrorCategory 到 ProviderErrorCategory 的显式跨值域映射，必须是全映射，缺项应以 KeyError 暴露
_PROVIDER_CATEGORY_BY_MODEL_CATEGORY: Final[
    Mapping[ErrorCategory, ProviderErrorCategory]
] = {
    ErrorCategory.MISSING_CONFIG: "missing_config",
    ErrorCategory.AUTH: "auth",
    ErrorCategory.BILLING: "billing",
    ErrorCategory.TRANSPORT: "transport_error",
    ErrorCategory.TIMEOUT: "timeout",
    ErrorCategory.RATE_LIMITED: "rate_limited",
    ErrorCategory.OVERLOADED: "overloaded",
    ErrorCategory.SERVER_ERROR: "server_error",
    ErrorCategory.CONTEXT_OVERFLOW: "context_overflow",
    ErrorCategory.PAYLOAD_TOO_LARGE: "payload_too_large",
    ErrorCategory.MODEL_NOT_FOUND: "model_not_found",
    ErrorCategory.FORMAT_ERROR: "invalid_request",
    ErrorCategory.PROVIDER_ERROR: "provider_error",
    ErrorCategory.INVALID_PROVIDER_RESPONSE: "invalid_provider_response",
    ErrorCategory.INVALID_PROTOCOL: "invalid_model_protocol",
    ErrorCategory.EMPTY_RESPONSE: "empty_response",
    ErrorCategory.UNKNOWN: "provider_error",
}
_HTTP_CLIENT_ERROR_MIN: Final[int] = 400
_HTTP_SERVER_ERROR_MIN: Final[int] = 500
_HTTP_STATUS_CEILING: Final[int] = 600


@dataclass(frozen=True, slots=True)
class ProviderError:
    """保存 Provider 调用边界的脱敏失败事实。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：category/stage 为闭合值域；其余字段定位失败调用且不得包含 secret
    返回：不可变 Provider 错误
    """

    category: ProviderErrorCategory
    stage: ProviderErrorStage
    retryable: bool
    summary: str
    provider: str
    model: str
    api_family: str
    http_status: int | None = None
    request_id: str = ""
    retry_after_seconds: float | None = None
    evidence_ref: str = ""

    def __post_init__(self) -> None:
        """校验 Provider 错误值域和安全字段。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法字段抛 ValueError
        """
        if self.category not in _ERROR_CATEGORIES:
            raise ValueError(f"unknown_provider_error_category:{self.category}")
        if self.stage not in _ERROR_STAGES:
            raise ValueError(f"unknown_provider_error_stage:{self.stage}")
        for name in ("summary", "provider", "model", "api_family"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"provider_error_{name}_required")
        if self.http_status is not None and not 100 <= self.http_status <= 599:
            raise ValueError("provider_error_http_status_invalid")
        if self.retry_after_seconds is not None and self.retry_after_seconds < 0:
            raise ValueError("provider_error_retry_after_invalid")
        object.__setattr__(self, "summary", _redact_summary(self.summary))


@dataclass(frozen=True, slots=True)
class ProviderCallResult:
    """保存流组装后的成功消息或显式失败。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：message 与 error 必须且只能存在一个；partial_message 保存失败前确认内容
    返回：Provider 调用的统一结果
    """

    message: AssistantMessage | None = None
    error: ProviderError | None = None
    partial_message: AssistantMessage | None = None
    usage: ModelUsage | None = None

    def __post_init__(self) -> None:
        """校验成功与失败结果互斥。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法组合抛 ValueError
        """
        if (self.message is None) == (self.error is None):
            raise ValueError("provider_result_requires_exactly_one_terminal")
        if self.message is not None and self.partial_message is not None:
            raise ValueError("successful_provider_result_has_no_partial_message")


def reported(value: int) -> UsageMeasurement:
    """构造 Provider 明确报告的用量值。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：value 为非负 token 数，包括零
    返回：reported 状态的 UsageMeasurement
    """
    return UsageMeasurement(MeasurementStatus.REPORTED, value)


def not_applicable() -> UsageMeasurement:
    """构造 Provider 明确不存在该概念的用量值。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：无
    返回：not_applicable 状态的 UsageMeasurement
    """
    return UsageMeasurement(MeasurementStatus.NOT_APPLICABLE, None)


class UsageAccumulator:
    """按累计 snapshot 合并五项 canonical usage。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：无
    返回：可接收多个 usage_update 的合并器
    """

    def __init__(self) -> None:
        """初始化五项 unknown measurement。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无
        """
        self._usage = ModelUsage()

    def merge(self, update: ModelUsage) -> None:
        """合并一个累计 usage snapshot，unknown 不覆盖已知值。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：update 为 Provider 最新累计 snapshot
        返回：无
        """
        values: dict[str, UsageMeasurement] = {}
        for name in _USAGE_FIELDS:
            current = getattr(self._usage, name)
            incoming = getattr(update, name)
            values[name] = (
                current if incoming.status is MeasurementStatus.UNKNOWN else incoming
            )
        self._usage = ModelUsage(**values)

    def apply(self, update: ModelUsage) -> None:
        """以明确的 accumulator 动词合并一个 usage snapshot。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：update 为 Provider 累计 snapshot
        返回：无
        """
        self.merge(update)

    def finalize(self) -> ModelUsage:
        """返回最终 usage，并在输入输出可靠时推导缺失 total。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：不可变 ModelUsage
        """
        if self._usage.total_tokens.status is not MeasurementStatus.UNKNOWN:
            return self._usage
        input_tokens = self._usage.input_tokens
        output_tokens = self._usage.output_tokens
        if input_tokens.value is None or output_tokens.value is None:
            return self._usage
        total = UsageMeasurement(
            MeasurementStatus.DERIVED,
            input_tokens.value + output_tokens.value,
            ("input_tokens", "output_tokens"),
        )
        return ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total,
            cache_read_input_tokens=self._usage.cache_read_input_tokens,
            cache_write_input_tokens=self._usage.cache_write_input_tokens,
        )


def normalize_provider_error(
    error: BaseException,
    *,
    provider: str,
    model: str,
    api_family: str,
    stage: ProviderErrorStage,
) -> ProviderError:
    """将明确的网络/HTTP/取消异常归一化，未知编程异常继续抛出。

    作者：xxx
    时间：2026-08-30 16:20:00
    传参：error 为已知边界异常；其余参数为调用身份和失败阶段
    返回：唯一 ProviderError；未知异常原样抛出
    """
    status = _status_code(error)
    summary = _safe_summary(error)
    category = _classify(error, status, summary)
    return ProviderError(
        category,
        stage,
        category in _RETRYABLE_ERROR_CATEGORIES,
        summary,
        provider,
        model,
        api_family,
        http_status=status,
        request_id=_request_id(error),
        retry_after_seconds=_retry_after(error),
    )


def _classify(
    error: BaseException, status: int | None, summary: str
) -> ProviderErrorCategory:
    """按结构、状态码、报文三层顺序判定失败分类。

    作者：xxx
    时间：2026-08-30 16:20:00
    传参：error 为边界异常；status 为已校验 HTTP 状态码或 None；summary 为脱敏后报文
    返回：ProviderErrorCategory；无法归类的编程异常原样抛出
    """
    # 1. 取消、超时、连接类异常按继承链类名判定，报文不参与，避免文案变化影响重试
    structural = _structural_category(error)
    if structural is not None:
        return structural
    # 2. 带 HTTP 状态码时以状态码为主，再交给报文精修配额重置、非模型 404 与上下文溢出
    if status is not None:
        return _http_category(status, summary)
    # 3. 无状态码的 SDK 边界异常退回关键字分类，覆盖端点把失败写在报文里的形态
    sdk_category = _sdk_message_category(error)
    if sdk_category is not None:
        return sdk_category
    # 4. 其余异常不属于 Provider 边界失败，原样抛出让编程错误清晰暴露
    raise error


def _structural_category(error: BaseException) -> ProviderErrorCategory | None:
    """按异常继承链类名判定取消、超时与传输失败。

    作者：xxx
    时间：2026-08-30 16:20:00
    传参：error 为边界异常
    返回：命中的分类；三类都不命中返回 None
    """
    # 取继承链全部类名而非叶子类名：httpx 的 WriteTimeout/PoolTimeout 只在父类上叫 TimeoutException，
    # RemoteProtocolError 等只在父类上叫 TransportError，只比叶子名会让它们逃逸成未归类异常
    names = _class_names(error)
    if names & _CANCELLED_ERROR_NAMES:
        return "cancelled"
    # 超时先于传输判定：ReadTimeout 的继承链同时含 TimeoutException 与 TransportError
    if isinstance(error, TimeoutError) or names & _TIMEOUT_ERROR_NAMES:
        return "timeout"
    if isinstance(error, ConnectionError) or names & _TRANSPORT_ERROR_NAMES:
        return "transport_error"
    return None


def _class_names(error: BaseException) -> frozenset[str]:
    """收集异常整条继承链上的类名。

    作者：xxx
    时间：2026-08-30 16:20:00
    传参：error 为任意异常实例
    返回：继承链类名集合
    """
    return frozenset(cls.__name__ for cls in type(error).__mro__)


def _sdk_message_category(error: BaseException) -> ProviderErrorCategory | None:
    """对无状态码的 Provider SDK 异常按报文关键字分类。

    作者：xxx
    时间：2026-08-30 16:20:00
    传参：error 为边界异常
    返回：映射后的分类；非 SDK 根类异常返回 None 以便原样抛出
    """
    if not _class_names(error) & _PROVIDER_SDK_ERROR_NAMES:
        return None
    # 【模型调用】【断流识别】HTTP 已建立后的 SSE error 没有状态码，明确的上游断流仍属于传输失败
    if _safe_summary(error).casefold() == "upstream stream disconnected":
        return "transport_error"
    return _provider_category(classify_provider_exception(error))


def _http_category(status: int, summary: str) -> ProviderErrorCategory:
    """按状态码定分类，并在报文能推翻结构判定时精修。

    作者：xxx
    时间：2026-08-30 16:20:00
    传参：status 为 HTTP 状态码；summary 为脱敏后报文
    返回：ProviderErrorCategory
    """
    structural = _STATUS_CATEGORIES.get(status) or _status_range_category(status)
    if structural not in _MESSAGE_REFINABLE_CATEGORIES:
        return structural
    # 402 带配额重置提示时是限流、404 不含模型语义时不是模型缺失、4xx 兜底命中溢出关键词时是上下文溢出，
    # 三条判据都由 error_classifier.classify_http_status 独占，此处只做跨值域映射不重抄关键字表
    return _provider_category(classify_http_status(status, summary))


def _status_range_category(status: int) -> ProviderErrorCategory:
    """为未列入状态码表的取值给出区间兜底分类。

    作者：xxx
    时间：2026-08-30 16:20:00
    传参：status 为 HTTP 状态码
    返回：ProviderErrorCategory
    """
    if _HTTP_CLIENT_ERROR_MIN <= status < _HTTP_SERVER_ERROR_MIN:
        return "invalid_request"
    if _HTTP_SERVER_ERROR_MIN <= status < _HTTP_STATUS_CEILING:
        return "server_error"
    return "provider_error"


def _provider_category(category: ErrorCategory) -> ProviderErrorCategory:
    """把 ErrorCategory 翻译成 Provider 边界值域。

    作者：xxx
    时间：2026-08-30 16:20:00
    传参：category 为 error_classifier 的判定结果
    返回：ProviderErrorCategory；映射缺项以 KeyError 暴露而不静默兜底
    """
    return _PROVIDER_CATEGORY_BY_MODEL_CATEGORY[category]


def _status_code(error: BaseException) -> int | None:
    value = getattr(error, "status_code", getattr(error, "status", None))
    return value if isinstance(value, int) and 100 <= value <= 599 else None


def _safe_summary(error: BaseException) -> str:
    """提取 Provider 公开错误消息并统一脱敏。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：error 为 SDK、HTTP 或网络边界异常
    返回：不含认证凭证的稳定错误摘要
    """
    body = getattr(error, "body", None)
    if isinstance(body, Mapping):
        payload = body.get("error", body)
        if isinstance(payload, Mapping):
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                return _redact_summary(message.strip())
    value = str(error).strip()
    if not value:
        return error.__class__.__name__
    return _redact_summary(value)


def _redact_summary(value: str) -> str:
    """移除异常文本中的常见认证凭证，避免进入错误摘要。"""
    patterns = (
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"(?i)(x-api-key\s*[:=]\s*)[^\s,;]+",
        r"(?i)(api[-_ ]?key\s*[:=]\s*)[^\s,;]+",
        r"(?i)(bearer\s+)[^\s,;]+",
    )
    redacted = value
    for pattern in patterns:
        redacted = re.sub(pattern, r"\1<redacted>", redacted)
    return redacted


def _request_id(error: BaseException) -> str:
    value = getattr(error, "request_id", "")
    return value if isinstance(value, str) else ""


def _retry_after(error: BaseException) -> float | None:
    """取限流退避秒数，属性缺失时回落到 HTTP 响应头。

    作者：xxx
    时间：2026-08-30 16:20:00
    传参：error 为边界异常
    返回：非负退避秒数；两条来源都取不到返回 None
    """
    # 1. 少数 SDK 把秒数直接挂在异常上，优先采信
    value = getattr(error, "retry_after", None)
    if isinstance(value, int | float) and not isinstance(value, bool) and value >= 0:
        return float(value)
    # 2. openai SDK 3.2.0 的 RateLimitError 没有 retry_after 属性，秒数只在 429 响应头里，
    #    不读响应头会让限流退避时间整体丢失，退化成固定退避
    return _retry_after_header(getattr(error, "response", None))


def _retry_after_header(response: object) -> float | None:
    """从 HTTP 响应头解析 Retry-After 秒数。

    作者：xxx
    时间：2026-08-30 16:20:00
    传参：response 为携带 headers 的响应对象或 None
    返回：非负退避秒数；缺头或不可解析返回 None
    """
    get = getattr(getattr(response, "headers", None), "get", None)
    if get is None:
        return None
    # httpx Headers 本身大小写无关，普通 dict 不是，两种大小写都取一次
    raw = get("retry-after") or get("Retry-After")
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


__all__ = [
    "ProviderCallResult",
    "ProviderError",
    "ProviderErrorCategory",
    "ProviderErrorStage",
    "UsageAccumulator",
    "normalize_provider_error",
    "not_applicable",
    "reported",
]
