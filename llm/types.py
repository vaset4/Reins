from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Final, Literal, Mapping, Sequence, cast

from runtime.types import RunToolsRequest

if TYPE_CHECKING:
    from llm.messages import AssistantMessage
    from tools.tool_registry import ToolRegistry


class CacheTier(str, Enum):
    STABLE = "stable"
    SEMI_STABLE = "semi_stable"
    DYNAMIC = "dynamic"


class ErrorCategory(str, Enum):
    MISSING_CONFIG = "missing_config"
    AUTH = "auth"
    BILLING = "billing"
    TRANSPORT = "transport_error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    OVERLOADED = "overloaded"
    SERVER_ERROR = "server_error"
    CONTEXT_OVERFLOW = "context_overflow"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    MODEL_NOT_FOUND = "model_not_found"
    FORMAT_ERROR = "format_error"
    PROVIDER_ERROR = "provider_error"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    INVALID_PROTOCOL = "invalid_model_protocol"
    EMPTY_RESPONSE = "empty_response"
    UNKNOWN = "unknown"

    @property
    def retryable(self) -> bool:
        return self in RETRYABLE_ERROR_CATEGORIES


RETRYABLE_ERROR_CATEGORIES: Final[frozenset[ErrorCategory]] = frozenset(
    {
        ErrorCategory.TRANSPORT,
        ErrorCategory.TIMEOUT,
        ErrorCategory.RATE_LIMITED,
        ErrorCategory.OVERLOADED,
        ErrorCategory.SERVER_ERROR,
    }
)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __post_init__(self) -> None:
        for value in (
            self.input_tokens,
            self.output_tokens,
            self.cache_read_input_tokens,
            self.cache_creation_input_tokens,
        ):
            if value < 0:
                raise ValueError("token counts must be non-negative")

    def __add__(self, other: object) -> TokenUsage:
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
        )


class MeasurementStatus(str, Enum):
    """定义模型用量单项指标的观测状态。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：枚举值由 Provider Adapter 或严格反序列化入口提供
    返回：可区分已报告、推导、未知和不适用的状态
    """

    REPORTED = "reported"
    DERIVED = "derived"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class UsageContractError(ValueError):
    """表示模型用量合同构造或反序列化失败。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：code 为稳定错误码；path 为失败字段路径；detail 为具体原因
    返回：携带 code、path 和 detail 的 ValueError
    """

    def __init__(self, code: str, path: str, detail: str) -> None:
        """初始化可定位的模型用量合同错误。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：code 为稳定错误码；path 为失败字段路径；detail 为具体原因
        返回：无；构造异常对象
        """
        self.code = code
        self.path = path
        self.detail = detail
        super().__init__(f"{code} at {path}: {detail}")


@dataclass(frozen=True, slots=True)
class UsageMeasurement:
    """保存一个用量指标的状态、数值和推导来源。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：status 为观测状态；value 为非负数值或 None；derived_from 为来源字段
    返回：校验后的不可变用量指标
    """

    status: MeasurementStatus
    value: int | None
    derived_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """校验状态、数值和推导来源之间的不变量。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法组合抛 UsageContractError
        """
        if not isinstance(self.status, MeasurementStatus):
            raise _usage_error("status", "status must be MeasurementStatus")
        if isinstance(self.derived_from, (str, bytes, bytearray)) or not isinstance(
            self.derived_from,
            Sequence,
        ):
            raise _usage_error(
                "derived_from",
                "sources must be an ordered sequence",
            )
        sources = tuple(self.derived_from)
        object.__setattr__(self, "derived_from", sources)
        _validate_measurement_sources(sources)
        if self.status is MeasurementStatus.DERIVED:
            _validate_derived_measurement(self.value, sources)
            return
        if self.status is MeasurementStatus.REPORTED:
            _validate_reported_measurement(self.value, sources)
            return
        if self.value is not None or sources:
            raise _usage_error(
                "value",
                f"{self.status.value} requires value=None and no derived_from",
            )


def _unknown_measurement() -> UsageMeasurement:
    return UsageMeasurement(MeasurementStatus.UNKNOWN, None)


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """保存一次模型调用的五项标准用量指标。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：各字段为 UsageMeasurement；省略时明确记为 unknown
    返回：不可变、不会把缺失值伪装为零的模型用量
    """

    input_tokens: UsageMeasurement = field(default_factory=_unknown_measurement)
    output_tokens: UsageMeasurement = field(default_factory=_unknown_measurement)
    total_tokens: UsageMeasurement = field(default_factory=_unknown_measurement)
    cache_read_input_tokens: UsageMeasurement = field(
        default_factory=_unknown_measurement
    )
    cache_write_input_tokens: UsageMeasurement = field(
        default_factory=_unknown_measurement
    )

    def __post_init__(self) -> None:
        """拒绝旧 TokenUsage 或其他弱类型指标。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；字段类型错误抛 UsageContractError
        """
        for name in _USAGE_FIELD_NAMES:
            if not isinstance(getattr(self, name), UsageMeasurement):
                raise UsageContractError(
                    "invalid_usage_measurement",
                    f"usage.{name}",
                    "field must be UsageMeasurement",
                )


_USAGE_FIELD_NAMES: Final[tuple[str, ...]] = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
)


def usage_to_mapping(usage: ModelUsage) -> dict[str, object]:
    """把模型用量转换为新的普通 JSON 映射。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：usage 为经过校验的 ModelUsage
    返回：保留 status、null 和 derived_from 的新映射
    """
    if not isinstance(usage, ModelUsage):
        raise UsageContractError(
            "invalid_usage_measurement",
            "usage",
            "value must be ModelUsage",
        )
    return {
        name: _measurement_to_mapping(getattr(usage, name))
        for name in _USAGE_FIELD_NAMES
    }


def usage_from_mapping(value: object) -> ModelUsage:
    """从严格 mapping 恢复状态化模型用量。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：value 为包含五项标准指标的 mapping
    返回：校验后的 ModelUsage；旧 TokenUsage 和未知字段会失败
    """
    raw = _usage_mapping(value, "usage")
    _validate_usage_keys(
        raw,
        set(_USAGE_FIELD_NAMES),
        set(_USAGE_FIELD_NAMES),
        path="usage",
    )
    measurements = {
        name: _measurement_from_mapping(raw[name], f"usage.{name}")
        for name in _USAGE_FIELD_NAMES
    }
    return ModelUsage(**measurements)


def _usage_error(path: str, detail: str) -> UsageContractError:
    return UsageContractError("invalid_usage_measurement", f"usage.{path}", detail)


def _validate_measurement_sources(sources: tuple[str, ...]) -> None:
    if any(not isinstance(source, str) or not source.strip() for source in sources):
        raise _usage_error("derived_from", "sources must be non-empty strings")
    if len(set(sources)) != len(sources):
        raise _usage_error("derived_from", "sources must be unique")


def _valid_usage_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_derived_measurement(value: int | None, sources: tuple[str, ...]) -> None:
    if not _valid_usage_count(value) or not sources:
        raise _usage_error(
            "value",
            "derived requires a non-negative integer and at least one source",
        )


def _validate_reported_measurement(value: int | None, sources: tuple[str, ...]) -> None:
    if not _valid_usage_count(value) or sources:
        raise _usage_error(
            "value",
            "reported requires a non-negative integer and no derived_from",
        )


def _measurement_to_mapping(measurement: UsageMeasurement) -> dict[str, object]:
    return {
        "status": measurement.status.value,
        "value": measurement.value,
        "derived_from": list(measurement.derived_from),
    }


def _measurement_from_mapping(value: object, path: str) -> UsageMeasurement:
    raw = _usage_mapping(value, path)
    allowed = {"status", "value", "derived_from"}
    _validate_usage_keys(raw, allowed, allowed, path=path)
    try:
        status = MeasurementStatus(raw["status"])
    except (TypeError, ValueError) as exc:
        raise UsageContractError(
            "invalid_usage_measurement",
            f"{path}.status",
            "unknown measurement status",
        ) from exc
    sources = raw["derived_from"]
    if not isinstance(sources, list):
        raise UsageContractError(
            "invalid_usage_measurement",
            f"{path}.derived_from",
            "derived_from must be a list",
        )
    try:
        return UsageMeasurement(
            status,
            cast(int | None, raw["value"]),
            tuple(cast(list[str], sources)),
        )
    except UsageContractError as exc:
        error_path = path
        if exc.path.startswith("usage."):
            error_path = f"{path}{exc.path[len('usage') :]}"
        raise UsageContractError(exc.code, error_path, exc.detail) from exc


def _usage_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise UsageContractError(
            "invalid_usage_measurement",
            path,
            "value must be a mapping",
        )
    return value


def _validate_usage_keys(
    raw: Mapping[str, object],
    allowed: set[str],
    required: set[str],
    *,
    path: str,
) -> None:
    invalid_keys = [key for key in raw if not isinstance(key, str)]
    if invalid_keys:
        raise UsageContractError("unknown_field", path, "field names must be strings")
    unknown = set(raw) - allowed
    missing = required - set(raw)
    if unknown:
        raise UsageContractError(
            "unknown_field", path, f"unknown fields: {sorted(unknown)}"
        )
    if missing:
        raise UsageContractError(
            "missing_field", path, f"missing fields: {sorted(missing)}"
        )


@dataclass(frozen=True, slots=True)
class ProviderHint:
    cache_tier: CacheTier | str = CacheTier.DYNAMIC

    def __post_init__(self) -> None:
        if isinstance(self.cache_tier, str):
            object.__setattr__(self, "cache_tier", CacheTier(self.cache_tier))


@dataclass(frozen=True, slots=True)
class PromptSection:
    name: str
    content: str
    hint: ProviderHint = ProviderHint()


@dataclass(slots=True)
class ModelError:
    category: str
    summary: str
    raw_summary: str = ""
    stage: str = ""
    retryable: bool = False
    retry_after: float | None = None

    @classmethod
    def create(
        cls,
        *,
        category: str,
        summary: str,
        raw_summary: str = "",
        stage: str = "",
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> "ModelError":
        if not retryable:
            retryable = category in {
                "transport_error",
                "timeout",
                "rate_limited",
                "overloaded",
                "server_error",
            }
        return cls(
            category,
            summary,
            raw_summary,
            stage,
            retryable,
            retry_after=retry_after,
        )

    def render_output(self) -> str:
        """生成面向运行与用户的错误说明；传参：无；返回：保留错误来源的正文。"""
        if self.stage == "response":
            return f"MODEL_RESPONSE_ERROR: {self.summary}"
        provider_categories = {
            "missing_config",
            "auth",
            "billing",
            "provider_error",
            "transport_error",
            "timeout",
            "rate_limited",
            "overloaded",
            "server_error",
            "context_overflow",
            "payload_too_large",
            "model_not_found",
        }
        if self.stage == "transport" or self.category in provider_categories:
            return f"MODEL_PROVIDER_ERROR: {self.summary}"
        return f"MODEL_PROTOCOL_ERROR: {self.summary}"


@dataclass(slots=True)
class ModelObservation:
    stage: str
    provider: str
    model: str
    started_at: str
    elapsed_ms: int
    attempt_count: int
    success: bool
    error_category: str | None = None
    was_retried: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost: float | None = None
    total_wait_seconds: float = 0.0
    retry_after_used: float | None = None
    config_source: str = ""
    credential_source: str = ""
    base_url_host: str = ""
    profile_name: str = ""
    credential_name: str = ""
    should_fallback: bool = False
    fallback_reason: str = ""
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelOutputDelta:
    """模型输出的一小片增量，供 Runtime 在模型仍在生成时透出给界面。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：channel 区分思考链与回答正文；text 为本片文本
    返回：不可变增量

    Runtime 不许读 Provider 的 Wire 字段（api_family / block_id / sequence 这些），
    所以 client 向上只交这两个字段，`ModelStreamEvent` 止步于 llm/ 内部。
    """

    channel: Literal["thinking", "text"]
    text: str


@dataclass(frozen=True, slots=True)
class ModelRetryNotice:
    """【模型调用】【等待重试】向界面说明下一次请求的等待，不携带模型正文。

    作者：xxx
    时间：2026-09-29 10:45:00
    传参：attempt_index 为下一次尝试序号；max_attempts 为总次数上限；
          wait_seconds 为等待秒数；error_category 为本次失败分类
    返回：不可变重试提示
    """

    attempt_index: int
    max_attempts: int
    wait_seconds: float
    error_category: str


@dataclass(frozen=True, slots=True)
class ModelAttemptStarted:
    """模型尝试已获预算并即将派发；参数：尝试序号和总上限；返回：独立状态通知。"""

    attempt_index: int
    max_attempts: int


ModelStreamOutput = ModelOutputDelta | ModelRetryNotice | ModelAttemptStarted


@dataclass(frozen=True, slots=True)
class ModelAttemptEvent:
    """保存一次真实模型尝试的开始或结束证据，重试共享请求身份。

    作者：xxx
    时间：2026-09-13 20:00:00
    传参：phase 区分发送前与结束；其余字段为身份、请求响应和实际用量
    返回：供运行时持久化的 Provider 无关记录
    """

    phase: Literal["started", "finished"]
    request_id: str
    attempt_id: str
    attempt_index: int
    provider: str
    model: str
    started_at: str
    elapsed_ms: int = 0
    request: Mapping[str, object] = field(default_factory=dict)
    response: Mapping[str, object] = field(default_factory=dict)
    usage: ModelUsage = field(default_factory=ModelUsage)
    error: ModelError | None = None
    input_ids: tuple[str, ...] = ()
    api_family: str = ""
    sources: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class LLMPlan:
    """模型单轮计划。final_output / run_tools_request 一轮只能有一个非空
    （由 parser 单 type 分派天然保证，本处仅记录不变量供消费侧信赖）。
    作者：LKX
    时间：2026-07-05 00:00:00"""

    final_output: str | None = None
    run_tools_request: RunToolsRequest | None = None
    reasoning_content: str = ""
    model_error: ModelError | None = None
    observation: ModelObservation | None = None
    prompt_context: dict[str, object] = field(default_factory=dict)
    render_text_to_model: str = ""
    raw_model_request: dict[str, object] = field(default_factory=dict)
    raw_model_response: dict[str, object] = field(default_factory=dict)
    protocol_mode: str = ""
    trim_delta: dict[str, object] | None = None
    request_bundle_evidence: dict[str, object] = field(default_factory=dict)
    request_id: str = ""
    model_attempts: tuple[ModelAttemptEvent, ...] = ()
    registry_snapshot: ToolRegistry | None = field(default=None, repr=False)
    assistant_message: AssistantMessage | None = field(default=None, repr=False)


__all__ = [
    "CacheTier",
    "ErrorCategory",
    "LLMPlan",
    "MeasurementStatus",
    "ModelError",
    "ModelAttemptEvent",
    "ModelObservation",
    "ModelOutputDelta",
    "ModelRetryNotice",
    "ModelAttemptStarted",
    "ModelStreamOutput",
    "ModelUsage",
    "ProviderHint",
    "PromptSection",
    "RETRYABLE_ERROR_CATEGORIES",
    "TokenUsage",
    "UsageContractError",
    "UsageMeasurement",
    "usage_from_mapping",
    "usage_to_mapping",
]
