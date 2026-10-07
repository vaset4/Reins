from __future__ import annotations
from scripts.testing.llm import from_test_stub

import asyncio

import httpx2
import openai
import pytest

from llm.provider_result import (
    ProviderError,
    UsageAccumulator,
    normalize_provider_error,
    reported,
)
from llm.provider_stream import ModelStreamEvent
from llm.types import (
    MeasurementStatus,
    ModelObservation,
    ModelUsage,
    UsageMeasurement,
)


def _normalize(error: BaseException) -> ProviderError:
    """以固定身份归一化一个边界异常，供分类断言复用。

    作者：xxx
    时间：2026-08-30 16:20:00
    传参：error 为待归一化异常
    返回：ProviderError
    """
    return normalize_provider_error(
        error,
        provider="fixture",
        model="fixture-model",
        api_family="openai_chat",
        stage="transport",
    )


def _status_error(
    status: int, message: str, *, headers: dict[str, str] | None = None
) -> openai.APIStatusError:
    """造一个带真实 httpx 响应的 openai 状态异常。

    作者：xxx
    时间：2026-08-30 16:20:00
    传参：status 为 HTTP 状态码；message 为端点报文；headers 为响应头
    返回：openai.APIStatusError
    """
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    body = {"error": {"message": message}}
    response = httpx2.Response(
        status, headers=headers or {}, request=request, json=body
    )
    return openai.APIStatusError(message, response=response, body=body)


def test_provider_error_is_typed_and_immutable() -> None:
    error = ProviderError(
        category="rate_limited",
        stage="transport",
        retryable=True,
        summary="request throttled",
        provider="fixture",
        model="fixture-model",
        api_family="openai_chat",
        http_status=429,
    )
    assert error.retryable is True


def test_provider_error_constructor_redacts_secret_bearing_summary() -> None:
    error = ProviderError(
        "auth",
        "transport",
        False,
        "x-api-key=provider-secret",
        "fixture",
        "fixture-model",
        "anthropic_messages",
    )
    assert error.summary == "x-api-key=<redacted>"


def test_usage_accumulator_preserves_reported_zero_and_derives_total() -> None:
    accumulator = UsageAccumulator()
    accumulator.merge(
        ModelUsage(
            input_tokens=reported(0),
            output_tokens=reported(3),
            cache_read_input_tokens=reported(0),
            cache_write_input_tokens=UsageMeasurement(
                MeasurementStatus.NOT_APPLICABLE, None
            ),
        )
    )
    usage = accumulator.finalize()
    assert usage.input_tokens.status is MeasurementStatus.REPORTED
    assert usage.input_tokens.value == 0
    assert usage.total_tokens.status is MeasurementStatus.DERIVED
    assert usage.total_tokens.value == 3
    assert usage.cache_write_input_tokens.status is MeasurementStatus.NOT_APPLICABLE


def test_unknown_snapshot_field_does_not_overwrite_reported_value() -> None:
    accumulator = UsageAccumulator()
    accumulator.merge(ModelUsage(input_tokens=reported(7)))
    accumulator.merge(ModelUsage())
    assert accumulator.finalize().input_tokens.value == 7


def _observation_with_usage(
    usage: ModelUsage, monkeypatch: pytest.MonkeyPatch
) -> ModelObservation:
    """从真实计划入口读取用量投影；传参：用量与替换器；返回：本次模型观测。"""
    client = from_test_stub("ok")
    adapter = client._adapter_registry.require("scripted_test")
    stream = adapter.stream

    def with_usage(request: object, **kwargs: object):
        """在脚本响应开始后交付计量；传参：原请求；返回：含用量的流。"""
        from dataclasses import replace

        for item in stream(request, **kwargs):
            if item.sequence == 0:
                yield item
                yield ModelStreamEvent(
                    kind="usage_update",
                    sequence=1,
                    api_family=item.api_family,
                    provider=item.provider,
                    model=item.model,
                    usage=usage,
                )
            else:
                yield replace(item, sequence=item.sequence + 1)

    monkeypatch.setattr(adapter, "stream", with_usage)
    observation = client.plan("读取用量").observation
    assert observation is not None
    return observation


def test_usage_projection_keeps_unreported_cache_as_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """厂商不报缓存用量时，投影到观测字段必须留 None，不能折成 0。"""
    tokens = _observation_with_usage(
        ModelUsage(
            input_tokens=reported(10),
            output_tokens=reported(5),
            cache_read_input_tokens=UsageMeasurement(MeasurementStatus.UNKNOWN, None),
            cache_write_input_tokens=UsageMeasurement(
                MeasurementStatus.NOT_APPLICABLE, None
            ),
        ),
        monkeypatch,
    )
    assert tokens.cache_read_input_tokens is None, "UNKNOWN 不得被读成 0"
    assert tokens.cache_creation_input_tokens is None, "NOT_APPLICABLE 不得被读成 0"


def test_usage_projection_keeps_reported_zero_distinct_from_unreported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """厂商明确报 0 时投影为 0，与"没报"的 None 区分开。"""
    tokens = _observation_with_usage(
        ModelUsage(
            input_tokens=reported(10),
            output_tokens=reported(5),
            cache_read_input_tokens=reported(0),
            cache_write_input_tokens=UsageMeasurement(MeasurementStatus.UNKNOWN, None),
        ),
        monkeypatch,
    )
    assert tokens.cache_read_input_tokens == 0, "报了 0 就是 0"
    assert tokens.cache_creation_input_tokens is None, "没报就是 None"


def test_observation_cache_fields_default_to_unreported() -> None:
    """没传 cache 字段的观测记录默认为未上报。

    缺省值在生产上承重：`MissingConfigurationLLMClient` 造观测时压根没调过
    Provider（llm/client.py:249 不传这两个字段），缺省成 0 会让下游写出一条
    "缓存读 0、写 0"的假事实。
    """
    observation = ModelObservation(
        stage="plan",
        provider="missing_config",
        model="(missing)",
        started_at="2026-09-01T00:00:00+00:00",
        elapsed_ms=0,
        attempt_count=1,
        success=False,
    )
    assert observation.cache_read_input_tokens is None
    assert observation.cache_creation_input_tokens is None


def test_normalized_error_summary_redacts_authentication_secrets() -> None:
    error = TimeoutError(
        "Authorization: Bearer secret-token, x-api-key=anthropic-secret api_key=openai-secret"
    )
    normalized = normalize_provider_error(
        error,
        provider="fixture",
        model="fixture-model",
        api_family="openai_chat",
        stage="transport",
    )
    assert "secret-token" not in normalized.summary
    assert "anthropic-secret" not in normalized.summary
    assert "openai-secret" not in normalized.summary
    assert normalized.summary.count("<redacted>") == 3


def test_bad_request_reporting_context_length_is_context_overflow_not_invalid_request() -> (
    None
):
    """400 报上下文超限必须归成 context_overflow，否则 client 的裁剪重试永不触发。"""
    error = _status_error(
        400,
        "This model's maximum context length is 8192 tokens, however you requested 9000 tokens",
    )
    normalized = _normalize(error)
    assert normalized.category == "context_overflow"
    assert normalized.retryable is False


def test_bad_request_without_overflow_markers_stays_invalid_request() -> None:
    error = _status_error(400, "Unrecognized request argument supplied: top_kk")
    assert _normalize(error).category == "invalid_request"


def test_payment_required_with_quota_reset_hint_is_rate_limited() -> None:
    error = _status_error(402, "Quota exceeded, please try again in 30s")
    normalized = _normalize(error)
    assert normalized.category == "rate_limited"
    assert normalized.retryable is True


def test_payment_required_without_retry_hint_stays_billing() -> None:
    error = _status_error(402, "Your credit balance is too low")
    normalized = _normalize(error)
    assert normalized.category == "billing"
    assert normalized.retryable is False


def test_not_found_without_model_semantics_is_provider_error_not_model_not_found() -> (
    None
):
    error = _status_error(404, "Invalid URL (POST /v1/chat/completion)")
    assert _normalize(error).category == "provider_error"


def test_not_found_naming_the_model_stays_model_not_found() -> None:
    error = _status_error(404, "The model `gpt-nonexistent` does not exist")
    assert _normalize(error).category == "model_not_found"


def test_service_unavailable_is_overloaded_not_generic_server_error() -> None:
    normalized = _normalize(_status_error(503, "Service temporarily unavailable"))
    assert normalized.category == "overloaded"
    assert normalized.retryable is True


def test_rate_limit_retry_after_header_survives_without_exception_attribute() -> None:
    """openai SDK 的 RateLimitError 没有 retry_after 属性，秒数只能从响应头取。"""
    error = _status_error(429, "Rate limit reached", headers={"retry-after": "7"})
    assert not hasattr(error, "retry_after")
    normalized = _normalize(error)
    assert normalized.category == "rate_limited"
    assert normalized.retry_after_seconds == 7.0


def test_retry_after_attribute_still_wins_when_present() -> None:
    error = _status_error(429, "Rate limit reached", headers={"retry-after": "7"})
    error.retry_after = 2.5  # type: ignore[attr-defined]
    assert _normalize(error).retry_after_seconds == 2.5


def test_missing_retry_after_header_reports_no_backoff_hint() -> None:
    assert (
        _normalize(_status_error(429, "Rate limit reached")).retry_after_seconds is None
    )


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (httpx2.WriteTimeout("write timed out"), "timeout"),
        (httpx2.PoolTimeout("pool exhausted"), "timeout"),
        (httpx2.ReadTimeout("read timed out"), "timeout"),
        (
            httpx2.RemoteProtocolError(
                "server disconnected without sending a response"
            ),
            "transport_error",
        ),
        (httpx2.ReadError("connection dropped mid stream"), "transport_error"),
        (httpx2.WriteError("failed to send body"), "transport_error"),
        (httpx2.ProxyError("proxy refused"), "transport_error"),
        (httpx2.ConnectError("connection refused"), "transport_error"),
    ],
)
def test_every_transport_subclass_classifies_instead_of_escaping(
    error: BaseException, category: str
) -> None:
    """Adapter 的 _known_errors 会捕获整棵 httpx TransportError 树，每一支都必须能归类。"""
    normalized = _normalize(error)
    assert normalized.category == category
    assert normalized.retryable is True


def test_cancellation_is_not_retryable() -> None:
    normalized = _normalize(asyncio.CancelledError())
    assert normalized.category == "cancelled"
    assert normalized.retryable is False


def test_bare_sdk_api_error_becomes_provider_error_instead_of_escaping() -> None:
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    normalized = _normalize(openai.APIError("provider failed", request, body=None))
    assert normalized.category == "provider_error"
    assert normalized.retryable is False


def test_sdk_api_error_message_can_still_reach_context_overflow_without_status() -> (
    None
):
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    error = openai.APIError(
        "prompt is too long: 9000 tokens exceed the context window", request, body=None
    )
    assert _normalize(error).category == "context_overflow"


def test_non_boundary_programming_error_is_raised_unchanged() -> None:
    error = TypeError("unsupported operand type")
    with pytest.raises(TypeError) as caught:
        _normalize(error)
    assert caught.value is error


def test_auth_and_permission_stay_distinct_across_401_and_403() -> None:
    assert _normalize(_status_error(401, "Invalid API key provided")).category == "auth"
    assert (
        _normalize(_status_error(403, "You are not allowed to use this model")).category
        == "permission_denied"
    )


def test_payload_too_large_keeps_413_semantics() -> None:
    normalized = _normalize(_status_error(413, "Request entity too large"))
    assert normalized.category == "payload_too_large"
    assert normalized.retryable is False


def test_conflict_and_too_early_remain_retryable_transport() -> None:
    for status in (409, 425):
        normalized = _normalize(_status_error(status, "please retry"))
        assert normalized.category == "transport_error"
        assert normalized.retryable is True


def test_request_id_and_status_are_carried_onto_the_error() -> None:
    error = _status_error(500, "internal error", headers={"x-request-id": "req-42"})
    normalized = _normalize(error)
    assert (normalized.http_status, normalized.request_id) == (500, "req-42")
    assert normalized.category == "server_error"
