"""Tests for explicit model profile diagnostics."""

from __future__ import annotations

from collections.abc import Iterator

from llm.doctor import run_model_doctor
from llm.model_registry import ModelDescriptor
from llm.model_request import ModelRequest
from llm.provider_adapter import AdapterRegistry
from llm.provider_connection import ResolvedConnection
from llm.provider_result import ProviderError
from llm.provider_stream import ModelStreamEvent
from llm.resolved_target import ResolvedModelTarget


# 生产 registry 由 target.api_mode=chat_completions 推出这个 family，桩必须同名才过 validate
_PROBE_FAMILY = "openai_chat"
_PROBE_MESSAGE_ID = "doctor-probe-message"


class _RecordingAdapter:
    """记录 doctor 实际发出的请求，并按脚本回一段合规事件流。

    只替换"事件从哪来"，ModelSelector、validate_adapter_target、StreamAssembler
    都仍是生产实现，所以断言核对的是真实装配链。
    """

    api_family = _PROBE_FAMILY

    def __init__(self, *, error: ProviderError | None = None) -> None:
        self.calls: list[tuple[ModelRequest, ModelDescriptor, ResolvedConnection]] = []
        self._error = error

    def stream(
        self,
        request: ModelRequest,
        *,
        model: ModelDescriptor,
        connection: ResolvedConnection,
        cancellation: object | None = None,
        prepared_body: object | None = None,
    ) -> Iterator[ModelStreamEvent]:
        """记录诊断请求并返回脚本响应；参数：请求、连接及发送快照；返回：实际响应事件。"""
        self.calls.append((request, model, connection))
        return iter(_events(model, self._error))


def _events(
    model: ModelDescriptor, error: ProviderError | None
) -> list[ModelStreamEvent]:
    started = _event("response_start", model, 0)
    if error is not None:
        return [started, _event("response_error", model, 1, error=error)]
    return [
        started,
        _event("content_start", model, 1, block_id="b1", content_kind="text"),
        _event("content_delta", model, 2, block_id="b1", delta="OK"),
        _event("content_end", model, 3, block_id="b1"),
        _event("response_done", model, 4, stop_reason="end_turn"),
    ]


def _event(
    kind: str,
    model: ModelDescriptor,
    sequence: int,
    *,
    block_id: str = "",
    content_kind: str | None = None,
    delta: str = "",
    stop_reason: str | None = None,
    error: ProviderError | None = None,
) -> ModelStreamEvent:
    return ModelStreamEvent(
        kind=kind,
        sequence=sequence,
        api_family=_PROBE_FAMILY,
        provider=model.provider,
        model=model.model_id,
        message_id=_PROBE_MESSAGE_ID,
        block_id=block_id,
        content_kind=content_kind,
        delta=delta,
        stop_reason=stop_reason,
        error=error,
    )


def _registry(adapter: _RecordingAdapter) -> AdapterRegistry:
    return AdapterRegistry([adapter])


def _provider_error(
    category: str, summary: str, *, http_status: int | None = None
) -> ProviderError:
    return ProviderError(
        category=category,
        stage="transport",
        retryable=False,
        summary=summary,
        provider="openai_compatible",
        model="glm-5.1",
        api_family=_PROBE_FAMILY,
        http_status=http_status,
    )


def _target(**overrides: object) -> ResolvedModelTarget:
    values = {
        "provider": "openai_compatible",
        "model": "glm-5.1",
        "base_url": "https://provider.example/v1",
        "api_mode": "chat_completions",
        "timeout_seconds": 30,
        "config_source": "profile",
        "credential_source": "secrets_vault",
        "api_key_present": True,
        "profile_name": "glm-main",
        "credential_name": "glm_gateway_key",
        "api_key": "secret-value",
    }
    values.update(overrides)
    return ResolvedModelTarget(**values)


def test_doctor_reports_missing_target():
    report = run_model_doctor(None)

    assert report.ok is False
    assert report.error_category == "missing_config"
    assert "not available" in report.message


def test_doctor_fails_before_network_when_remote_key_missing():
    adapter = _RecordingAdapter()

    report = run_model_doctor(
        _target(api_key_present=False, api_key=None, credential_source="none"),
        adapter_registry=_registry(adapter),
    )

    assert report.ok is False
    assert adapter.calls == []
    assert report.error_category == "missing_config"
    assert report.credential_present is False


def test_doctor_probes_through_typed_stack():
    adapter = _RecordingAdapter()

    report = run_model_doctor(_target(), adapter_registry=_registry(adapter))

    assert report.ok is True
    assert report.error_category == ""
    assert report.profile_name == "glm-main"
    assert report.credential_name == "glm_gateway_key"
    _, model, connection = adapter.calls[0]
    # 模型名与凭据都由 doctor 从 target 现推出来，注入的只有 adapter
    assert model.model_id == "glm-5.1"
    assert connection.credential == "secret-value"
    assert connection.headers["Authorization"] == "Bearer secret-value"


def test_doctor_sends_typed_probe_request_without_tools():
    adapter = _RecordingAdapter()

    run_model_doctor(_target(), adapter_registry=_registry(adapter))

    request = adapter.calls[0][0]
    # 旧 dict 路径交的是 {"messages": [...]}，这里必须是 canonical ModelRequest
    assert isinstance(request, ModelRequest)
    assert request.tools == ()
    assert request.stream is True
    message = request.messages[0]
    assert message.kind == "user"
    assert message.content[0].text == "Reply with OK."


def test_doctor_reports_provider_error_category():
    adapter = _RecordingAdapter(error=_provider_error("auth", "unauthorized"))

    report = run_model_doctor(_target(), adapter_registry=_registry(adapter))

    assert report.ok is False
    assert report.error_category == "auth"
    assert "unauthorized" in report.message


def test_doctor_reports_http_status_code():
    adapter = _RecordingAdapter(
        error=_provider_error("auth", "error code: 1010", http_status=403)
    )

    report = run_model_doctor(_target(), adapter_registry=_registry(adapter))

    assert report.ok is False
    assert report.error_category == "auth"
    assert report.status_code == 403
