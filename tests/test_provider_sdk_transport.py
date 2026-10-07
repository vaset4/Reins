from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import httpx2
import pytest

from llm.model_registry import ModelDescriptor
from llm.model_request import Capability, PreferenceKind
from llm.provider_connection import (
    ConnectionProfile,
    HeaderPolicy,
    ResolvedConnection,
    resolve_connection,
)
from llm.provider_stream import StreamAssembler
from llm.provider_adapter import ProviderAdapterError
from llm.providers.anthropic_messages import AnthropicMessagesAdapter
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.providers.openai_responses import OpenAIResponsesAdapter
from tests.provider_test_support import text_request


FIXTURES = Path(__file__).parent / "fixtures" / "llm" / "providers"


def _error_fixture(family: str, name: str) -> dict[str, object]:
    return json.loads((FIXTURES / family / name).read_text(encoding="utf-8"))


class Credentials:
    def resolve(self, credential_ref: str) -> str:
        return "synthetic-sdk-token"


def _connection(family: str, base_url: str) -> ResolvedConnection:
    profile = ConnectionProfile(
        "sdk",
        base_url,
        10,
        "fixture/sdk",
        organization="org-fixture",
        project="project-fixture",
        tenant="tenant-fixture",
        feature_headers=("prompt-caching-2024-07-31",),
    )
    return resolve_connection(profile, Credentials(), HeaderPolicy(family))


def _model(family: str, provider: str, model_id: str) -> ModelDescriptor:
    return ModelDescriptor(
        "sdk-model",
        provider,
        model_id,
        family,
        {Capability.STREAMING: True},
        {PreferenceKind.LATENCY_PRIORITY: frozenset()},
        "sdk",
    )


def test_openai_chat_sdk_mock_transport_captures_final_wire() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.update(
            url=str(request.url),
            headers=dict(request.headers),
            body=json.loads(request.content),
        )
        sse = 'data: {"id":"chat-sdk","object":"chat.completion.chunk","created":1,"model":"gpt-fixture","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\ndata: {"id":"chat-sdk","object":"chat.completion.chunk","created":1,"model":"gpt-fixture","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse
        )

    connection = _connection("openai_chat", "https://sdk.openai.test/v1")
    adapter = OpenAIChatAdapter(
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    result = StreamAssembler().assemble(
        list(
            adapter.stream(
                text_request(),
                model=_model("openai_chat", "openai", "gpt-fixture"),
                connection=connection,
            )
        )
    )
    assert result.message is not None
    assert str(captured["url"]).endswith("/v1/chat/completions")
    headers = captured["headers"]
    assert headers["authorization"] == "Bearer synthetic-sdk-token"
    assert headers["openai-project"] == "project-fixture"
    assert headers["x-reins-tenant"] == "tenant-fixture"
    assert "reins/0.1.0" in headers["user-agent"].lower()
    assert captured["body"]["model"] == "gpt-fixture"


def test_openai_responses_sdk_http_error_becomes_typed_event() -> None:
    captured: dict[str, object] = {}
    fixture = _error_fixture("openai_responses", "error_auth.json")

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.update(headers=dict(request.headers), body=json.loads(request.content))
        return httpx2.Response(
            fixture["status"],
            headers={"x-request-id": fixture["request_id"]},
            json=fixture["body"],
        )

    connection = _connection("openai_responses", "https://sdk.openai.test/v1")
    adapter = OpenAIResponsesAdapter(
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    events = list(
        adapter.stream(
            text_request(),
            model=_model("openai_responses", "openai", "gpt-fixture"),
            connection=connection,
        )
    )
    result = StreamAssembler().assemble(events)
    assert result.error is not None and result.error.category == "auth"
    assert result.error.summary == "synthetic authentication failure"
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer synthetic-sdk-token"
    assert headers["openai-project"] == "project-fixture"
    assert headers["x-reins-tenant"] == "tenant-fixture"
    assert "reins/0.1.0" in headers["user-agent"].lower()
    assert captured["body"]["model"] == "gpt-fixture"


def test_anthropic_sdk_rate_limit_captures_version_feature_and_identity() -> None:
    captured: dict[str, object] = {}
    fixture = _error_fixture("anthropic_messages", "error_rate_limit.json")

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(
            url=str(request.url),
            headers=dict(request.headers),
            body=json.loads(request.content),
        )
        return httpx.Response(
            fixture["status"],
            headers={"request-id": fixture["request_id"]},
            json=fixture["body"],
        )

    connection = _connection("anthropic_messages", "https://sdk.anthropic.test")
    adapter = AnthropicMessagesAdapter(
        http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    events = list(
        adapter.stream(
            text_request(),
            model=_model("anthropic_messages", "anthropic", "claude-fixture"),
            connection=connection,
        )
    )
    result = StreamAssembler().assemble(events)
    assert result.error is not None and result.error.category == "rate_limited"
    headers = captured["headers"]
    assert headers["x-api-key"] == "synthetic-sdk-token"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "prompt-caching-2024-07-31" in headers["anthropic-beta"]
    assert headers["x-reins-tenant"] == "tenant-fixture"
    assert "reins/0.1.0" in headers["user-agent"].lower()


def test_openai_chat_auth_fixture_maps_through_sdk_boundary() -> None:
    fixture = _error_fixture("openai_chat", "error_auth.json")

    def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            fixture["status"],
            headers={"x-request-id": fixture["request_id"]},
            json=fixture["body"],
        )

    connection = _connection("openai_chat", "https://sdk.openai.test/v1")
    adapter = OpenAIChatAdapter(
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler))
    )
    result = StreamAssembler().assemble(
        list(
            adapter.stream(
                text_request(),
                model=_model("openai_chat", "openai", "gpt-fixture"),
                connection=connection,
            )
        )
    )
    assert result.error is not None and result.error.category == "auth"
    assert result.error.request_id == "req-chat-auth"


def test_chat_partial_timeout_is_preserved_and_programming_error_propagates() -> None:
    first = {
        "id": "partial",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-fixture",
        "choices": [
            {"index": 0, "delta": {"content": "partial"}, "finish_reason": None}
        ],
    }

    def interrupted(_body: object, _connection: object) -> Iterator[dict[str, object]]:
        yield first
        raise httpx2.ReadTimeout("synthetic timeout")

    connection = _connection("openai_chat", "https://sdk.openai.test/v1")
    adapter = OpenAIChatAdapter(stream_factory=interrupted)
    result = StreamAssembler().assemble(
        list(
            adapter.stream(
                text_request(),
                model=_model("openai_chat", "openai", "gpt-fixture"),
                connection=connection,
            )
        )
    )
    assert result.error is not None and result.error.category == "timeout"
    assert result.partial_message is not None
    broken = OpenAIChatAdapter(
        client_factory=lambda _connection: (_ for _ in ()).throw(
            RuntimeError("programming bug")
        )
    )
    with pytest.raises(RuntimeError, match="programming bug"):
        list(
            broken.stream(
                text_request(),
                model=_model("openai_chat", "openai", "gpt-fixture"),
                connection=connection,
            )
        )


@pytest.mark.parametrize(
    "family,adapter_type,http",
    [
        ("openai_chat", OpenAIChatAdapter, httpx2),
        ("openai_responses", OpenAIResponsesAdapter, httpx2),
        ("anthropic_messages", AnthropicMessagesAdapter, httpx),
    ],
)
def test_sdk_reports_html_maintenance_response_as_wrong_content_type(
    family, adapter_type, http
):
    """HTTP成功却返回维护页时说明真实响应类型；传参：三类SDK与协议；返回：无，不把维护页当空模型回答。"""

    def maintenance(_request):
        """复现远端维护页；传参：实际SDK请求；返回：HTTP200的HTML正文。"""
        return http.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html>Service maintenance</html>",
        )

    with http.Client(transport=http.MockTransport(maintenance)) as client:
        adapter = adapter_type(http_client=client)
        with pytest.raises(
            ProviderAdapterError,
            match="received text/html.*HTTP 200.*text/event-stream",
        ):
            list(
                adapter.stream(
                    text_request(),
                    model=_model(family, "fixture", "configured-model"),
                    connection=_connection(family, "https://fixture.invalid/v1"),
                )
            )
