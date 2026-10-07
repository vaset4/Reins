from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from llm.providers.anthropic_messages import AnthropicMessagesAdapter
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.providers.openai_responses import OpenAIResponsesAdapter
from llm.provider_stream import StreamAssembler
from tests.provider_test_support import text_request, tool_roundtrip_request


ROOT = Path(__file__).parent / "fixtures" / "llm" / "providers"


@pytest.mark.parametrize(
    "family,adapter,text_model,tool_model",
    [
        ("openai_chat", OpenAIChatAdapter(), "gpt-fixture", "gpt-fixture"),
        ("openai_responses", OpenAIResponsesAdapter(), "gpt-fixture", "gpt-fixture"),
        (
            "anthropic_messages",
            AnthropicMessagesAdapter(),
            "claude-fixture",
            "claude-fixture",
        ),
    ],
)
def test_request_golden_matches_full_adapter_semantics(
    family: str,
    adapter: object,
    text_model: str,
    tool_model: str,
) -> None:
    text_expected = json.loads(
        (ROOT / family / "request_text.json").read_text(encoding="utf-8")
    )
    tool_expected = json.loads(
        (ROOT / family / "request_tool_roundtrip.json").read_text(encoding="utf-8")
    )
    assert adapter.build_request(text_request(), model_id=text_model) == text_expected  # type: ignore[attr-defined]
    assert (
        adapter.build_request(tool_roundtrip_request(), model_id=tool_model)
        == tool_expected
    )  # type: ignore[attr-defined]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _assert_expected_event(event: object, expected: dict[str, Any]) -> None:
    assert getattr(event, "kind") == expected["kind"]
    assert getattr(event, "sequence") == expected["sequence"]
    for field in (
        "message_id",
        "block_id",
        "content_kind",
        "delta",
        "call_id",
        "tool_name",
    ):
        if field in expected:
            assert getattr(event, field) == expected[field]
    if "stop_reason" in expected:
        stop_reason = getattr(event, "stop_reason")
        assert getattr(stop_reason, "value", stop_reason) == expected["stop_reason"]
    usage_fields = {
        name: expected[name]
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_read_input_tokens",
            "cache_write_input_tokens",
        )
        if name in expected
    }
    if usage_fields:
        usage = getattr(event, "usage")
        assert usage is not None
        for name, value in usage_fields.items():
            assert getattr(getattr(usage, name), "value") == value
    if "error_category" in expected:
        assert getattr(event, "error") is not None
        assert getattr(event, "error").category == expected["error_category"]


@pytest.mark.parametrize(
    "family, sdk_type, adapter, stream_file, expected_file",
    [
        (
            "openai_chat",
            "chat",
            OpenAIChatAdapter(),
            "stream_text.jsonl",
            "expected_stream_text.jsonl",
        ),
        (
            "openai_chat",
            "chat",
            OpenAIChatAdapter(),
            "stream_tool_fragmented.jsonl",
            "expected_stream_tool_fragmented.jsonl",
        ),
        (
            "openai_responses",
            "responses",
            OpenAIResponsesAdapter(),
            "stream_text.jsonl",
            "expected_stream_text.jsonl",
        ),
        (
            "openai_responses",
            "responses",
            OpenAIResponsesAdapter(),
            "stream_tool_fragmented.jsonl",
            "expected_stream_tool_fragmented.jsonl",
        ),
        (
            "openai_responses",
            "responses",
            OpenAIResponsesAdapter(),
            "stream_reasoning_state.jsonl",
            "expected_stream_reasoning_state.jsonl",
        ),
        (
            "openai_responses",
            "responses",
            OpenAIResponsesAdapter(),
            "stream_failed.jsonl",
            "expected_stream_failed.jsonl",
        ),
        (
            "openai_responses",
            "responses",
            OpenAIResponsesAdapter(),
            "stream_incomplete.jsonl",
            "expected_stream_incomplete.jsonl",
        ),
        (
            "anthropic_messages",
            "anthropic",
            AnthropicMessagesAdapter(),
            "stream_text.jsonl",
            "expected_stream_text.jsonl",
        ),
        (
            "anthropic_messages",
            "anthropic",
            AnthropicMessagesAdapter(),
            "stream_thinking_signature.jsonl",
            "expected_stream_thinking_signature.jsonl",
        ),
        (
            "anthropic_messages",
            "anthropic",
            AnthropicMessagesAdapter(),
            "stream_thinking.jsonl",
            "expected_stream_thinking.jsonl",
        ),
        (
            "anthropic_messages",
            "anthropic",
            AnthropicMessagesAdapter(),
            "stream_cache_usage.jsonl",
            "expected_stream_cache_usage.jsonl",
        ),
    ],
)
def test_raw_fixture_provenance_and_public_sdk_type_validation(
    family: str,
    sdk_type: str,
    adapter: object,
    stream_file: str,
    expected_file: str,
) -> None:
    manifest = json.loads((ROOT / family / "manifest.json").read_text(encoding="utf-8"))
    expected_package = "anthropic" if sdk_type == "anthropic" else "openai"
    assert manifest["sdk_package"] == expected_package
    assert manifest["sdk_version"] == version(expected_package)
    assert manifest["source_tag"] in manifest["source_url"]
    assert manifest["captured_at"] == "2026-08-19"
    assert manifest["fixture_kind"] == "synthetic_provider_protocol_golden"
    assert manifest["contains_secrets"] is False
    raw = _jsonl(ROOT / family / stream_file)
    raw_event_types = {str(event.get("type", event.get("object", ""))) for event in raw}
    assert raw_event_types <= set(manifest["raw_event_types"])
    if sdk_type == "chat":
        from openai.types.chat import ChatCompletionChunk

        validator = TypeAdapter(ChatCompletionChunk)
    elif sdk_type == "responses":
        from openai.types.responses import ResponseStreamEvent

        validator = TypeAdapter(ResponseStreamEvent)
    else:
        from anthropic.types.raw_message_stream_event import RawMessageStreamEvent

        validator = TypeAdapter(RawMessageStreamEvent)
    for event in raw:
        validator.validate_python(event)
    events = tuple(
        adapter.translate_stream(raw, provider="fixture", model="fixture-model")
    )  # type: ignore[attr-defined]
    expected = _jsonl(ROOT / family / expected_file)
    assert len(events) == len(expected)
    for event, expected_event in zip(events, expected):
        _assert_expected_event(event, expected_event)
    assert events[-1].kind in {"response_done", "response_error"}
    StreamAssembler().assemble(list(events))


@pytest.mark.parametrize(
    "family,sdk_type,adapter,stream_file,error_code",
    [
        (
            "openai_chat",
            "chat",
            OpenAIChatAdapter(),
            "stream_interrupted_without_usage.jsonl",
            "incomplete_chat_stream",
        ),
        (
            "anthropic_messages",
            "anthropic",
            AnthropicMessagesAdapter(),
            "stream_missing_stop.jsonl",
            "missing_message_stop",
        ),
    ],
)
def test_incomplete_raw_fixture_fails_closed_after_sdk_validation(
    family: str,
    sdk_type: str,
    adapter: object,
    stream_file: str,
    error_code: str,
) -> None:
    raw = _jsonl(ROOT / family / stream_file)
    if sdk_type == "chat":
        from openai.types.chat import ChatCompletionChunk

        validator = TypeAdapter(ChatCompletionChunk)
    else:
        from anthropic.types.raw_message_stream_event import RawMessageStreamEvent

        validator = TypeAdapter(RawMessageStreamEvent)
    for event in raw:
        validator.validate_python(event)
    with pytest.raises(Exception, match=error_code):
        tuple(adapter.translate_stream(raw, provider="fixture", model="fixture-model"))  # type: ignore[attr-defined]
