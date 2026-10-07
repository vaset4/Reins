from __future__ import annotations

from llm.messages import ThinkingPart, UserMessage, TextPart
from llm.model_request import Capability, CapabilityRequirement, ModelRequest
from llm.provider_stream import StreamAssembler
from llm.providers.anthropic_messages import AnthropicMessagesAdapter
from tests.provider_test_support import tool_roundtrip_request


def test_anthropic_request_maps_tool_result_to_user_content_block() -> None:
    body = AnthropicMessagesAdapter().build_request(
        tool_roundtrip_request(), model_id="claude-fixture"
    )
    assert body["system"] == [{"type": "text", "text": "You are Reins."}]
    assert body["messages"][1]["content"][0]["type"] == "tool_use"
    assert body["messages"][2]["role"] == "user"
    assert body["messages"][2]["content"][0]["tool_use_id"] == "c1"


def test_anthropic_stream_maps_input_json_delta_and_cache_usage() -> None:
    raw = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-fixture",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "c1",
                "name": "read",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"path":"a.txt"}'},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {
                "output_tokens": 1,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 2,
            },
        },
        {"type": "message_stop"},
    ]
    events = tuple(
        AnthropicMessagesAdapter().translate_stream(
            raw, provider="anthropic", model="claude-fixture"
        )
    )
    assert events[-2].kind == "usage_update"
    assert events[-2].usage is not None
    assert events[-2].usage.cache_write_input_tokens.value == 2
    assert events[-1].kind == "response_done"


def test_anthropic_thinking_signature_round_trips_through_provider_state() -> None:
    raw = [
        {
            "type": "message_start",
            "message": {
                "id": "msg-thinking",
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "considering"},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "synthetic-signature"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    adapter = AnthropicMessagesAdapter()
    result = StreamAssembler().assemble(
        list(
            adapter.translate_stream(raw, provider="anthropic", model="claude-fixture")
        )
    )
    assert result.message is not None
    assert result.message.content == (ThinkingPart("considering", "visible"),)
    assert result.message.provider_state is not None
    assert (
        result.message.provider_state.payload["signatures"]["considering"]
        == "synthetic-signature"
    )

    request = ModelRequest(
        instructions=(),
        messages=(UserMessage("m1", (TextPart("continue"),)), result.message),
        tools=(),
        required_capabilities=frozenset(
            {
                CapabilityRequirement(Capability.REASONING),
                CapabilityRequirement(Capability.PROVIDER_STATE_ROUND_TRIP),
            }
        ),
    )
    body = adapter.build_request(request, model_id="claude-fixture")
    assert body["messages"][1]["content"][0] == {
        "type": "thinking",
        "thinking": "considering",
        "signature": "synthetic-signature",
    }
