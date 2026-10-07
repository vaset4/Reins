from __future__ import annotations

import pytest
from openai.types.responses import (
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryPartDoneEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
)

from llm.messages import TextPart, ThinkingPart
from llm.providers.openai_responses import OpenAIResponsesAdapter
from llm.provider_stream import StreamAssembler
from tests.provider_test_support import tool_roundtrip_request


def test_responses_request_uses_function_call_items() -> None:
    body = OpenAIResponsesAdapter().build_request(
        tool_roundtrip_request(), model_id="gpt-fixture"
    )
    assert body["instructions"] == "You are Reins."
    assert body["input"][1]["type"] == "function_call"
    assert body["input"][2] == {
        "type": "function_call_output",
        "call_id": "c1",
        "output": "content",
    }


def test_responses_stream_maps_fragmented_function_arguments() -> None:
    raw = [
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {"id": "resp-1"},
        },
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": {
                "id": "fc-1",
                "type": "function_call",
                "call_id": "c1",
                "name": "read",
                "arguments": "",
                "status": "in_progress",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 2,
            "item_id": "fc-1",
            "output_index": 0,
            "delta": '{"path":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "sequence_number": 3,
            "item_id": "fc-1",
            "output_index": 0,
            "delta": '"a.txt"}',
        },
        {
            "type": "response.function_call_arguments.done",
            "sequence_number": 4,
            "item_id": "fc-1",
            "output_index": 0,
            "arguments": '{"path":"a.txt"}',
        },
        {
            "type": "response.completed",
            "sequence_number": 5,
            "response": {
                "id": "resp-1",
                "status": "completed",
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            },
        },
    ]
    events = tuple(
        OpenAIResponsesAdapter().translate_stream(
            raw, provider="openai", model="gpt-fixture"
        )
    )
    assert [item.kind for item in events].count("content_delta") == 2
    assert events[-1].kind == "response_done"


def test_responses_preserves_minimal_encrypted_reasoning_state() -> None:
    response = {
        "id": "resp-1",
        "created_at": 1,
        "model": "gpt-fixture",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "status": "completed",
    }
    reasoning = {
        "id": "rs-1",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "synthetic-encrypted-state",
        "status": "completed",
    }
    raw = [
        {"type": "response.created", "sequence_number": 0, "response": response},
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "output_index": 0,
            "item": reasoning,
        },
        {
            "type": "response.reasoning_text.delta",
            "sequence_number": 2,
            "item_id": "rs-1",
            "output_index": 0,
            "content_index": 0,
            "delta": "considering",
        },
        {
            "type": "response.reasoning_text.done",
            "sequence_number": 3,
            "item_id": "rs-1",
            "output_index": 0,
            "content_index": 0,
            "text": "considering",
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 4,
            "output_index": 0,
            "item": reasoning,
        },
        {"type": "response.completed", "sequence_number": 5, "response": response},
    ]
    events = tuple(
        OpenAIResponsesAdapter().translate_stream(
            raw, provider="openai", model="gpt-fixture"
        )
    )
    result = StreamAssembler().assemble(events)
    assert result.message is not None
    assert result.message.provider_state is not None
    input_items = result.message.provider_state.payload["input_items"]
    assert input_items[0]["encrypted_content"] == "synthetic-encrypted-state"


@pytest.mark.parametrize("summaries", [("",), ("检查收款", "核对退款")])
def test_summary_parts_keep_text_and_independent_blocks(
    summaries: tuple[str, ...],
) -> None:
    """标准摘要容器不应打断响应或混入重复文字；传参：空摘要或多段摘要；返回：无。"""
    raw = [
        {"type": "response.created", "response": {"id": "resp-summary"}},
        {
            "type": "response.reasoning_text.delta",
            "item_id": "rs-1",
            "content_index": 0,
            "delta": "完整推理",
        },
        {
            "type": "response.reasoning_text.done",
            "item_id": "rs-1",
            "content_index": 0,
            "text": "完整推理",
        },
    ]
    for index, text in enumerate(summaries):
        # 1. 【模型协议】【推理摘要】官方SDK校验新增、增量及完成事件，摘要序号独立于正文序号
        common = {"item_id": "rs-1", "output_index": 0, "summary_index": index}
        parts = [
            (
                ResponseReasoningSummaryPartAddedEvent,
                {
                    "type": "response.reasoning_summary_part.added",
                    "part": {"type": "summary_text", "text": ""},
                },
            ),
            (
                ResponseReasoningSummaryTextDeltaEvent,
                {"type": "response.reasoning_summary_text.delta", "delta": text},
            ),
            (
                ResponseReasoningSummaryTextDoneEvent,
                {"type": "response.reasoning_summary_text.done", "text": text},
            ),
            (
                ResponseReasoningSummaryPartDoneEvent,
                {
                    "type": "response.reasoning_summary_part.done",
                    "part": {"type": "summary_text", "text": text},
                },
            ),
        ]
        for event_type, payload in parts:
            if payload["type"] == "response.reasoning_summary_text.delta" and not text:
                continue
            event = event_type.model_validate(
                {**common, **payload, "sequence_number": len(raw)}
            )
            raw.append(event.model_dump(mode="json", exclude_none=True))
    raw.extend(
        [
            {
                "type": "response.output_text.delta",
                "item_id": "message-1",
                "content_index": 0,
                "delta": "结果已核对",
            },
            {
                "type": "response.output_text.done",
                "item_id": "message-1",
                "content_index": 0,
                "text": "结果已核对",
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp-summary",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            },
        ]
    )
    events = OpenAIResponsesAdapter().translate_stream(
        raw, provider="openai", model="gpt-fixture"
    )
    result = StreamAssembler().assemble(events)
    assert result.message is not None
    thinking = tuple(
        part.text
        for part in result.message.content
        if isinstance(part, ThinkingPart) and part.text
    )
    assert thinking == ("完整推理", *(text for text in summaries if text))
    assert tuple(
        part.text for part in result.message.content if isinstance(part, TextPart)
    ) == ("结果已核对",)
    assert result.usage.total_tokens.value == 15
