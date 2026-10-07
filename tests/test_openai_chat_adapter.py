from __future__ import annotations

import pytest

from llm.messages import TextPart, ThinkingPart, ToolCallPart
from llm.provider_stream import StreamAssembler, StreamInterruptedError
from llm.providers.openai_chat import OpenAIChatAdapter
from tests.provider_test_support import text_request, tool_roundtrip_request


def test_chat_request_maps_canonical_text_and_tool_roundtrip() -> None:
    adapter = OpenAIChatAdapter()
    body = adapter.build_request(tool_roundtrip_request(), model_id="gpt-fixture")
    assert body["model"] == "gpt-fixture"
    assert body["messages"][0] == {"role": "system", "content": "You are Reins."}
    assert (
        body["messages"][2]["tool_calls"][0]["function"]["arguments"]
        == '{"path":"a.txt"}'
    )
    assert body["messages"][3]["tool_call_id"] == "c1"


def test_chat_stream_maps_text_and_usage_without_fabricating_cache_write() -> None:
    adapter = OpenAIChatAdapter()
    raw = [
        {
            "id": "chat-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-fixture",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hi"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chat-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-fixture",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 1,
                "total_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        },
    ]
    events = tuple(
        adapter.translate_stream(raw, provider="openai", model="gpt-fixture")
    )
    assert [item.kind for item in events] == [
        "response_start",
        "content_start",
        "content_delta",
        "usage_update",
        "content_end",
        "response_done",
    ]
    assert events[3].usage is not None
    assert events[3].usage.input_tokens.value == 0


def test_chat_stream_without_finish_reason_is_not_reported_as_success() -> None:
    raw = [
        {
            "id": "chat-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-fixture",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "partial"},
                    "finish_reason": None,
                }
            ],
        }
    ]
    with pytest.raises(StreamInterruptedError, match="incomplete_chat_stream"):
        tuple(
            OpenAIChatAdapter().translate_stream(
                raw, provider="openai", model="gpt-fixture"
            )
        )
    assert text_request().stream is True


def _chunk(
    delta: dict[str, object], finish_reason: str | None = None
) -> dict[str, object]:
    """构造一个 Chat Completions SSE chunk 的 dict 形态。

    作者：xxx
    时间：2026-08-30 00:00:00
    传参：delta 为 choices[0].delta 内容；finish_reason 为该 chunk 的收尾原因
    返回：可直接喂给 translate_stream 的 chunk 映射
    """
    return {
        "id": "chat-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-fixture",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _assemble(chunks: list[dict[str, object]]) -> tuple[object, ...]:
    """把 chunk 流经 Adapter 翻译后交给 StreamAssembler 组装。

    作者：xxx
    时间：2026-08-30 00:00:00
    传参：chunks 为按到达顺序排列的 SSE chunk 列表
    返回：组装出的 AssistantMessage.content 内容块元组
    """
    events = tuple(
        OpenAIChatAdapter().translate_stream(
            chunks, provider="openai", model="gpt-fixture"
        )
    )
    result = StreamAssembler().assemble(events)
    assert result.message is not None
    return result.message.content


def test_chat_stream_translates_reasoning_content_into_thinking_part() -> None:
    """只带 reasoning_content 的 chunk 流必须译出可见的思考内容块。

    作者：xxx
    时间：2026-08-30 00:00:00
    传参：无
    返回：无
    """
    content = _assemble(
        [
            _chunk({"role": "assistant", "reasoning_content": "先看清用户要什么"}),
            _chunk({}, "stop"),
        ]
    )
    assert content == (ThinkingPart("先看清用户要什么", "visible"),)


def test_chat_stream_keeps_reasoning_and_text_as_separate_ordered_parts() -> None:
    """同流内先推理后回答时两个块都在，且顺序与到达顺序一致。

    作者：xxx
    时间：2026-08-30 00:00:00
    传参：无
    返回：无
    """
    content = _assemble(
        [
            _chunk({"role": "assistant", "reasoning_content": "先推理"}),
            _chunk({"content": "再回答"}),
            _chunk({}, "stop"),
        ]
    )
    assert content == (ThinkingPart("先推理", "visible"), TextPart("再回答"))


def test_chat_stream_merges_reasoning_fragments_into_one_thinking_part() -> None:
    """跨 chunk 分片到达的推理正文必须拼成一个块而非多个碎片。

    作者：xxx
    时间：2026-08-30 00:00:00
    传参：无
    返回：无
    """
    content = _assemble(
        [
            _chunk({"role": "assistant", "reasoning_content": "第一段"}),
            _chunk({"reasoning_content": "第二段"}),
            _chunk({"reasoning_content": "第三段", "content": "答案"}),
            _chunk({}, "stop"),
        ]
    )
    assert content == (ThinkingPart("第一段第二段第三段", "visible"), TextPart("答案"))


def test_chat_stream_translates_reasoning_alias_field() -> None:
    """端点改用 reasoning 别名时推理正文同样要译出，不得静默丢失。

    作者：LKX
    时间：2026-08-30 20:10:00
    传参：无
    返回：无

    本仓 179 份运行证据里 reasoning_content 独占 18 份、reasoning 独占 14 份，同一端点会在
    两个字段名之间切换，只认其中一个会在另一半响应上丢掉推理正文。
    """
    content = _assemble(
        [
            _chunk({"role": "assistant", "reasoning": "别名字段里的推理"}),
            _chunk({"content": "答案"}),
            _chunk({}, "stop"),
        ]
    )
    assert content == (ThinkingPart("别名字段里的推理", "visible"), TextPart("答案"))


def test_chat_stream_without_reasoning_content_emits_no_thinking_part() -> None:
    """缺失或空串的 reasoning_content 不得造出空的思考块。

    作者：xxx
    时间：2026-08-30 00:00:00
    传参：无
    返回：无
    """
    content = _assemble(
        [
            _chunk(
                {"role": "assistant", "reasoning_content": "", "content": "只有回答"}
            ),
            _chunk({"content": "继续"}),
            _chunk({}, "stop"),
        ]
    )
    assert content == (TextPart("只有回答继续"),)
    assert not any(isinstance(part, ThinkingPart) for part in content)


def test_chat_stream_drops_whitespace_only_text_block_and_keeps_the_rest() -> None:
    """正文分片只有一个空格时丢弃该块，思考链与工具调用照常产出，整轮不再崩。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：无
    返回：无

    这是 2026-09-03 线上故障的复现：推理模型边想边发工具调用，正文那一路只吐了个空格，
    组装出的 TextPart("") 被消息合同拒收，整轮 run 直接失败。空格没有语义，丢掉即可。
    """
    content = _assemble(
        [
            _chunk({"role": "assistant", "reasoning_content": "先想清楚要读哪个文件"}),
            _chunk({"content": " "}),
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c1",
                            "function": {
                                "name": "read",
                                "arguments": '{"path":"a.txt"}',
                            },
                        }
                    ]
                }
            ),
            _chunk({}, "tool_calls"),
        ]
    )
    assert content == (
        ThinkingPart("先想清楚要读哪个文件", "visible"),
        ToolCallPart("c1", "read", {"path": "a.txt"}),
    )
    assert not any(isinstance(part, TextPart) for part in content)


def test_chat_stream_drops_whitespace_only_reasoning_block() -> None:
    """推理分片全是空白时不产出思考块，回答正文不受影响。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：无
    返回：无

    与空串不同，`"\\n "` 这种非空白空串能通过适配器的开块判断，会一路走到 ThinkingPart
    的非空白校验前才炸，所以丢弃判断必须放在组装层收尾处。
    """
    content = _assemble(
        [
            _chunk({"role": "assistant", "reasoning_content": "\n "}),
            _chunk({"content": "答案"}),
            _chunk({}, "stop"),
        ]
    )
    assert content == (TextPart("答案"),)
    assert not any(isinstance(part, ThinkingPart) for part in content)
