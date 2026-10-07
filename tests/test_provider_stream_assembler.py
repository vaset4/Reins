from __future__ import annotations

import pytest

from llm.messages import TextPart, ToolCallPart
from llm.provider_result import ProviderError
from llm.provider_stream import ModelStreamEvent, StreamAssembler, StreamProtocolError


def event(kind: str, sequence: int, **values: object) -> ModelStreamEvent:
    return ModelStreamEvent(
        kind=kind,
        sequence=sequence,
        api_family="fixture",
        provider="fixture",
        model="fixture-model",
        **values,
    )


def test_assembler_preserves_text_and_fragmented_tool_arguments() -> None:
    events = [
        event("response_start", 0, message_id="a1"),
        event("content_start", 1, block_id="b1", content_kind="text"),
        event("content_delta", 2, block_id="b1", delta="hello"),
        event("content_end", 3, block_id="b1"),
        event(
            "content_start",
            4,
            block_id="b2",
            content_kind="tool_call",
            call_id="c1",
            tool_name="read",
        ),
        event("content_delta", 5, block_id="b2", delta='{"path":'),
        event("content_delta", 6, block_id="b2", delta='"a.txt"}'),
        event("content_end", 7, block_id="b2"),
        event("response_done", 8, stop_reason="tool_call"),
    ]
    result = StreamAssembler().assemble(events)
    assert result.error is None
    assert result.message is not None
    assert result.message.content == (
        TextPart("hello"),
        ToolCallPart("c1", "read", {"path": "a.txt"}),
    )


@pytest.mark.parametrize(
    "events, code",
    [
        (
            [event("content_delta", 0, block_id="b1", delta="bad")],
            "response_not_started",
        ),
        (
            [
                event("response_start", 0),
                event("content_start", 1, block_id="b1", content_kind="text"),
                event("content_delta", 2, block_id="b1", delta="ok"),
                event("content_end", 3, block_id="b1"),
                event("response_done", 4),
                event("response_done", 5),
            ],
            "event_after_terminal",
        ),
        ([event("response_start", 0)], "missing_terminal_event"),
    ],
)
def test_assembler_rejects_invalid_event_order(
    events: list[ModelStreamEvent], code: str
) -> None:
    with pytest.raises(StreamProtocolError, match=code):
        StreamAssembler().assemble(events)


@pytest.mark.parametrize(
    "delta, code",
    [
        ('{"path":', "invalid_tool_arguments_json"),
        ("[1, 2]", "tool_arguments_must_be_object"),
        ("null", "tool_arguments_must_be_object"),
        ('"path=a.txt"', "tool_arguments_must_be_object"),
    ],
)
def test_assembler_rejects_tool_arguments_that_are_not_json_object(
    delta: str, code: str
) -> None:
    """端点交来的工具参数不是 JSON object 时，在装配阶段就判协议错误，不放行到下游。

    作者：LKX
    时间：2026-08-31 22:40:00
    传参：delta 为工具调用块收到的参数文本；code 为期望的协议错误码
    返回：无

    这道闸是畸形工具参数唯一的生产入口：typed 栈把校验前移到 _finish_block，
    ToolCallPart 拿到的参数已经是校验过的 mapping，所以构造器那层的用例覆盖不到
    真实入站路径。闸门失守时模型会拿着残缺参数去执行工具，故单独钉死。
    """
    events = [
        event("response_start", 0, message_id="a1"),
        event(
            "content_start",
            1,
            block_id="b1",
            content_kind="tool_call",
            call_id="c1",
            tool_name="read",
        ),
        event("content_delta", 2, block_id="b1", delta=delta),
        event("content_end", 3, block_id="b1"),
        event("response_done", 4, stop_reason="tool_call"),
    ]

    with pytest.raises(StreamProtocolError, match=code):
        StreamAssembler().assemble(events)


def test_response_error_preserves_confirmed_open_text_as_partial_message() -> None:
    error = ProviderError(
        "timeout",
        "transport",
        True,
        "request timed out",
        "fixture",
        "fixture-model",
        "fixture",
    )
    result = StreamAssembler().assemble(
        [
            event("response_start", 0, message_id="a1"),
            event("content_start", 1, block_id="b1", content_kind="text"),
            event("content_delta", 2, block_id="b1", delta="confirmed partial"),
            event("response_error", 3, error=error),
        ]
    )
    assert result.message is None
    assert result.error == error
    assert result.partial_message is not None
    assert result.partial_message.content == (TextPart("confirmed partial"),)


def test_finish_without_terminal_event_refuses_to_synthesize_success() -> None:
    """逐片消费路径下流中途断掉，仍必须报 missing_terminal_event。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：无
    返回：无

    生产已改成 accept() 逐事件喂 + finish() 收口，只测 assemble() 就覆盖不到这条路。
    半截的流绝不能被当成成功的一轮交上去。
    """
    assembler = StreamAssembler()
    for item in (
        event("response_start", 0, message_id="a1"),
        event("content_start", 1, block_id="b1", content_kind="text"),
        event("content_delta", 2, block_id="b1", delta="半截答案"),
    ):
        assembler.accept(item)

    with pytest.raises(StreamProtocolError, match="missing_terminal_event"):
        assembler.finish()


def test_assembler_reports_empty_message_when_every_block_is_blank() -> None:
    """整条消息只剩空白块时报协议错误，而不是把空白塞进消息合同里炸掉。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：无
    返回：无

    丢弃空白块是无损规范化，但"一条内容都没有"仍是端点异常，必须按原协议错误上报。
    """
    events = [
        event("response_start", 0, message_id="a1"),
        event("content_start", 1, block_id="b1", content_kind="text"),
        event("content_delta", 2, block_id="b1", delta=" "),
        event("content_end", 3, block_id="b1"),
        event("response_done", 4, stop_reason="end_turn"),
    ]
    with pytest.raises(StreamProtocolError, match="empty_assistant_message"):
        StreamAssembler().assemble(events)


def test_response_error_drops_blank_open_block_from_partial_message() -> None:
    """未闭合块只剩空白时不进 partial message，错误路径同样不碰合同校验。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：无
    返回：无
    """
    error = ProviderError(
        "timeout",
        "transport",
        True,
        "request timed out",
        "fixture",
        "fixture-model",
        "fixture",
    )
    result = StreamAssembler().assemble(
        [
            event("response_start", 0, message_id="a1"),
            event("content_start", 1, block_id="b1", content_kind="text"),
            event("content_delta", 2, block_id="b1", delta="\n  "),
            event("response_error", 3, error=error),
        ]
    )
    assert result.error == error
    assert result.partial_message is None
