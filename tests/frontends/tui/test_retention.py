"""长连接只保留阅读窗口，已保存的旧工具正文仍可从真实历史读回。

作者：xxx
时间：2026-09-29 20:00:00
"""

from dataclasses import asdict

from app.background.sessions import session_history_page
from frontends.tui.projection import ConversationProjection
from llm.messages import (
    AssistantMessage,
    StopReason,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
)
from runtime.session_message_store import SessionMessageStore
from runtime.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    ToolExecutionCompleted,
)
from runtime.stream_events import AssistantStreamClosed, ToolExecutionStarted

ROUND_COUNT = 90
RESULT_CHARACTERS = 40000
GIANT_CHARACTERS = 1000000
CACHE_CARDS = 12


def deliver(projection, event):
    """交付有序真实事件形状；参数：投影和事件；返回：无。"""
    projection.apply(
        {
            "session_id": "s",
            "events": [
                {
                    "sequence": projection.cursor + 1,
                    "run_id": "run",
                    "type": type(event).__name__,
                    "data": asdict(event),
                }
            ],
        }
    )


def append_round(store, index):
    """保存一轮实际消息和完整工具结果；参数：存储和轮次；返回：对应显示事件。"""
    text = f"第{index}轮回答"
    message_id, call_id = f"message-{index}", f"call-{index}"
    entry = store.append_message(
        "s",
        AssistantMessage(
            message_id,
            (TextPart(text), ToolCallPart(call_id, "read", {"path": f"file-{index}"})),
            stop_reason=StopReason.TOOL_CALL,
        ),
        run_id="run",
    )
    output = f"结果{index}:" + "原" * (
        GIANT_CHARACTERS if index == 0 else RESULT_CHARACTERS
    )
    store.append_message(
        "s",
        ToolResultMessage(
            f"result-{index}", call_id, "read", (TextPart(output),), "success"
        ),
        run_id="run",
    )
    return (
        AssistantTurnComplete(text, message_id=message_id, entry_id=entry.entry_id),
        ToolExecutionCompleted("read", output, call_id),
    )


def test_persisted_bodies_are_released_and_history_recovers_full_results(tmp_path):
    """多轮持久历史增长时缓存稳定，旧大结果完整可回查；参数：隔离根；返回：无。"""
    store = SessionMessageStore(tmp_path)
    projection = ConversationProjection(retained_cards=CACHE_CARDS)
    projection.apply({"session_id": "s", "history": []})
    unbounded = ConversationProjection()
    unbounded.apply({"session_id": "s", "history": []})
    first_message = None
    with store.database.transaction():
        for index in range(ROUND_COUNT):
            completion, tool = append_round(store, index)
            if first_message is None:
                first_message = completion
            deliver(projection, completion)
            deliver(projection, tool)
            deliver(unbounded, completion)
            deliver(unbounded, tool)
            assert len(projection.cards) <= CACHE_CARDS
    assert not any("结果0:" in card.text for card in projection.cards.values())
    retained = sum(
        len(card.text) + len(card.reasoning) + len(card.args)
        for card in projection.cards.values()
    )
    assert retained < CACHE_CARDS * RESULT_CHARACTERS
    projection.apply(
        {
            **session_history_page(tmp_path, "s", limit=CACHE_CARDS),
            "event_epoch": "reconnected",
        }
    )

    # 【TUI】【长连接缓存】1. 已淘汰消息收到迟到增量或重复收尾时不能重新出现
    for event in (
        AssistantTextDelta("迟到片段", message_id=first_message.message_id),
        first_message,
    ):
        projection.apply(
            {
                "session_id": "s",
                "event_epoch": "reconnected",
                "events": [
                    {
                        "sequence": projection.cursor + 1,
                        "run_id": "run",
                        "type": type(event).__name__,
                        "data": asdict(event),
                    }
                ],
            }
        )
    assert first_message.entry_id not in projection.cards
    before, restored, pages = None, None, 0
    while True:
        page = session_history_page(tmp_path, "s", before=before, limit=CACHE_CARDS)
        pages += 1
        restored = next(
            (
                row["text"]
                for row in page["history"]
                if row.get("tool_call_id") == "call-0"
            ),
            restored,
        )
        before = page["next_before"]
        if before is None:
            break
    assert restored == "结果0:" + "原" * GIANT_CHARACTERS
    assert pages > 1
    reloaded = ConversationProjection(retained_cards=CACHE_CARDS)
    reloaded.apply(session_history_page(tmp_path, "s", limit=ROUND_COUNT * 2))
    assert len(reloaded.cards) == CACHE_CARDS
    assert "tool:s:run:call-0" not in reloaded.cards
    baseline = sum(
        len(card.text) + len(card.reasoning) + len(card.args)
        for card in unbounded.cards.values()
    )
    assert len(unbounded.cards) == ROUND_COUNT * 2
    print(
        f"retention: rounds={ROUND_COUNT}, cards={len(projection.cards)}, "
        f"retained_chars={retained}, unbounded_chars={baseline}, history_pages={pages}"
    )


def test_reading_window_and_uncommitted_stream_survive_cache_rotation():
    """正在阅读和未提交输出不因后续保存而消失；参数：无；返回：无。"""
    projection = ConversationProjection(retained_cards=2)
    projection.apply({"session_id": "s", "history": []})
    deliver(projection, AssistantTextDelta("尚未提交", message_id="active"))
    deliver(
        projection,
        AssistantTurnComplete("最早回答", message_id="first", entry_id="saved-first"),
    )
    projection.retain_window(["saved-first", "stream:s:active"])
    for index in range(12):
        deliver(
            projection,
            AssistantTurnComplete(
                str(index), message_id=f"m-{index}", entry_id=f"e-{index}"
            ),
        )
    assert projection.cards["saved-first"].text == "最早回答"
    assert projection.cards["stream:s:active"].text == "尚未提交"
    deliver(projection, AssistantTextDelta("后半段", message_id="active"))
    deliver(
        projection,
        AssistantTurnComplete(
            "尚未提交后半段", message_id="active", entry_id="saved-active"
        ),
    )
    assert "stream:s:active" not in projection.cards
    assert projection.cards["saved-active"].text == "尚未提交后半段"
    projection.retain_window([])
    assert "saved-first" not in projection.cards


def test_pending_tools_and_unsaved_cancellation_are_not_evicted_as_saved_history():
    """缓存轮换保留待收工具和未保存片段，重连只认事实；参数：无；返回：无。"""
    projection = ConversationProjection(retained_cards=2)
    projection.apply({"session_id": "s", "history": []})
    projection.accepted("input", "已接纳输入")
    deliver(projection, AssistantTextDelta("中断前的局部正文", message_id="cancelled"))
    deliver(
        projection,
        ToolExecutionStarted("read", {"path": "等待完成"}, "pending", "safe"),
    )
    for index in range(8):
        deliver(
            projection,
            AssistantTurnComplete(
                str(index), message_id=f"m-{index}", entry_id=f"e-{index}"
            ),
        )
    assert "input" not in projection.cards
    projection.accepted("input", "重复回执")
    assert "input" not in projection.cards
    assert projection.cards["tool:s:run:pending"].state == "执行中"
    deliver(projection, AssistantStreamClosed("cancelled", "cancelled"))
    deliver(projection, AssistantTextDelta("迟到片段", message_id="cancelled"))
    assert (
        projection.cards["stream:s:cancelled"].text
        == "【输出已中断，未保存】\n\n中断前的局部正文"
    )
    projection.apply({"session_id": "s", "event_epoch": "new", "history": []})
    assert not projection.cards


def test_batch_completion_maps_reading_identity_before_releasing_old_cards():
    """同一刷新前批量收尾后仍保护刚提交的阅读卡；参数：无；返回：无。"""
    projection = ConversationProjection(retained_cards=2)
    projection.apply({"session_id": "s", "history": []})
    deliver(projection, AssistantTextDelta("正在阅读", message_id="reading"))
    projection.retain_window(["stream:s:reading"])
    completions = [
        AssistantTurnComplete(
            "正在阅读的完整回答", message_id="reading", entry_id="saved-reading"
        )
    ]
    completions.extend(
        AssistantTurnComplete(
            str(index), message_id=f"m-{index}", entry_id=f"e-{index}"
        )
        for index in range(12)
    )
    events = [
        {
            "sequence": projection.cursor + index + 1,
            "run_id": "run",
            "type": type(event).__name__,
            "data": asdict(event),
        }
        for index, event in enumerate(completions)
    ]
    projection.apply({"session_id": "s", "events": events})
    assert projection.cards["saved-reading"].text == "正在阅读的完整回答"
    assert projection.current_key("stream:s:reading") == "saved-reading"
    assert len(projection.cards) == 3
