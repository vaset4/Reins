"""真实持久消息与流投影之间的重连窗口。

作者：xxx
时间：2026-09-29 22:30:00
"""

import pytest

from app.background.events import EventBuffer
from app.background.sessions import session_history
from frontends.tui.projection import ConversationProjection
from runtime.session_messages import append_assistant_message
from runtime.stream_events import (
    AssistantTextDelta,
    AssistantReasoningDelta,
    AssistantTurnComplete,
)


@pytest.mark.parametrize("commit_before_attach", [False, True])
def test_reconnect_merges_stream_and_persisted_answer_once(
    tmp_path, commit_before_attach
):
    """重连跨越提交窗口时保留全文和思考且只显示一次；传参：隔离根与提交时刻；返回：无。"""
    events = EventBuffer()
    events.emit(
        AssistantTextDelta("回答前半", message_id="request-one"), run_id="run-one"
    )
    events.emit(
        AssistantReasoningDelta("可见思考", message_id="request-one"), run_id="run-one"
    )
    checkpoint = events.read(None, include_streams=True)
    entry_id = None
    if commit_before_attach:
        entry_id = append_assistant_message(
            tmp_path,
            "session",
            "回答前半后半",
            run_id="run-one",
            message_id="request-one",
            reasoning="可见思考",
        )
    projection = ConversationProjection()
    projection.apply(
        {
            "session_id": "session",
            "history": session_history(tmp_path, "session"),
            **checkpoint,
        }
    )
    assert len(projection.cards) == 1
    assert next(iter(projection.cards.values())).text.startswith("回答前半")
    events.emit(AssistantTextDelta("后半", message_id="request-one"), run_id="run-one")
    projection.apply({"session_id": "session", **events.read(checkpoint["cursor"])})
    if entry_id is None:
        entry_id = append_assistant_message(
            tmp_path,
            "session",
            "回答前半后半",
            run_id="run-one",
            message_id="request-one",
            reasoning="可见思考",
        )
    events.emit(
        AssistantTurnComplete(
            "回答前半后半", message_id="request-one", entry_id=entry_id
        ),
        run_id="run-one",
    )
    projection.apply({"session_id": "session", **events.read(projection.cursor)})
    assert list(projection.cards) == [entry_id]
    assert projection.cards[entry_id].text == "回答前半后半"
    assert projection.cards[entry_id].reasoning == "可见思考"
    projection.apply(
        {
            "session_id": "session",
            "history": session_history(tmp_path, "session"),
            **events.read(None, include_streams=True),
        }
    )
    assert list(projection.cards) == [entry_id]
    assert not events.read(None, include_streams=True)["streams"]


def test_gap_retains_full_in_progress_answer_beyond_event_cache(monkeypatch):
    """事件缓存淘汰旧增量时，当前输出仍可完整重建；传参：缓存容量替换；返回：无。"""
    monkeypatch.setattr("app.background.events.EVENT_CACHE_SIZE", 2)
    events = EventBuffer()
    for text in ("一", "二", "三", "四"):
        events.emit(AssistantTextDelta(text, message_id="request"), run_id="run")
    checkpoint = events.read(0)
    assert checkpoint["gap"]
    projection = ConversationProjection()
    projection.apply({"session_id": "session", "history": [], **checkpoint})
    assert [card.text for card in projection.cards.values()] == ["一二三四"]
