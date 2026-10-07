"""核对入站回执和已保存历史在界面上共用同一输入身份。

作者：xxx
时间：2026-09-29 23:00:00
"""

import asyncio

import pytest
from textual.widgets import Static

from app.background.sessions import session_history
from frontends.tui.interactive import InteractiveTui
from frontends.tui.widgets import MessageCard
from runtime.session_message_store import SessionMessageStore
from tests.frontends.tui.test_interactive import DisplayBridge


@pytest.mark.parametrize("ack_first", [True, False])
def test_input_ack_and_saved_history_render_once_without_deduplicating_text(
    tmp_path, ack_first
):
    """两种到达次序均只显示一张卡，独立发送相同正文保留两张；参数：目录与次序；返回：无。"""
    store = SessionMessageStore(tmp_path)
    store.accept_input("session", "你是谁", input_id="input-one")
    store.deliver_inputs("session", run_id="run-one", task_id=None)
    first_history = session_history(tmp_path, "session")
    store.accept_input("session", "你是谁", input_id="input-two")
    store.deliver_inputs("session", run_id="run-two", task_id=None)
    full_history = session_history(tmp_path, "session")

    async def scenario():
        """驱动真实控件消息循环，验证每个持久输入各出现一次；参数：无；返回：无。"""
        app = InteractiveTui(DisplayBridge())
        async with app.run_test(size=(120, 40)) as pilot:
            app.receive("snapshot", {"session_id": "session", "history": []})
            await pilot.pause()
            updates = [
                (
                    "accepted",
                    {
                        "identity": "input-one",
                        "text": "你是谁",
                        "session_id": "session",
                    },
                ),
                ("snapshot", {"session_id": "session", "history": first_history}),
            ]
            for kind, payload in updates if ack_first else reversed(updates):
                app.receive(kind, payload)
                await pilot.pause()
            await app.refresh_cards()
            cards = list(app.query(MessageCard))
            assert len(cards) == 1
            assert cards[0].card.text == "你是谁"
            assert len(cards[0].query(".message-author")) == 1
            assert len(cards[0].query(".message-body")) == 1
            assert str(cards[0].query_one(".message-body", Static).render()) == "你是谁"
            app.receive("snapshot", {"session_id": "session", "history": full_history})
            app.receive(
                "accepted",
                {"identity": "input-two", "text": "你是谁", "session_id": "session"},
            )
            app.receive(
                "accepted",
                {"identity": "input-one", "text": "你是谁", "session_id": "session"},
            )
            await pilot.pause()
            await app.refresh_cards()
            assert [card.card.key for card in app.query(MessageCard)] == [
                "input-one",
                "input-two",
            ]

    asyncio.run(scenario())
