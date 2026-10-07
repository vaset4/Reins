"""正式界面的固定历史页与实时消息隔离。

作者：xxx
时间：2026-09-29 23:30:00
"""

import asyncio

from app.background.sessions import session_history_page
from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui
from llm.messages import TextPart, UserMessage
from runtime.session_message_store import SessionMessageStore
from tests.frontends.tui.test_interactive import DisplayBridge

RENDER_WAIT_SECONDS = 5


class PagedBridge(DisplayBridge):
    """通过真实存储分页，只替换进程连接。"""

    def __init__(self, data_root):
        """保存隔离数据根；传参：目录；返回：无。"""
        super().__init__()
        self.data_root = data_root

    def browse(self, method, **params):
        """调用正式历史投影；传参：后台动作及游标；返回：真实页面。"""
        if method == "history_page":
            options = {
                key: value for key, value in params.items() if key != "session_id"
            }
            return session_history_page(
                self.data_root, params["session_id"], limit=2, **options
            )
        return super().browse(method, **params)


def test_history_pages_keep_draft_and_do_not_follow_live_output(tmp_path):
    """翻阅固定历史时新输出不抢回页面，返回实时后可见；传参：隔离根；返回：无。"""
    store = SessionMessageStore(tmp_path)
    for index in range(6):
        store.append_message(
            "s", UserMessage(f"m-{index}", (TextPart(f"历史{index}"),))
        )

    async def scenario():
        bridge = PagedBridge(tmp_path)
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.receive("snapshot", session_history_page(tmp_path, "s", limit=2))
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("翻页时继续写草稿")
            await pilot.click("#history-open")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.history_projection is not None, str(
                app.query_one("#notice").render()
            )
            assert app._history_page["next_before"] is not None
            assert [card.card.text for card in app.cards.values()] == ["历史4", "历史5"]
            await pilot.click("#history-older")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert [card.card.text for card in app.cards.values()] == [
                "历史2",
                "历史3",
            ], (
                app._history_cursors,
                app._history_pending,
                [card.text for card in app.history_projection.cards.values()],
                str(app.query_one("#notice").render()),
            )
            app.receive(
                "snapshot",
                {
                    "session_id": "s",
                    "events": [
                        {
                            "sequence": 1,
                            "type": "AssistantTextDelta",
                            "data": {"text": "新的实时回答", "message_id": "new"},
                        }
                    ],
                },
            )
            await pilot.pause()
            assert [card.card.text for card in app.cards.values()] == ["历史2", "历史3"]
            assert composer.text == "翻页时继续写草稿"
            await pilot.click("#history-newer")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert [card.card.text for card in app.cards.values()] == ["历史4", "历史5"]
            await pilot.click("#history-live")
            await pilot.pause()
            assert any(card.card.text == "新的实时回答" for card in app.cards.values())
            assert composer.text == "翻页时继续写草稿"

    asyncio.run(scenario())


def test_initial_background_snapshot_does_not_materialize_full_history(
    tmp_path, monkeypatch
):
    """正式快照只查询一页正文，普通poll只读取叶身份；传参：隔离根；返回：无。"""
    from scripts.testing.llm import from_test_sequence
    from tests.test_background_sessions import session_for
    from runtime.session_message_store import DEFAULT_HISTORY_PAGE_SIZE

    session = session_for(tmp_path, from_test_sequence(["unused"]))
    identity = session.record.session_id
    with session.messages.database.transaction():
        for index in range(DEFAULT_HISTORY_PAGE_SIZE + 1):
            session.messages.append_message(
                identity, UserMessage(f"m-{index}", (TextPart(str(index)),))
            )

    def full_read_forbidden(*args, **kwargs):
        """禁止快照退回全量读取；传参：查询参数；返回：明确失败。"""
        raise AssertionError("snapshot must not load full history")

    monkeypatch.setattr(SessionMessageStore, "read_entries", full_read_forbidden)
    monkeypatch.setattr(SessionMessageStore, "materialize", full_read_forbidden)
    try:
        snapshot = session.snapshot(history=True)
        assert len(snapshot["history"]) == DEFAULT_HISTORY_PAGE_SIZE
        assert snapshot["next_before"] is not None
        assert "history" not in session.snapshot(after=snapshot["cursor"])
    finally:
        session.close()


def test_live_render_window_is_bounded_and_reading_position_is_preserved():
    """长会话仅挂载当前窗口，上翻后新消息不移动阅读起点；传参：无；返回：无。"""
    from textual.containers import VerticalScroll
    from frontends.tui.interactive import LIVE_CARD_WINDOW

    async def scenario():
        app = InteractiveTui(DisplayBridge())
        async with app.run_test(size=(100, 32)) as pilot:
            await pilot.pause()
            rows = [
                {
                    "entry_id": f"row-{index}",
                    "role": "user",
                    "text": f"历史{index}\n多行正文",
                }
                for index in range(LIVE_CARD_WINDOW + 10)
            ]
            app.receive("snapshot", {"session_id": "s", "history": rows, "cursor": 0})
            await pilot.pause()
            await app.refresh_cards()
            async with asyncio.timeout(RENDER_WAIT_SECONDS):
                while app._rendering or app.changed:
                    await pilot.pause()
            assert len(app.cards) == LIVE_CARD_WINDOW
            pane = app.query_one("#conversation", VerticalScroll)
            pane.scroll_home(animate=False, immediate=True)
            await pilot.pause()
            first_key = next(iter(app.cards))
            app.receive(
                "snapshot",
                {
                    "session_id": "s",
                    "events": [
                        {
                            "sequence": 1,
                            "type": "AssistantTextDelta",
                            "data": {"text": "新输出", "message_id": "new"},
                        }
                    ],
                },
            )
            await pilot.pause()
            await app.refresh_cards()
            async with asyncio.timeout(RENDER_WAIT_SECONDS):
                while app._rendering or app.changed:
                    await pilot.pause()
            assert len(app.cards) == LIVE_CARD_WINDOW
            assert next(iter(app.cards)) == first_key
            assert pane.scroll_y == 0

    asyncio.run(scenario())
