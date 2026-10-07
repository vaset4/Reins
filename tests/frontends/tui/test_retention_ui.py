"""真实保存的大结果在持续接收和回收旧正文时仍可阅读、编辑。

作者：xxx
时间：2026-09-29 20:00:00
"""

import asyncio
from dataclasses import asdict
from time import perf_counter

from textual.containers import VerticalScroll
from textual.widgets import Collapsible

from app.background.sessions import session_history_page
from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui, LIVE_CARD_WINDOW
from frontends.tui.paged_text import PagedText
from runtime.session_message_store import SessionMessageStore
from runtime.stream_events import AssistantTextDelta
from tests.frontends.tui.test_history_window import PagedBridge
from tests.frontends.tui.test_retention import append_round, GIANT_CHARACTERS

INITIAL_ROUNDS = 18
ADDED_ROUNDS = 54
INPUT_RESPONSE_SECONDS = 1.5


def prepare_traffic(data_root):
    """在界面外准备真实提交后的初始页和后续事件；参数：隔离根；返回：快照与事件。"""
    store = SessionMessageStore(data_root)
    with store.database.transaction():
        for index in range(INITIAL_ROUNDS):
            append_round(store, index)
    initial = session_history_page(data_root, "s", limit=INITIAL_ROUNDS * 2)
    later = []
    with store.database.transaction():
        for index in range(INITIAL_ROUNDS, INITIAL_ROUNDS + ADDED_ROUNDS):
            later.extend(append_round(store, index))
    return initial, later


async def settle(app, pilot):
    """等待当前显示批次完成而非固定睡眠；参数：界面和驾驶器；返回：无。"""
    await pilot.pause()
    await app.refresh_cards()
    async with asyncio.timeout(5):
        while app._rendering or app.changed:
            await pilot.pause()


def test_large_saved_reading_window_survives_live_retention_and_input(tmp_path):
    """读旧大结果时持续收尾不跳页，回到实时释放旧正文；参数：隔离根；返回：无。"""
    initial, later = prepare_traffic(tmp_path)

    async def scenario():
        """驱动真实控件验证阅读锚点、草稿与保留量；参数：无；返回：无。"""
        app = InteractiveTui(PagedBridge(tmp_path))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.receive("snapshot", initial)
            await settle(app, pilot)
            pane = app.query_one("#conversation", VerticalScroll)
            pane.scroll_home(animate=False, immediate=True)
            giant = app.cards["tool:s:run:call-0"]
            giant.query_one(Collapsible).collapsed = False
            await pilot.pause()
            body = giant.query_one(".tool-output", PagedText)
            body.page = body.page_count - 1
            body.render_page()
            composer = app.query_one(Composer)
            composer.load_text("草稿")
            composer.move_cursor((0, len(composer.text)))
            composer.focus()
            await pilot.pause()
            first_key, position = next(iter(app.cards)), pane.scroll_y
            started = perf_counter()
            for sequence, event in enumerate(later, 1):
                app.receive(
                    "snapshot",
                    {
                        "session_id": "s",
                        "events": [
                            {
                                "sequence": sequence,
                                "run_id": "run",
                                "type": type(event).__name__,
                                "data": asdict(event),
                            }
                        ],
                    },
                )
            await pilot.press("x")
            elapsed = perf_counter() - started
            await settle(app, pilot)
            assert next(iter(app.cards)) == first_key
            assert pane.scroll_y == position
            assert app.cards[giant.card.key] is giant
            assert body.page == body.page_count - 1
            assert body.text == "结果0:" + "原" * GIANT_CHARACTERS
            assert len(app.projection.cards) <= LIVE_CARD_WINDOW * 2
            assert composer.text == "草稿x"
            assert elapsed < INPUT_RESPONSE_SECONDS
            delta = AssistantTextDelta("继续输出", message_id="active")
            app.receive(
                "snapshot",
                {
                    "session_id": "s",
                    "events": [
                        {
                            "sequence": len(later) + 1,
                            "run_id": "run",
                            "type": type(delta).__name__,
                            "data": asdict(delta),
                        }
                    ],
                },
            )
            await settle(app, pilot)
            assert pane.scroll_y == position
            app.return_to_live()
            await settle(app, pilot)
            assert len(app.cards) == LIVE_CARD_WINDOW
            assert len(app.projection.cards) == LIVE_CARD_WINDOW
            assert giant.card.key not in app.projection.cards
            assert any(
                card.text == "继续输出" for card in app.projection.cards.values()
            )
            assert composer.text == "草稿x"
            retained = sum(
                len(card.text) + len(card.args) + len(card.reasoning)
                for card in app.projection.cards.values()
            )
            print(
                f"retention-ui: persisted_rounds={INITIAL_ROUNDS + ADDED_ROUNDS}, cards={len(app.cards)}, "
                f"retained_chars={retained}, key_response_seconds={elapsed:.3f}"
            )

    asyncio.run(scenario())
