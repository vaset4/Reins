"""【TUI】【原件搜索】查看命中保留草稿，明确打开才切换会话。

作者：xxx
时间：2026-09-30 16:00:00
"""

import asyncio

from textual.widgets import OptionList

from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui
from frontends.tui.paged_text import RemotePagedText
from frontends.tui.search_results import SearchResultsScreen
from tests.frontends.tui.test_interactive import DisplayBridge


def test_search_reading_preserves_draft_and_requires_explicit_switch():
    """点击正文命中只打开阅读浮层；参数：无；返回：无。"""
    source = {
        "session_id": "other",
        "kind": "session_entry",
        "record_id": "entry",
        "sequence": 1,
        "source_path": "workspaces/a/sessions/other/events.jsonl",
        "source_offset": 10,
        "entry_id": "entry",
        "label": "对话原文",
    }

    class SearchBridge(DisplayBridge):
        """提供目录与固定原件，不执行运行请求。"""

        def browse(self, method, **params):
            """返回选定只读页面；参数：方法和查询；返回：目录或正文。"""
            if method == "session_search":
                if params["action"] == "matches":
                    return {"matches": [source], "next_after": None}
                return {
                    "text": "历史原件正文",
                    "total_chars": 6,
                    "retention": "captured",
                }
            return {
                "sessions": [
                    {
                        "session_id": "other",
                        "title": "历史会话",
                        "status": "idle",
                        "match": source,
                    }
                ],
                "next_before": None,
            }

    async def scenario():
        """操作实际Textual消息循环；参数：无；返回：无。"""
        app = InteractiveTui(SearchBridge())
        selected = []
        app.switch_session = selected.append
        async with app.run_test(size=(120, 45)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("未发送草稿")
            app.query_one("#sessions", OptionList).focus()
            app.query_one("#sessions", OptionList).highlighted = 0
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, SearchResultsScreen)
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.screen.query_one(RemotePagedText).text == "历史原件正文"
            assert selected == [] and composer.text == "未发送草稿"
            await pilot.click("#search-close")
            await pilot.pause()
            assert composer.text == "未发送草稿" and selected == []
            app.query_one("#sessions", OptionList).focus()
            app.query_one("#sessions", OptionList).highlighted = 0
            await pilot.press("enter")
            await pilot.pause()
            await pilot.click("#search-open-session")
            await pilot.pause()
            assert selected == ["other"]

    asyncio.run(scenario())
