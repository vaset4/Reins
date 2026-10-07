"""【会话】【目录分页】验证跨页可达与查询不读取全部正文。

作者：xxx
时间：2026-09-29 23:00:00
"""

import asyncio
from threading import Event

import pytest
from textual.widgets import Button, Input, OptionList, Static

from app.background.navigation import list_sessions
from frontends.tui.interactive import InteractiveTui
from runtime.session_message_store import SessionMessageStore
from tests.frontends.tui.test_interactive import DisplayBridge


def test_directory_pages_and_search_without_loading_history(tmp_path, monkeypatch):
    """超过百条仍可逐页访问，搜索保留中文与引号；参数：隔离库；返回：无。"""
    store = SessionMessageStore(tmp_path)
    with store.database.transaction():
        for index in range(105):
            store.accept_input(f"session-{index:03}", f"中文标题 '{index}")

    def forbidden(*args, **kwargs):
        """禁止目录物化会话历史；参数：任意；返回：直接失败。"""
        raise AssertionError("directory loaded full history")

    monkeypatch.setattr(SessionMessageStore, "read_entries", forbidden)
    first = list_sessions(tmp_path)
    second = list_sessions(tmp_path, before=first["next_before"])
    assert len(first["sessions"]) == 100 and len(second["sessions"]) == 5
    assert not second["has_more"]
    identities = [item["session_id"] for item in first["sessions"] + second["sessions"]]
    assert len(set(identities)) == 105
    found = list_sessions(tmp_path, "中文标题 '0")
    assert [item["session_id"] for item in found["sessions"]] == ["session-000"]
    with pytest.raises(ValueError, match="cursor"):
        list_sessions(tmp_path, before=["bad"])


def test_directory_controls_keep_search_and_return_to_first_page():
    """下一页和上一页沿同一检索游标读取；参数：无；返回：无。"""

    class DirectoryBridge(DisplayBridge):
        """提供两页可预测目录，界面走真实消息循环。"""

        def browse(self, method, **params):
            """返回所选页；参数：目录查询；返回：摘要页。"""
            before = params.get("before")
            number = "two" if before else "one"
            return {
                "sessions": [
                    {
                        "session_id": number,
                        "title": params["query"] + number,
                        "status": "idle",
                    }
                ],
                "next_before": None if before else ["time", "one"],
            }

    async def scenario():
        """执行搜索和翻页；参数：无；返回：无。"""
        app = InteractiveTui(DirectoryBridge())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            search = app.query_one("#session-search", Input)
            search.value = "中文"
            search.focus()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.click("#sessions-next")
            await pilot.pause()
            assert (
                app.query_one("#sessions", OptionList).get_option_at_index(0).id
                == "two"
            )
            assert app.query_one("#sessions-next", Button).disabled
            await pilot.click("#sessions-previous")
            await pilot.pause()
            assert (
                app.query_one("#sessions", OptionList).get_option_at_index(0).id
                == "one"
            )
            assert app._session_query == "中文"

    asyncio.run(scenario())


def test_directory_ignores_old_failure_but_reports_current_failure():
    """旧搜索失败不污染新结果，当前失败仍可见并能重新检索；参数：无；返回：无。"""
    entered, release = Event(), Event()

    class DirectoryBridge(DisplayBridge):
        """控制目录请求完成顺序，保留实际界面线程交接。"""

        def browse(self, method, **params):
            """旧查询等待后失败，当前错误直接返回；参数：方法与查询；返回：目录页。"""
            query = params.get("query", "")
            if query == "旧搜索":
                entered.set()
                assert release.wait(5)
                raise OSError("旧请求失败")
            if query == "当前错误":
                raise OSError("当前请求失败")
            return {
                "sessions": [
                    {"session_id": "current", "title": query, "status": "idle"}
                ],
                "next_before": None,
            }

    async def scenario():
        """新搜索完成后释放旧失败，再检查当前失败和重新检索；参数：无；返回：无。"""
        app = InteractiveTui(DirectoryBridge())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            await app.workers.wait_for_complete()
            app.load_sessions("旧搜索")
            assert await asyncio.to_thread(entered.wait, 2)
            app.load_sessions("新搜索")
            await pilot.pause()
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._session_query == "新搜索"
            assert (
                app.query_one("#sessions", OptionList).get_option_at_index(0).id
                == "current"
            )
            assert "旧请求失败" not in str(app.query_one("#notice", Static).render())
            assert not any("旧请求失败" in value for value in app.command_outputs)
            app.load_sessions("当前错误")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "当前请求失败" in str(app.query_one("#notice", Static).render())
            app.load_sessions("再次搜索")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "再次搜索" in str(
                app.query_one("#sessions", OptionList).get_option_at_index(0).prompt
            )

    try:
        asyncio.run(scenario())
    finally:
        release.set()
