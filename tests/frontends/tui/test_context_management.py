"""【TUI】【上下文管理】正式界面的分页、控制和跨会话回复隔离。

作者：xxx
时间：2026-10-01 15:00:00
"""

import asyncio
from threading import Event

import pytest
from textual.widgets import Button, OptionList, Static

from frontends.tui.composer import Composer
from frontends.tui.context_management import ContextPanel
from frontends.tui.interactive import InteractiveTui
from frontends.tui.paged_text import RemotePagedText, TEXT_PAGE_CHARACTERS
from tests.frontends.tui.test_interactive import DisplayBridge
from tests.frontends.tui.test_request_inspection import until

WAIT_SECONDS = 5
DETAIL = "已核验的来源与记忆修改\n" * TEXT_PAGE_CHARACTERS


class ContextBridge(DisplayBridge):
    """仅替换RPC边界，界面和异步事件使用正式实现。"""

    def __init__(self):
        """准备独立控制和读取记录；参数：无；返回：无。"""
        super().__init__()
        self.release.set()
        self.calls = []
        self.enabled = {"history": True, "knowledge": True}
        self.cancelled = False
        self.admission = None
        self.slow_action = None
        self.fail_old = False
        self.slow_entered, self.slow_release = Event(), Event()

    def query_context_management(self, **params):
        """返回有归属的原件页并允许冻结旧请求；参数：RPC请求；返回：对应结果。"""
        self.calls.append(dict(params))
        owner = {key: params[key] for key in ("data_space_id", "session_id")}
        action = params["action"]
        if params["session_id"] == "old" and action == self.slow_action:
            self.slow_entered.set()
            assert self.slow_release.wait(WAIT_SECONDS)
            if self.fail_old:
                raise OSError("旧会话请求失败")
        if action == "configure":
            self.enabled[params["domain"]] = params["enabled"]
        if action in {"overview", "configure"}:
            return {
                **owner,
                **{
                    domain: {
                        "enabled": enabled,
                        **(
                            {"admission": self.admission}
                            if domain == "knowledge"
                            else {}
                        ),
                    }
                    for domain, enabled in self.enabled.items()
                },
            }
        if action == "cancel":
            self.cancelled = True
            return {**owner, "work": {"state": "cancelled"}}
        if action == "list":
            second = params.get("before") is not None
            return {
                **owner,
                "items": [
                    {
                        "work_id": owner["session_id"] + ("-second" if second else ""),
                        "domain": "knowledge",
                        "title": "自动知识维护",
                        "created_at": "2026-10-01 15:00:00",
                        "state": "cancelled" if self.cancelled else "running",
                        "source_count": 2,
                    }
                ],
                "next_before": None
                if second
                else ["2026-10-01 15:00:00", owner["session_id"]],
                "total": 2,
            }
        assert action == "detail"
        offset, limit = params["offset"], params["limit"]
        return {
            **owner,
            "execution": {"session_id": "auxiliary", "run_id": "worker-run"},
            "commit": 12,
            "text": DETAIL[offset : offset + limit],
            "total_chars": len(DETAIL),
        }


def test_context_paging_controls_and_request_navigation_preserve_draft():
    """查看完整版本、分页与请求不发送草稿，两个开关独立；参数：无；返回：无。"""

    async def scenario():
        bridge = ContextBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(140, 50)) as pilot:
            app.receive(
                "snapshot",
                {"session_id": "session", "data_space_id": "space", "history": []},
            )
            await until(pilot, lambda: app.projection.session_id == "session")
            composer = app.query_one(Composer)
            composer.load_text("继续保留这个草稿")
            app.action_context_management()
            panel = app.query_one(ContextPanel)
            await until(pilot, lambda: bool(panel.rows) and bool(panel.enabled))
            panel.query_one("#context-next", Button).press()
            await until(pilot, lambda: "session-second" in panel.rows)
            panel.query_one("#context-previous", Button).press()
            await until(pilot, lambda: "session" in panel.rows)
            panel.query_one(OptionList).focus()
            await pilot.press("enter")
            body = panel.query_one(RemotePagedText)
            await until(pilot, lambda: bool(body.text))
            assert body.text == DETAIL[:TEXT_PAGE_CHARACTERS]
            body.query_one(".paged-next", Button).press()
            await until(pilot, lambda: body.page == 1)
            assert all(
                call.get("commit") == 12
                for call in bridge.calls
                if call["action"] == "detail" and call["limit"] > 1
            )
            panel.query_one("#context-history", Button).press()
            await until(pilot, lambda: panel.enabled.get("history") is False)
            assert panel.enabled["knowledge"] is True
            panel.query_one(OptionList).focus()
            await pilot.press("enter")
            await until(pilot, lambda: panel.execution is not None)
            panel.query_one("#context-requests", Button).press()
            await until(pilot, lambda: not panel.display)
            assert composer.text == "继续保留这个草稿" and not bridge.submitted
            assert app.projection.session_id == "session" and not bridge.cancelled

    asyncio.run(scenario())


@pytest.mark.parametrize("fail_old", [False, True])
def test_context_late_list_and_error_cannot_replace_new_session(fail_old):
    """旧目录成功和错误均不能覆盖新会话；参数：旧请求是否失败；返回：无。"""

    async def scenario():
        bridge = ContextBridge()
        bridge.slow_action, bridge.fail_old = "list", fail_old
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one(ContextPanel)
            panel.open({"session_id": "old", "data_space_id": "space"})
            assert await asyncio.to_thread(bridge.slow_entered.wait, WAIT_SECONDS)
            panel.open({"session_id": "new", "data_space_id": "space"})
            await until(pilot, lambda: "new" in panel.rows)
            app.query_one(Composer).load_text("新会话的草稿")
            bridge.slow_release.set()
            await app.workers.wait_for_complete()
            assert "old" not in panel.rows
            assert "旧会话请求失败" not in str(
                panel.query_one("#context-notice", Static).render()
            )
            assert app.query_one(Composer).text == "新会话的草稿"

    asyncio.run(scenario())


def test_context_cancel_does_not_wait_for_slow_detail():
    """正文读取未返回时仍可取消工作且旧正文不重新出现；参数：无；返回：无。"""

    async def scenario():
        bridge = ContextBridge()
        bridge.slow_action = "detail"
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one(ContextPanel)
            panel.open({"session_id": "old", "data_space_id": "space"})
            await until(pilot, lambda: bool(panel.rows))
            panel.query_one(OptionList).focus()
            await pilot.press("enter")
            assert await asyncio.to_thread(bridge.slow_entered.wait, WAIT_SECONDS)
            panel.query_one("#context-cancel", Button).press()
            await until(pilot, lambda: bridge.cancelled and not panel.selection)
            bridge.slow_release.set()
            await app.workers.wait_for_complete()
            assert panel.query_one(RemotePagedText).text == ""
            assert panel.query_one("#context-cancel", Button).disabled

    asyncio.run(scenario())


@pytest.mark.parametrize("fail_old", [False, True])
def test_context_old_control_does_not_leave_new_session_controls_disabled(fail_old):
    """会话切换期间旧控制仍独立结束，新会话保持可操作；参数：旧控制是否失败；返回：无。"""

    async def scenario():
        bridge = ContextBridge()
        bridge.slow_action, bridge.fail_old = "cancel", fail_old
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one(ContextPanel)
            panel.open({"session_id": "old", "data_space_id": "space"})
            await until(pilot, lambda: bool(panel.rows) and bool(panel.enabled))
            panel.query_one(OptionList).focus()
            await pilot.press("enter")
            panel.query_one("#context-cancel", Button).press()
            assert await asyncio.to_thread(bridge.slow_entered.wait, WAIT_SECONDS)
            panel.invalidate()
            panel.open({"session_id": "new", "data_space_id": "space"})
            await until(pilot, lambda: "new" in panel.rows)
            bridge.slow_release.set()
            await app.workers.wait_for_complete()
            assert not panel.query_one("#context-history", Button).disabled
            assert not panel.query_one("#context-knowledge", Button).disabled
            assert "旧会话请求失败" not in str(
                panel.query_one("#context-notice", Static).render()
            )

    asyncio.run(scenario())


def test_context_paging_does_not_discard_pending_settings_query():
    """慢设置查询期间翻页，返回后仍可控制后台工作；参数：无；返回：无。"""

    async def scenario():
        bridge = ContextBridge()
        bridge.slow_action = "overview"
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one(ContextPanel)
            panel.open({"session_id": "old", "data_space_id": "space"})
            assert await asyncio.to_thread(bridge.slow_entered.wait, WAIT_SECONDS)
            await until(pilot, lambda: bool(panel.rows))
            panel.query_one("#context-next", Button).press()
            await until(pilot, lambda: "old-second" in panel.rows)
            bridge.slow_release.set()
            await app.workers.wait_for_complete()
            assert not panel.query_one("#context-history", Button).disabled
            assert panel.enabled == {"history": True, "knowledge": True}

    asyncio.run(scenario())


def test_unaccepted_knowledge_is_visible_without_disabling_main_composer():
    """未接纳原因在面板可见且保留主聊天草稿；参数：无；返回：无。"""

    async def scenario():
        bridge = ContextBridge()
        bridge.admission = {
            "state": "not_accepted",
            "reason": "background_knowledge_requires_saved_model_credentials",
        }
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            composer = app.query_one(Composer)
            composer.load_text("主请求的草稿")
            panel = app.query_one(ContextPanel)
            panel.open({"session_id": "old", "data_space_id": "space"})
            await until(pilot, lambda: bool(panel.enabled))
            assert "临时凭据" in str(
                panel.query_one("#context-status", Static).render()
            )
            assert composer.text == "主请求的草稿" and not bridge.submitted

    asyncio.run(scenario())


def test_late_control_receipt_keeps_new_selection_in_same_session():
    """控制执行期间改看另一项工作，旧回执不能清掉新阅读；参数：无；返回：无。"""

    async def scenario():
        bridge = ContextBridge()
        bridge.slow_action = "cancel"
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            panel = app.query_one(ContextPanel)
            panel.open({"session_id": "old", "data_space_id": "space"})
            await until(pilot, lambda: bool(panel.rows) and bool(panel.enabled))
            panel.query_one(OptionList).focus()
            await pilot.press("enter")
            panel.query_one("#context-cancel", Button).press()
            assert await asyncio.to_thread(bridge.slow_entered.wait, WAIT_SECONDS)
            panel.query_one("#context-next", Button).press()
            await until(pilot, lambda: "old-second" in panel.rows)
            panel.query_one(OptionList).focus()
            await pilot.press("enter")
            await until(pilot, lambda: bool(panel.query_one(RemotePagedText).text))
            bridge.slow_release.set()
            await app.workers.wait_for_complete()
            assert panel.selection == {"domain": "knowledge", "work_id": "old-second"}
            assert panel.query_one(RemotePagedText).text

    asyncio.run(scenario())
