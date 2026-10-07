"""请求面板的真实控件分页、并发输入及旧响应隔离。

作者：xxx
时间：2026-09-30 20:00:00
"""

import asyncio
from threading import Event

import pytest
from textual.widgets import Button, Input, OptionList, Static

from frontends.tui.approval_dialog import ApprovalScreen
from frontends.tui.composer import Composer
from frontends.tui.evidence import EvidencePanel
from frontends.tui.interactive import InteractiveTui
from frontends.tui.paged_text import RemotePagedText, TEXT_PAGE_CHARACTERS
from tests.frontends.tui.test_interactive import DisplayBridge

WAIT_SECONDS = 5
LARGE_TEXT = "中文材料" * 250000 + "完整最后一行"


class EvidenceBridge(DisplayBridge):
    """仅替换RPC边界，面板与分页控件保持正式实现。"""

    def __init__(self):
        """准备可控制的长读和导出；参数：无；返回：无。"""
        super().__init__()
        self.reads = []
        self.slow_entered, self.slow_release = Event(), Event()
        self.slow = False
        self.fail_old = False
        self.cancelled = False
        self.release.set()

    def query_evidence(self, **params):
        """返回真实协议形状并记录按需读取量；参数：查询；返回：对应页面。"""
        self.reads.append(dict(params))
        owner = {key: params[key] for key in ("data_space_id", "session_id")}
        if params["action"] == "detail":
            if self.slow and params["attempt_id"] == "old":
                self.slow_entered.set()
                assert self.slow_release.wait(WAIT_SECONDS)
                if self.fail_old:
                    raise OSError("旧选择读取失败")
            source = LARGE_TEXT if params["attempt_id"] != "new" else "新选择的真实正文"
            offset, limit = params["offset"], params["limit"]
            return {
                **owner,
                "text": source[offset : offset + limit],
                "total_chars": len(source),
                "status": "completed",
            }
        key = {"runs": "run_id", "requests": "request_id", "attempts": "attempt_id"}[
            params["action"]
        ]
        value = {"run_id": "run", "request_id": "request", "attempt_id": "attempt"}[key]
        return {
            **owner,
            "items": [{key: value, "status": "completed"}],
            "next_cursor": None,
        }

    def export_evidence(self, **params):
        """导出等待明确取消，不触碰模型取消通道；参数：动作；返回：独立任务状态。"""
        if params["action"] == "cancel":
            self.cancelled = True
        return {
            "job_id": "job",
            "status": "cancelled" if self.cancelled else "running",
            "cutoff": {"fact_id": 4},
        }


async def until(pilot, predicate):
    """等待可观察控件结果，超时暴露失败；参数：驾驶器和条件；返回：无。"""
    async with asyncio.timeout(WAIT_SECONDS):
        while not predicate():
            await pilot.pause(0.02)


def test_million_character_page_allows_chat_approval_export_and_full_copy():
    """百万正文按页读取，输入和审批继续可用，导出取消不停止模型；参数：无；返回：无。"""

    async def scenario():
        bridge = EvidenceBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(140, 50)) as pilot:
            app.receive(
                "snapshot",
                {"session_id": "session", "data_space_id": "space", "history": []},
            )
            await until(pilot, lambda: app.projection.session_id == "session")
            composer = app.query_one(Composer)
            composer.load_text("未发送草稿")
            await pilot.press("f9")
            panel = app.query_one(EvidencePanel)
            await until(pilot, lambda: bool(panel.rows))
            for level in ("requests", "attempts"):
                panel.query_one(OptionList).focus()
                await pilot.press("enter")
                await until(
                    pilot,
                    lambda: (
                        panel.level == level
                        and next(iter(panel.rows))
                        != ("run" if level == "requests" else "request")
                    ),
                )
            panel.query_one(OptionList).focus()
            await pilot.press("enter")
            body = panel.query_one(RemotePagedText)
            await until(pilot, lambda: bool(body.text))
            assert body.text == LARGE_TEXT[:TEXT_PAGE_CHARACTERS]
            assert len([row for row in bridge.reads if row["action"] == "detail"]) == 1
            page_input = body.query_one(Input)
            page_input.value = str(body.page_count)
            page_input.focus()
            await pilot.press("enter")
            await until(pilot, lambda: body.text.endswith("完整最后一行"))
            last_page = body.page
            app.receive(
                "snapshot",
                {
                    "session_id": "session",
                    "data_space_id": "space",
                    "approval": {
                        "identity": "approval",
                        "tool": "exec",
                        "args": {},
                        "force_confirmation": True,
                    },
                },
            )
            await until(pilot, lambda: isinstance(app.screen, ApprovalScreen))
            await pilot.press("escape")
            assert body.page == last_page and composer.text == "未发送草稿"
            panel.open_export()
            panel.start_export()
            await until(pilot, lambda: panel.job.get("status") == "running")
            composer.focus()
            composer.load_text("读取期间继续发送")
            await pilot.press("enter")
            await until(pilot, lambda: bridge.submitted == ["读取期间继续发送"])
            panel.query_one("#export-cancel", Button).press()
            await until(pilot, lambda: panel.job.get("status") == "cancelled")
            body.copy_source(body.reader, body._copy_generation)
            await until(pilot, lambda: app.clipboard == LARGE_TEXT)
            assert bridge.cancelled and bridge.submitted == ["读取期间继续发送"]
            await pilot.resize_terminal(70, 24)
            composer.load_text("缩放后草稿")
            panel.query_one("#evidence-close", Button).press()
            await pilot.pause()
            assert composer.text == "缩放后草稿" and not panel.display

    asyncio.run(scenario())


@pytest.mark.parametrize("old_error", [False, True])
def test_late_detail_success_and_error_cannot_replace_new_selection(old_error):
    """较慢旧查询无论成功失败都不能污染新正文；参数：旧结果是否失败；返回：无。"""

    async def scenario():
        bridge = EvidenceBridge()
        bridge.slow, bridge.fail_old = True, old_error
        app = InteractiveTui(bridge)
        try:
            async with app.run_test(size=(120, 40)) as pilot:
                await pilot.pause()
                panel = app.query_one(EvidencePanel)
                owner = {"session_id": "session", "data_space_id": "space"}
                panel.open(
                    owner,
                    {"run_id": "run", "request_id": "request", "attempt_id": "old"},
                )
                panel.show_section("request")
                assert await asyncio.to_thread(bridge.slow_entered.wait, WAIT_SECONDS)
                panel.selection = {**panel.selection, "attempt_id": "new"}
                panel.show_section("request")
                body = panel.query_one(RemotePagedText)
                await until(pilot, lambda: body.text == "新选择的真实正文")
                bridge.slow_release.set()
                await pilot.pause()
                assert body.text == "新选择的真实正文"
                assert "旧选择" not in str(
                    body.query_one(".paged-caption", Static).render()
                )
        finally:
            bridge.slow_release.set()

    asyncio.run(scenario())


def test_request_page_failure_keeps_reading_and_retry_restores_navigation():
    """同来源翻页失败保留原页，重试成功后隐藏重试入口；参数：无；返回：无。"""

    async def scenario():
        bridge = EvidenceBridge()
        app = InteractiveTui(bridge)
        failed = True

        def read(offset, limit):
            """模拟实际第二页读取故障；参数：字符位置与页长；返回：规范正文页。"""
            if failed and offset:
                raise OSError("原件暂不可读")
            return {
                "text": LARGE_TEXT[offset : offset + limit],
                "total_chars": len(LARGE_TEXT),
            }

        async with app.run_test(size=(120, 40)) as pilot:
            body = app.query_one(RemotePagedText)
            body.set_reader(read)
            await until(pilot, lambda: bool(body.text))
            body.page = 1
            body.render_page()
            retry = body.query_one(".paged-retry", Button)
            await until(pilot, lambda: retry.display)
            assert body.page == 0 and body.text == LARGE_TEXT[:TEXT_PAGE_CHARACTERS]
            failed = False
            retry.press()
            await until(pilot, lambda: body.page == 1)
            assert not retry.display
            assert (
                body.text == LARGE_TEXT[TEXT_PAGE_CHARACTERS : TEXT_PAGE_CHARACTERS * 2]
            )

    asyncio.run(scenario())


def test_changing_request_clears_display_before_new_body_arrives():
    """慢查询期间不把旧正文挂到新来源，草稿保持；参数：无；返回：无。"""

    async def scenario():
        bridge = EvidenceBridge()
        app = InteractiveTui(bridge)
        bridge.slow = True
        try:
            async with app.run_test(size=(120, 40)) as pilot:
                panel = app.query_one(EvidencePanel)
                owner = {"session_id": "session", "data_space_id": "space"}
                panel.open(
                    owner,
                    {"run_id": "run", "request_id": "request", "attempt_id": "new"},
                )
                panel.show_section("request")
                body = panel.query_one(RemotePagedText)
                await until(pilot, lambda: bool(body.text))
                panel.selection = {**panel.selection, "attempt_id": "old"}
                panel.show_section("request")
                assert await asyncio.to_thread(bridge.slow_entered.wait, WAIT_SECONDS)
                assert "新选择的真实正文" not in str(
                    body.query_one(".paged-body", Static).render()
                )
        finally:
            bridge.slow_release.set()

    asyncio.run(scenario())


def test_reconnect_preserves_reading_but_new_space_retires_old_selection():
    """宿主换实例保留阅读，重置空间保留草稿且不接纳旧响应；参数：无；返回：无。"""

    async def scenario():
        bridge = EvidenceBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            snapshot = {
                "session_id": "session",
                "data_space_id": "space",
                "event_epoch": "first",
                "history": [],
                "cursor": 7,
                "project_root": "原工作区",
            }
            app.receive("snapshot", snapshot)
            await until(pilot, lambda: app.projection.session_id == "session")
            composer = app.query_one(Composer)
            composer.load_text("重连期间未发送草稿")
            panel = app.query_one(EvidencePanel)
            panel.open(
                {"session_id": "session", "data_space_id": "space"},
                {"run_id": "run", "request_id": "request", "attempt_id": "attempt"},
            )
            panel.show_section("request")
            body = panel.query_one(RemotePagedText)
            await until(pilot, lambda: bool(body.text))
            body.page = 1
            body.render_page()
            await until(pilot, lambda: body._loaded_page == 1)
            app.receive("snapshot", {**snapshot, "event_epoch": "second"})
            await pilot.pause()
            assert (
                panel.display
                and body.page == 1
                and composer.text == "重连期间未发送草稿"
            )
            app.receive(
                "snapshot",
                {**snapshot, "data_space_id": "new-space", "event_epoch": "third"},
            )
            await pilot.pause()
            assert not panel.display and not body.text and not panel.owner
            assert composer.text == "重连期间未发送草稿" and not bridge.submitted
            assert "原工作区" in str(app.query_one("#notice", Static).render())
            app.receive("snapshot", snapshot)
            await pilot.pause()
            assert app.projection.data_space_id == "new-space" and not panel.display

    asyncio.run(scenario())


def test_pending_directory_keeps_old_rows_inert_until_new_identity_is_confirmed():
    """新目录等待期旧摘要不能被解释为新层级身份；参数：无；返回：无。"""

    async def scenario():
        bridge = EvidenceBridge()
        entered, release = Event(), Event()
        original_query = bridge.query_evidence

        def query(**params):
            """延迟请求目录响应以覆盖真实竞态；参数：查询参数；返回：原规范页。"""
            if params["action"] == "requests":
                entered.set()
                assert release.wait(WAIT_SECONDS)
            return original_query(**params)

        bridge.query_evidence = query
        app = InteractiveTui(bridge)
        try:
            async with app.run_test(size=(120, 40)) as pilot:
                panel = app.query_one(EvidencePanel)
                panel.open({"session_id": "session", "data_space_id": "space"})
                await until(pilot, lambda: bool(panel.rows))
                directory = panel.query_one(OptionList)
                directory.focus()
                await pilot.press("enter")
                assert await asyncio.to_thread(entered.wait, WAIT_SECONDS)
                assert directory.disabled
                assert list(panel.rows) == ["run"] and panel.selection == {
                    "run_id": "run"
                }
                await pilot.press("enter")
                assert panel.selection == {"run_id": "run"}
                release.set()
                await until(pilot, lambda: not directory.disabled)
                assert list(panel.rows) == ["request"]
        finally:
            release.set()

    asyncio.run(scenario())
