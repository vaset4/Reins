"""全屏界面的草稿、浮层与线程交互回归。

作者：xxx
时间：2026-09-29 21:00:00
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from threading import Event
from pathlib import Path

from app.repl.slash_commands import create_default_registry
from frontends.tui.interactive import InteractiveTui
from frontends.tui.composer import Composer
from textual.widgets import Input


class DisplayBridge:
    """仅替换外部后台连接，测试真实控件与消息循环。"""

    def __init__(self):
        """初始化受控接纳边界；传参：无；返回：无。"""
        self.commands = create_default_registry()
        self.submitted = []
        self.entered = Event()
        self.release = Event()
        self.project_root = Path("test-workspace")
        self.data_root = Path("test-data")

    def query_evidence(self, **params):
        """提供受控只读空请求目录；参数：查询；返回：对应空间的空页。"""
        return {
            "items": [],
            "next_cursor": None,
            "data_space_id": params["data_space_id"],
            "session_id": params["session_id"],
        }

    def export_evidence(self, **params):
        """未注入导出时明确暴露错误；参数：导出动作；返回：不返回。"""
        raise RuntimeError("测试未注入导出服务")

    def query_file_restore(self, **params):
        """提供独立的空恢复目录；参数：查询；返回：目录或无作业状态。"""
        return (
            {}
            if params["action"] == "status"
            else {"points": [], "turns": [], "next_cursor": None}
        )

    def execute_file_restore(self, **params):
        """未注入恢复执行时明确报错；参数：确认；返回：不返回。"""
        raise RuntimeError("测试未注入文件恢复执行服务")

    def cancel_file_restore(self, **params):
        """未注入恢复作业时明确报错；参数：操作身份；返回：不返回。"""
        raise RuntimeError("测试未注入文件恢复作业")

    def start(self):
        """测试无需启动模型；传参：无；返回：无。"""

    def model_label(self):
        """返回明确测试模型标签；传参：无；返回：标签。"""
        return "测试模型"

    def require_host(self):
        """提供受控后台边界；传参：无；返回：自身。"""
        return self

    def browse(self, method, **params):
        """返回空目录；传参：查询；返回：目录快照。"""
        return {"sessions": []}

    def submit(self, text):
        """在明确释放后接纳；传参：用户原文；返回：不退出。"""
        self.submitted.append(text)
        self.entered.set()
        assert self.release.wait(5)
        return False


def test_multiline_draft_survives_prompt_resize_and_session_switch():
    """中文多行草稿经过追问、缩放及会话切换仍保留；传参：无；返回：无。"""

    async def scenario():
        app = InteractiveTui(DisplayBridge())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.receive(
                "snapshot", {"session_id": "one", "status": "idle", "history": []}
            )
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("中文第一行\n第二行")
            answer = Future()
            app.receive("prompt", {"label": "请选择模型", "answer": answer})
            await pilot.pause()
            app.screen.query_one(Input).value = "选择甲"
            await pilot.press("enter")
            await pilot.pause()
            assert answer.result() == "选择甲"
            assert composer.text == "中文第一行\n第二行"
            await pilot.resize_terminal(70, 22)
            app.receive(
                "snapshot", {"session_id": "two", "status": "idle", "history": []}
            )
            await pilot.pause()
            composer.load_text("另一个草稿")
            app.receive(
                "snapshot", {"session_id": "one", "status": "idle", "history": []}
            )
            await pilot.pause()
            assert composer.text == "中文第一行\n第二行"
            assert app.drafts["two"] == "另一个草稿"

    asyncio.run(scenario())


def test_submit_preserves_edits_made_while_acceptance_is_pending():
    """等待接纳时重复 Enter 不重投，新编辑不被回执清除；传参：无；返回：无。"""

    async def scenario():
        bridge = DisplayBridge()
        app = InteractiveTui(bridge)
        async with app.run_test() as pilot:
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("发送原文")
            await pilot.press("enter", "enter")
            assert await asyncio.to_thread(bridge.entered.wait, 2)
            composer.load_text("下一份草稿")
            bridge.release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert bridge.submitted == ["发送原文"]
            assert composer.text == "下一份草稿"

    asyncio.run(scenario())


def test_first_connection_keeps_text_typed_during_startup():
    """后台首次返回会话身份时保留连接期间草稿；传参：无；返回：无。"""

    async def scenario():
        app = InteractiveTui(DisplayBridge())
        async with app.run_test() as pilot:
            composer = app.query_one(Composer)
            composer.load_text("正在连接时写下的草稿")
            app.receive(
                "snapshot", {"session_id": "first", "status": "idle", "history": []}
            )
            await pilot.pause()
            assert composer.text == "正在连接时写下的草稿"

    asyncio.run(scenario())


def test_reconnected_snapshot_restores_submission_without_losing_draft():
    """重新握手取得快照后恢复发送，期间草稿保留；传参：无；返回：无。"""

    async def scenario():
        bridge = DisplayBridge()
        bridge.release.set()
        app = InteractiveTui(bridge)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.receive(
                "snapshot",
                {"session_id": "s", "event_epoch": "old", "history": [], "cursor": 7},
            )
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("断开期间草稿")
            app.receive("disconnected", "网络中断")
            await pilot.pause()
            assert not app.connected
            app.receive(
                "snapshot",
                {"session_id": "s", "event_epoch": "new", "history": [], "cursor": 7},
            )
            await pilot.pause()
            assert app.connected and composer.text == "断开期间草稿"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert bridge.submitted == ["断开期间草稿"]

    asyncio.run(scenario())


def test_complete_command_output_remains_readable_after_later_notices():
    """长命令回执不会被提示区裁切或后续回执覆盖，查看时草稿保留；传参：无；返回：无。"""
    from frontends.tui.paged_text import PagedText

    async def scenario():
        app = InteractiveTui(DisplayBridge())
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("继续写的草稿")
            original = "\n".join(f"配置项{index}" for index in range(30))
            app.receive("output", original)
            app.receive("output", "后续连接状态")
            await pilot.pause()
            await pilot.press("f8")
            await pilot.pause()
            text = app.screen.query_one(PagedText).text
            assert original in text and "后续连接状态" in text
            await pilot.press("escape")
            await pilot.pause()
            assert composer.text == "继续写的草稿"

    asyncio.run(scenario())
