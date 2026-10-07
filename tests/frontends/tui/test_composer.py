"""通过真实Textual按键与粘贴事件验证输入行为。

作者：xxx
时间：2026-09-29 21:30:00
"""

import asyncio

from textual import events
from textual.app import App, ComposeResult

from frontends.tui.composer import Composer


class EditorApp(App):
    """提供真实消息循环，记录发送和Esc是否继续冒泡。"""

    BINDINGS = [("f2", "mode", "模式"), ("escape", "cancel", "停止")]

    def __init__(self):
        """初始化观测值；传参：无；返回：无。"""
        super().__init__()
        self.sent = []
        self.cancelled = 0

    def compose(self) -> ComposeResult:
        """挂载生产输入控件；传参：无；返回：控件。"""
        yield Composer()

    def action_mode(self):
        """切换生产模式；传参：无；返回：无。"""
        self.query_one(Composer).toggle_vim()

    def action_cancel(self):
        """记录冒泡后的运行取消；传参：无；返回：无。"""
        self.cancelled += 1

    def on_composer_submitted(self, event):
        """记录用户发送；传参：整份草稿；返回：无。"""
        self.sent.append(event.text)


def test_history_returns_to_unsent_multiline_draft():
    """前后浏览已接纳输入后还原未发送的中文多行草稿；传参：无；返回：无。"""

    async def scenario():
        app = EditorApp()
        async with app.run_test() as pilot:
            editor = app.query_one(Composer)
            editor.load_text("未发送\n继续编辑")
            editor.set_history(["第一条", "第二条\n多行"])
            await pilot.press("ctrl+up")
            assert editor.text == "第二条\n多行"
            await pilot.press("ctrl+up", "ctrl+up")
            assert editor.text == "第一条"
            await pilot.press("ctrl+down", "ctrl+down", "ctrl+down")
            assert editor.text == "未发送\n继续编辑"
            assert app.sent == []

    asyncio.run(scenario())


def test_command_completion_cycles_without_sending_or_changing_arguments():
    """Tab循环真实候选且保留参数正文；传参：无；返回：无。"""

    async def scenario():
        app = EditorApp()
        async with app.run_test() as pilot:
            editor = app.query_one(Composer)
            editor.commands = ("/model", "/mode", "/mcp")
            editor.load_text("/m")
            await pilot.press("tab")
            assert editor.text == "/model"
            await pilot.press("tab")
            assert editor.text == "/mode"
            await pilot.press("shift+tab")
            assert editor.text == "/model"
            editor.load_text("/model 中文参数")
            await pilot.press("tab")
            assert editor.text == "/model 中文参数"
            assert app.sent == []

    asyncio.run(scenario())


def test_vim_navigation_editing_and_copy_keep_mode_boundaries():
    """导航不插入字母，编辑可撤销，Esc先回导航再取消运行；传参：无；返回：无。"""

    async def scenario():
        app = EditorApp()
        async with app.run_test() as pilot:
            editor = app.query_one(Composer)
            editor.load_text("甲乙丙\n第二行")
            await pilot.press("f2", "l", "x")
            assert editor.text == "甲丙\n第二行"
            await pilot.press("u")
            assert editor.text == "甲乙丙\n第二行"
            await pilot.press("g", "g", "y", "y")
            assert "甲乙丙" in app.clipboard
            await pilot.press("j", "d", "d")
            assert "第二行" not in editor.text
            await pilot.press("p")
            assert editor.text == "甲乙丙\n第二行"
            await pilot.press("g", "g")
            await pilot.press("A", "补", "escape")
            assert editor.text.startswith("甲乙丙补")
            assert editor.vim_state == "normal" and app.cancelled == 0
            await pilot.press("escape")
            assert app.cancelled == 1
            await pilot.press("f2")
            assert not editor.vim and "普通输入" in editor.hints
            assert app.sent == []

    asyncio.run(scenario())


def test_chinese_multiline_paste_never_submits_in_either_mode():
    """普通及Vim下的粘贴保持一个草稿，显式Enter才发送；传参：无；返回：无。"""

    async def scenario():
        app = EditorApp()
        async with app.run_test() as pilot:
            editor = app.query_one(Composer)
            editor.post_message(events.Paste("中文一\r\n中文二"))
            await pilot.pause()
            assert editor.text == "中文一\n中文二" and not app.sent
            await pilot.press("f2")
            editor.move_cursor(editor.document.end)
            editor.post_message(events.Paste("\r第三行"))
            await pilot.pause()
            assert editor.text == "中文一\n中文二\n第三行"
            assert editor.vim_state == "insert" and not app.sent
            await pilot.press("enter")
            assert app.sent == ["中文一\n中文二\n第三行"]

    asyncio.run(scenario())
