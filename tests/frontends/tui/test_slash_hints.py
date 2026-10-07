"""输入命令时的即时提示与草稿边界。

作者：xxx
时间：2026-09-30 00:10:00
"""

import asyncio
from unittest.mock import Mock

from textual.containers import VerticalScroll
from textual.widgets import Input, OptionList

from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui
from tests.frontends.tui.test_interactive import DisplayBridge


def option_text(options):
    """读取用户可见候选正文；参数：候选控件；返回：各项提示。"""
    return "\n".join(
        str(options.get_option_at_index(index).prompt)
        for index in range(options.option_count)
    )


def test_command_arrows_select_scroll_and_enter_only_fills_first():
    """方向键选择始终可见，首次回车只填草稿，再次才执行；参数：无；返回：无。"""

    async def scenario():
        bridge = DisplayBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(70, 22)) as pilot:
            editor = app.query_one(Composer)
            await pilot.press("/", *(["down"] * 12), "up")
            assert editor.hint_index == 11
            selected = editor.hint_matches[editor.hint_index]
            assert app.query_one("#command-hints-body").scroll_y > 0
            assert editor.text == "/" and editor.has_focus and not bridge.submitted
            await pilot.press("enter")
            assert editor.text == selected and not bridge.submitted
            bridge.release.set()
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            assert bridge.submitted == [selected]

    asyncio.run(scenario())


def test_bare_slash_enter_and_escape_never_submit_or_cancel():
    """孤立斜线回车先补全，Esc收起且保留草稿；参数：无；返回：无。"""

    async def scenario():
        bridge = DisplayBridge()
        bridge.cancel = Mock()
        app = InteractiveTui(bridge)
        async with app.run_test() as pilot:
            editor = app.query_one(Composer)
            await pilot.press("/", "enter")
            assert editor.text != "/" and not bridge.submitted
            editor.load_text("/m")
            await pilot.pause()
            await pilot.press("escape")
            assert editor.text == "/m" and not app.query_one("#command-hints").display
            assert not bridge.submitted
            bridge.cancel.assert_not_called()

    asyncio.run(scenario())


def test_mouse_selection_fills_without_execution_or_focus_loss():
    """点击候选只补全草稿并保持编辑焦点；参数：无；返回：无。"""

    async def scenario():
        bridge = DisplayBridge()
        app = InteractiveTui(bridge)
        async with app.run_test() as pilot:
            editor = app.query_one(Composer)
            await pilot.press("/")
            await pilot.click("#command-hints-body", offset=(2, 1))
            assert editor.text in editor.commands and editor.text != "/"
            assert editor.has_focus and not bridge.submitted
            assert not app.query_one("#command-hints").display

    asyncio.run(scenario())


def test_slash_hints_filter_without_executing_and_preserve_completion():
    """输入斜线立即列出真实命令，过滤补全均不执行；参数：无；返回：无。"""

    async def scenario():
        bridge = DisplayBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(70, 26)) as pilot:
            editor = app.query_one(Composer)
            await pilot.press("/")
            panel = app.query_one("#command-hints", VerticalScroll)
            body = app.query_one("#command-hints-body", OptionList)
            assert panel.display
            for command in bridge.commands.visible_commands():
                assert f"/{command.name}" in option_text(body)
                if command.name != "pause":
                    assert command.description in option_text(body)
            assert "/pause — 停止当前运行" in option_text(body)
            assert editor.has_focus and editor.region.bottom <= 26
            await pilot.press("m")
            assert "/model" in option_text(body)
            assert "/help" not in option_text(body)
            assert bridge.submitted == []
            await pilot.press("tab")
            first = editor.text
            await pilot.press("tab", "shift+tab")
            assert editor.text == first and bridge.submitted == []
            await pilot.press("space")
            assert not panel.display
            editor.load_text("普通草稿")
            await pilot.pause()
            assert not panel.display and bridge.submitted == []
            editor.load_text("/help")
            await pilot.pause()
            bridge.release.set()
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert bridge.submitted == ["/help"]
            assert editor.text == "" and not panel.display

    asyncio.run(scenario())


def test_slash_hints_resize_scroll_and_multiline_keep_editor_available():
    """窄屏提示可滚动且聊天和草稿仍可见，多行草稿隐藏提示；参数：无；返回：无。"""

    async def scenario():
        """通过真实控件检查缩放、滚动和多行输入；参数：无；返回：无。"""
        bridge = DisplayBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            editor = app.query_one(Composer)
            await pilot.press("/")
            await pilot.resize_terminal(70, 22)
            await pilot.pause()
            panel = app.query_one("#command-hints", VerticalScroll)
            conversation = app.query_one("#conversation", VerticalScroll)
            assert panel.display and panel.region.height <= 6
            assert conversation.region.height > 0
            assert panel.region.bottom <= editor.region.y
            assert editor.region.bottom <= 22 and editor.has_focus
            app.query_one("#command-hints-body", OptionList).scroll_end(animate=False)
            await pilot.pause()
            assert app.query_one("#command-hints-body", OptionList).scroll_y > 0
            assert editor.has_focus and editor.text == "/"
            editor.load_text("/help\n补充说明")
            await pilot.pause()
            assert not panel.display and bridge.submitted == []
            editor.load_text("")
            await pilot.pause()
            assert not panel.display

    asyncio.run(scenario())


def test_slash_hints_vim_cancel_and_history_keep_draft():
    """Vim退出编辑和历史恢复只改变草稿与提示；参数：无；返回：无。"""

    async def scenario():
        bridge = DisplayBridge()
        app = InteractiveTui(bridge)
        async with app.run_test() as pilot:
            editor = app.query_one(Composer)
            await pilot.press("f2", "i", "/", "m")
            panel = app.query_one("#command-hints", VerticalScroll)
            assert panel.display
            await pilot.press("escape")
            assert editor.text == "/m" and editor.vim_state == "normal"
            assert not panel.display
            await pilot.press("i")
            assert panel.display
            editor.set_history(["旧草稿"])
            await pilot.press("ctrl+up")
            assert not panel.display
            await pilot.press("ctrl+down")
            assert editor.text == "/m" and panel.display
            editor.load_text("/不存在")
            await pilot.pause()
            assert "没有匹配" in option_text(
                app.query_one("#command-hints-body", OptionList)
            )
            app.query_one("#session-search", Input).focus()
            await pilot.pause()
            assert not panel.display and editor.text == "/不存在"
            editor.focus()
            await pilot.pause()
            assert panel.display
            assert bridge.submitted == []

    asyncio.run(scenario())
