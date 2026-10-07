"""正式界面中阅读控件与命令草稿的焦点隔离。

作者：xxx
"""

import asyncio

from textual.containers import VerticalScroll
from textual.widgets import Button, Input

from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui
from frontends.tui.paged_text import PagedText, TEXT_PAGE_CHARACTERS
from tests.frontends.tui.test_interactive import DisplayBridge


def test_reading_page_and_copy_do_not_submit_command_draft():
    """阅读末页和复制保留命令草稿，回到输入恢复提示；参数：无；返回：无。"""

    async def scenario():
        """在正式布局中操作分页与输入焦点；参数：无；返回：无。"""
        bridge = DisplayBridge()
        app = InteractiveTui(bridge)
        source = "正文" * TEXT_PAGE_CHARACTERS + "末页真实内容"
        async with app.run_test(size=(70, 26)) as pilot:
            body = PagedText(source, title="思考过程")
            await app.query_one("#conversation", VerticalScroll).mount(body)
            editor = app.query_one(Composer)
            editor.focus()
            await pilot.press("/", "m")
            assert app.query_one("#command-hints").display
            page_input = body.query_one(Input)
            page_input.value = str(body.page_count)
            page_input.focus()
            await pilot.pause()
            assert not app.query_one("#command-hints").display
            await pilot.press("enter")
            assert body.page_text == "末页真实内容"
            copy = next(
                button for button in body.query(Button) if button.name == "copy"
            )
            copy.focus()
            await pilot.press("enter")
            assert app.clipboard == source and bridge.submitted == []
            editor.focus()
            await pilot.pause()
            assert editor.text == "/m" and app.query_one("#command-hints").display
            assert editor.region.bottom <= 26
            await pilot.press("down", "enter")
            assert editor.text in editor.commands and bridge.submitted == []

    asyncio.run(scenario())
