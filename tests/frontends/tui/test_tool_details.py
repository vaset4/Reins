"""工具详情懒加载与完整结果分页验证。

作者：xxx
时间：2026-09-30 00:00:00
"""

import asyncio
from dataclasses import replace
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Button, Collapsible, Input, Static

import frontends.tui.widgets as widget_module
from frontends.tui.composer import Composer
from frontends.tui.paged_text import PagedText, TEXT_PAGE_CHARACTERS
from frontends.tui.projection import Card
from frontends.tui.widgets import MessageCard

DETAIL_READY_TIMEOUT_SECONDS = 5


async def wait_for_details(widget, pilot):
    """等待异步展开完成首屏排版后再发键；参数：卡片与驾驶器；返回：无。"""
    async with asyncio.timeout(DETAIL_READY_TIMEOUT_SECONDS):
        while True:
            bodies = list(widget.query(PagedText))
            if bodies and all(
                body.is_mounted and body.query_one(".paged-body", Static).render()
                for body in bodies
            ):
                await pilot.pause()
                return
            await pilot.pause()


class DetailApp(App):
    """使用生产样式挂载工具卡片和可编辑草稿。"""

    CSS_PATH = Path(widget_module.__file__).with_name("interactive.tcss")

    def __init__(self, card):
        """保存测试内容；传参：真实卡片投影；返回：无。"""
        super().__init__()
        self.card = card

    def compose(self) -> ComposeResult:
        """挂载生产控件；传参：无；返回：控件。"""
        yield MessageCard(self.card)
        yield Composer()


def test_large_tool_output_is_lazy_complete_and_keeps_page_on_updates():
    """折叠不排版全文，展开可读末页并保留更新前阅读页；传参：无；返回：无。"""
    source = "甲" * TEXT_PAGE_CHARACTERS + "乙" * TEXT_PAGE_CHARACTERS + "末尾真实内容"
    card = Card(
        "tool",
        "tool",
        text=source,
        args='{"path": "中文.txt"}',
        title="read",
        state="完成",
    )

    async def scenario():
        app = DetailApp(card)
        async with app.run_test(size=(120, 40)) as pilot:
            widget = app.query_one(MessageCard)
            details = widget.query_one(Collapsible)
            editor = app.query_one(Composer)
            editor.load_text("正在编辑的草稿")
            assert not widget.query(PagedText)
            details.collapsed = False
            await pilot.pause()
            await wait_for_details(widget, pilot)
            output = widget.query_one(".tool-output", PagedText)
            assert output.page_text == "甲" * TEXT_PAGE_CHARACTERS
            page_input = output.query_one(Input)
            page_input.value = "3"
            page_input.focus()
            await pilot.press("enter")
            assert output.page_text == "末尾真实内容"
            page_input.value = str(output.page_count + 1)
            await pilot.press("enter")
            assert output.page == 2
            await widget.update_card(replace(card, text=source + "追加结果"))
            assert output.page == 2 and output.page_text == "末尾真实内容追加结果"
            details.collapsed = True
            await pilot.pause()
            await widget.update_card(replace(card, text=source + "新的最终结果"))
            details.collapsed = False
            await pilot.pause()
            assert len(widget.query(PagedText)) == 2
            assert output.page_text == "末尾真实内容新的最终结果"
            copy = next(
                button for button in output.query(Button) if button.name == "copy"
            )
            copy.press()
            await pilot.pause()
            assert app.clipboard == source + "新的最终结果"
            assert editor.text == "正在编辑的草稿"

    asyncio.run(scenario())


def test_reasoning_is_not_rendered_until_expanded_and_uses_latest_text():
    """思考折叠时不排版，展开读取最新完整内容；传参：无；返回：无。"""

    async def scenario():
        card = Card("assistant", "assistant", text="正式回答", reasoning="初始思考")
        app = DetailApp(card)
        async with app.run_test() as pilot:
            widget = app.query_one(MessageCard)
            assert not widget.query(PagedText)
            await widget.update_card(replace(card, reasoning="更新后的完整思考"))
            widget.query_one(".reasoning", Collapsible).collapsed = False
            await pilot.pause()
            assert widget.query_one(PagedText).text == "更新后的完整思考"

    asyncio.run(scenario())


def test_short_details_have_no_pagination_and_copy_is_keyboard_accessible():
    """短思考只占一行操作区且键盘可复制全文；参数：无；返回：无。"""

    async def scenario():
        source = "短思考的完整原文"
        app = DetailApp(Card("assistant", "assistant", reasoning=source))
        async with app.run_test(size=(60, 24)) as pilot:
            app.query_one(".reasoning", Collapsible).collapsed = False
            await pilot.pause()
            await wait_for_details(app.query_one(MessageCard), pilot)
            body = app.query_one(PagedText)
            assert not body.query_one(".paged-actions").display
            assert body.query_one(".paged-toolbar").size.height == 1
            copy = next(
                button for button in body.query(Button) if button.name == "copy"
            )
            assert copy.size.width == 3 and copy.size.height == 1
            assert "复制全文" in str(copy.tooltip)
            copy.focus()
            await pilot.press("shift+tab", "tab")
            assert app.focused is copy
            await pilot.press("enter")
            assert app.clipboard == source

    asyncio.run(scenario())


def test_pagination_appears_for_growth_and_keyboard_navigation_preserves_full_copy():
    """文本跨页后显示导航，可键盘翻至末页并在缩短后收起；参数：无；返回：无。"""

    async def scenario():
        app = DetailApp(Card("assistant", "assistant", reasoning="短内容"))
        async with app.run_test(size=(60, 24)) as pilot:
            app.query_one(".reasoning", Collapsible).collapsed = False
            await pilot.pause()
            await wait_for_details(app.query_one(MessageCard), pilot)
            body = app.query_one(PagedText)
            source = "甲" * TEXT_PAGE_CHARACTERS + "末页"
            body.update_text(source)
            await pilot.pause()
            assert body.query_one(".paged-actions").display
            assert body.query_one(".paged-toolbar").size.height == 1
            body.query_one(".paged-next", Button).focus()
            await pilot.press("enter")
            assert body.page_text == "末页"
            assert body.query_one(".paged-next", Button).disabled
            copy = next(
                button for button in body.query(Button) if button.name == "copy"
            )
            copy.focus()
            await pilot.press("enter")
            assert app.clipboard == source
            body.query_one(".paged-previous", Button).focus()
            await pilot.press("enter")
            assert body.page == 0
            body.update_text("缩短后的原文")
            assert not body.query_one(".paged-actions").display
            assert body.page_text == "缩短后的原文"

    asyncio.run(scenario())
