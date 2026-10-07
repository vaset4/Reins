"""鼠标选区经过消息留白和动态布局后仍能复制。

作者：xxx
时间：2026-09-30 12:00:00
"""

from __future__ import annotations

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from frontends.tui.composer import Composer
from frontends.tui.paged_text import PagedText
from frontends.tui.projection import Card
from frontends.tui.widgets import MessageCard


class SelectionApp(App):
    """使用真实 Textual 消息循环复现文本坐标与屏幕坐标混用。"""

    CSS = """
    Screen { padding: 8 2; }
    Static { height: 1; }
    #body { margin-top: 2; }
    """

    def compose(self) -> ComposeResult:
        """构造与消息作者相同的单行标题及下方正文；参数：无；返回：控件。"""
        yield Static("运行状态", id="author", classes="message-author")
        yield Static("下面是正在生成的正文", id="body")


def test_copy_backwards_across_author_and_blank_space():
    """从正文向上拖选再经过标题留白，复制不产生屏幕坐标越界；参数：无；返回：无。"""

    async def scenario():
        app = SelectionApp()
        async with app.run_test(size=(116, 30)) as pilot:
            await pilot.mouse_down("#body", offset=(8, 0))
            await pilot.hover("#author", offset=(0, 0))
            # 1. 鼠标离开标题进入其下方空白，旧版把标题右下角屏幕坐标当作文本坐标
            await pilot.hover(offset=(20, 9))
            await pilot.mouse_up(offset=(20, 9))
            app.screen.action_copy_text()
            assert "下面" in app.clipboard

    asyncio.run(scenario())


class ConversationSelectionApp(App):
    """使用生产卡片验证流式更新和输入区复制。"""

    CSS = """
    MessageCard { height: auto; margin-bottom: 1; }
    .message-author { height: 1; }
    .message-body { height: auto; }
    Composer { height: 3; }
    """

    def compose(self) -> ComposeResult:
        """挂载消息、详情与输入框；参数：无；返回：生产控件。"""
        yield MessageCard(Card("answer", "assistant", text="中文正文持续生成"))
        yield PagedText("完整工具结果", title="结果")
        yield Composer(id="composer")


def test_selection_survives_stream_update_resize_and_card_removal():
    """流式增长、缩放和旧卡片移除后鼠标复制保持正常；参数：无；返回：无。"""

    async def scenario():
        app = ConversationSelectionApp()
        async with app.run_test(size=(90, 30)) as pilot:
            card = app.query_one(MessageCard)
            await pilot.mouse_down(".message-author", offset=(0, 0))
            await pilot.hover(".message-body", offset=(8, 0))
            await card.update_card(
                Card("answer", "assistant", text="中文正文持续生成并且增加新内容")
            )
            await pilot.resize_terminal(65, 30)
            await pilot.mouse_up(".message-body", offset=(8, 0))
            app.screen.action_copy_text()
            assert app.clipboard == "Reins\n中文正"
            # 1. 移除被选中的旧卡片后，下一次选择应只读取仍存在的工具正文
            await card.remove()
            await pilot.pause()
            await pilot.mouse_down(".paged-body", offset=(0, 0))
            await pilot.hover(".paged-body", offset=(8, 0))
            await pilot.mouse_up(".paged-body", offset=(8, 0))
            app.screen.action_copy_text()
            assert app.clipboard == "完整工具结"
            # 2. 输入框复制与显式全文复制继续使用各自现有动作
            app.screen.clear_selection()
            composer = app.query_one(Composer)
            composer.focus()
            composer.load_text("保留输入草稿")
            await pilot.press("f7", "ctrl+c")
            assert app.clipboard == "保留输入草稿"
            await pilot.click(app.query_one(Button))
            assert app.clipboard == "完整工具结果"

    asyncio.run(scenario())
