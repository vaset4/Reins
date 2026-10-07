"""按卡片更新的对话控件与多行输入。

作者：xxx
时间：2026-09-29 18:00:00
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Collapsible, Markdown, Static

from frontends.tui.projection import Card, ToolResultSource
from frontends.tui.paged_text import PagedText


class MessageCard(Vertical):
    """卡片保持身份和展开状态，只更新变化的正文。"""

    class InspectRequested(Message):
        """请求阅读沿已知运行和请求身份定位。"""

        def __init__(self, selection: dict[str, str]) -> None:
            """携带持久关联；参数：运行及可用请求身份；返回：无。"""
            super().__init__()
            self.selection = selection

    class RestoreRequested(Message):
        """文件恢复入口只携带真实工具关联，不推断文件版本。"""

        def __init__(self, selection: dict[str, str]) -> None:
            """保存原运行和工具身份；参数：来源；返回：消息。"""
            super().__init__()
            self.selection = selection

    def __init__(
        self,
        card: Card,
        *,
        read_result: Callable[[ToolResultSource], str] | None = None,
    ) -> None:
        """保存投影和原件读取入口；传参：消息卡片、按来源读取函数；返回：无。"""
        super().__init__(classes=f"message-card {card.role}-card")
        self.card = card
        self.read_result = read_result
        self._saved_result_page = 0

    def compose(self) -> ComposeResult:
        """根据消息种类构建控件；传参：无；返回：标题和正文。"""
        card = self.card
        if card.role in {"assistant", "tool"}:
            yield Button(
                "请求记录",
                classes="card-requests",
                disabled=not bool(card.run_id),
                tooltip="查看所属运行及实际请求"
                if card.run_id
                else "该消息未留存运行关联",
            )
        if card.role == "tool":
            yield Button(
                "文件恢复",
                classes="card-restore",
                disabled=not bool(card.run_id and card.call_id),
                tooltip="查看本次工具的恢复点；未保存原件的旧记录不可恢复",
            )
            yield Collapsible(
                title=self._tool_title(card), collapsed=True, classes="tool-details"
            )
            return
        yield Static(
            {"user": "你", "status": "运行状态"}.get(card.role, "Reins"),
            classes="message-author",
        )
        reasoning = Collapsible(title="思考过程", collapsed=True, classes="reasoning")
        reasoning.display = bool(card.reasoning)
        yield reasoning
        if card.role == "user":
            yield Static(Text(card.text), classes="message-body")
        else:
            yield Markdown(card.text, classes="message-body")

    async def update_card(self, card: Card) -> None:
        """保留组件实例与折叠状态更新内容；传参：新投影；返回：无。"""
        previous, self.card = self.card, card
        for button in self.query(".card-requests").results(Button):
            button.disabled = not bool(card.run_id)
        for button in self.query(".card-restore").results(Button):
            button.disabled = not bool(card.run_id and card.call_id)
        if card.role == "tool":
            details = self.query_one(Collapsible)
            details.title = self._tool_title(card)
            if not details.collapsed:
                for body in details.query(PagedText):
                    if body.has_class("tool-args"):
                        body.update_text(card.args)
                    elif card.result_source is None:
                        body.update_text(card.text)
            return
        if card.reasoning != previous.reasoning:
            reasoning = self.query_one(".reasoning", Collapsible)
            reasoning.display = bool(card.reasoning)
            if not reasoning.collapsed:
                for body in reasoning.query(PagedText):
                    body.update_text(card.reasoning)
        if card.text != previous.text:
            if card.role == "user":
                self.query_one(".message-body", Static).update(Text(card.text))
            else:
                await self.query_one(".message-body", Markdown).update(card.text)

    async def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        """用户展开后才排版详情，后续展开读取当前完整内容；传参：折叠控件；返回：无。"""
        event.stop()
        details = event.collapsible
        if self.card.result_source is not None and details.has_class("tool-details"):
            await self._show_saved_result(details)
            return
        bodies = list(details.query(PagedText))
        if bodies:
            for body in bodies:
                text = (
                    self.card.reasoning
                    if details.has_class("reasoning")
                    else (
                        self.card.args
                        if body.has_class("tool-args")
                        else self.card.text
                    )
                )
                body.update_text(text)
            return
        content = details.query_one(Collapsible.Contents)
        if details.has_class("reasoning"):
            await content.mount(PagedText(self.card.reasoning, title="思考过程"))
        else:
            language = (
                "diff"
                if self.card.text.startswith(("diff --git", "@@ ", "--- "))
                else None
            )
            await content.mount(
                PagedText(
                    self.card.args, title="参数", language="json", classes="tool-args"
                ),
                PagedText(
                    self.card.text,
                    title="结果",
                    language=language,
                    classes="tool-output",
                ),
            )

    async def _show_saved_result(self, details: Collapsible) -> None:
        """展开时才读取已保存原件，预览不冒充全文；参数：工具折叠区；返回：无。"""
        content = details.query_one(Collapsible.Contents)
        if not details.query(".tool-args"):
            await content.mount(
                PagedText(
                    self.card.args, title="参数", language="json", classes="tool-args"
                )
            )
        if details.query(".tool-output"):
            return
        if not details.query(".saved-result-status"):
            await content.mount(Static("", classes="saved-result-status", markup=False))
        details.query_one(".saved-result-status", Static).update(
            "正在读取已保存的完整结果…"
        )
        assert self.card.result_source is not None
        self.load_saved_result(self.card.result_source)

    @work(exclusive=True, group="saved-tool-result")
    async def load_saved_result(self, source: ToolResultSource) -> None:
        """在线程读取原件，离开卡片后的迟到结果不再展示；参数：固定来源；返回：无。"""
        text, error = "", ""
        try:
            if self.read_result is None:
                raise RuntimeError("未连接工具原件读取服务")
            text = await asyncio.to_thread(self.read_result, source)
        except Exception as exc:
            error = f"原件读取失败：{exc}；收起后重新展开可重试"
        if not self.is_mounted or self.card.result_source != source:
            return
        details = self.query_one(Collapsible)
        if details.collapsed:
            return
        status = details.query_one(".saved-result-status", Static)
        if error:
            status.update(error)
            return
        status.update("")
        language = "diff" if text.startswith(("diff --git", "@@ ", "--- ")) else None
        body = PagedText(
            text, title="完整已保存结果", language=language, classes="tool-output"
        )
        body.page = min(self._saved_result_page, body.page_count - 1)
        await details.query_one(Collapsible.Contents).mount(body)

    async def on_collapsible_collapsed(self, event: Collapsible.Collapsed) -> None:
        """收起历史原件时释放正文，仅保留阅读页码；参数：折叠事件；返回：无。"""
        if self.card.result_source is None or not event.collapsible.has_class(
            "tool-details"
        ):
            return
        event.stop()
        for body in event.collapsible.query(".tool-output").results(PagedText):
            self._saved_result_page = body.page
            await body.remove()

    @staticmethod
    def _tool_title(card: Card) -> str:
        """展示真实状态和可观察耗时；传参：工具投影；返回：折叠标题。"""
        elapsed = (
            f" · {card.elapsed:.2f}s" if card.elapsed is not None else " · 耗时未知"
        )
        return f"{card.title} · {card.state}{elapsed}"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """打开真实关联的请求，缺失关系不按时间猜测；参数：卡片按钮；返回：无。"""
        if event.button.has_class("card-restore"):
            event.stop()
            if self.card.run_id and self.card.call_id:
                self.post_message(
                    self.RestoreRequested(
                        {"run_id": self.card.run_id, "call_id": self.card.call_id}
                    )
                )
            return
        if not event.button.has_class("card-requests"):
            return
        event.stop()
        if self.card.run_id:
            selection = {"run_id": self.card.run_id}
            if self.card.role == "tool" and self.card.call_id:
                selection["call_id"] = self.card.call_id
            elif self.card.message_id:
                selection["message_id"] = self.card.message_id
            elif self.card.entry_id:
                selection["entry_id"] = self.card.entry_id
            self.post_message(self.InspectRequested(selection))
