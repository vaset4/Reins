"""完整工具文本的按页显示，不截掉已保存内容。

作者：xxx
时间：2026-09-30 00:00:00
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rich.syntax import Syntax
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

TEXT_PAGE_CHARACTERS = 8000


class PagedText(Vertical):
    """只为当前文本页创建排版对象，完整原文仍可逐页查阅或复制。"""

    DEFAULT_CSS = """
    PagedText, PagedText .paged-body { height: auto; }
    PagedText .paged-toolbar { height: 1; color: #a6b6c5; }
    PagedText .paged-caption { width: 1fr; height: 1; }
    PagedText .paged-actions { width: auto; height: 1; }
    PagedText .paged-toolbar Button {
        min-width: 3; width: 3; height: 1; padding: 0; border: none;
        background: transparent; color: #a6b6c5;
    }
    PagedText .paged-toolbar Button:hover, PagedText .paged-toolbar Button:focus {
        background: #354452; color: #dae2eb; text-style: bold;
    }
    PagedText .paged-number {
        width: 5; height: 1; padding: 0; border: none; background: transparent;
    }
    PagedText .paged-number:focus { background: #354452; }
    PagedText .paged-total { width: auto; height: 1; }
    """

    def __init__(
        self, text: str, *, title: str, language: str | None = None, classes: str = ""
    ) -> None:
        """保存原文引用与显示方式；传参：完整文本、标题、语法及样式；返回：无。"""
        super().__init__(classes=f"paged-text {classes}")
        self.text, self.heading, self.language = text, title, language
        self.page = 0

    @property
    def page_count(self) -> int:
        """计算实际页数；传参：无；返回：含空文本显示页的页数。"""
        return max(
            1, (len(self.text) + TEXT_PAGE_CHARACTERS - 1) // TEXT_PAGE_CHARACTERS
        )

    @property
    def page_text(self) -> str:
        """只切出当前页文本；传参：无；返回：未省略的原文片段。"""
        start = self.page * TEXT_PAGE_CHARACTERS
        return self.text[start : start + TEXT_PAGE_CHARACTERS]

    def compose(self) -> ComposeResult:
        """创建一页正文及导航；传参：无；返回：轻量文本控件。"""
        with Horizontal(classes="paged-toolbar"):
            yield Static(self.heading, classes="paged-caption", markup=False)
            navigation = Horizontal(classes="paged-actions")
            navigation.display = self.page_count > 1
            with navigation:
                yield Button(
                    "‹", name="previous", classes="paged-previous", tooltip="上一页"
                )
                yield Input(
                    "1",
                    type="integer",
                    classes="paged-number",
                    tooltip="输入页码后按回车",
                )
                yield Static("", classes="paged-total", markup=False)
                yield Button("›", name="next", classes="paged-next", tooltip="下一页")
            yield Button("⧉", name="copy", tooltip="复制全文（Tab 聚焦后按回车）")
        yield Static("", classes="paged-body", markup=False)

    def on_mount(self) -> None:
        """首次只排版一页；传参：无；返回：无。"""
        self.render_page()

    def render_page(self) -> None:
        """更新当前页正文与真实范围；传参：无；返回：无。"""
        body: Syntax | Text = (
            Syntax(self.page_text, self.language, word_wrap=True)
            if self.language
            else Text(self.page_text)
        )
        self.query_one(".paged-body", Static).update(body)
        self.query_one(".paged-caption", Static).tooltip = f"共 {len(self.text):,} 字符"
        self.query_one(".paged-actions").display = self.page_count > 1
        self.query_one(".paged-total", Static).update(f"/ {self.page_count}")
        self.query_one(Input).value = str(self.page + 1)
        self.query_one(".paged-previous", Button).disabled = self.page == 0
        self.query_one(".paged-next", Button).disabled = (
            self.page + 1 == self.page_count
        )

    def update_text(self, text: str) -> None:
        """更新原文并保留仍有效的阅读页；传参：完整新文本；返回：无。"""
        if text == self.text:
            return
        self.text = text
        self.page = min(self.page, self.page_count - 1)
        self.render_page()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """翻页或明确复制完整原文；传参：按钮；返回：无。"""
        event.stop()
        if event.button.name == "copy":
            self.app.copy_to_clipboard(self.text)
            return
        direction = -1 if event.button.name == "previous" else 1
        self.page = max(0, min(self.page_count - 1, self.page + direction))
        self.render_page()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """按实际页码跳转，不把无效输入当第一页；传参：页码；返回：无。"""
        event.stop()
        try:
            page = int(event.value)
        except ValueError:
            self.notify("请输入整数页码", severity="error")
            return
        if not 1 <= page <= self.page_count:
            self.notify(f"页码范围为 1–{self.page_count}", severity="error")
            return
        self.page = page - 1
        self.render_page()


class RemotePagedText(PagedText):
    """复用文本导航，仅持有后台已提交正文的一页。"""

    def __init__(self, *, title: str) -> None:
        """创建尚未选择来源的分页正文；参数：标题；返回：无。"""
        super().__init__("", title=title)
        self.reader: Callable[[int, int], dict[str, Any]] | None = None
        self.total_chars = 0
        self._generation = 0
        self._loaded_page = 0
        self._copy_generation = 0
        self._failed_page = 0

    def compose(self) -> ComposeResult:
        """复用已有排版与页码，仅补充当前失败的明确重试；参数：无；返回：控件。"""
        yield from super().compose()
        retry = Button("重试读取此页", name="retry", classes="paged-retry")
        retry.display = False
        yield retry

    @property
    def page_count(self) -> int:
        """按后台完整长度计算页数；参数：无；返回：可查询的页数。"""
        return max(
            1, (self.total_chars + TEXT_PAGE_CHARACTERS - 1) // TEXT_PAGE_CHARACTERS
        )

    @property
    def page_text(self) -> str:
        """返回当前已读取正文页；参数：无；返回：未省略片段。"""
        return self.text

    def set_reader(self, reader: Callable[[int, int], dict[str, Any]] | None) -> None:
        """切换固定来源并使旧成功和错误失效；参数：指定来源的读取器；返回：无。"""
        self.reader = reader
        self._generation += 1
        self.query_one(".paged-retry", Button).display = False
        self._copy_generation += 1
        self.text, self.total_chars, self.page, self._loaded_page = "", 0, 0, 0
        # 【TUI】【请求阅读】1. 改选即移除旧正文，不能在新来源标题下继续显示旧材料
        super().render_page()
        self.render_page()

    def render_page(self) -> None:
        """保持成功页直到新页返回，不在界面线程读取全文；参数：无；返回：无。"""
        if self.reader is None:
            super().render_page()
            return
        self._generation += 1
        self.query_one(".paged-caption", Static).update(
            f"{self.heading} · 读取第 {self.page + 1} 页…"
        )
        self.load_page(self.reader, self.page, self._generation)

    @work(exclusive=True, group="remote-text-page")
    async def load_page(
        self, reader: Callable[[int, int], dict[str, Any]], page: int, generation: int
    ) -> None:
        """异步读取一页，失败保留最后成功正文；参数：来源、页号、代次；返回：无。"""
        try:
            result = await asyncio.to_thread(
                reader, page * TEXT_PAGE_CHARACTERS, TEXT_PAGE_CHARACTERS
            )
            text, total = result["text"], result["total_chars"]
            if not isinstance(text, str) or not isinstance(total, int) or total < 0:
                raise ValueError("请求详情未返回有效的正文和完整长度")
        except Exception as exc:
            if generation == self._generation and self.is_mounted:
                self._failed_page = page
                self.page = self._loaded_page
                super().render_page()
                self.query_one(".paged-caption", Static).update(f"读取失败：{exc}")
                self.query_one(".paged-retry", Button).display = True
            return
        if generation != self._generation or not self.is_mounted:
            return
        self.text, self.total_chars, self.page, self._loaded_page = (
            text,
            total,
            page,
            page,
        )
        super().render_page()
        self.query_one(".paged-retry", Button).display = False
        status = str(result.get("status", ""))
        retention = {
            "protected": "已按保护规则遮蔽",
            "captured": "已留存",
            "not_returned": "尚无已提交返回",
        }.get(str(result.get("retention", "")), str(result.get("retention", "")))
        self.query_one(".paged-caption", Static).update(
            f"{self.heading} · {status} · {retention} · 共 {total:,} 字符"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """沿用页码导航，全文复制显式后台分段读取；参数：按钮；返回：无。"""
        event.prevent_default()
        if event.button.name == "retry":
            event.stop()
            self.page = self._failed_page
            self.render_page()
            return
        if event.button.name != "copy":
            super().on_button_pressed(event)
            return
        event.stop()
        if self.reader is not None:
            self._copy_generation += 1
            self.copy_source(self.reader, self._copy_generation)

    @work(exclusive=True, group="remote-text-copy")
    async def copy_source(
        self, reader: Callable[[int, int], dict[str, Any]], generation: int
    ) -> None:
        """明确复制时取回全部分页，改选后不覆盖剪贴板；参数：来源和复制代次；返回：无。"""
        parts: list[str] = []
        offset = 0
        try:
            while True:
                result = await asyncio.to_thread(reader, offset, TEXT_PAGE_CHARACTERS)
                if generation != self._copy_generation or not self.is_mounted:
                    return
                parts.append(result["text"])
                offset += len(result["text"])
                if offset >= result["total_chars"]:
                    break
                if not result["text"]:
                    raise ValueError("正文分页未前进，复制未完成")
            text = await asyncio.to_thread("".join, parts)
        except Exception as exc:
            if generation == self._copy_generation and self.is_mounted:
                self.notify(f"复制失败：{exc}", severity="error")
            return
        if generation == self._copy_generation and self.is_mounted:
            self.app.copy_to_clipboard(text)
            self.notify("已复制完整已保存正文")
