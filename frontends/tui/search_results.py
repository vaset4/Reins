"""【TUI】【历史搜索】分页查看命中原件，显式选择才切换会话。

作者：xxx
时间：2026-09-30 16:00:00
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, OptionList, Static
from textual.widgets.option_list import Option

from frontends.tui.paged_text import RemotePagedText


class SearchResultsScreen(ModalScreen[str | None]):
    """搜索阅读范围独立于正在执行的会话。"""

    BINDINGS = [("escape", "dismiss(None)", "关闭搜索")]
    DEFAULT_CSS = """
    SearchResultsScreen .search-dialog { width: 90%; height: 90%; }
    SearchResultsScreen #search-matches { height: 7; }
    SearchResultsScreen #search-original { height: 1fr; }
    SearchResultsScreen #search-source { height: auto; max-height: 4; }
    SearchResultsScreen .search-pages { height: 3; }
    """

    def __init__(
        self, row: dict[str, Any], query: str, *, browse: Callable[..., dict[str, Any]]
    ) -> None:
        """冻结检索归属；参数：目录行、查询和后台读取器；返回：无。"""
        super().__init__()
        self.row, self.query_text, self.browse = row, query, browse
        self.matches: list[dict[str, Any]] = []
        self.cursors = [0]
        self.next_after: int | None = None
        self.generation = 0

    def compose(self) -> ComposeResult:
        """复用已有远程正文分页；参数：无；返回：命中目录和只读正文控件。"""
        with Vertical(classes="dialog search-dialog"):
            yield Label(Text(f"搜索原件 · {self.row['title']}"), classes="dialog-title")
            yield Static(
                Text(
                    f"查询：{self.query_text}\n工作区：{self.row.get('project_root') or '未记录'}"
                )
            )
            yield OptionList(id="search-matches")
            with Horizontal(classes="search-pages"):
                yield Button("上一页命中", id="search-previous", disabled=True)
                yield Button("下一页命中", id="search-next", disabled=True)
            yield Static("正在读取命中位置…", id="search-source", markup=False)
            retry = Button("重试查询", id="search-retry")
            retry.display = False
            yield retry
            with VerticalScroll(id="search-original"):
                yield RemotePagedText(title="命中原件")
            with Horizontal(classes="dialog-actions"):
                yield Button(
                    "打开所属会话", id="search-open-session", variant="primary"
                )
                yield Button("关闭", id="search-close")

    def on_mount(self) -> None:
        """读取第一页命中，不切换当前会话；参数：无；返回：无。"""
        self.request_matches([0])

    def request_matches(self, cursors: list[int]) -> None:
        """冻结本次分页路径；参数：游标栈；返回：无，旧结果被代次隔离。"""
        self.generation += 1
        self.query_one("#search-matches", OptionList).disabled = True
        self.query_one("#search-previous", Button).disabled = True
        self.query_one("#search-next", Button).disabled = True
        self.query_one("#search-retry", Button).display = False
        self.load_matches(cursors, self.generation)

    @work(thread=True)
    def load_matches(self, cursors: list[int], generation: int) -> None:
        """在后台查命中页；参数：游标与代次；返回：通过界面线程应用结果。"""
        try:
            result = self.browse(
                "session_search",
                action="matches",
                session_id=self.row["session_id"],
                query=self.query_text,
                after=cursors[-1],
            )
        except Exception as exc:
            result = {"error": str(exc)}
        if self.is_mounted:
            self.app.call_from_thread(self.apply_matches, result, cursors, generation)

    def apply_matches(
        self, result: dict[str, Any], cursors: list[int], generation: int
    ) -> None:
        """更新当前检索目录；参数：后台结果、分页路径、代次；返回：无，失败保留已有正文。"""
        if not self.is_mounted or generation != self.generation:
            return
        if "error" in result:
            self.query_one("#search-source", Static).update(
                f"读取失败：{result['error']}"
            )
            self.query_one("#search-retry", Button).display = True
            return
        self.matches, self.cursors = result["matches"], cursors
        self.next_after = result.get("next_after")
        options = self.query_one("#search-matches", OptionList)
        options.clear_options().add_options(
            Option(
                Text(f"{row['label']} · {row.get('entry_id') or row['record_id']}"),
                id=str(index),
            )
            for index, row in enumerate(self.matches)
        )
        options.disabled = False
        self.query_one("#search-previous", Button).disabled = len(cursors) == 1
        self.query_one("#search-next", Button).disabled = self.next_after is None
        if self.matches:
            self.select_source(self.matches[0])
        else:
            self.query_one("#search-source", Static).update("当前检索没有可读命中")

    def select_source(self, source: dict[str, Any]) -> None:
        """以固定来源和截止点查看正文；参数：命中位置；返回：无，不更改运行叶。"""
        self.query_one("#search-source", Static).update(
            f"{source['label']}\n原件：{source['source_path']} · 字节位置 {source['source_offset']}"
        )
        self.query_one(RemotePagedText).set_reader(
            partial(self.read_source, dict(source))
        )

    def read_source(
        self, source: dict[str, Any], offset: int, limit: int
    ) -> dict[str, Any]:
        """转交原件字符页读取；参数：冻结来源、位置、长度；返回：正文和完整长度。"""
        return self.browse(
            "session_search",
            action="content",
            source=source,
            offset=offset,
            limit=limit,
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """只切换所读原件；参数：命中选项；返回：无。"""
        if event.option_list.id == "search-matches" and event.option.id is not None:
            event.stop()
            self.select_source(self.matches[int(event.option.id)])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """分别处理阅读分页与明确打开会话；参数：按钮；返回：无。"""
        identity = event.button.id
        if identity == "search-close":
            self.dismiss(None)
        elif identity == "search-open-session":
            self.dismiss(self.row["session_id"])
        elif identity == "search-next" and self.next_after is not None:
            self.request_matches([*self.cursors, self.next_after])
        elif identity == "search-previous" and len(self.cursors) > 1:
            self.request_matches(self.cursors[:-1])
        elif identity == "search-retry":
            self.request_matches(self.cursors)
