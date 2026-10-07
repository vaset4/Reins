"""【TUI】【上下文与记忆】复用后台原件查看、暂停和取消自动整理。

作者：xxx
时间：2026-10-01 12:35:00
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from typing import Any

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option

from frontends.tui.paged_text import RemotePagedText

PAGE_SIZE = 20
STATE_LABELS = {
    "queued": "等待执行",
    "running": "正在执行",
    "published": "历史已发布",
    "completed": "知识已更新",
    "no_op": "已核验，无需保存",
    "failed": "失败",
    "cancelled": "已取消",
    "cancelling": "正在取消",
}
CANCELLABLE_STATES = frozenset({"queued", "running"})


class ContextPanel(Vertical):
    """只保存阅读位置和查询代次，关闭面板不取消后台工作。"""

    BINDINGS = [("escape", "close", "收起上下文与记忆")]
    DEFAULT_CSS = """
    ContextPanel { height: 1fr; min-height: 8; border-top: solid #80d8c0; }
    ContextPanel .context-bar { height: auto; }
    ContextPanel .context-bar Button { width: auto; min-width: 6; height: 1; border: none; padding: 0 1; }
    ContextPanel #context-status, ContextPanel #context-notice { height: auto; max-height: 3; }
    ContextPanel #context-content { height: 1fr; }
    ContextPanel #context-directory { width: 35%; min-width: 15; }
    ContextPanel #context-items, ContextPanel #context-body { height: 1fr; }
    ContextPanel #context-reading { width: 1fr; }
    """

    class Closed(Message):
        """收起面板后把焦点交回草稿。"""

    class RequestSelected(Message):
        """将后台工作的实际请求交给现有请求查看器。"""

        def __init__(self, source: dict[str, str]) -> None:
            """携带已核对的请求归属；参数：空间、会话与运行；返回：无。"""
            super().__init__()
            self.source = source

    def __init__(self, *, query: Callable[..., dict[str, Any]]) -> None:
        """注入后台查询，界面不读取存储；参数：查询函数；返回：无。"""
        super().__init__(id="context-panel")
        self.query_context = query
        self.owner: dict[str, str] = {}
        self.rows: dict[str, dict[str, Any]] = {}
        self.selection: dict[str, Any] = {}
        self.execution: dict[str, str] | None = None
        self.enabled: dict[str, bool] = {}
        self.cursors: list[list[str] | None] = [None]
        self.next_before: list[str] | None = None
        self.generation = 0
        self.overview_generation = 0
        self.detail_generation = 0
        self.control_generation = 0
        self.control_pending = False
        self.display = False

    def compose(self) -> ComposeResult:
        """组织独立开关、工作目录和版本详情；参数：无；返回：控件。"""
        with Horizontal(classes="context-bar"):
            yield Button("历史整理", id="context-history", disabled=True)
            yield Button("知识维护", id="context-knowledge", disabled=True)
            yield Button("刷新", id="context-refresh")
            yield Button("收起", id="context-close")
        yield Static("正在读取设置…", id="context-status", markup=False)
        with Horizontal(id="context-content"):
            with Vertical(id="context-directory"):
                yield OptionList(id="context-items", markup=False)
                with Horizontal(classes="context-bar"):
                    yield Button("上一页", id="context-previous", disabled=True)
                    yield Button("下一页", id="context-next", disabled=True)
            with Vertical(id="context-reading"):
                with Horizontal(classes="context-bar"):
                    yield Button("取消所选工作", id="context-cancel", disabled=True)
                    yield Button(
                        "查看模型请求及用量", id="context-requests", disabled=True
                    )
                with VerticalScroll(id="context-body"):
                    yield RemotePagedText(title="来源、处理结果与记忆版本")
        yield Static("", id="context-notice", markup=False)

    def open(self, owner: dict[str, str]) -> None:
        """打开所选会话，原草稿和执行叶保持；参数：持久空间与会话；返回：无。"""
        if owner != self.owner:
            self.invalidate()
        self.owner = dict(owner)
        self.display = True
        self.refresh_view()

    def invalidate(self) -> None:
        """会话切换使旧回复失效；参数：无；返回：无，不取消后端工作。"""
        self.generation += 1
        self.overview_generation += 1
        self.detail_generation += 1
        self.control_generation += 1
        self.control_pending = False
        self.enabled = {}
        self.owner, self.rows, self.selection, self.execution = {}, {}, {}, None
        for domain in ("history", "knowledge"):
            self.query_one(f"#context-{domain}", Button).disabled = True
        self.query_one(RemotePagedText).set_reader(None)
        self.display = False

    def refresh_view(self) -> None:
        """刷新设置和目录，不把旧阅读当当前状态；参数：无；返回：无。"""
        self.request_list([None])
        self.overview_generation += 1
        self.load_overview(dict(self.owner), self.overview_generation)

    def request_list(self, cursors: list[list[str] | None]) -> None:
        """以稳定创建游标取一页工作；参数：页面栈；返回：无。"""
        self.generation += 1
        self.detail_generation += 1
        self.selection, self.execution = {}, None
        self.query_one(RemotePagedText).set_reader(None)
        self.query_one("#context-items", OptionList).disabled = True
        self.query_one("#context-cancel", Button).disabled = True
        self.query_one("#context-requests", Button).disabled = True
        self.load_list(
            {**self.owner, "action": "list", "before": cursors[-1], "limit": PAGE_SIZE},
            cursors,
            self.generation,
        )

    @work(exclusive=True, group="context-overview")
    async def load_overview(self, owner: dict[str, str], generation: int) -> None:
        """读取真实开关，失败不显示猜测的状态；参数：原范围与代次；返回：无。"""
        try:
            result = await asyncio.to_thread(self.read, {**owner, "action": "overview"})
        except Exception as exc:
            if generation == self.overview_generation:
                self.show_error(exc, self.generation)
            return
        if generation != self.overview_generation or not self.is_mounted:
            return
        self.enabled = {
            domain: result[domain]["enabled"] for domain in ("history", "knowledge")
        }
        for domain, title in (("history", "历史整理"), ("knowledge", "知识维护")):
            button = self.query_one(f"#context-{domain}", Button)
            button.label = ("暂停" if self.enabled[domain] else "启用") + title
            button.disabled = self.control_pending
        status = (
            "开关作用于当前数据空间。暂停后已接纳工作继续；历史原文和已有记忆仍可使用。"
        )
        admission = result["knowledge"].get("admission")
        if admission is not None and admission["state"] == "not_accepted":
            status += "\n知识维护尚未接纳：当前模型使用临时凭据，需保存模型凭据；主聊天可正常继续。"
        self.query_one("#context-status", Static).update(status)

    @work(exclusive=True, group="context-list")
    async def load_list(
        self, options: dict[str, Any], cursors: list[list[str] | None], generation: int
    ) -> None:
        """应用当前页且拒绝旧会话回复；参数：范围、游标与代次；返回：无。"""
        try:
            result = await asyncio.to_thread(self.read, options)
        except Exception as exc:
            self.show_error(exc, generation)
            return
        if generation != self.generation or not self.is_mounted:
            return
        self.rows = {row["work_id"]: row for row in result["items"]}
        self.cursors, self.next_before = cursors, result["next_before"]
        items = self.query_one("#context-items", OptionList)
        items.clear_options().add_options(
            Option(
                f"{row['title']} · {STATE_LABELS.get(row['state'], row['state'])}\n"
                f"{row['source_count']} 条来源 · {row['created_at']}",
                id=identity,
            )
            for identity, row in self.rows.items()
        )
        items.highlighted = 0 if self.rows else None
        items.disabled = False
        self.query_one("#context-previous", Button).disabled = len(cursors) == 1
        self.query_one("#context-next", Button).disabled = self.next_before is None
        self.query_one("#context-notice", Static).update(
            f"第 {len(cursors)} 页 · 共 {result['total']} 项；刷新可查看最新状态"
        )

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """按实际工作身份展开详情；参数：目录选择；返回：无。"""
        event.stop()
        if event.option.id not in self.rows:
            return
        row = self.rows[event.option.id]
        self.detail_generation += 1
        self.selection = {"domain": row["domain"], "work_id": row["work_id"]}
        self.execution = None
        self.query_one("#context-cancel", Button).disabled = (
            self.control_pending or row["state"] not in CANCELLABLE_STATES
        )
        self.query_one("#context-requests", Button).disabled = True
        self.query_one(RemotePagedText).set_reader(None)
        self.load_detail(
            {**self.owner, **self.selection, "action": "detail"}, self.detail_generation
        )

    @work(exclusive=True, group="context-detail")
    async def load_detail(self, options: dict[str, Any], generation: int) -> None:
        """先固定提交位置再提供分页正文；参数：工作来源与代次；返回：无。"""
        try:
            result = await asyncio.to_thread(
                self.read, {**options, "offset": 0, "limit": 1}
            )
        except Exception as exc:
            if generation == self.detail_generation:
                self.show_error(exc, self.generation)
            return
        if generation != self.detail_generation or not self.is_mounted:
            return
        self.execution = result["execution"]
        self.query_one("#context-requests", Button).disabled = self.execution is None
        self.query_one(RemotePagedText).set_reader(
            partial(self.read_detail, {**options, "commit": result["commit"]})
        )

    def read_detail(
        self, options: dict[str, Any], offset: int, limit: int
    ) -> dict[str, Any]:
        """读取固定工作及记忆版本正文；参数：来源与页范围；返回：后端正文页。"""
        return self.read({**options, "offset": offset, "limit": limit})

    def read(self, options: dict[str, Any]) -> dict[str, Any]:
        """核对后台返回的空间和会话；参数：固定请求；返回：真实结果。"""
        result = self.query_context(**options)
        if any(result[key] != options[key] for key in ("data_space_id", "session_id")):
            raise ValueError("上下文记录归属已变化，请重新打开")
        return result

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """只转发用户明确选择，关闭不影响工作；参数：按钮事件；返回：无。"""
        identity = event.button.id or ""
        if not identity.startswith("context-"):
            return
        event.stop()
        if identity == "context-close":
            self.action_close()
        elif identity == "context-refresh":
            self.refresh_view()
        elif identity == "context-previous" and len(self.cursors) > 1:
            self.request_list(self.cursors[:-1])
        elif identity == "context-next" and self.next_before is not None:
            self.request_list([*self.cursors, self.next_before])
        elif identity == "context-requests" and self.execution:
            self.post_message(
                self.RequestSelected(
                    {"data_space_id": self.owner["data_space_id"], **self.execution}
                )
            )
        elif identity == "context-cancel" and self.selection:
            self.start_control({**self.owner, **self.selection, "action": "cancel"})
        elif identity in {"context-history", "context-knowledge"}:
            domain = identity.removeprefix("context-")
            if domain in self.enabled:
                self.start_control(
                    {
                        **self.owner,
                        "action": "configure",
                        "domain": domain,
                        "enabled": not self.enabled[domain],
                    }
                )

    def start_control(self, options: dict[str, Any]) -> None:
        """固定控制对象并防止重复点击；参数：明确动作；返回：无。"""
        if self.control_pending:
            return
        self.control_pending = True
        self.control_generation += 1
        for identity in ("history", "knowledge", "cancel"):
            self.query_one(f"#context-{identity}", Button).disabled = True
        self.apply_control(options, self.generation, self.control_generation)

    @work(group="context-control")
    async def apply_control(
        self, options: dict[str, Any], generation: int, control_generation: int
    ) -> None:
        """控制与阅读各自执行，旧回执不覆盖新选择；参数：原动作与代次；返回：无。"""
        try:
            await asyncio.to_thread(self.read, options)
        except Exception as exc:
            self.show_error(exc, generation)
            return
        finally:
            # 1. 【TUI】【控制归属】换会话不取消已接纳控制，旧回执也不能解锁新会话的控制
            if control_generation == self.control_generation:
                self.control_pending = False
        if control_generation == self.control_generation and self.is_mounted:
            if generation == self.generation:
                self.refresh_view()
            else:
                # 2. 【TUI】【控制归属】用户已翻页或选择新工作时只更新开关，保留新的阅读位置
                self.overview_generation += 1
                self.load_overview(dict(self.owner), self.overview_generation)
                selected = self.rows.get(self.selection.get("work_id", ""))
                self.query_one("#context-cancel", Button).disabled = (
                    selected is None or selected["state"] not in CANCELLABLE_STATES
                )

    def show_error(self, error: Exception, generation: int) -> None:
        """展示当前查询错误，不伪装为空目录或未执行；参数：异常与代次；返回：无。"""
        if generation == self.generation and self.is_mounted:
            self.query_one("#context-notice", Static).update(
                f"操作未完成：{error} · 可刷新核对实际状态"
            )

    def action_close(self) -> None:
        """收起阅读并保持所有后台工作；参数：无；返回：无。"""
        self.invalidate()
        self.post_message(self.Closed())
