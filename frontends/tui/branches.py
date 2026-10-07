"""历史分支的分页浏览与明确继续选择。

作者：xxx
时间：2026-09-30 00:00:00
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static, Tree
from textual.widgets.tree import TreeNode

from frontends.tui.paged_text import PagedText

ENTRY_LABELS = {
    "message": "对话消息",
    "branch": "分支起点",
    "inbound": "已接纳输入",
    "delivery": "输入交接",
}


class BranchPageLoaded(Message):
    """线程只返回查询结果，所有选择和控件状态仍归界面线程。"""

    def __init__(
        self,
        kind: str,
        generation: int,
        *,
        page: dict[str, Any],
        cursors: list[Any],
        error: str = "",
    ) -> None:
        """携带查询代次及游标；参数：种类、代次、结果或错误；返回：消息。"""
        super().__init__()
        self.kind, self.generation = kind, generation
        self.page, self.cursors, self.error = page, cursors, error


class BranchScreen(ModalScreen[tuple[str, str] | None]):
    """只保存当前目录页和正文页，确认继续才返回后台动作。"""

    BINDINGS = [("escape", "dismiss(None)", "关闭历史")]
    DEFAULT_CSS = """
    BranchScreen .branch-dialog { height: 85%; width: 90%; }
    BranchScreen #branch-content { height: 1fr; }
    BranchScreen #branch-directory { width: 40%; }
    BranchScreen #branch-tree { width: 100%; height: 1fr; }
    BranchScreen #branch-history { width: 60%; }
    BranchScreen #branch-body { height: 1fr; }
    BranchScreen .branch-caption { height: auto; }
    BranchScreen .branch-pages { height: 3; }
    BranchScreen .branch-pages Button { min-width: 8; width: 1fr; }
    BranchScreen #branch-filter { height: 3; }
    """

    def __init__(
        self, snapshot: dict[str, Any], *, browse: Callable[..., dict[str, Any]]
    ) -> None:
        """保存第一页和只读查询入口；参数：摘要页、后台查询；返回：无。"""
        super().__init__()
        self.snapshot, self.browse = snapshot, browse
        self.entries: dict[str, dict[str, Any]] = {}
        self.selected_entry: str | None = None
        self.selection_anchor: str | None = None
        self.tree_view = snapshot.get("view", "turns")
        self.tree_query = ""
        self.tree_cursors = [0]
        self.history_cursors: list[str | None] = [None]
        self.history_page: dict[str, Any] = {}
        self.generations = {"tree": 0, "history": 0}

    def compose(self) -> ComposeResult:
        """将节点分页与选中分支正文分页分开；参数：无；返回：控件。"""
        with Vertical(classes="dialog branch-dialog"):
            yield Label(
                "会话分支 · 选择只查看历史，确认后从此处继续", classes="dialog-title"
            )
            with Horizontal(id="branch-content"):
                with Vertical(id="branch-directory"):
                    yield Static(
                        "",
                        id="branch-tree-caption",
                        classes="branch-caption",
                        markup=False,
                    )
                    yield Input(placeholder="筛选历史摘要", id="branch-filter")
                    yield Button("显示完整记录", id="branch-view")
                    yield Tree(Text("对话历史"), id="branch-tree")
                    with Horizontal(classes="branch-pages"):
                        yield Button(
                            "上一页节点", id="branch-tree-previous", disabled=True
                        )
                        yield Button("下一页节点", id="branch-tree-next", disabled=True)
                with Vertical(id="branch-history"):
                    yield Static(
                        "请选择历史节点",
                        id="branch-detail",
                        classes="branch-caption",
                        markup=False,
                    )
                    with Horizontal(classes="branch-pages"):
                        yield Button(
                            "更早历史", id="branch-history-older", disabled=True
                        )
                        yield Button(
                            "较新历史", id="branch-history-newer", disabled=True
                        )
                    with VerticalScroll(id="branch-body"):
                        yield PagedText("", title="已保存历史")
            with Horizontal(classes="dialog-actions"):
                yield Button(
                    "从选中位置继续",
                    id="continue-branch",
                    disabled=True,
                    variant="primary",
                )
                yield Button("关闭历史", id="close-branch")

    def on_mount(self) -> None:
        """显示已取得的一页节点，避免加载整棵树；参数：无；返回：无。"""
        self.render_tree_page()
        self.query_one(Tree).focus()

    def render_tree_page(self) -> None:
        """按本页父关系建树，页外父节点显示真实身份；参数：无；返回：无。"""
        tree = self.query_one(Tree)
        tree.clear()
        self.entries = {row["entry_id"]: row for row in self.snapshot["entries"]}
        nodes: dict[str | None, TreeNode[Any]] = {None: tree.root}
        for identity, row in self.entries.items():
            parent = row.get("display_parent_id", row["parent_id"])
            if parent not in nodes:
                nodes[parent] = tree.root.add(
                    Text("前页父节点"), data=parent, expand=True
                )
            marker = " ← 当前" if identity == self.snapshot["leaf_id"] else ""
            label = row.get("label", ENTRY_LABELS[row["type"]]) + marker
            owner = nodes[parent]
            # 1. 【会话历史】【线性平铺】只有真实可见分叉增加缩进，普通消息链保持同一阅读列
            if (
                parent in self.entries
                and self.entries[parent].get("visible_children") == 1
            ):
                owner = owner.parent or tree.root
            nodes[identity] = owner.add(Text(label), data=identity, expand=True)
        tree.root.expand()
        selection = self.snapshot.get("selected_entry_id") or self.selected_entry
        if selection in nodes:
            tree.move_cursor(nodes[selection])
        self.query_one("#branch-tree-caption", Static).update(
            f"第 {len(self.tree_cursors)} 页 · {len(self.entries)} 项 · "
            f"原始记录 {self.snapshot.get('raw_count', len(self.entries))} 条"
        )
        self.query_one("#branch-tree-previous", Button).disabled = (
            len(self.tree_cursors) == 1
        )
        self.query_one("#branch-tree-next", Button).disabled = (
            self.snapshot.get("next_after") is None
        )

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        """固定所选叶读取历史，旧查询不能覆盖新选择；参数：节点事件；返回：无。"""
        if event.node.data is None:
            return
        self.selected_entry = event.node.data
        self.selection_anchor = event.node.data
        self.history_page = {}
        body = self.query_one(PagedText)
        body.page = 0
        body.update_text("")
        self.request_history([None])

    def request_history(self, cursors: list[str | None]) -> None:
        """发出固定分支的正文查询，读取期间允许改选；参数：分页路径；返回：无。"""
        self.generations["history"] += 1
        self.query_one("#continue-branch", Button).disabled = True
        self.query_one("#branch-history-older", Button).disabled = True
        self.query_one("#branch-history-newer", Button).disabled = True
        self.query_one("#branch-detail", Static).update(
            f"读取节点：{self.selected_entry}"
        )
        # 【会话】【分支浏览】1. 翻页查询期间保留成功页，失败不会丢失阅读位置
        self.load_page(
            "history",
            self.generations["history"],
            cursors=cursors,
            options={"leaf_id": self.selected_entry, "before": cursors[-1]},
        )

    def request_tree(
        self, cursors: list[int], *, preserve_selection: bool = False
    ) -> None:
        """读取另一页目录而不累积整棵树；参数：分页路径；返回：无。"""
        self.generations["tree"] += 1
        self.query_one("#branch-tree-previous", Button).disabled = True
        self.query_one("#branch-tree-next", Button).disabled = True
        options: dict[str, Any] = {
            "after": cursors[-1],
            "view": self.tree_view,
            "query": self.tree_query,
        }
        if preserve_selection:
            options["selected_entry"] = self.selection_anchor
        self.load_page(
            "tree", self.generations["tree"], cursors=cursors, options=options
        )

    def on_input_changed(self, event: Input.Changed) -> None:
        """只筛选历史目录，空结果保留原选择和正文；参数：筛选输入；返回：无。"""
        if event.input.id == "branch-filter":
            self.tree_query = event.value
            self.request_tree([0], preserve_selection=True)

    @work(thread=True)
    def load_page(
        self, kind: str, generation: int, *, cursors: list[Any], options: dict[str, Any]
    ) -> None:
        """在线程查询真实后台，失败原样传回；参数：种类、代次、游标、查询；返回：无。"""
        try:
            page = self.browse(
                "session_tree" if kind == "tree" else "history_page",
                session_id=self.snapshot["session_id"],
                **options,
            )
        except Exception as exc:
            self.post_message(
                BranchPageLoaded(
                    kind, generation, page={}, cursors=cursors, error=str(exc)
                )
            )
            return
        self.post_message(
            BranchPageLoaded(kind, generation, page=page, cursors=cursors)
        )

    def on_branch_page_loaded(self, event: BranchPageLoaded) -> None:
        """只应用当前代次的页或错误，不抢回后来选中的分支；参数：查询结果；返回：无。"""
        if event.generation != self.generations[event.kind]:
            return
        if event.error:
            self.query_one("#branch-detail", Static).update(
                f"历史读取失败：{event.error}"
            )
            if event.kind == "tree":
                self.render_tree_page()
            elif self.history_page:
                self.query_one("#continue-branch", Button).disabled = False
                self.query_one("#branch-history-older", Button).disabled = (
                    self.history_page.get("next_before") is None
                )
                self.query_one("#branch-history-newer", Button).disabled = (
                    len(self.history_cursors) == 1
                )
            return
        if event.kind == "tree":
            self.snapshot, self.tree_cursors = event.page, event.cursors
            self.render_tree_page()
            return
        self.history_page, self.history_cursors = event.page, event.cursors
        self.query_one("#branch-detail", Static).update(
            f"查看节点：{self.selected_entry} · 历史第 {len(self.history_cursors)} 页"
        )
        body = self.query_one(PagedText)
        body.page = 0
        body.update_text(history_text(event.page["history"]))
        self.query_one("#branch-body", VerticalScroll).scroll_home(animate=False)
        self.query_one("#continue-branch", Button).disabled = False
        self.query_one("#branch-history-older", Button).disabled = (
            event.page.get("next_before") is None
        )
        self.query_one("#branch-history-newer", Button).disabled = (
            len(self.history_cursors) == 1
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """明确区分只读翻页、关闭与继续动作；参数：按钮事件；返回：无。"""
        action = event.button.id
        if action == "branch-view":
            self.tree_view = "all" if self.tree_view == "turns" else "turns"
            event.button.label = (
                "只看对话轮次" if self.tree_view == "all" else "显示完整记录"
            )
            self.request_tree([0], preserve_selection=True)
        elif action == "close-branch":
            self.dismiss(None)
        elif action == "continue-branch" and self.selected_entry is not None:
            self.dismiss((self.snapshot["session_id"], self.selected_entry))
        elif (
            action == "branch-tree-next" and self.snapshot.get("next_after") is not None
        ):
            self.request_tree([*self.tree_cursors, self.snapshot["next_after"]])
        elif action == "branch-tree-previous" and len(self.tree_cursors) > 1:
            self.request_tree(self.tree_cursors[:-1])
        elif action == "branch-history-older" and self.history_page.get("next_before"):
            self.request_history(
                [*self.history_cursors, self.history_page["next_before"]]
            )
        elif action == "branch-history-newer" and len(self.history_cursors) > 1:
            self.request_history(self.history_cursors[:-1])


def history_text(rows: list[dict[str, Any]]) -> str:
    """整理本页完整消息与工具字段，正文仍分屏排版；参数：历史页；返回：可阅读原文。"""
    parts = []
    for row in rows:
        body = [
            f"{row['role']} · {row['entry_id']} · 运行 {row.get('run_id') or '无'}",
            row["text"],
        ]
        for key, label in (
            ("reasoning", "思考"),
            ("tool_calls", "工具调用"),
            ("status", "状态"),
            ("error", "错误"),
            ("artifact_refs", "产物"),
        ):
            if row.get(key):
                value = (
                    row[key]
                    if isinstance(row[key], str)
                    else json.dumps(row[key], ensure_ascii=False, indent=2)
                )
                body.append(f"{label}：{value}")
        parts.append("\n".join(body))
    return "\n\n".join(parts) or "此位置尚无可见消息"
