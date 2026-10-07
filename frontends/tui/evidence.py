"""正式聊天界面的请求阅读和按需导出。

作者：xxx
时间：2026-09-30 19:00:00
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

from frontends.tui.paged_text import RemotePagedText

LIST_PAGE_SIZE = 30
EXPORT_POLL_SECONDS = 0.3
EXPORT_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
LEVEL_IDENTITIES = {
    "runs": "run_id",
    "requests": "request_id",
    "attempts": "attempt_id",
    "tools": "operation_id",
    "locate": "attempt_id",
}


class EvidencePanel(Vertical):
    """阅读状态局限于面板，聊天和审批继续使用原界面。"""

    BINDINGS = [("escape", "close", "收起请求记录")]

    DEFAULT_CSS = """
    EvidencePanel { height: 1fr; min-height: 8; border-top: solid #80d8c0; overflow-y: auto; }
    EvidencePanel .evidence-bar { height: 1; }
    EvidencePanel .evidence-bar Button { min-width: 6; width: auto; height: 1; border: none; padding: 0 1; }
    EvidencePanel #evidence-title, EvidencePanel #evidence-notice { height: auto; max-height: 2; }
    EvidencePanel #evidence-content { height: 1fr; min-height: 4; }
    EvidencePanel #evidence-directory { width: 35%; min-width: 15; }
    EvidencePanel #evidence-items, EvidencePanel #evidence-body { height: 1fr; }
    EvidencePanel #evidence-reading { width: 1fr; }
    EvidencePanel #evidence-export { height: auto; }
    EvidencePanel #export-path { width: 1fr; height: 1; border: none; }
    """

    class Closed(Message):
        """关闭阅读面板时将焦点交回原草稿。"""

    class ExportChanged(Message):
        """导出结果可在面板收起后继续显示给用户。"""

        def __init__(self, text: str) -> None:
            """携带实际导出状态；参数：结果文本；返回：无。"""
            super().__init__()
            self.text = text

    def __init__(
        self,
        *,
        query: Callable[..., dict[str, Any]],
        export: Callable[..., dict[str, Any]],
        data_root: Path,
    ) -> None:
        """注入共享查询服务，不访问数据库；参数：查询、导出、建议根；返回：无。"""
        super().__init__(id="evidence-panel")
        self.query_evidence, self.export_evidence, self.data_root = (
            query,
            export,
            data_root,
        )
        self.owner: dict[str, str] = {}
        self.selection: dict[str, str] = {}
        self.level = "runs"
        self.rows: dict[str, dict[str, Any]] = {}
        self.cursors: list[str | None] = [None]
        self.next_cursor: str | None = None
        self.generation = 0
        self.section = "request"
        self.job: dict[str, Any] = {}
        self._export_pending = False
        self._export_generation = 0
        self._export_scope: dict[str, Any] = {}
        self.display = False

    def compose(self) -> ComposeResult:
        """组合分页目录、按需正文和导出区域；参数：无；返回：控件。"""
        with Horizontal(classes="evidence-bar"):
            yield Button("上一级", id="evidence-up")
            yield Button("刷新", id="evidence-refresh")
            yield Button("工具", id="evidence-tools")
            yield Button("导出", id="evidence-export-open")
            yield Button("收起", id="evidence-close")
        yield Static("", id="evidence-title", markup=False)
        with Horizontal(id="evidence-content"):
            with Vertical(id="evidence-directory"):
                yield OptionList(id="evidence-items", markup=False)
                with Horizontal(classes="evidence-bar"):
                    yield Button("上一页", id="evidence-previous")
                    yield Button("下一页", id="evidence-next")
            with Vertical(id="evidence-reading"):
                with Horizontal(classes="evidence-bar"):
                    yield Button("输入", id="evidence-request")
                    yield Button("返回", id="evidence-response")
                    yield Button("来源", id="evidence-sources")
                with VerticalScroll(id="evidence-body"):
                    yield RemotePagedText(title="实际留存正文")
        yield Static("", id="evidence-notice", markup=False)
        with Vertical(id="evidence-export"):
            yield Static("", id="export-scope", markup=False)
            with Horizontal(classes="evidence-bar"):
                yield Input(placeholder="导出到新的包目录", id="export-path")
                yield Button("开始导出", id="export-start")
                yield Button("取消导出", id="export-cancel", disabled=True)
            yield Static("", id="export-status", markup=False)

    def on_mount(self) -> None:
        """隐藏尚未选择的导出区域并轮询真实任务；参数：无；返回：无。"""
        self.query_one("#evidence-export").display = False
        self.set_interval(EXPORT_POLL_SECONDS, self.poll_export)

    def open(
        self, owner: dict[str, str], selection: dict[str, str] | None = None
    ) -> None:
        """打开指定持久归属，不改变执行会话；参数：空间/会话、可选请求定位；返回：无。"""
        self.display = True
        if owner == self.owner and selection is None and self.rows:
            self.query_one("#evidence-items", OptionList).focus()
            return
        self.owner, self.selection = dict(owner), dict(selection or {})
        self.level = (
            "attempts"
            if self.selection.get("request_id")
            else ("requests" if self.selection.get("run_id") else "runs")
        )
        if any(
            self.selection.get(key) for key in ("entry_id", "message_id", "call_id")
        ):
            self.level = "locate"
        self.clear_detail()
        self.request_list([None])

    def invalidate(self) -> None:
        """空间或会话改变后作废旧查询，保留独立导出任务；参数：无；返回：无。"""
        self.generation += 1
        self.owner, self.selection, self.rows = {}, {}, {}
        self.clear_detail()
        self.display = False

    def clear_detail(self) -> None:
        """改选时清除前一对象正文，避免错认来源；参数：无；返回：无。"""
        self.query_one(RemotePagedText).set_reader(None)
        self.query_one("#evidence-body", VerticalScroll).scroll_home(animate=False)

    def request_list(self, cursors: list[str | None]) -> None:
        """分页只取摘要，选择代次同时约束成功和错误；参数：游标链；返回：无。"""
        self.generation += 1
        options: dict[str, Any] = {
            **self.owner,
            **self.selection,
            "action": self.level,
            "cursor": cursors[-1],
            "limit": LIST_PAGE_SIZE,
        }
        # 【TUI】【请求目录】1. 新层级确认前旧摘要只保留展示，不能按新身份继续下钻
        self.query_one("#evidence-items", OptionList).disabled = True
        self.query_one("#evidence-notice", Static).update("正在读取请求记录…")
        self.load_list(options, cursors, self.generation)

    @work(exclusive=True, group="evidence-list")
    async def load_list(
        self, options: dict[str, Any], cursors: list[str | None], generation: int
    ) -> None:
        """读取当前页并验证空间归属；参数：固定查询、游标、代次；返回：无。"""
        try:
            result = await asyncio.to_thread(self.query_evidence, **options)
            if (
                result["data_space_id"] != options["data_space_id"]
                or result["session_id"] != options["session_id"]
            ):
                raise ValueError("请求记录的数据空间或会话已变化，请重新连接")
        except Exception as exc:
            if generation == self.generation and self.is_mounted:
                self.query_one("#evidence-notice", Static).update(
                    f"读取失败：{exc} · 可刷新重试"
                )
            return
        if generation != self.generation or not self.is_mounted:
            return
        key = LEVEL_IDENTITIES[self.level]
        self.rows = {str(row[key]): row for row in result["items"]}
        self.cursors, self.next_cursor = cursors, result.get("next_cursor")
        options_list = self.query_one("#evidence-items", OptionList)
        options_list.clear_options().add_options(
            Option(self.row_label(row), id=identity)
            for identity, row in self.rows.items()
        )
        options_list.disabled = False
        options_list.highlighted = 0 if self.rows else None
        self.query_one("#evidence-title", Static).update(
            f"请求记录 · {self.owner['session_id']} · {' / '.join(self.selection.values()) or '全部分支运行'}"
        )
        self.query_one("#evidence-notice", Static).update(
            f"第 {len(cursors)} 页 · {len(self.rows)} 条 · 刷新可查看新记录"
            if self.rows
            else "此范围没有已留存记录"
        )
        self.query_one("#evidence-previous", Button).disabled = len(cursors) == 1
        self.query_one("#evidence-next", Button).disabled = self.next_cursor is None

    def row_label(self, row: dict[str, Any]) -> str:
        """展示后端提供的状态和来源，不推断最近请求；参数：摘要；返回：标签。"""
        key = LEVEL_IDENTITIES[self.level]
        values = [str(row[key]), str(row.get("status", "状态未记录"))]
        for field in (
            "request_index",
            "attempt_index",
            "model",
            "tool_name",
            "branch_entry_id",
            "error_category",
        ):
            if row.get(field) is not None:
                values.append(f"{field}: {row[field]}")
        return " · ".join(values)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """只沿持久身份下钻，浏览不改变执行叶；参数：所选摘要；返回：无。"""
        event.stop()
        identity = event.option.id
        if identity is None or identity not in self.rows:
            return
        self.selection = {**self.selection, LEVEL_IDENTITIES[self.level]: identity}
        self.clear_detail()
        if self.level == "locate":
            self.selection = {
                key: str(self.rows[identity][key])
                for key in ("run_id", "request_id", "attempt_id")
            }
            self.level = "attempts"
            self.request_list([None])
            self.show_section("request")
            return
        if self.level in {"attempts", "tools"}:
            self.show_section("tool_result" if self.level == "tools" else "request")
            return
        self.level = "requests" if self.level == "runs" else "attempts"
        self.request_list([None])

    def show_section(self, section: str) -> None:
        """按固定尝试或操作读取一个区块；参数：内容区块；返回：无。"""
        if not self.selection.get("attempt_id") and not self.selection.get(
            "operation_id"
        ):
            self.query_one("#evidence-notice", Static).update(
                "请先选择一次实际尝试或工具记录"
            )
            return
        self.section = section
        options = {
            **self.owner,
            **self.selection,
            "action": "detail",
            "section": section,
        }
        self.query_one(RemotePagedText).set_reader(partial(self.read_detail, options))

    def read_detail(
        self, options: dict[str, Any], offset: int, limit: int
    ) -> dict[str, Any]:
        """查询正文并核对原空间与会话；参数：固定来源、位置、长度；返回：真实正文页。"""
        result = self.query_evidence(**options, offset=offset, limit=limit)
        if (
            result["data_space_id"] != options["data_space_id"]
            or result["session_id"] != options["session_id"]
        ):
            raise ValueError("正文归属已变化，请重新打开请求记录")
        return result

    def go_up(self) -> None:
        """回到上级目录，不改变聊天阅读或草稿；参数：无；返回：无。"""
        if self.level == "runs":
            return
        if self.level in {"attempts", "locate"}:
            self.selection = {
                key: value for key, value in self.selection.items() if key == "run_id"
            }
            self.level = "requests"
        elif self.level == "tools" and self.selection.get("request_id"):
            self.selection = {
                key: value
                for key, value in self.selection.items()
                if key in {"run_id", "request_id"}
            }
            self.level = "attempts"
        else:
            self.selection, self.level = {}, "runs"
        self.clear_detail()
        self.request_list([None])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """分派只读导航及独立导出，避免事件进入聊天提交；参数：按钮；返回：无。"""
        event.stop()
        action = event.button.id or ""
        if action == "evidence-close":
            self.action_close()
        elif action == "evidence-up":
            self.go_up()
        elif action == "evidence-refresh":
            self.request_list([None])
        elif action == "evidence-next" and self.next_cursor is not None:
            self.request_list([*self.cursors, self.next_cursor])
        elif action == "evidence-previous" and len(self.cursors) > 1:
            self.request_list(self.cursors[:-1])
        elif action == "evidence-tools":
            if self.selection.get("run_id"):
                self.level = "tools"
                self.request_list([None])
        elif action in {"evidence-request", "evidence-response", "evidence-sources"}:
            section = action.removeprefix("evidence-")
            if self.level == "tools":
                section = {
                    "request": "tool_result",
                    "response": "tool_feedback",
                    "sources": "sources",
                }[section]
            self.show_section(section)
        elif action == "evidence-export-open":
            self.open_export()
        elif action == "export-start":
            self.start_export()
        elif action == "export-cancel":
            self.run_export({"action": "cancel", "job_id": self.job["job_id"]})

    def action_close(self) -> None:
        """阅读区Escape仅收起详情，原运行和导出继续；参数：无；返回：无。"""
        self.display = False
        self.post_message(self.Closed())

    def open_export(self) -> None:
        """显示明确范围和新包目录，用户确认后才生成；参数：无；返回：无。"""
        if not self.selection.get("run_id"):
            self.query_one("#evidence-notice", Static).update(
                "请先选择需要导出的运行或请求"
            )
            return
        self.query_one("#evidence-export").display = True
        self._export_scope = {**self.owner, "run_id": self.selection["run_id"]}
        if self.selection.get("request_id"):
            self._export_scope["request_id"] = self.selection["request_id"]
        identity = self._export_scope.get("request_id", self._export_scope["run_id"])
        self.query_one("#export-scope", Static).update(
            f"范围：{'请求及全部尝试' if 'request_id' in self._export_scope else '整个运行'} · {identity}\n"
            "包含 Markdown、JSON 及实际留存附件；运行中按已提交截止快照导出"
        )
        self.query_one("#export-path", Input).value = str(
            self.data_root
            / "exports"
            / f"{identity}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

    def start_export(self) -> None:
        """提交当前明确范围和目标，导出不占输入通道；参数：无；返回：无。"""
        target = self.query_one("#export-path", Input).value.strip()
        if not target or not self._export_scope:
            self.query_one("#export-status", Static).update("请选择范围并填写新包目录")
            return
        if self.job.get("status") in {"pending", "running"}:
            self.query_one("#export-status", Static).update(
                "已有导出正在生成，请完成或取消后再开始"
            )
            return
        self.run_export({**self._export_scope, "action": "start", "target_dir": target})

    def poll_export(self) -> None:
        """只轮询本面板已启动任务，不等待文件复制；参数：无；返回：无。"""
        if (
            self.job.get("job_id")
            and self.job.get("status") not in EXPORT_TERMINAL_STATES
            and not self._export_pending
        ):
            self.run_export({"action": "status", "job_id": self.job["job_id"]})

    @work(group="evidence-export")
    async def run_export(self, options: dict[str, Any]) -> None:
        """异步接纳、轮询或取消导出，真实完成后才报告成功；参数：任务动作；返回：无。"""
        if self._export_pending and options["action"] != "cancel":
            return
        self._export_pending = True
        self._export_generation += 1
        generation = self._export_generation
        self.query_one("#export-start", Button).disabled = True
        try:
            result = await asyncio.to_thread(self.export_evidence, **options)
        except Exception as exc:
            if generation == self._export_generation:
                self.query_one("#export-status", Static).update(f"导出失败：{exc}")
                self.query_one("#export-start", Button).disabled = False
            return
        finally:
            if generation == self._export_generation:
                self._export_pending = False
        if generation != self._export_generation or not self.is_mounted:
            return
        self.job = result
        status = str(result["status"])
        text = f"导出：{status} · {result.get('path', '')}"
        if result.get("cutoff"):
            text += "\n截止：" + json.dumps(result["cutoff"], ensure_ascii=False)
        if result.get("error"):
            text += f"\n原因：{result['error']}"
        self.query_one("#export-status", Static).update(text)
        terminal = status in EXPORT_TERMINAL_STATES
        self.query_one("#export-start", Button).disabled = not terminal
        self.query_one("#export-cancel", Button).disabled = terminal
        if terminal:
            self.post_message(self.ExportChanged(text))
