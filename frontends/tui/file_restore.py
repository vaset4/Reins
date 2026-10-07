"""独立文件恢复的选择、预览、明确确认与后台进度。

作者：xxx
时间：2026-09-30 19:45:00
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
from textual.widgets import Button, Checkbox, OptionList, Static
from textual.widgets.option_list import Option

from frontends.tui.paged_text import RemotePagedText

RESTORE_PAGE_SIZE = 30
RESTORE_POLL_SECONDS = 0.5
TERMINAL_STATES = frozenset(
    {"completed", "partial", "cancelled", "failed", "needs_reconciliation"}
)
CHOICE_LABELS = {"restore": "恢复", "keep": "保留", "copy": "另存副本"}
STATE_LABELS = {
    "pending": "待执行",
    "started": "执行中",
    "restored": "已恢复",
    "unchanged": "无需修改",
    "conflict": "有后续修改",
    "failed": "失败",
    "cancelled": "已取消",
    "unknown": "结果待核对",
    "completed": "完成",
    "partial": "部分完成",
    "needs_reconciliation": "需要核对",
    "running": "进行中",
    "unavailable": "无法恢复",
    "ready": "可恢复",
    "source_unknown": "来源待确认",
    "unconfirmed": "来源待确认",
    "queued": "等待执行",
    "complete": "捕获完整",
    "incomplete": "恢复记录不完整",
    "before_saved": "已保存修改前版本",
    "legacy": "仅有变化记录，无法恢复",
}


class FileRestorePanel(Vertical):
    """界面保存选择和阅读状态，文件操作与作业生命周期归后台。"""

    BINDINGS = [("escape", "close", "收起文件恢复")]
    DEFAULT_CSS = """
    FileRestorePanel { height: 1fr; min-height: 12; border-top: solid #80d8c0; overflow-y: auto; }
    FileRestorePanel .restore-bar { height: 1; }
    FileRestorePanel .restore-bar Button { min-width: 6; width: auto; height: 1; border: none; padding: 0 1; }
    FileRestorePanel #restore-title, FileRestorePanel #restore-notice { height: auto; max-height: 3; }
    FileRestorePanel #restore-content { height: 1fr; min-height: 5; }
    FileRestorePanel #restore-directory { width: 42%; min-width: 15; }
    FileRestorePanel #restore-items { height: 1fr; }
    FileRestorePanel #restore-reading { width: 1fr; height: 1fr; }
    FileRestorePanel #restore-confirmation { height: auto; }
    FileRestorePanel Checkbox { height: auto; border: none; padding: 0; }
    FileRestorePanel #restore-progress { height: auto; max-height: 5; overflow-y: auto; }
    """

    class Closed(Message):
        """面板收起后焦点回到聊天草稿。"""

    class OperationChanged(Message):
        """收起面板后仍可收到实际作业终态。"""

        def __init__(self, text: str) -> None:
            """保存后台结果文字；参数：逐项结果；返回：消息。"""
            super().__init__()
            self.text = text

    def __init__(
        self,
        *,
        query: Callable[..., dict[str, Any]],
        execute: Callable[..., dict[str, Any]],
        cancel: Callable[..., dict[str, Any]],
    ) -> None:
        """注入共享后台接口；参数：查询、确认执行及取消；返回：恢复面板。"""
        super().__init__(id="file-restore-panel")
        self.query_restore, self.execute_restore, self.cancel_restore = (
            query,
            execute,
            cancel,
        )
        self.owner: dict[str, str] = {}
        self.source: dict[str, str] = {}
        self.filters: dict[str, str] = {}
        self.rows: dict[str, dict[str, Any]] = {}
        self.choices: dict[str, str] = {}
        self.plan: dict[str, Any] = {}
        self.job: dict[str, Any] = {}
        self.job_owner: dict[str, str] = {}
        self.cursors: list[Any] = [None]
        self.next_cursor: Any = None
        self.generation = 0
        self._operation_generation = 0
        self._pending = False
        self._operation_pending = False
        self._operation_action = ""
        self.display = False

    def compose(self) -> ComposeResult:
        """组合文件选择和明确确认，保留聊天输入空间；参数：无；返回：控件。"""
        with Horizontal(classes="restore-bar"):
            yield Button("恢复点", id="restore-up")
            yield Button("刷新", id="restore-refresh")
            yield Button("收起", id="restore-close")
        yield Static("", id="restore-title", markup=False)
        with Horizontal(id="restore-content"):
            with Vertical(id="restore-directory"):
                yield OptionList(id="restore-items", markup=False)
                with Horizontal(classes="restore-bar"):
                    yield Button("上一页", id="restore-previous", disabled=True)
                    yield Button("下一页", id="restore-next", disabled=True)
            with VerticalScroll(id="restore-reading"):
                yield RemotePagedText(title="当前文件与恢复目标差异")
        with Horizontal(classes="restore-bar"):
            yield Button("选中恢复", id="restore-choice-restore", disabled=True)
            yield Button("保留当前", id="restore-choice-keep", disabled=True)
            yield Button("另存副本", id="restore-choice-copy", disabled=True)
            yield Button("预览所选", id="restore-preview", disabled=True)
        yield Static(
            "请选择恢复点；查看和预览不会修改文件", id="restore-notice", markup=False
        )
        with Vertical(id="restore-confirmation"):
            yield Static("", id="restore-impact", markup=False)
            yield Checkbox("我确认按以上预览恢复选中文件", id="restore-confirm")
            yield Checkbox(
                "确认本次受保护配置恢复；密文仅保证当前 Windows 账户可解密",
                id="restore-sensitive",
            )
        with Horizontal(classes="restore-bar"):
            yield Button("执行已确认恢复", id="restore-execute", disabled=True)
            yield Button("取消剩余文件", id="restore-cancel", disabled=True)
            yield Button("刷新进度", id="restore-status")
        yield Static("", id="restore-progress", markup=False)

    def on_mount(self) -> None:
        """进度查询与目录阅读分开，不随面板收起停止作业；参数：无；返回：无。"""
        self.query_one("#restore-confirmation").display = False
        self.set_interval(RESTORE_POLL_SECONDS, self.poll_operation)

    def open(
        self, owner: dict[str, str], selection: dict[str, str] | None = None
    ) -> None:
        """绑定原会话与空间后浏览恢复点；参数：持久归属、可选工具来源；返回：无。"""
        self.invalidate()
        self.owner, self.filters = dict(owner), dict(selection or {})
        self.display = True
        self.refresh_list([None])
        self.query_one("#restore-items", OptionList).focus()
        if not self.job or self.job_owner != self.owner:
            self.request_operation("status", dict(self.owner))

    def invalidate(self) -> None:
        """切换会话时废弃旧预览和晚到回复，不取消后台作业；参数：无；返回：无。"""
        self.generation += 1
        self.source, self.rows, self.choices, self.plan = {}, {}, {}, {}
        self._pending = False
        self.display = False
        if self.is_mounted:
            self.query_one(RemotePagedText).set_reader(None)
            self.query_one("#restore-confirmation").display = False
            self.query_one("#restore-items", OptionList).clear_options()
            self.query_one("#restore-progress", Static).update("")
            self.update_controls()

    def action_close(self) -> None:
        """仅收起阅读，不向后台发送取消；参数：无；返回：无。"""
        self.display = False
        self.post_message(self.Closed())

    def refresh_list(self, cursors: list[Any]) -> None:
        """按固定游标读取目录页；参数：当前页路径；返回：无。"""
        self.source, self.choices = {}, {}
        self.cursors = cursors
        self.request_page(
            "list", {**self.filters, "cursor": cursors[-1], "limit": RESTORE_PAGE_SIZE}
        )

    def request_page(self, action: str, options: dict[str, Any]) -> None:
        """读取期间使旧计划失效，成功和错误共用代次；参数：动作、查询条件；返回：无。"""
        self.generation += 1
        self._pending = True
        self.clear_plan()
        self.query_one("#restore-items", OptionList).disabled = True
        self.query_one("#restore-notice", Static).update("正在读取文件恢复记录…")
        self.fetch_page(action, {**self.owner, **options}, self.generation)

    @work(thread=True)
    def fetch_page(self, action: str, options: dict[str, Any], generation: int) -> None:
        """在线程查询原件，结果只由UI线程接纳；参数：动作、固定条件、代次；返回：无。"""
        try:
            result = self.query_restore(action=action, **options)
        except Exception as exc:
            self.app.call_from_thread(
                self.receive_page, action, generation, {}, str(exc)
            )
        else:
            self.app.call_from_thread(self.receive_page, action, generation, result, "")

    def receive_page(
        self, action: str, generation: int, result: dict[str, Any], error: str
    ) -> None:
        """只显示当前来源的成功或失败，不让旧回复替换新选择；参数：查询回执；返回：无。"""
        if generation != self.generation or not self.is_mounted:
            return
        self._pending = False
        directory = self.query_one("#restore-items", OptionList)
        directory.disabled = False
        if error:
            self.query_one("#restore-notice", Static).update(
                f"读取失败：{error}；可刷新重试"
            )
            self.update_controls()
            return
        if action == "list":
            self.show_points(result)
        else:
            self.show_entries(result)
            if action == "preview":
                self.plan = result
                self.query_one("#restore-impact", Static).update(
                    str(result["confirmation_text"])
                )
                self.query_one("#restore-confirmation").display = True
                self.query_one("#restore-sensitive").display = any(
                    row.get("sensitive") and self.choices.get(key) != "keep"
                    for key, row in self.rows.items()
                )
        self.update_controls()

    def show_points(self, result: dict[str, Any]) -> None:
        """按轮次并列出具体动作，完整路径由后台返回；参数：恢复点目录；返回：无。"""
        self.rows = {}
        options = []
        for row in result.get("turns", []):
            key = "turn:" + str(row["input_id"])
            self.rows[key] = row
            options.append(
                Option(
                    f"本轮改动 · {row.get('started_at', '')} · {row['input_id']}",
                    id=key,
                )
            )
        for row in result["points"]:
            key = "point:" + str(row["point_id"])
            self.rows[key] = row
            status = str(row.get("status", "unknown"))
            options.append(
                Option(
                    f"  {row.get('started_at', '')} · {STATE_LABELS.get(status, status)} · "
                    f"{row.get('entry_count', len(row.get('entries', [])))} 个文件 · {row['point_id']}",
                    id=key,
                )
            )
        directory = self.query_one("#restore-items", OptionList)
        directory.clear_options().add_options(options)
        directory.highlighted = 0 if options else None
        self.next_cursor = result.get("next_cursor")
        self.query_one("#restore-previous", Button).disabled = len(self.cursors) == 1
        self.query_one("#restore-next", Button).disabled = self.next_cursor is None
        self.query_one("#restore-title", Static).update(
            f"文件恢复 · {result.get('workspace_root', '')}"
        )
        self.query_one("#restore-notice", Static).update(
            "选择一轮或具体文件动作"
            if options
            else "尚无恢复点；启用前只记录哈希的文件不能恢复旧内容"
        )

    def show_entries(self, result: dict[str, Any]) -> None:
        """显示后端已核实条目，来源不明和冲突遵循默认不选；参数：详情或预览；返回：无。"""
        self.rows = {str(row["entry_id"]): row for row in result["entries"]}
        self.choices = {
            key: self.choices.get(
                key, "restore" if row.get("default_selected") else "keep"
            )
            for key, row in self.rows.items()
        }
        self.render_entries()
        self.next_cursor = None
        self.query_one("#restore-previous", Button).disabled = True
        self.query_one("#restore-next", Button).disabled = True
        self.query_one("#restore-title", Static).update(
            f"文件恢复 · {result.get('workspace_root', '')}"
        )
        scope = result.get("coverage_text", "")
        notice = (
            str(scope)
            or "逐项选择恢复、保留或另存副本，再预览；来源待确认可能包含外部编辑"
        )
        if result.get("blocked_by"):
            notice += f"\n当前无法执行：{result['blocked_by']}；可在当前活动中查看或停止占用运行"
        self.query_one("#restore-notice", Static).update(notice)

    def render_entries(self) -> None:
        """保留所选行并刷新逐文件选择标记；参数：无；返回：无。"""
        directory = self.query_one("#restore-items", OptionList)
        highlighted = directory.highlighted
        directory.clear_options().add_options(
            [
                Option(
                    f"[{CHOICE_LABELS[self.choices[key]]}] {row['path']} · "
                    f"{STATE_LABELS.get(str(row.get('state')), str(row.get('state', '')))}"
                    f"{' · 二进制' if row.get('binary') else ''}{' · 受保护配置' if row.get('sensitive') else ''}"
                    f"{(' · ' + str(row['error'])) if row.get('error') else ''}",
                    id=key,
                )
                for key, row in self.rows.items()
            ]
        )
        if highlighted is not None and highlighted < len(self.rows):
            directory.highlighted = highlighted
        elif self.rows:
            directory.highlighted = 0

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """目录选择只下钻或读差异，不触发写入；参数：已核实条目；返回：无。"""
        if event.option_list.id != "restore-items" or event.option.id is None:
            return
        event.stop()
        key = event.option.id
        if not self.source:
            row = self.rows[key]
            field = "input_id" if key.startswith("turn:") else "point_id"
            self.source = {field: str(row[field])}
            self.choices = {}
            self.request_page("detail", self.source)
            return
        self.update_controls()
        if self.plan:
            self.query_one(RemotePagedText).set_reader(
                partial(
                    self.read_diff,
                    {**self.owner, "plan_id": self.plan["plan_id"], "entry_id": key},
                )
            )

    def read_diff(
        self, options: dict[str, Any], offset: int, limit: int
    ) -> dict[str, Any]:
        """按计划与条目读取差异页，不读取任意磁盘路径；参数：来源、偏移、长度；返回：正文页。"""
        return self.query_restore(action="diff", **options, offset=offset, limit=limit)

    def clear_plan(self) -> None:
        """选择变化后移除旧授权和差异，必须重新预览；参数：无；返回：无。"""
        self.plan = {}
        self.query_one("#restore-confirm", Checkbox).value = False
        self.query_one("#restore-sensitive", Checkbox).value = False
        self.query_one("#restore-confirmation").display = False
        self.query_one(RemotePagedText).set_reader(None)
        self.update_controls()

    def update_controls(self) -> None:
        """执行按钮绑定当前预览和确认，繁忙时仍可收起或取消；参数：无；返回：无。"""
        has_entries = bool(self.source and self.rows) and not self._pending
        for choice in CHOICE_LABELS:
            self.query_one(
                f"#restore-choice-{choice}", Button
            ).disabled = not has_entries
        self.query_one("#restore-preview", Button).disabled = not has_entries
        confirmed = self.query_one("#restore-confirm", Checkbox).value
        sensitive = self.query_one("#restore-sensitive", Checkbox)
        active = (
            self.job_owner == self.owner
            and self.job.get("status") not in TERMINAL_STATES
            and bool(self.job)
        )
        control_pending = self._operation_pending and self._operation_action != "status"
        self.query_one("#restore-execute", Button).disabled = not (
            self.plan.get("can_execute")
            and confirmed
            and (not sensitive.display or sensitive.value)
            and not self._pending
            and not control_pending
            and not active
        )
        self.query_one("#restore-cancel", Button).disabled = (
            not active or control_pending
        )

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """只调整本次预览确认，不将其存为长期授权；参数：确认选择；返回：无。"""
        event.stop()
        self.update_controls()

    def choose_file(self, choice: str) -> None:
        """对当前条目明确选择，冲突条目不批量自动选中；参数：恢复/保留/副本；返回：无。"""
        index = self.query_one("#restore-items", OptionList).highlighted
        if not self.source or index is None:
            return
        key = list(self.rows)[index]
        self.choices = {**self.choices, key: choice}
        self.clear_plan()
        self.render_entries()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """分派明确用户动作，取消仅针对恢复作业；参数：按钮事件；返回：无。"""
        identity = event.button.id or ""
        if not identity.startswith("restore-"):
            return
        event.stop()
        action = identity.removeprefix("restore-")
        if action.startswith("choice-"):
            self.choose_file(action.removeprefix("choice-"))
        elif action == "close":
            self.action_close()
        elif action in {"up", "refresh"}:
            self.refresh_list([None])
        elif action == "previous" and len(self.cursors) > 1:
            self.refresh_list(self.cursors[:-1])
        elif action == "next" and self.next_cursor is not None:
            self.refresh_list([*self.cursors, self.next_cursor])
        elif action == "preview":
            self.request_page(
                "preview",
                {
                    **self.source,
                    "selection": [
                        {"entry_id": key, "choice": choice}
                        for key, choice in self.choices.items()
                    ],
                },
            )
        elif action == "execute" and self.plan:
            self.request_operation(
                "execute",
                {
                    **self.owner,
                    "plan_id": self.plan["plan_id"],
                    "confirmation": {
                        "accepted": self.query_one("#restore-confirm", Checkbox).value,
                        "sensitive": self.query_one(
                            "#restore-sensitive", Checkbox
                        ).value,
                    },
                },
            )
        elif action == "cancel" and self.job:
            self.request_operation(
                "cancel", {**self.job_owner, "operation_id": self.job["operation_id"]}
            )
        elif action == "status":
            self.request_operation("status", dict(self.owner))

    def request_operation(self, action: str, options: dict[str, Any]) -> None:
        """固定作业归属，进度读取不使文件选择失效；参数：动作、固定身份；返回：无。"""
        if self._operation_pending and (
            action == "status" or self._operation_action != "status"
        ):
            return
        self._operation_pending = True
        self._operation_action = action
        self._operation_generation += 1
        self.update_controls()
        self.fetch_operation(action, options, self._operation_generation)

    @work()
    async def fetch_operation(
        self, action: str, options: dict[str, Any], generation: int
    ) -> None:
        """异步转发后台作业命令，断开显示不终止恢复；参数：动作、归属、代次；返回：无。"""
        try:
            callbacks: dict[str, Callable[..., dict[str, Any]]] = {
                "execute": self.execute_restore,
                "cancel": self.cancel_restore,
                "status": partial(self.query_restore, action="status"),
            }
            callback = callbacks[action]
            result = await asyncio.to_thread(callback, **options)
        except Exception as exc:
            if generation == self._operation_generation and self.is_mounted:
                text = f"文件恢复{action}失败：{exc}；请刷新进度核对，勿假定文件未变化"
                if all(options.get(key) == value for key, value in self.owner.items()):
                    self.query_one("#restore-progress", Static).update(text)
                self.post_message(self.OperationChanged(text))
            return
        finally:
            if generation == self._operation_generation:
                self._operation_pending = False
                if self.is_mounted:
                    self.update_controls()
        if (
            generation != self._operation_generation
            or not self.is_mounted
            or not result
        ):
            return
        self.job = result
        self.job_owner = {
            key: str(options[key])
            for key in ("session_id", "data_space_id")
            if key in options
        }
        text = operation_text(result)
        if self.job_owner == self.owner:
            self.query_one("#restore-progress", Static).update(text)
            self.update_controls()
        if result.get("status") in TERMINAL_STATES:
            self.post_message(self.OperationChanged(text))

    def poll_operation(self) -> None:
        """只轮询未结束作业，面板收起仍能观察真实终态；参数：无；返回：无。"""
        if (
            self.job
            and self.job.get("status") not in TERMINAL_STATES
            and not self._operation_pending
        ):
            self.request_operation(
                "status", {**self.job_owner, "operation_id": self.job["operation_id"]}
            )


def operation_text(result: dict[str, Any]) -> str:
    """将逐项真实状态转成可读回执；参数：后台操作；返回：完整结果文字。"""
    state = str(result.get("status", "unknown"))
    lines = [
        f"文件恢复 · {STATE_LABELS.get(state, state)} · {result.get('operation_id', '')}"
    ]
    for row in result.get("entries", []):
        status = str(row.get("status", "unknown"))
        lines.append(
            f"{row['path']}：{STATE_LABELS.get(status, status)} {row.get('error') or ''}"
        )
        if row.get("backup_path"):
            lines.append(f"保护副本：{row['backup_path']}")
    if result.get("error"):
        lines.append(str(result["error"]))
    return "\n".join(lines)
