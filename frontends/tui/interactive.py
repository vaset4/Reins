"""后台驱动的全屏聊天界面。

作者：xxx
时间：2026-09-29 21:00:00
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Any

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Input,
    OptionList,
    Select,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

from frontends.tui.bridge import TuiBridge
from frontends.tui.branches import BranchScreen
from frontends.tui.dialogs import ApprovalScreen, DetailScreen, PromptScreen
from frontends.tui.projection import ConversationProjection, ToolResultSource
from frontends.tui.composer import Composer
from frontends.tui.widgets import MessageCard
from frontends.tui.settings import SettingsScreen
from frontends.tui.activity import ActivityScreen
from frontends.tui.evidence import EvidencePanel
from frontends.tui.file_restore import FileRestorePanel
from frontends.tui.context_management import ContextPanel
from frontends.tui.search_results import SearchResultsScreen

NARROW_COLUMNS = 100
COMPACT_ROWS = 32
REFRESH_SECONDS = 0.1
LIVE_CARD_WINDOW = 60


class BackendUpdate(Message):
    """后台线程只发送消息，由界面线程更新控件。"""

    def __init__(self, kind: str, payload: Any) -> None:
        """保存事件；传参：种类和内容；返回：无。"""
        super().__init__()
        self.kind, self.payload = kind, payload


class InteractiveTui(App[int]):
    """界面只拥有草稿和显示状态，执行由注入的后台桥接负责。"""

    CSS_PATH = "interactive.tcss"
    BINDINGS = [
        ("ctrl+q", "quit", "断开界面"),
        ("escape", "cancel_run", "停止当前运行"),
        ("ctrl+b", "sidebar", "会话"),
        ("ctrl+n", "new_session", "新会话"),
        ("f1", "help", "帮助"),
        ("f2", "input_mode", "输入模式"),
        ("f4", "approval", "审批"),
        ("f3", "branches", "分支历史"),
        ("f5", "history", "分页历史"),
        Binding("f6", "settings", "设置", priority=True),
        Binding("f7", "activity", "活动", priority=True),
        ("f8", "outputs", "界面消息"),
        Binding("f9", "requests", "请求记录", priority=True),
        Binding("f10", "file_restore", "文件恢复", priority=True),
        ("ctrl+g", "cancel_run", "停止运行"),
        ("ctrl+o", "attach_file", "文件附件"),
    ]

    def __init__(self, bridge: TuiBridge) -> None:
        """保存后台连接；传参：共用桥接；返回：无。"""
        super().__init__()
        self.bridge = bridge
        self.projection = ConversationProjection(retained_cards=LIVE_CARD_WINDOW)
        self.drafts: dict[str, str] = {}
        self.input_history: dict[str, list[str]] = {}
        self.cards: dict[str, MessageCard] = {}
        self.changed: set[str] = set()
        self.submitting = False
        self.connected = False
        self._bridge_ready = False
        self.approval_identity: str | None = None
        self._rendering = False
        self.history_projection: ConversationProjection | None = None
        self._history_page: dict[str, Any] = {}
        self._history_cursors: list[str | None] = []
        self._history_pending = False
        self._history_generation = 0
        self._history_scroll_top = False
        self.command_outputs: list[str] = []
        self._session_query = ""
        self._session_workspace: str | None = None
        self._session_purpose = "chat"
        self._workspace_options: list[tuple[str, str]] = []
        self._workspace_paths: dict[str, str] = {}
        self._session_cursors: list[list[str] | None] = [None]
        self._session_next: list[str] | None = None
        self._session_generation = 0
        self._session_results: dict[str, dict[str, Any]] = {}
        self._switch_generation = 0
        self._switch_lock = Lock()
        self._command_descriptions: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        """建立固定输入与独立滚动正文；传参：无；返回：界面控件。"""
        yield Static("Reins · 正在连接后台", id="heading", markup=False)
        with Horizontal(id="workspace"):
            with Vertical(id="sidebar"):
                yield Select(
                    [("全部工作区", "")],
                    value="",
                    allow_blank=False,
                    id="workspace-filter",
                    tooltip="筛选下方会话，不改变当前聊天的工作区",
                )
                yield Static("", id="workspace-filter-path", markup=False)
                yield Select(
                    [
                        ("聊天会话", "chat"),
                        ("自动知识维护", "maintenance"),
                        ("全部记录", "all"),
                    ],
                    value="chat",
                    allow_blank=False,
                    id="session-purpose",
                )
                yield Input(placeholder="搜索会话、正文或工具结果", id="session-search")
                yield OptionList(id="sessions", markup=False)
                with Horizontal(id="session-pages"):
                    yield Button("上一页", id="sessions-previous", disabled=True)
                    yield Button("下一页", id="sessions-next", disabled=True)
                with Horizontal(classes="sidebar-actions"):
                    yield Button("新会话", id="new-session")
                    yield Button("分支历史", id="show-branches")
                with Collapsible(title="更多操作", collapsed=True, id="sidebar-tools"):
                    with Horizontal(classes="sidebar-actions"):
                        yield Button("设置", id="show-settings")
                        yield Button("当前活动", id="show-activity")
                    with Horizontal(classes="sidebar-actions"):
                        yield Button("请求记录", id="show-requests")
                        yield Button("界面消息", id="show-outputs")
                    with Horizontal(classes="sidebar-actions"):
                        yield Button("文件恢复", id="show-file-restore")
                        yield Button("上下文与记忆", id="show-context")
            with Vertical(id="chat"):
                with Horizontal(id="history-actions"):
                    yield Button("历史", id="history-open")
                    yield Button("更早", id="history-older", disabled=True)
                    yield Button("较新", id="history-newer", disabled=True)
                    yield Button("返回实时", id="history-live")
                    yield Button("请求记录", id="history-requests")
                yield Static("实时对话", id="history-caption", markup=False)
                yield VerticalScroll(id="conversation")
                yield EvidencePanel(
                    query=self.bridge.query_evidence,
                    export=self.bridge.export_evidence,
                    data_root=self.bridge.data_root,
                )
                yield FileRestorePanel(
                    query=self.bridge.query_file_restore,
                    execute=self.bridge.execute_file_restore,
                    cancel=self.bridge.cancel_file_restore,
                )
                yield ContextPanel(query=self.query_context_management)
                yield Static("", id="notice", markup=False)
                with VerticalScroll(id="command-hints", can_focus=False):
                    yield Static(
                        "↑↓ 选择 · Enter 确认 · Tab 补全 · Esc 收起", markup=False
                    )
                    yield OptionList(
                        id="command-hints-body", markup=False, compact=True
                    )
                yield Composer(id="composer")
                yield Static("", id="input-hints", markup=False)
                yield Static("用量未知", id="status", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        """启动连接并合并流式刷新；传参：无；返回：无。"""
        composer = self.query_one(Composer)
        self._command_descriptions = {
            f"/{name}": item.description
            for item in self.bridge.commands.visible_commands()
            for name in (item.name, *item.aliases)
        }
        # 1. 补充由后台宿主及输入控件直接接纳的命令，业务仍由原入口执行
        self._command_descriptions.update(
            {
                "/background": "查看、启动或停止后台服务",
                "/notifications": "查看、标记或重试通知",
                "/approve": "回答当前运行的审批请求",
                "/stop": "停止当前运行",
                "/pause": "停止当前运行",
                "/mode": "查看或切换权限模式",
                "/revoke": "撤回授权",
                "/vim": "切换普通输入与 Vim 编辑模式",
            }
        )
        composer.commands = tuple(sorted(self._command_descriptions))
        self.query_one("#command-hints-body", OptionList).can_focus = False
        composer.focus()
        self.on_composer_hints_changed()
        self.set_interval(REFRESH_SECONDS, self.refresh_cards)
        self.set_interval(1, self.refresh_activity)
        self.connect_backend()

    def receive(self, kind: str, payload: Any) -> None:
        """供后台线程投递更新；传参：种类和内容；返回：无。"""
        self.post_message(BackendUpdate(kind, payload))

    @work(thread=True)
    def connect_backend(self) -> None:
        """在线程内装配后台，失败显示真实原因；传参：无；返回：无。"""
        try:
            self.bridge.start()
            self.receive("connected", self.bridge.model_label())
            self.load_sessions()
        except Exception as exc:
            self.receive("disconnected", str(exc))

    def on_backend_update(self, event: BackendUpdate) -> None:
        """处理后台事实与命令回执；传参：后台消息；返回：无。"""
        kind, payload = event.kind, event.payload
        if kind == "snapshot":
            self.apply_snapshot(payload)
        elif kind == "accepted":
            if payload["session_id"] != self.projection.session_id:
                return
            self.projection.accepted(payload["identity"], payload["text"])
            self.changed.add(payload["identity"])
        elif kind in {"connected", "input-model"}:
            self._bridge_ready = self._bridge_ready or kind == "connected"
            self.connected = self.connected or kind == "connected"
            self.query_one("#status", Static).update(
                f"后续新运行：{payload} · 用量未知 · Enter 发送 / Ctrl+J 换行"
            )
        elif kind == "disconnected":
            self.connected = False
            self.query_one("#notice", Static).update(
                f"连接中断：{payload} · /background start 重新连接"
            )
        elif kind == "output":
            self.command_outputs.append(
                f"会话：{self.projection.session_id or '连接中'}\n{payload}"
            )
            self.query_one("#notice", Static).update(str(payload))
            self.query_one(
                "#show-outputs", Button
            ).label = f"界面消息（{len(self.command_outputs)}）"
        elif kind in {"sessions", "branches", "history-page", "settings", "activity"}:
            self.receive_query_result(kind, payload)
        elif kind == "submitted":
            self.finish_submit(payload)
        elif kind == "prompt":
            self.answer_prompt(payload)
        elif kind == "activity-controlled":
            if (
                payload["session_id"] == self.projection.session_id
                and "input_id" in payload
            ):
                self.projection.accepted(payload["input_id"], payload["text"])
                self.changed.add(payload["input_id"])
            self.receive("output", payload["message"])

    def receive_query_result(self, kind: str, payload: dict[str, Any]) -> None:
        """将只读查询回执交给所属浏览视图，拒绝迟到会话选择；参数：种类及结果；返回：无。"""
        if kind == "sessions":
            if payload["generation"] != self._session_generation:
                return
            if "error" in payload:
                self.receive("output", f"会话列表读取失败：{payload['error']}")
                return
            if "workspaces" in payload:
                self.update_workspace_options(payload["workspaces"])
            self._session_results = {
                row["session_id"]: row for row in payload["sessions"]
            }
            options = []
            for row in payload["sessions"]:
                location = row.get("workspace_name") or "工作区未记录"
                if row.get("purpose") == "maintenance":
                    location += f" · 来源 {row.get('owner_title') or '聊天会话'}"
                elif row.get("maintenance_count"):
                    location += f" · {row['maintenance_count']} 项维护"
                location += " · 正文命中" if row.get("match") else ""
                options.append(
                    Option(
                        Text(
                            f"{row['title']} · {row['status']}\n{location}",
                            overflow="ellipsis",
                            no_wrap=True,
                        ),
                        id=row["session_id"],
                    )
                )
            self.query_one("#sessions", OptionList).clear_options().add_options(options)
            self.query_one("#sessions", OptionList).disabled = False
            self._session_next = payload.get("next_before")
            self.query_one("#sessions-next", Button).disabled = (
                self._session_next is None
            )
            self.query_one("#sessions-previous", Button).disabled = (
                len(self._session_cursors) == 1
            )
        elif kind == "branches":
            if (
                payload["session_id"] == self.projection.session_id
                and len(self.screen_stack) == 1
            ):
                self.push_screen(
                    BranchScreen(payload, browse=self.bridge.require_host().browse),
                    self.branch_selected,
                )
        elif kind == "history-page":
            self.receive_history_page(payload)
        elif kind == "settings":
            if (
                payload["session_id"] == self.projection.session_id
                and len(self.screen_stack) == 1
            ):
                self.push_screen(SettingsScreen(payload), self.settings_selected)
        elif kind == "activity":
            if (
                payload["session_id"] == self.projection.session_id
                and len(self.screen_stack) == 1
            ):
                self.push_screen(ActivityScreen(payload), self.activity_selected)

    def apply_snapshot(self, snapshot: dict[str, Any]) -> None:
        """切换时保存独立草稿，再更新真实状态；传参：后台快照；返回：无。"""
        composer = self.query_one(Composer)
        previous_session = self.projection.session_id
        previous_space = self.projection.snapshot.get("data_space_id")
        previous_workspace = self.projection.snapshot.get(
            "project_root", self.bridge.project_root
        )
        try:
            changed = self.projection.apply(snapshot)
        except ValueError as exc:
            self.connected = False
            self.query_one("#notice", Static).update(str(exc))
            return
        if self.projection.snapshot is not snapshot:
            return
        self.connected = self._bridge_ready
        space_changed = previous_space is not None and previous_space != snapshot.get(
            "data_space_id"
        )
        if snapshot["session_id"] != previous_session or space_changed:
            self.query_one(EvidencePanel).invalidate()
            self.query_one(FileRestorePanel).invalidate()
            self.query_one(ContextPanel).invalidate()
            self.return_to_live()
            self.drafts[previous_session] = composer.text
            initial_draft = composer.text if not previous_session else ""
            if space_changed:
                self._session_workspace = None
                self.update_workspace_options([])
                self._session_generation += 1
                self._session_cursors, self._session_next = [None], None
                self.query_one("#sessions", OptionList).clear_options()
                initial_draft = composer.text
                self.drafts = {snapshot["session_id"]: initial_draft}
                self.input_history = {}
                self.query_one("#notice", Static).update(
                    f"数据空间已变化，草稿尚未发送。原工作区：{previous_workspace}"
                )
                self.load_sessions(self._session_query)
            composer.load_text(self.drafts.get(snapshot["session_id"], initial_draft))
            history = self.input_history.setdefault(
                snapshot["session_id"],
                [
                    row["text"]
                    for row in snapshot.get("history", [])
                    if row["role"] == "user"
                ],
            )
            composer.set_history(history)
        self.changed.update(changed)
        self.refresh_activity()
        project_root = str(snapshot.get("project_root", self.bridge.project_root))
        new_session = self.query_one("#new-session", Button)
        new_session.label = f"新会话 · {Path(project_root).name}"
        new_session.tooltip = f"在当前聊天的工作区新建：{project_root}"
        request = snapshot.get("approval")
        if (
            request
            and request["identity"] != self.approval_identity
            and len(self.screen_stack) == 1
        ):
            self.action_approval()

    def refresh_activity(self) -> None:
        """刷新等待秒数而不改动消息与焦点；参数：无；返回：无。"""
        if self.connected:
            self.query_one("#heading", Static).update(
                f"Reins · {self.projection.session_id} · {self.projection.activity_status()}\n"
                f"工作区：{self.projection.snapshot.get('project_root', self.bridge.project_root)} · "
                f"数据空间：{self.projection.snapshot.get('data_root', self.bridge.data_root)}"
            )

    async def refresh_cards(self) -> None:
        """只更新变化卡片，上翻时保留阅读位置；传参：无；返回：无。"""
        if self._rendering or not self.changed:
            return
        self._rendering = True
        try:
            changed, self.changed = self.changed, set()
            pane = self.query_one("#conversation", VerticalScroll)
            following = pane.is_vertical_scroll_end
            view = self.history_projection or self.projection
            keys = list(view.cards)
            if self.history_projection is None:
                previous_keys = [view.current_key(key) for key in self.cards]
                anchor = next((key for key in previous_keys if key in view.cards), None)
                start = (
                    keys.index(anchor)
                    if anchor is not None and not following
                    else max(0, len(keys) - LIVE_CARD_WINDOW)
                )
                keys = keys[start : start + LIVE_CARD_WINDOW]
                self.query_one("#history-caption", Static).update(
                    "实时对话 · 较早内容可打开历史查阅"
                    if view.has_older_cards
                    else "实时对话"
                )
            self.projection.retain_window(
                keys if self.history_projection is None else []
            )
            visible = {key: view.cards[key] for key in keys}
            for key in list(self.cards):
                if key not in visible:
                    await self.cards.pop(key).remove()
            for position, (key, card) in enumerate(visible.items()):
                if key not in changed and key in self.cards:
                    continue
                if key in self.cards:
                    await self.cards[key].update_card(card)
                else:
                    widget = MessageCard(card, read_result=self.read_tool_result)
                    following_widget = next(
                        (
                            self.cards[item]
                            for item in keys[position + 1 :]
                            if item in self.cards
                        ),
                        None,
                    )
                    self.cards[key] = widget
                    await pane.mount(widget, before=following_widget)
            if self._history_scroll_top:
                pane.scroll_home(animate=False)
                self._history_scroll_top = False
            elif following and self.history_projection is None:
                pane.scroll_end(animate=False, immediate=False)
        finally:
            self._rendering = False

    def read_tool_result(self, source: ToolResultSource) -> str:
        """供详情后台线程读取指定消息原件；参数：保存来源；返回：完整已保存正文。"""
        result = self.bridge.require_host().browse(
            "tool_result_detail", **asdict(source)
        )
        text = result["text"]
        if not isinstance(text, str):
            raise ValueError("工具原件查询没有返回正文")
        return text

    def on_composer_submitted(self, event: Composer.Submitted) -> None:
        """保留草稿直到后台接纳，不因连按 Enter 重投；传参：完整草稿；返回：无。"""
        if self.submitting:
            return
        if event.text.strip() == "/vim":
            self.action_input_mode()
            self.query_one(Composer).clear()
            return
        if not self.connected and event.text != "/background start":
            self.query_one("#notice", Static).update("尚未连接，草稿已保留")
            return
        if event.text.strip() == "/model":
            self.query_one(Composer).clear()
            self.action_settings()
            return
        if event.text.strip() == "/requests":
            self.query_one(Composer).clear()
            self.action_requests()
            return
        if event.text.strip() == "/restore":
            self.query_one(Composer).clear()
            self.action_file_restore()
            return
        self.submitting = True
        self.submit_text(event.text, self.projection.session_id)

    @work(thread=True)
    def submit_text(self, text: str, session_id: str) -> None:
        """通过共用服务执行命令和输入；传参：原文及草稿所属会话；返回：无。"""
        try:
            should_exit = self.bridge.submit(text)
            self.receive(
                "submitted", {"text": text, "session_id": session_id, "ok": True}
            )
            if should_exit:
                self.call_from_thread(self.exit, 0)
        except (Exception, KeyboardInterrupt) as exc:
            self.receive("output", f"提交未完成：{exc}")
            self.receive(
                "submitted", {"text": text, "session_id": session_id, "ok": False}
            )

    def finish_submit(self, result: dict[str, Any]) -> None:
        """只清理已接纳的原草稿，保留发送期间新编辑；传参：提交结果；返回：无。"""
        self.submitting = False
        if not result["ok"]:
            return
        identity = result["session_id"]
        self.input_history.setdefault(identity, []).append(result["text"])
        if identity == self.projection.session_id:
            composer = self.query_one(Composer)
            if composer.text == result["text"]:
                composer.clear()
            composer.set_history(self.input_history[identity])
        elif self.drafts.get(identity) == result["text"]:
            self.drafts[identity] = ""

    @work(exclusive=False)
    async def answer_prompt(self, payload: dict[str, Any]) -> None:
        """浮层回答交还等待中的命令；传参：问题及结果通道；返回：无。"""
        answer: Future[str] = payload["answer"]
        value = await self.push_screen_wait(PromptScreen(payload["label"]))
        if answer.done():
            return
        if value is None:
            answer.set_exception(EOFError("用户取消命令"))
        else:
            answer.set_result(value)

    @work(exclusive=True, group="attachment-picker")
    async def action_attach_file(self) -> None:
        """将选中的路径放入所属会话草稿；参数：无；返回：无，读取失败时仍可修改重发。"""
        if len(self.screen_stack) != 1:
            return
        session_id = self.projection.session_id
        value = await self.push_screen_wait(
            PromptScreen(
                '文件路径（文本、PDF、PNG、JPEG、静态GIF/WebP）。只引用文件可输入 @ref "路径"。'
            )
        )
        if value is None or not value.strip():
            return
        value = value.strip()
        directive = value if value.startswith("@ref ") else f"@file {value}"
        if session_id == self.projection.session_id:
            composer = self.query_one(Composer)
            composer.load_text(f"{composer.text}\n{directive}".lstrip("\n"))
            composer.focus()
        else:
            previous = self.drafts.get(session_id, "")
            self.drafts[session_id] = f"{previous}\n{directive}".lstrip("\n")

    def action_approval(self) -> None:
        """打开当前真实审批，不修改输入；传参：无；返回：无。"""
        request = self.projection.snapshot.get("approval")
        if request and len(self.screen_stack) == 1:
            self.approval_identity = request["identity"]
            self.push_screen(ApprovalScreen(request), self.approval_answered)

    def approval_answered(self, command: str | None) -> None:
        """只提交明确审批或补充要求；传参：浮层选择；返回：无。"""
        if command:
            self.submit_text(command, self.projection.session_id)

    def load_sessions(
        self, query: str = "", *, cursors: list[list[str] | None] | None = None
    ) -> None:
        """经后台读取目录，不访问会话文件；传参：检索词；返回：无。"""
        self._session_query = query
        self._session_cursors = cursors or [None]
        self._session_generation += 1
        self.query_one("#sessions-next", Button).disabled = True
        self.query_one("#sessions-previous", Button).disabled = True
        self.query_one("#sessions", OptionList).disabled = True
        self.fetch_sessions(
            query,
            self._session_cursors[-1],
            self._session_generation,
            self._session_workspace,
            purpose=self._session_purpose,
        )

    @work(thread=True)
    def fetch_sessions(
        self,
        query: str,
        before: list[str] | None,
        generation: int,
        workspace_id: str | None,
        *,
        purpose: str,
    ) -> None:
        """在线程查询固定工作区目录页；传参：搜索词、锚点、代次、工作区；返回：发送目录结果。"""
        try:
            result = self.bridge.require_host().browse(
                "sessions",
                query=query,
                before=before,
                workspace_id=workspace_id,
                purpose=purpose,
            )
            self.receive("sessions", {**result, "generation": generation})
        except Exception as exc:
            self.receive("sessions", {"generation": generation, "error": str(exc)})

    def update_workspace_options(self, rows: list[dict[str, Any]]) -> None:
        """使用持久完整目录作为工作区名称；参数：工作区摘要；返回：无，不改实体身份。"""
        options = [
            ("全部工作区", ""),
            *[(str(row["project_root"]), str(row["workspace_id"])) for row in rows],
        ]
        self._workspace_paths = {
            str(row["workspace_id"]): str(row["project_root"]) for row in rows
        }
        picker = self.query_one("#workspace-filter", Select)
        if options != self._workspace_options:
            with self.prevent(Select.Changed):
                picker.set_options(
                    (Text(label), identity) for label, identity in options
                )
                picker.value = self._session_workspace or ""
            self._workspace_options = options
        path = self._workspace_paths.get(self._session_workspace or "", "")
        picker.tooltip = path or "筛选下方会话，不改变当前聊天的工作区"
        location = self.query_one("#workspace-filter-path", Static)
        location.update(path)
        location.tooltip = path or "当前显示全部工作区的会话"
        location.display = bool(path)

    def on_select_changed(self, event: Select.Changed) -> None:
        """切换目录过滤并重置分页，不切换执行会话；参数：工作区选择；返回：无。"""
        if event.select.id == "session-purpose" and isinstance(event.value, str):
            event.stop()
            self._session_purpose = event.value
            self.load_sessions(self._session_query)
            return
        if event.select.id != "workspace-filter" or not isinstance(event.value, str):
            return
        workspace_id = event.value or None
        if workspace_id == self._session_workspace:
            return
        event.stop()
        self._session_workspace = workspace_id
        self.query_one("#sessions", OptionList).clear_options()
        location = self.query_one("#workspace-filter-path", Static)
        path = self._workspace_paths.get(event.value, "")
        event.select.tooltip = path or "筛选下方会话，不改变当前聊天的工作区"
        location.update(path)
        location.tooltip, location.display = path, bool(path)
        self.load_sessions(self._session_query)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """提交会话检索；传参：搜索输入；返回：无。"""
        if event.input.id == "session-search":
            self.load_sessions(event.value)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """选择目录中的真实会话；传参：选项；返回：无。"""
        if event.option_list.id == "command-hints-body":
            if event.option.id:
                self.query_one(Composer).accept_command_hint(event.option_index)
                self.query_one(Composer).focus()
                self.refresh_command_hints()
            event.stop()
            return
        if event.option_list.id == "sessions" and event.option.id:
            row = self._session_results.get(event.option.id)
            if row is not None and row.get("match"):
                self.push_screen(
                    SearchResultsScreen(
                        row,
                        self._session_query,
                        browse=self.bridge.require_host().browse,
                    ),
                    self.search_source_selected,
                )
            else:
                self.switch_session(event.option.id)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        """显示目录项的完整工作区与维护归属，不占据列表正文；参数：高亮项；返回：无。"""
        if (
            event.option_list.id == "sessions"
            and event.option.id in self._session_results
        ):
            row = self._session_results[event.option.id]
            detail = str(row.get("project_root") or "工作区未记录")
            if row.get("owner_session_id"):
                detail += f"\n所属聊天：{row.get('owner_title') or ''} · {row['owner_session_id']}"
            event.option_list.tooltip = detail

    def search_source_selected(self, identity: str | None) -> None:
        """只执行阅读框中明确选择的会话切换；参数：身份或关闭；返回：无，关闭保留草稿。"""
        if identity is not None:
            self.switch_session(identity)
        else:
            self.query_one(Composer).focus()

    def switch_session(self, identity: str | None) -> None:
        """按用户选择顺序分配代次，旧排队选择不覆盖新会话；参数：会话或新建；返回：无。"""
        self._switch_generation += 1
        self.attach_selected_session(identity, self._switch_generation)

    @work(thread=True)
    def attach_selected_session(self, identity: str | None, generation: int) -> None:
        """切换显示连接，原后台工作继续；传参：会话编号；返回：无。"""
        try:
            with self._switch_lock:
                if generation != self._switch_generation:
                    return
                self.bridge.attach_session(identity)
            if generation == self._switch_generation:
                self.call_from_thread(self.load_sessions)
        except Exception as exc:
            if generation == self._switch_generation:
                self.receive("output", f"切换失败：{exc}")

    def action_new_session(self) -> None:
        """创建空会话连接；传参：无；返回：无。"""
        if self.connected and not self.submitting:
            # 【TUI】【新建会话】1. 新建仍属于当前聊天工作区，解除其他工作区过滤使新会话可见
            if self._session_workspace != self.projection.snapshot.get("workspace_id"):
                self.query_one("#workspace-filter", Select).value = ""
            self.switch_session(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """响应新会话按钮；传参：按钮事件；返回：无。"""
        actions: dict[str, Callable[[], object]] = {
            "new-session": self.action_new_session,
            "show-branches": self.action_branches,
            "show-outputs": self.action_outputs,
            "show-settings": self.action_settings,
            "show-activity": self.action_activity,
            "show-requests": self.action_requests,
            "show-file-restore": self.action_file_restore,
            "show-context": self.action_context_management,
            "history-requests": self.action_requests,
            "history-open": self.action_history,
            "history-live": self.return_to_live,
        }
        action = actions.get(event.button.id or "")
        if action is not None:
            action()
        elif event.button.id == "sessions-next" and self._session_next is not None:
            self.load_sessions(
                self._session_query,
                cursors=[*self._session_cursors, self._session_next],
            )
        elif event.button.id == "sessions-previous" and len(self._session_cursors) > 1:
            self.load_sessions(self._session_query, cursors=self._session_cursors[:-1])
        elif event.button.id == "history-older" and self._history_page.get(
            "next_before"
        ):
            self.request_history(
                [*self._history_cursors, self._history_page["next_before"]]
            )
        elif event.button.id == "history-newer" and len(self._history_cursors) > 1:
            self.request_history(self._history_cursors[:-1])

    def action_history(self) -> None:
        """打开最新已保存历史页，实时输出继续接收；传参：无；返回：无。"""
        if self.connected and not self._history_pending:
            self._history_page = {}
            self.request_history([None])

    def request_history(self, cursors: list[str | None]) -> None:
        """固定历史叶并发起一页读取；传参：页游标路径；返回：无。"""
        if self._history_pending:
            return
        self._history_pending = True
        self._history_generation += 1
        options = {"leaf_id": self._history_page.get("leaf_id"), "before": cursors[-1]}
        self.fetch_history(
            self.projection.session_id, options, (self._history_generation, cursors)
        )

    @work(thread=True)
    def fetch_history(
        self,
        session_id: str,
        options: dict[str, Any],
        request: tuple[int, list[str | None]],
    ) -> None:
        """后台分页查询不阻塞输入；传参：会话、锚点与界面请求编号；返回：无。"""
        generation, cursors = request
        try:
            page = self.bridge.require_host().browse(
                "history_page", session_id=session_id, **options
            )
            self.receive(
                "history-page",
                {"generation": generation, "page": page, "cursors": cursors},
            )
        except Exception as exc:
            self.receive("history-page", {"generation": generation, "error": str(exc)})

    def receive_history_page(self, result: dict[str, Any]) -> None:
        """只展示当前请求的固定历史页；传参：分页结果；返回：无。"""
        if result["generation"] != self._history_generation:
            return
        self._history_pending = False
        if "error" in result:
            self.query_one("#notice", Static).update(f"历史读取失败：{result['error']}")
            return
        page = result["page"]
        if page["session_id"] != self.projection.session_id:
            return
        self._history_page, self._history_cursors = page, result["cursors"]
        self.history_projection = ConversationProjection()
        self.history_projection.apply(page)
        self.changed.update(self.cards.keys() | self.history_projection.cards.keys())
        self._history_scroll_top = True
        self.query_one("#history-caption", Static).update(
            f"正在查看历史第 {len(self._history_cursors)} 页 · 新输出继续在后台接收"
        )
        self.query_one("#history-older", Button).disabled = page["next_before"] is None
        self.query_one("#history-newer", Button).disabled = (
            len(self._history_cursors) < 2
        )
        self.call_after_refresh(self.refresh_cards)

    def return_to_live(self) -> None:
        """退出历史分页并显示已收到的最新输出，不改草稿；传参：无；返回：无。"""
        self._history_generation += 1
        self._history_pending = False
        self.history_projection = None
        self._history_page, self._history_cursors = {}, []
        self.query_one("#history-caption", Static).update("实时对话")
        self.changed.update(self.cards.keys() | self.projection.cards.keys())
        self.query_one("#conversation", VerticalScroll).scroll_end(
            animate=False, immediate=True
        )
        self.query_one("#history-older", Button).disabled = True
        self.query_one("#history-newer", Button).disabled = True
        self.call_after_refresh(self.refresh_cards)

    def action_branches(self) -> None:
        """打开当前会话的只读树；传参：无；返回：无。"""
        if self.connected:
            self.load_branches(self.projection.session_id)

    @work(thread=True)
    def load_branches(self, session_id: str) -> None:
        """后台读取首屏节点摘要，后续目录和正文按需查询；传参：会话编号；返回：无。"""
        try:
            self.receive(
                "branches",
                self.bridge.require_host().browse(
                    "session_tree", session_id=session_id
                ),
            )
        except Exception as exc:
            self.receive("output", f"分支历史读取失败：{exc}")

    def branch_selected(self, choice: tuple[str, str] | None) -> None:
        """转交明确继续选择，关闭历史不产生动作；传参：会话和节点；返回：无。"""
        if choice is not None:
            self.continue_branch(*choice)

    @work(thread=True)
    def continue_branch(self, session_id: str, entry_id: str) -> None:
        """复用后台分支切换并显示实际结果；传参：原会话及节点；返回：无。"""
        try:
            self.bridge.require_host().browse(
                "branch", session_id=session_id, entry_id=entry_id
            )
        except Exception as exc:
            self.receive("output", f"从历史继续失败：{exc}")

    @work(thread=True)
    def action_cancel_run(self) -> None:
        """只取消当前连接运行；传参：无；返回：无。"""
        if not self.connected:
            return
        try:
            self.bridge.require_host().cancel()
        except Exception as exc:
            self.receive("output", f"停止失败：{exc}")

    def action_sidebar(self) -> None:
        """切换会话面板；传参：无；返回：无。"""
        sidebar = self.query_one("#sidebar")
        sidebar.display = not sidebar.display

    def on_resize(self, event: events.Resize) -> None:
        """窄屏腾出正文与输入空间；传参：终端尺寸；返回：无。"""
        if self.query("#sidebar"):
            sidebar = self.query_one("#sidebar")
            sidebar.display = event.size.width >= NARROW_COLUMNS
            sidebar.set_class(event.size.width < NARROW_COLUMNS, "narrow")
            sidebar.set_class(event.size.height <= COMPACT_ROWS, "compact")
            self.query_one("#sidebar-tools", Collapsible).collapsed = (
                event.size.height <= COMPACT_ROWS
            )

    def action_help(self) -> None:
        """显示快捷键与共用命令目录；传参：无；返回：无。"""
        commands = "\n".join(
            f"/{item.name} — {item.description}"
            for item in self.bridge.commands.visible_commands()
        )
        self.push_screen(
            DetailScreen(
                "帮助",
                self.query_one(Composer).hints + "\n"
                "Ctrl+B 会话 · Ctrl+N 新会话 · F4 审批 · F5 分页历史 · F6 设置 · F7 活动 · Ctrl+G 停止 · Ctrl+Q 断开（后台继续）\n"
                "/mode read_only|workspace|auto 权限 · /revoke 撤回授权 · /background 后台 · /notifications 通知\n"
                "普通模式：Ctrl+C 复制选区，Ctrl+V 粘贴；Ctrl+Z 撤销，Ctrl+Y 重做\n"
                "Vim：i/a/I/A/o/O 编辑，Esc 返回导航；h/j/k/l、w/b、0/$、gg/G 导航\n"
                "v 选择，y/yy 复制，d/dd 删除，x/D 修改，p 粘贴，u 撤销，Ctrl+R 重做\n"
                "Vim 导航模式再次 Esc 停止运行；浮层内 Esc 优先关闭浮层\n\n" + commands,
            )
        )

    def action_input_mode(self) -> None:
        """切换普通和Vim编辑，保留当前草稿；传参：无；返回：无。"""
        self.query_one(Composer).toggle_vim()

    @work(thread=True, exclusive=True, group="settings")
    def action_settings(self) -> None:
        """在后台查询设置，不阻塞输入或触碰审批；参数：无；返回：无。"""
        if not self.connected or self.submitting or len(self.screen_stack) != 1:
            return
        try:
            self.receive("settings", self.bridge.settings())
        except Exception as exc:
            self.receive("output", f"设置读取失败：{exc}")

    def settings_selected(self, command: str | None) -> None:
        """转交设置选择，保留草稿并避免并发命令；参数：明确命令；返回：无。"""
        if command == "refresh":
            self.action_settings()
        elif command == "context_management":
            self.action_context_management()
        elif command and not self.submitting:
            self.submitting = True
            self.submit_text(command, self.projection.session_id)

    def action_outputs(self) -> None:
        """查看本次连接收到的完整命令回执，不改变草稿；传参：无；返回：无。"""
        body = (
            "\n\n".join(self.command_outputs)
            if self.command_outputs
            else "本次连接尚无界面消息"
        )
        self.push_screen(DetailScreen("界面消息 · 完整命令回执", body))

    def query_context_management(self, **params: Any) -> dict[str, Any]:
        """向桥接传递后台查询，执行与显示分离；参数：原范围与动作；返回：真实记录。"""
        return self.bridge.query_context_management(**params)

    def action_context_management(self) -> None:
        """打开整理与记忆的同一查看入口；参数：无；返回：无，不改变草稿或执行叶。"""
        space = self.projection.snapshot.get("data_space_id")
        if not self.connected or not space:
            self.query_one("#notice", Static).update(
                "后台尚未确认数据空间，请连接后查看上下文与记忆"
            )
            return
        self.query_one(ContextPanel).open(
            {"data_space_id": space, "session_id": self.projection.session_id}
        )

    def on_context_panel_closed(self) -> None:
        """关闭整理详情后回到原草稿；参数：无；返回：无。"""
        self.query_one(Composer).focus()

    def on_context_panel_request_selected(
        self, event: ContextPanel.RequestSelected
    ) -> None:
        """打开所选后台工作的实际请求，不切换执行会话；参数：已核对来源；返回：无。"""
        self.query_one(ContextPanel).action_close()
        self.query_one(EvidencePanel).open(
            {key: event.source[key] for key in ("data_space_id", "session_id")},
            {"run_id": event.source["run_id"]},
        )

    def action_requests(self) -> None:
        """打开当前会话全部分支的请求目录，保留聊天位置；参数：无；返回：无。"""
        if self.connected:
            self.open_requests()

    def open_requests(self, selection: dict[str, str] | None = None) -> None:
        """以后台持久空间身份打开只读记录；参数：可选卡片来源；返回：无。"""
        space = self.projection.snapshot.get("data_space_id")
        if not space:
            self.query_one("#notice", Static).update(
                "后台尚未确认数据空间，请重新连接后查看请求"
            )
            return
        self.query_one(EvidencePanel).open(
            {"data_space_id": space, "session_id": self.projection.session_id},
            selection,
        )

    def on_message_card_inspect_requested(
        self, event: MessageCard.InspectRequested
    ) -> None:
        """从消息卡片的真实身份定位运行或请求；参数：卡片来源；返回：无。"""
        self.open_requests(event.selection)

    def on_evidence_panel_closed(self) -> None:
        """收起阅读面板后回到未提交草稿；参数：无；返回：无。"""
        self.query_one(Composer).focus()

    def action_file_restore(self) -> None:
        """打开当前工作区的独立恢复入口，保留草稿；参数：无；返回：无。"""
        if self.connected:
            self.open_file_restore()

    def open_file_restore(self, selection: dict[str, str] | None = None) -> None:
        """携带当前持久身份查看恢复点，浏览不修改文件；参数：可选工具来源；返回：无。"""
        space = self.projection.snapshot.get("data_space_id")
        if not space:
            self.query_one("#notice", Static).update(
                "后台尚未确认数据空间，请重新连接后查看文件恢复"
            )
            return
        self.query_one(FileRestorePanel).open(
            {"data_space_id": space, "session_id": self.projection.session_id},
            selection,
        )

    def on_message_card_restore_requested(
        self, event: MessageCard.RestoreRequested
    ) -> None:
        """沿实际工具身份定位恢复点，不按消息位置猜版本；参数：卡片来源；返回：无。"""
        self.open_file_restore(event.selection)

    def on_file_restore_panel_closed(self) -> None:
        """收起恢复面板后回到原草稿；参数：无；返回：无。"""
        self.query_one(Composer).focus()

    def on_file_restore_panel_operation_changed(
        self, event: FileRestorePanel.OperationChanged
    ) -> None:
        """保留后台恢复的完整终态回执，不主动发送模型输入；参数：结果；返回：无。"""
        self.receive("output", event.text)

    def on_evidence_panel_export_changed(
        self, event: EvidencePanel.ExportChanged
    ) -> None:
        """导出即使收起仍显示实际完成状态；参数：导出结果；返回：无。"""
        self.receive("output", event.text)

    @work(thread=True, exclusive=True, group="activity")
    def action_activity(self) -> None:
        """在线程查询活动，聊天输入与审批保持可用；参数：无；返回：无。"""
        if not self.connected or len(self.screen_stack) != 1:
            return
        try:
            self.receive(
                "activity",
                self.bridge.require_host().browse(
                    "activity", session_id=self.projection.session_id
                ),
            )
        except Exception as exc:
            self.receive("output", f"活动读取失败：{exc}")

    def activity_selected(self, choice: dict[str, str] | None) -> None:
        """将明确选择交给原会话，不改变聊天草稿；参数：动作或关闭；返回：无。"""
        if choice is None or choice["session_id"] != self.projection.session_id:
            return
        if choice["action"] == "refresh":
            self.action_activity()
        elif choice["action"] == "context_management":
            self.action_context_management()
        elif choice["action"] == "open":
            self.switch_session(choice["target"])
        else:
            self.control_activity(choice)

    @work(thread=True)
    def control_activity(self, choice: dict[str, str]) -> None:
        """控制真实执行者并显示实际接纳结果；参数：原会话及运行选择；返回：无。"""
        try:
            result = self.bridge.require_host().control_activity(choice)
            self.receive("activity-controlled", result)
        except Exception as exc:
            self.receive("output", f"活动操作失败：{exc}")

    def on_composer_hints_changed(self) -> None:
        """显示当前模式和补全候选；传参：无；返回：无。"""
        # 【TUI】【界面退出】1. 卸载控件会产生失焦消息，退出后不再更新已销毁的提示区
        if not self.is_running:
            return
        self.query_one("#input-hints", Static).update(self.query_one(Composer).hints)
        self.refresh_command_hints()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """草稿编辑、粘贴和历史恢复后刷新命令提示；参数：文本变更；返回：无。"""
        if isinstance(event.text_area, Composer) and event.text_area.is_attached:
            self.refresh_command_hints()

    def refresh_command_hints(self) -> None:
        """按未提交的首个命令词显示用途，不执行命令或更改焦点；参数：无；返回：无。"""
        composer = self.query_one(Composer)
        panel = self.query_one("#command-hints", VerticalScroll)
        panel.display = composer.command_hints_visible()
        if not panel.display:
            return
        options = self.query_one("#command-hints-body", OptionList)
        current = tuple(
            options.get_option_at_index(index).id
            for index in range(options.option_count)
        )
        if current != composer.hint_matches:
            options.clear_options()
            options.add_options(
                Option(f"{name} — {self._command_descriptions[name]}", id=name)
                for name in composer.hint_matches
            )
            if not composer.hint_matches:
                options.add_option(Option("没有匹配的命令", disabled=True))
        options.highlighted = composer.hint_index if composer.hint_matches else None
        options.scroll_to_highlight()


def run_interactive_tui(
    *, project_root: Path, data_root: Path, llm_client: Any, startup_error: str = ""
) -> int:
    """运行正式全屏入口并释放前台连接；传参：目录和模型；返回：退出码。"""
    bridge = TuiBridge(
        project_root=project_root,
        data_root=data_root,
        llm_client=llm_client,
        event_sink=lambda kind, payload: app.receive(kind, payload),
        startup_error=startup_error,
    )
    app = InteractiveTui(bridge)
    try:
        return app.run() or 0
    finally:
        bridge.close()
