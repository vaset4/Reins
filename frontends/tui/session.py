from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import approval
from approval.batch import register_batch_backend
from approval.batch_types import ApprovalBatch, BatchDecision, format_batch
from approval.channel import ApprovalChannel
from approval.session import ApprovalSession
from app.completion_ui import CompletionMenu, completion_service
from app.approval_ui import approval_control
from app.repl import (
    _describe_model,
    _describe_provider,
)
from app.repl.console import capture_console
from app.repl.slash_commands import (
    ReplState,
    SlashCommandContext,
    create_default_registry,
)
from llm.base import LLMClient
from llm.messages import model_visible_text
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.application import Application
from runtime.cancellation import CancellationToken
from runtime.session_message_store import SessionMessageStore
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import ToolRegistry

from frontends.tui.approval_view import approval_body, parse_approval_decision
from frontends.tui.footer import (
    FOOTER_FALLBACK_WIDTH,
    FooterView,
    render_footer,
    render_pending_approval_footer,
)
from frontends.tui.transcript import TranscriptBuffer
from frontends.tui.turn import drive_agent_turn
from frontends.tui.viewport import TranscriptViewport

TAIL_WORKING_LINES = 2
VIEWPORT_RESERVED_ROWS = 3
SCROLL_STEP = 4
SCROLL_PAGE = 16
SPINNER_FRAMES = "|/-\\"


@dataclass(slots=True)
class PendingApproval:
    request: approval.ApprovalRequest
    event: threading.Event
    decision: approval.ApprovalDecision | None = None


class FullscreenTui:
    def __init__(
        self,
        *,
        project_root: Path,
        data_root: Path,
        llm_client: LLMClient,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        """创建独立全屏会话并保存初始工作区；传参：启动目录、数据根和执行依赖；返回：无。"""
        self.data_root = data_root
        self.llm_client = llm_client
        self.state = ReplState()
        # 【全屏会话】【工作区绑定】1. 只为刚创建的会话登记目录，首条输入接纳前归属已持久化
        workspace = WorkspaceStore(data_root).bind_session(
            self.state.session_id, project_root
        )
        self.project_root = workspace.project_root
        self.store = TaskStore(data_root)
        self.registry = tool_registry or build_tool_registry(
            repo_root=self.project_root, data_root=data_root
        )
        self._owns_registry = tool_registry is None
        self.slash_registry = create_default_registry()
        self.transcript = TranscriptBuffer()
        self._app: Application[object] | None = None
        self._busy = False
        self._pending: PendingApproval | None = None
        self._run_started_at: float | None = None
        self._viewport = TranscriptViewport()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._cancellation = CancellationToken()
        self._completion_menu = CompletionMenu()
        self.approval_session = ApprovalSession()
        self._approval_session_id = ""
        self._batch_identity: str | None = None
        self.batch_approvals = ApprovalChannel(
            lambda _identity, _request: None, present_batch=self._present_batch
        )

    def attach_app(self, app: Application[object]) -> None:
        self._app = app

    def detach_app(self) -> None:
        self._app = None

    def start_approval_backend(self) -> None:
        approval.register_approval_backend(self._approval_backend)
        register_batch_backend(self._batch_approval_backend)

    def stop_approval_backend(self) -> None:
        approval.register_approval_backend(None)
        register_batch_backend(None)

    def submit(self, raw: str) -> None:
        text = raw.strip()
        if not text:
            return
        self._viewport.follow()
        self.transcript.append("user", "you", text)
        if text.split(maxsplit=1)[0] in {"/mode", "/revoke"}:
            try:
                message = approval_control(
                    text,
                    self.approval_session,
                    data_root=self.data_root,
                    session_id=self.state.session_id,
                    task_id=self.state.current_task_id,
                )
                self.transcript.append("system", "授权管理", message)
                if len(text.split()) > 1:
                    self.batch_approvals.interrupt()
            except (OSError, ValueError) as exc:
                self.transcript.append("error", "授权管理", str(exc), state="error")
            self._invalidate()
            return
        if self._handle_batch_approval(text):
            self._invalidate()
            return
        if self._handle_pending_approval(text):
            self._invalidate()
            return
        choice = self._completion_menu.choose(text) if not self._busy else None
        if choice is not None:
            try:
                with completion_service(self.data_root) as confirmations:
                    event = confirmations.decide(self.state.session_id, **choice)
                input_id = str(event.payload["source_input_id"])
                entry = next(
                    row
                    for row in SessionMessageStore(self.data_root).read_entries(
                        self.state.session_id
                    )
                    if row.entry_id == input_id
                )
                assert entry.message is not None
                self._start_turn(model_visible_text(entry.message), input_id=input_id)
            except (OSError, ValueError) as exc:
                self.transcript.append("error", "完成确认失败", str(exc), state="error")
            self._invalidate()
            return
        if text.startswith("/"):
            self._dispatch_slash(text)
            self._invalidate()
            return
        self._start_turn(text)
        self._invalidate()

    def footer_fragments(self) -> StyleAndTextTuples:
        if self._pending is not None:
            return render_pending_approval_footer()
        state = "running" if self._busy else "ready"
        view = FooterView(
            project_root=self.project_root,
            session_id=self.state.session_id,
            state=state,
            run_id=self.state.current_run_id,
            provider=_describe_provider(self.llm_client),
            model=_describe_model(self.llm_client),
        )
        return render_footer(view, self.terminal_width())

    def terminal_width(self) -> int:
        if self._app is None:
            return FOOTER_FALLBACK_WIDTH
        return max(1, self._app.output.get_size().columns)

    def terminal_height(self) -> int:
        if self._app is None:
            return FOOTER_FALLBACK_WIDTH
        return max(1, self._app.output.get_size().rows)

    def transcript_fragments(self) -> StyleAndTextTuples:
        width = self.terminal_width()
        height = max(1, self.terminal_height() - VIEWPORT_RESERVED_ROWS)
        self._viewport.jump_to_latest(self._max_scroll(width, height))
        return self.transcript.fragments(
            width,
            scroll_offset=self._viewport.offset,
            height=height,
            tail=self._working_fragments(),
        )

    def scroll_up(self, lines: int = SCROLL_PAGE) -> None:
        self._viewport.scroll_up(lines)
        self._invalidate()

    def scroll_down(self, lines: int = SCROLL_PAGE) -> None:
        width = self.terminal_width()
        height = max(1, self.terminal_height() - VIEWPORT_RESERVED_ROWS)
        self._viewport.scroll_down(lines, self._max_scroll(width, height))
        self._invalidate()

    def exit(self, result: int = 0) -> None:
        self._cancellation.cancel("interface_closed")
        self.batch_approvals.close()
        pending = self._pending
        if pending is not None:
            pending.decision = approval.ApprovalDecision.DENY
            pending.event.set()
        if self._app is not None:
            self._app.exit(result=result)

    def close(self) -> None:
        """等待执行边界交接后释放自己创建的连接；传参：无；返回：无，外部注入目录仍归调用者。"""
        self.exit()
        if self._worker is not None:
            self._worker.join()
        try:
            if self._owns_registry:
                self.registry.close()
        finally:
            self.store.close()
            self.approval_session.close()

    def _dispatch_slash(self, raw: str) -> None:
        ctx = SlashCommandContext(
            repl_state=self.state,
            store=self.store,
            registry=self.registry,
            llm_client=self.llm_client,
            project_root=self.project_root,
            data_root=self.data_root,
            prompt_fn=input,
        )
        # slash command 可能通过 Rich console 直接打印表格；全屏 TUI 必须捕获后
        # 放回 transcript，避免绕过 prompt_toolkit 破坏 alternate screen。
        with capture_console() as console:
            result = self.slash_registry.dispatch(raw, ctx)
        captured = console.export_text().strip()
        self._render_slash_result(raw, captured, result)

    def _render_slash_result(self, raw: str, captured: str, result: object) -> None:
        message = getattr(result, "message", None)
        if getattr(result, "clear_screen", False):
            self.transcript.clear()
        if captured:
            self.transcript.append("system", raw, captured)
        if message:
            self.transcript.append("system", raw, str(message))
        if getattr(result, "enter_dashboard", False):
            self.transcript.append("system", "dashboard", self._dashboard_summary())
        if getattr(result, "should_exit", False):
            self.exit(0)
        self._refresh_confirmations()

    def _dashboard_summary(self) -> str:
        focus = self.state.current_task_id or "(none)"
        run_id = self.state.current_run_id or "(none)"
        lines = [
            f"session: {self.state.session_id}",
            f"run: {run_id}",
            f"focus task: {focus}",
            f"data root: {self.data_root}",
            "Use /status for detailed recovery and task state.",
        ]
        return "\n".join(lines)

    def _start_turn(self, user_message: str, *, input_id: str | None = None) -> None:
        with self._lock:
            if self._busy:
                self.transcript.append("status", "busy", "A run is already active.")
                return
            self._busy = True
            self._run_started_at = time.monotonic()
            self._cancellation = CancellationToken()
        # AgentLoop 是阻塞生成器；放到后台线程，UI 线程只负责输入和重绘。
        self._worker = threading.Thread(
            target=self._run_turn, args=(user_message, input_id), daemon=True
        )
        self._worker.start()

    def _run_turn(self, user_message: str, input_id: str | None = None) -> None:
        succeeded = False
        try:
            if input_id is None:
                self._drive_agent_turn(user_message)
            else:
                self._drive_agent_turn(user_message, input_id=input_id)
            succeeded = True
        except Exception as exc:
            self.transcript.append("error", "agent loop error", str(exc), state="error")
        finally:
            self._append_run_finished(succeeded)
            with self._lock:
                self._busy = False
                self._run_started_at = None
            self._invalidate()

    def _drive_agent_turn(
        self, user_message: str, *, input_id: str | None = None
    ) -> None:
        """按会话原目录执行并刷新确认菜单；传参：正文及已接纳输入身份；返回：无。"""
        # 【全屏会话】【工作区恢复】1. 恢复会话只读原归属，缺失目录或归属必须在接纳输入前报错
        workspace = WorkspaceStore(self.data_root).for_session(self.state.session_id)
        workspace.require_available()
        # 【全屏会话】【工作区恢复】2. 宿主管理的文件工具随原目录切换，外部注入依赖仍归调用者
        if self._owns_registry and workspace.project_root != self.project_root:
            self.registry.close()
            self.registry = build_tool_registry(
                repo_root=workspace.project_root, data_root=self.data_root
            )
        self.project_root = workspace.project_root
        if (
            self._approval_session_id
            and self._approval_session_id != self.state.session_id
        ):
            self.approval_session.close()
            self.approval_session = ApprovalSession()
        drive_agent_turn(
            state=self.state,
            transcript=self.transcript,
            project_root=self.project_root,
            data_root=self.data_root,
            llm_client=self.llm_client,
            tool_registry=self.registry,
            user_message=user_message,
            invalidate=self._invalidate,
            cancellation=self._cancellation,
            input_id=input_id,
            approval_session=self.approval_session,
        )
        self._approval_session_id = self.state.session_id
        self._refresh_confirmations()

    def _refresh_confirmations(self) -> None:
        """恢复当前会话中已展示的确认提案；传参：无；返回：无，不创建新会话。"""
        if not self.state.session_id or not SessionMessageStore(self.data_root).exists(
            self.state.session_id
        ):
            return
        with completion_service(self.data_root) as confirmations:
            text = self._completion_menu.update(
                confirmations.pending(self.state.session_id)
            )
        if text:
            self.transcript.append("system", "完成确认", text)

    def _approval_backend(
        self, req: approval.ApprovalRequest
    ) -> approval.ApprovalDecision:
        pending = PendingApproval(req, threading.Event())
        self._pending = pending
        # 审批必须显式等待用户输入；TUI 退出或桥接异常时由 approval 层 fail closed。
        self.transcript.append(
            "approval", "approval required", approval_body(req), state="pending"
        )
        self._invalidate()
        pending.event.wait()
        return pending.decision or approval.ApprovalDecision.DENY

    def _batch_approval_backend(self, batch: ApprovalBatch) -> BatchDecision:
        """等待一次完整逐项提交，关闭和新要求均可中断；传参：批次；返回：完整回执。"""
        try:
            return self.batch_approvals.request_batch(batch)
        finally:
            self._batch_identity = None
            self._invalidate()

    def _present_batch(self, identity: str, batch: ApprovalBatch) -> None:
        """显示同批所有动作及授权范围；传参：交互身份和批次；返回：无。"""
        self._batch_identity = identity
        self.transcript.append(
            "approval",
            "逐项授权",
            f"{format_batch(batch)}\n输入 /approve {identity} 后跟全部选择。",
            state="pending",
        )
        self._invalidate()

    def _handle_batch_approval(self, text: str) -> bool:
        """将明确选项或新要求交回等待中的批次；传参：界面输入；返回：是否消费。"""
        if self._batch_identity is None:
            return False
        try:
            if text.startswith("/approve"):
                self.batch_approvals.answer(text)
            elif text == "/stop":
                self._cancellation.cancel()
                self.batch_approvals.interrupt()
            else:
                SessionMessageStore(self.data_root).accept_input(
                    self.state.session_id, text
                )
                self.batch_approvals.interrupt()
        except (ValueError, approval.ApprovalUnavailable) as exc:
            self.transcript.append("error", "逐项授权", str(exc), state="error")
        return True

    def _handle_pending_approval(self, text: str) -> bool:
        pending = self._pending
        if pending is None:
            return False
        decision = parse_approval_decision(text)
        if decision is None:
            self.transcript.append("status", "approval", "Use /approve or /deny first.")
            return True
        pending.decision = decision
        pending.event.set()
        self._pending = None
        state: Literal["error", "success"]
        state = "error" if decision is approval.ApprovalDecision.DENY else "success"
        self.transcript.append(
            "status", "approval", f"decision: {decision.value}", state=state
        )
        return True

    def _invalidate(self) -> None:
        if self._app is not None:
            self._app.invalidate()

    def _append_run_finished(self, succeeded: bool) -> None:
        run_id = self.state.current_run_id or "-"
        state: Literal["success", "error"] = "success" if succeeded else "error"
        title = "run finished" if succeeded else "run failed"
        elapsed = self._elapsed_seconds()
        self.transcript.append(
            "status", title, f"run={run_id} elapsed={elapsed}", state=state
        )

    def _max_scroll(self, width: int, height: int) -> int:
        tail_lines = TAIL_WORKING_LINES if self._busy else 0
        return max(0, self.transcript.line_count(width, tail_lines=tail_lines) - height)

    def _working_fragments(self) -> StyleAndTextTuples | None:
        if not self._busy:
            return None
        frame = SPINNER_FRAMES[int(time.monotonic() * 4) % len(SPINNER_FRAMES)]
        return [
            ("", "\n"),
            ("class:working", f"{frame} Working... {self._elapsed_seconds()}"),
        ]

    def _elapsed_seconds(self) -> str:
        if self._run_started_at is None:
            return "0s"
        elapsed = int(time.monotonic() - self._run_started_at)
        minutes, seconds = divmod(elapsed, 60)
        return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


__all__ = ["FullscreenTui", "parse_approval_decision"]
