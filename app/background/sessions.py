"""后台会话持有执行者；持久元数据只引用正文、运行和授权快照。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from threading import RLock
from typing import Any

from app.background.events import EventBuffer
from app.background.attachments import compose_attachment_input
from app.completion_ui import completion_service
from app.approval_ui import approval_control
from app.session_assembly import (
    SessionBinding,
    accepted_input_text,
    current_session_binding,
    ensure_session_binding,
    record_run_error,
    user_context,
    user_lease,
)
from app.run_task import execute_context
from approval import ApprovalDecision, ApprovalRequest
from approval.channel import ApprovalChannel
from approval.batch_types import ApprovalBatch, BatchDecision, format_batch
from approval.session import ApprovalSession
from llm.base import LLMClient
from llm.public_config import (
    PUBLIC_MODEL_FIELDS,
    public_model_config,
    restore_model_config,
)
from llm.messages import (
    AssistantMessage,
    UserMessage,
    ThinkingPart,
    ToolCallPart,
    ToolResultMessage,
    model_visible_text,
    thaw_json_value,
)
from runtime.checkpoint import (
    load_latest_checkpoint_for_run,
    load_latest_checkpoint_for_session,
)
from runtime.lease import load_snapshot
from runtime.run_facts import RunFactStore, latest_lifecycle_from_facts
from runtime.session_message_store import (
    DEFAULT_HISTORY_PAGE_SIZE,
    SessionEntry,
    SessionMessageStore,
)
from runtime.persistence import RuntimeStore, record_key
from runtime.workspaces import WorkspaceStore
from runtime.session_runtime import RecoveryIntent, SessionRun, SessionRuntime
from runtime.session_state import SessionStateStore
from runtime.tool_operations import ToolOperationStore
from runtime.tool_results import saved_execution_details
from runtime.shared_budget import BudgetOwner, SharedRunBudget
from runtime.types import RunContext, Trigger, new_run_id
from schedules.notifications import NotificationStore
from tasks.ids import utc_now
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry
from tools.redacted_files import RedactedFiles
from triggers.resume import make_run_context as make_resume_context


@dataclass(frozen=True, slots=True)
class SessionServices:
    """组合根提供的目录和工厂，数据库连接在使用线程内创建。"""

    project_root: Path
    data_root: Path
    make_llm: Callable[[dict[str, object]], LLMClient]
    make_registry: Callable[[], ToolRegistry]
    workspace_factory: Callable[[Path], SessionServices] | None = None

    def for_workspace(self, project_root: Path) -> SessionServices:
        """取得会话原目录的执行工厂；参数：持久工作区路径；返回：独立目录依赖，不创建模型连接。"""
        if project_root == self.project_root:
            return self
        if self.workspace_factory is None:
            raise ValueError("workspace execution factory is not configured")
        return self.workspace_factory(project_root)


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """执行接纳元数据；正文及运行结果始终使用 Session 和 Run 主存。"""

    session_id: str
    compatibility_task_id: str | None = None
    status: str = "idle"
    stopped: bool = False
    intent: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""
    last_input_at: str = ""
    error: str | None = None
    pending_notice: str | None = None


class BackgroundSession:
    """一个会话的后台执行责任；界面连接的存在与否不影响运行。"""

    def __init__(self, record: SessionRecord, services: SessionServices) -> None:
        """连接持久正文、执行工厂和审批；传参：记录与依赖；返回：无。"""
        self.record, self.services = record, services
        self._lock = RLock()
        self.events = EventBuffer()
        self.approvals = ApprovalChannel(
            self._present_approval, present_batch=self._present_batch
        )
        self.approval_session = ApprovalSession()
        self.redacted_files = RedactedFiles()
        self.pending_approval: dict[str, Any] | None = None
        self._api_keys: dict[str, str] = {}
        self._execution_registry: ToolRegistry | None = None
        self._execution_model: dict[str, object] = {}
        self.records = BackgroundSessionRecords(services.data_root)
        self.messages = SessionMessageStore(services.data_root)
        self.workspace = WorkspaceStore(services.data_root).bind_session(
            record.session_id, services.project_root
        )
        self.facts = RunFactStore(services.data_root)
        self.runtime = SessionRuntime(
            record.session_id, messages=self.messages, facts=self.facts, run=self._run
        )

    def branch(self, entry_id: str) -> None:
        """空闲时切换会话分支，和输入接纳互斥；参数：目标条目；返回：无。"""
        with self._lock:
            if self.runtime.active or self.pending_approval is not None:
                raise ValueError(
                    "cannot change session branch while a run or approval is active"
                )
            self.messages.branch(self.record.session_id, entry_id)
            # 【会话】【分支切换】1. 新路径使用独立显示实例，旧执行回调仍指向旧缓存
            self.events = EventBuffer()

    def submit(
        self,
        text: str,
        *,
        input_id: str,
        model_config: dict[str, object],
        api_key: str | None = None,
        task_id: str | None = None,
        attachment_paths: tuple[str, ...] = (),
        reference_paths: tuple[str, ...] = (),
    ) -> str:
        """先保存配置和输入再唤醒后台，重复输入身份只保存一次；传参：正文、引用与配置；返回：输入编号。"""
        if not text.strip() and not attachment_paths and not reference_paths:
            raise ValueError("input must be non-empty")
        if set(model_config) - PUBLIC_MODEL_FIELDS:
            raise ValueError("background model config accepts public fields only")
        self.workspace.require_available()
        with self._lock:
            prepared = compose_attachment_input(
                text,
                attachment_paths=attachment_paths,
                reference_paths=reference_paths,
                project_root=self.services.project_root,
                session_id=self.record.session_id,
                redacted_files=self.redacted_files,
                lease=user_lease(
                    task_id=task_id
                    or self.record.compatibility_task_id
                    or self.record.session_id,
                    project_root=self.services.project_root,
                    data_root=self.services.data_root,
                ),
            )
            text = prepared.text
            if self.record.pending_notice is not None:
                self._notify(self.record.pending_notice)
            existing = next(
                (
                    row
                    for row in self.messages.read_entries_or_create(
                        self.record.session_id
                    )
                    if row.entry_id == input_id
                ),
                None,
            )
            if existing is not None:
                self.messages.accept_input(
                    self.record.session_id,
                    text,
                    input_id=input_id,
                    content=prepared.content,
                )
                return input_id
            state = self._state()
            change_focus = task_id is not None and not self.runtime.active
            if change_focus:
                state = replace(state, focus_task_id=task_id)
            with closing(TaskStore(self.services.data_root)) as store:
                state = ensure_session_binding(state, store, text)
            if change_focus:
                SessionStateStore(self.services.data_root).update_focus(
                    self.record.session_id,
                    focus_task_id=state.focus_task_id,
                    compatibility_task_id=state.compatibility_task_id,
                    summary="用户选择了当前工作",
                )
            if api_key is not None:
                self._api_keys[input_id] = api_key
            updated = replace(
                self.record,
                compatibility_task_id=state.compatibility_task_id,
                stopped=False,
                status=self.record.status if self.runtime.active else "queued",
                last_input_at=utc_now(),
                updated_at=utc_now(),
                error=None,
            )
            # 【后台会话】【输入接纳】正文与恢复配置同一提交，成功后才唤醒执行线程
            with self.records.database.transaction():
                self.records.save(updated)
                self.messages.accept_input(
                    self.record.session_id,
                    text,
                    input_id=input_id,
                    task_id=state.focus_task_id,
                    content=prepared.content,
                )
                self.records.save_input(
                    self.record.session_id,
                    input_id,
                    model_config,
                    needs_ephemeral_key=api_key is not None,
                )
            self.record = updated
            return self.runtime.submit(
                text,
                input_id=input_id,
                task_id=state.focus_task_id,
                content=prepared.content,
            )

    def recover(self) -> None:
        """重启只接续未结束工作，已完成、等待和明确停止均不重放；传参：无；返回：无。"""
        with self._lock:
            if self.runtime.active:
                return
            if self.record.pending_notice is not None:
                self._notify(self.record.pending_notice)
            if self.record.stopped or self.record.status not in {"queued", "running"}:
                return
            intent = self.record.intent
            if self.record.status == "running" and intent:
                previous = str(intent["run_id"])
                boundary = latest_lifecycle_from_facts(
                    self.facts.read_session_run(self.record.session_id, previous)
                )
                if boundary:
                    self._save(
                        replace(
                            self.record,
                            status=str(boundary["lifecycle"]),
                            pending_notice=previous,
                        )
                    )
                    self._notify(previous)
                    return
                checkpoint = load_latest_checkpoint_for_run(
                    previous, data_root=self.services.data_root
                )
                recovery = RecoveryIntent(
                    f"background-resume-{previous}",
                    previous,
                    str(intent["input_id"]),
                    "host_interrupted",
                    checkpoint.checkpoint_id if checkpoint else None,
                )
                self.runtime.resume(recovery)
                return
            self.runtime.resume()

    def confirm_completion(
        self,
        *,
        question_id: str,
        action_id: str,
        accepted: bool,
        model_config: dict[str, object],
        api_key: str | None = None,
    ) -> str:
        """将认证前端的明确选择接回原会话；传参：问题、动作、选择和临时凭据；返回：原输入编号。"""
        self.workspace.require_available()
        if set(model_config) - PUBLIC_MODEL_FIELDS:
            raise ValueError("background model config accepts public fields only")
        with self._lock:
            with (
                completion_service(self.services.data_root) as confirmations,
                self.records.database.transaction(),
            ):
                event = confirmations.decide(
                    self.record.session_id,
                    question_id,
                    action_id=action_id,
                    accepted=accepted,
                )
                self.records.save_input(
                    self.record.session_id,
                    str(event.payload["source_input_id"]),
                    model_config,
                    needs_ephemeral_key=api_key is not None,
                )
            if api_key is not None:
                self._api_keys[str(event.payload["source_input_id"])] = api_key
            self._save(
                replace(self.record, stopped=False, last_input_at=utc_now(), error=None)
            )
            self.runtime.resume()
            return str(event.payload["source_input_id"])

    def cancel(self, *, expected_run_id: str | None = None) -> None:
        """保存停止意图并传播取消；参数：可选的界面所见运行；返回：无，旧运行选择明确拒绝。"""
        with self._lock:
            if (
                expected_run_id is not None
                and expected_run_id != self.record.intent.get("run_id", "")
            ):
                raise ValueError("运行已变化，请刷新活动后再停止")
            try:
                if self.records.load(
                    self.record.session_id
                ) is not None or self.messages.exists(self.record.session_id):
                    self._save(
                        replace(
                            self.record,
                            stopped=True,
                            status="stopping" if self.runtime.active else "stopped",
                        )
                    )
            finally:
                # 【本机后台】【明确停止】持久化故障不能阻止向当前执行和审批传播取消
                self.runtime.cancel()
                self.approvals.interrupt()

    def close(self) -> None:
        """显式停止宿主时等待实际运行释放资源；传参：无；返回：无，失败直接上抛。"""
        # 【后台会话】【资源释放】关闭故障不能跳过后续清理，已接纳工作的停止和交接顺序保持
        with ExitStack() as cleanup:
            cleanup.callback(self.redacted_files.close)
            cleanup.callback(self.approval_session.close)
            cleanup.callback(self.approvals.close)
            cleanup.callback(self.runtime.wait_idle)
            cleanup.callback(self.runtime.close, cancel=True)
            cleanup.callback(self.cancel)

    def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        """为本会话及其子执行者保留可重连审批；传参：真实请求；返回：用户决定。"""
        try:
            return self.approvals.request(request)
        finally:
            with self._lock:
                self.pending_approval = None
                if not self.record.stopped:
                    self._save(replace(self.record, status="running"))

    def snapshot(
        self, *, after: int | None = None, history: bool = False
    ) -> dict[str, Any]:
        """投影界面状态，正文只从主存读取；传参：事件游标和历史开关；返回：可展示状态。"""
        with self._lock:
            state = self._state()
            result = {
                "session_id": self.record.session_id,
                "current_task_id": state.focus_task_id,
                "compatibility_task_id": state.compatibility_task_id,
                "current_run_id": self.record.intent.get("run_id", ""),
                "root_run_id": self.record.intent.get(
                    "root_run_id", self.record.intent.get("run_id", "")
                ),
                "status": self.record.status,
                "active": self.runtime.active,
                "stopped": self.record.stopped,
                "error": self.record.error,
                "approval": self.pending_approval,
                **self.events.read(after, include_streams=history),
            }
            result.update(
                self.workspace.snapshot(),
                data_space_id=self.records.database.data_space_id,
                data_root=str(self.services.data_root),
            )
            result["leaf_id"] = self.messages.current_leaf(self.record.session_id)
            if history or result["gap"]:
                result.update(
                    session_history_page(
                        self.services.data_root, self.record.session_id
                    )
                )
        if history or result["gap"]:
            result["questions"] = waiting_questions(
                self.services.data_root,
                self.record.session_id,
                str(self.record.intent.get("run_id", "")),
            )
        result["completion_requests"] = []
        if result["status"] in {"paused", "waiting_user"}:
            with completion_service(self.services.data_root) as confirmations:
                result["completion_requests"] = confirmations.pending(
                    self.record.session_id
                )
        return result

    def approve_batch(self, request: ApprovalBatch) -> BatchDecision:
        """同宿主保存整批交互，前端重连继续提交原编号；传参：待决定批次；返回：逐项决定。"""
        try:
            return self.approvals.request_batch(request)
        finally:
            with self._lock:
                self.pending_approval = None
                if not self.record.stopped:
                    self._save(replace(self.record, status="running"))

    def settings(self) -> dict[str, Any]:
        """读取实际执行模型、连接与权限，不创建新连接；参数：无；返回：只读设置快照。"""
        with self._lock:
            registry = self._execution_registry
            model = dict(self._execution_model)
            active = self.runtime.active
        return {
            "session_id": self.record.session_id,
            "active": active,
            "model": model if registry is not None else None,
            "approval_mode": self.approval_session.mode.value,
            "mcp": registry.source_status().get("mcp")
            if registry is not None
            else None,
            "execution_attached": registry is not None,
        }

    def approval_control(self, command: str, action_id: str) -> str:
        """仅接受前端显式模式与撤销动作；传参：命令和传输身份；返回：实际权限结果。"""
        result = approval_control(
            command,
            self.approval_session,
            data_root=self.services.data_root,
            session_id=self.record.session_id,
            task_id=self._state().focus_task_id,
            action_id=action_id,
        )
        if len(command.split()) > 1:
            self.approvals.interrupt()
        return result

    def _run(self, request: SessionRun) -> None:
        """在工作线程装配同一个 AgentLoop，实时事件交给界面缓存；传参：已接纳输入；返回：无。"""
        context = None
        try:
            self.workspace.require_available()
            with self._lock:
                context, budget = self._context(request)
                intent = _context_reference(
                    context,
                    root_run_id=budget.owner.run_id if budget else context.run_id,
                )
                self._save(
                    replace(self.record, status="running", intent=intent, error=None)
                )
                accepted = self.records.load_input(
                    self.record.session_id, request.input_id
                )
                options = restore_model_config(accepted["model_config"])
                api_key = self._api_keys.get(request.input_id)
                if accepted["needs_ephemeral_key"] and api_key is None:
                    self._save(
                        replace(
                            self.record,
                            status="paused",
                            pending_notice=context.run_id,
                            error="临时模型凭据在重启后不可用，请重新配置再继续",
                        )
                    )
                    self._notify(context.run_id)
                    return
                if api_key is not None:
                    options["api_key"] = api_key
            with closing(self.services.make_registry()) as registry:
                registry.bind_redacted_files(self.redacted_files)
                client = self.services.make_llm(options)
                with self._lock:
                    self._execution_registry = registry
                    self._execution_model = public_model_config(client)
                result = execute_context(
                    context,
                    data_root=self.services.data_root,
                    llm_client=client,
                    registry=registry,
                    cancellation=request.cancellation,
                    shared_budget=budget,
                    event_sink=partial(self.events.emit, run_id=context.run_id),
                    approval_session=self.approval_session,
                )
            with self._lock:
                self._save(
                    replace(
                        self.record,
                        status="stopped" if self.record.stopped else result.status,
                        pending_notice=context.run_id,
                    )
                )
        except Exception as exc:
            if context is not None:
                record_run_error(self.services.data_root, context, exc)
            with self._lock:
                self._save(
                    replace(
                        self.record,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        pending_notice=context.run_id if context else request.input_id,
                    )
                )
        finally:
            with self._lock:
                self._execution_registry = None
                self._execution_model = {}
        if self.record.pending_notice is not None:
            self._notify(self.record.pending_notice)

    def _context(
        self, request: SessionRun
    ) -> tuple[RunContext, SharedRunBudget | None]:
        """区分自动恢复与新的用户输入，恢复不刷新权限或预算；传参：输入引用；返回：上下文和共享账目。"""
        text = accepted_input_text(
            self.services.data_root, self.record.session_id, request.input_id
        )
        if request.recovery is not None:
            return self._recovery_context(request.recovery, text)
        checkpoint = load_latest_checkpoint_for_session(
            self.record.session_id, data_root=self.services.data_root
        )
        if checkpoint is not None and checkpoint.state.upper() != "DONE":
            context = make_resume_context(
                checkpoint=checkpoint, data_root=self.services.data_root
            )
            context.payload.update(
                message=text, input_message_id=request.input_id, resume_action="inspect"
            )
            return context, None
        state = self._state()
        lease = user_lease(
            task_id=state.focus_task_id or state.compatibility_task_id or "",
            project_root=self.services.project_root,
            data_root=self.services.data_root,
        )
        return user_context(
            state,
            lease,
            data_root=self.services.data_root,
            run_id=new_run_id(),
            input_id=request.input_id,
        ), None

    def _recovery_context(
        self, recovery: RecoveryIntent, text: str
    ) -> tuple[RunContext, SharedRunBudget]:
        """从恢复引用和原输入重建执行边界；传参：已接纳恢复及原文；返回：原权限及余额的上下文。"""
        intent = self.record.intent
        previous = recovery.previous_run_id
        if (
            str(intent["run_id"]) != previous
            or str(intent["input_id"]) != recovery.input_id
        ):
            raise ValueError("background recovery source changed before execution")
        checkpoint = load_latest_checkpoint_for_run(
            previous, data_root=self.services.data_root
        )
        if (checkpoint.checkpoint_id if checkpoint else None) != recovery.checkpoint_id:
            raise ValueError("background recovery checkpoint changed before execution")
        lease = load_snapshot(intent["lease"])
        if checkpoint is not None:
            context = make_resume_context(
                checkpoint=checkpoint, data_root=self.services.data_root
            )
            context.capability_lease = lease
            context.trigger = Trigger(intent["trigger"])
        else:
            context = RunContext(
                session_id=self.record.session_id,
                trigger=Trigger(intent["trigger"]),
                task_id=intent["task_id"],
                focus_task_id=intent["focus_task_id"],
                compatibility_task_id=self.record.compatibility_task_id,
                capability_lease=lease,
                payload={},
            )
        context.payload.update(
            message=text,
            input_message_id=recovery.input_id,
            previous_run_id=previous,
            resume_action="inspect",
            recovery_intent=asdict(recovery),
        )
        root = BudgetOwner(self.record.session_id, str(intent["root_run_id"]), lease)
        return context, SharedRunBudget.restore(root, self.facts)

    def _state(self) -> SessionBinding:
        """读取中立的持久会话投影；传参：无；返回：当前焦点与兼容身份。"""
        return current_session_binding(
            SessionBinding(
                self.record.session_id,
                compatibility_task_id=self.record.compatibility_task_id,
            ),
            self.services.data_root,
        )

    def _save(self, record: SessionRecord) -> None:
        """原子发布元数据后更新内存；传参：新记录；返回：无，失败不冒充接纳成功。"""
        updated = replace(record, updated_at=utc_now())
        self.records.save(updated)
        self.record = updated

    def _present_approval(self, identity: str, request: ApprovalRequest) -> None:
        """保存等待边界并交付准确审批范围；传参：请求编号与内容；返回：无。"""
        with self._lock:
            self.pending_approval = {
                "identity": identity,
                "tool": request.tool,
                "args": dict(request.args),
                "resource": asdict(request.resource) if request.resource else None,
                "session_id": request.session_id,
                "run_id": request.run_id,
                "force_confirmation": request.force_confirmation,
                "task_id": request.lease.task_id,
            }
            self._save(replace(self.record, status="waiting_approval"))
        NotificationStore(self.services.data_root).enqueue(
            f"approval-{identity}",
            title="Reins 需要授权",
            message=f"{request.tool} 等待你的决定，请重新打开聊天窗口查看具体范围。",
            source={
                "session_id": self.record.session_id,
                "run_id": request.run_id,
                "approval_id": identity,
            },
        )

    def _present_batch(self, identity: str, request: ApprovalBatch) -> None:
        """保存一次整批展示供前端重连，不丢逐项范围；传参：交互编号和批次；返回：无。"""
        with self._lock:
            self.pending_approval = {
                "identity": identity,
                "batch_id": request.batch_id,
                "body": format_batch(request),
                "session_id": self.record.session_id,
                "requests": [
                    {
                        "tool": item.tool,
                        "args": dict(item.args),
                        "resource": asdict(item.resource) if item.resource else None,
                        "operation_id": item.operation_id,
                        "session_id": item.session_id,
                        "task_id": item.lease.task_id,
                        "force_confirmation": item.force_confirmation,
                    }
                    for item in request.requests
                ],
            }
            self._save(replace(self.record, status="waiting_approval"))
        NotificationStore(self.services.data_root).enqueue(
            f"approval-{identity}",
            title="Reins 需要逐项授权",
            message=f"{len(request.requests)} 项操作等待你的选择，请打开聊天窗口查看具体范围。",
            source={
                "session_id": self.record.session_id,
                "run_id": request.requests[0].run_id,
                "approval_id": identity,
            },
        )

    def _history(self) -> list[dict[str, Any]]:
        """读取最近的用户和模型正文供重连展示；传参：无；返回：已保存的对话。"""
        return session_history(self.services.data_root, self.record.session_id)

    def _notify(self, run_id: str) -> None:
        """用稳定运行身份交付结束或等待通知，结果正文仍引用 Session；传参：运行编号；返回：无。"""
        labels = {
            "done": "本次运行结束",
            "paused": "工作待继续",
            "waiting_user": "等待你的答复",
            "waiting_approval": "等待授权",
            "stopped": "已请求停止",
            "failed": "运行失败",
        }
        boundary = (
            latest_lifecycle_from_facts(
                self.facts.read_session_run(self.record.session_id, run_id)
            )
            if run_id.startswith("run-")
            else {}
        )
        status = (
            "stopped"
            if self.record.stopped
            else str(boundary.get("lifecycle", self.record.status))
        )
        history = self._history()
        # 【后台通知】【运行归属】1. 工具调用没有回答正文，停止通知使用实际边界，不复用旧运行回答
        outputs = [
            row["text"]
            for row in history
            if row["role"] == "assistant"
            and row["run_id"] == run_id
            and row["text"].strip()
        ]
        questions = waiting_questions(
            self.services.data_root, self.record.session_id, run_id
        )
        message = (
            self.record.error
            or "\n".join(questions)
            or (outputs[-1] if outputs else labels.get(status, status))
        )
        notices = NotificationStore(self.services.data_root)
        identity = f"background-{run_id}"
        if notices.load(identity) is None:
            notices.enqueue(
                identity,
                title=f"Reins · {labels.get(status, status)}",
                message=message,
                source={
                    "session_id": self.record.session_id,
                    "run_id": run_id if run_id.startswith("run-") else None,
                    "input_id": None if run_id.startswith("run-") else run_id,
                },
            )
        with self._lock:
            self._save(replace(self.record, pending_notice=None))


class BackgroundSessionRecords:
    """后台接纳与恢复元数据，不复制消息或工具正文。"""

    def __init__(self, data_root: Path) -> None:
        """初始化后台接纳实体；参数：数据根；返回：无。"""
        self.database = RuntimeStore(data_root)

    def save_input(
        self,
        session_id: str,
        input_id: str,
        model_config: dict[str, object],
        *,
        needs_ephemeral_key: bool,
    ) -> None:
        """保存该输入冻结的公开模型选择；参数：会话、输入、配置及临时密钥标志；返回：无，不保存密钥。"""
        with self.database.transaction() as batch:
            identity = record_key(session_id, input_id)
            if batch.get("background_input", identity) is not None:
                return
            # 1. 【后台会话】【接纳输入】冻结配置只能引用本批次中实际接纳的输入
            if batch.get("session_entry", identity) is None:
                raise ValueError("accepted input entry is missing")
            batch.put(
                "background_input",
                identity,
                {
                    "model_config": model_config,
                    "needs_ephemeral_key": needs_ephemeral_key,
                },
                session_id=session_id,
                expected_revision=0,
            )

    def load_input(self, session_id: str, input_id: str) -> dict[str, Any]:
        """读取原输入的模型快照；参数：会话和输入编号；返回：冻结配置，缺失明确失败。"""
        with self.database.snapshot() as source:
            value = source.get("background_input", record_key(session_id, input_id))
            if value is None:
                raise ValueError("accepted input model snapshot is missing")
            if not isinstance(value, dict):
                raise ValueError("accepted input model snapshot is invalid")
            return value

    def load(self, session_id: str) -> SessionRecord | None:
        """读取恢复记录；参数：会话编号；返回：记录或尚未接纳。"""
        with self.database.snapshot() as source:
            row = source.get("background_session", session_id)
            return SessionRecord(**row) if row is not None else None

    def save(self, record: SessionRecord) -> None:
        """事务提交执行配置和意图；参数：恢复记录；返回：无。"""
        with self.database.transaction() as batch:
            batch.put(
                "background_session",
                record.session_id,
                asdict(record),
                session_id=record.session_id,
            )

    def list_recent(self) -> list[SessionRecord]:
        """列出已接纳会话；参数：无；返回：按最近输入排序的记录。"""
        with self.database.snapshot() as source:
            rows = sorted(
                source.list("background_session"),
                key=lambda row: (row["last_input_at"], row["updated_at"]),
                reverse=True,
            )
            return [SessionRecord(**row) for row in rows]


def load_session_records(data_root: Path) -> list[SessionRecord]:
    """枚举后台接纳的会话引用；参数：数据根；返回：按最近使用排列的记录。"""
    return BackgroundSessionRecords(data_root).list_recent()


def session_history_page(
    data_root: Path,
    session_id: str,
    *,
    leaf_id: str | None = None,
    before: str | None = None,
    limit: int = DEFAULT_HISTORY_PAGE_SIZE,
) -> dict[str, Any]:
    """读取一页当前或固定分支正文；传参：目录、会话及分页锚点；返回：历史和后续游标。"""
    messages = SessionMessageStore(data_root)
    if not messages.exists(session_id):
        if leaf_id is not None or before is not None:
            raise ValueError("history anchor refers to a missing session")
        return {
            "session_id": session_id,
            "leaf_id": None,
            "history": [],
            "next_before": None,
        }
    page = messages.history_page(
        session_id, leaf_id=leaf_id, before=before, limit=limit
    )
    rows = [
        history_entry(entry)
        for entry in page.entries
        if entry.message is not None
        and entry.input_source != "agent"
        and entry.input_kind is None
    ]
    return {
        "session_id": session_id,
        "leaf_id": page.leaf_id,
        "history": rows,
        "next_before": page.next_before,
    }


def session_history(data_root: Path, session_id: str) -> list[dict[str, Any]]:
    """从当前分支返回稳定消息身份，入站只显示一次；参数：目录与会话；返回：已保存历史。"""
    messages = SessionMessageStore(data_root)
    if not messages.exists(session_id):
        return []
    view = messages.materialize(session_id)
    return [
        history_entry(row)
        for row in view.entries
        if row.message is not None
        and row.input_source != "agent"
        and row.input_kind is None
    ]


def history_entry(entry: SessionEntry) -> dict[str, Any]:
    """把一条事实转换为界面消息及工具详情；参数：条目；返回：稳定引用与完整可见正文。"""
    message = entry.message
    row: dict[str, Any] = {
        "entry_id": entry.entry_id,
        "parent_id": entry.parent_id,
        "type": entry.type,
        "run_id": entry.run_id,
        "message_id": message.message_id if message is not None else None,
        "role": "branch",
        "text": model_visible_text(message) if message is not None else "",
    }
    if isinstance(message, UserMessage):
        row["role"] = "user"
    elif isinstance(message, AssistantMessage):
        row["role"] = "assistant"
        row["reasoning"] = "".join(
            part.text
            for part in message.content
            if isinstance(part, ThinkingPart) and part.visibility == "visible"
        )
        row["tool_calls"] = [
            {
                "call_id": part.call_id,
                "tool_name": part.tool_name,
                "args": thaw_json_value(part.arguments),
            }
            for part in message.content
            if isinstance(part, ToolCallPart)
        ]
    elif isinstance(message, ToolResultMessage):
        from app.background.tool_results import retained_result_source

        row.update(
            role="tool",
            tool_call_id=message.call_id,
            tool_name=message.tool_name,
            status=message.status,
            error=message.error,
            artifact_refs=list(message.artifact_refs),
        )
        row["execution"] = saved_execution_details(
            row["text"], tool_name=message.tool_name, call_id=message.call_id
        )
        row["result_source"] = retained_result_source(entry)
    return row


def waiting_questions(data_root: Path, session_id: str, run_id: str) -> list[str]:
    """读取当前运行实际提出而尚待用户答复的问题；传参：存储和运行引用；返回：问题正文。"""
    if not run_id:
        return []
    boundary = latest_lifecycle_from_facts(
        RunFactStore(data_root).read_session_run(session_id, run_id)
    )
    if boundary.get("lifecycle") != "waiting_user":
        return []
    return [
        str(row["call"]["args"]["question"])
        for row in ToolOperationStore(data_root).for_session(session_id)
        if row["run_id"] == run_id
        and row["state"] == "waiting_user"
        and row["call"]["tool_name"] == "ask_user"
    ]


def _context_reference(context: RunContext, *, root_run_id: str) -> dict[str, Any]:
    """保存重启所需身份和原权限，排除所有对话正文；传参：上下文与预算根；返回：引用对象。"""
    return {
        "run_id": context.run_id,
        "root_run_id": root_run_id,
        "task_id": context.task_id,
        "focus_task_id": context.focus_task_id,
        "input_id": context.payload["input_message_id"],
        "trigger": context.trigger.value,
        "lease": asdict(context.capability_lease),
    }
