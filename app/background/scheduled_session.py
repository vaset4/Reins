"""定时会话的界面接续视图，执行始终由原发生认领者持有。

作者：xxx
时间：2026-09-14 23:50:00
"""

from __future__ import annotations

from contextlib import closing
from typing import TYPE_CHECKING, Any

from app.background.events import EventBuffer
from app.completion_ui import completion_service
from app.background.sessions import session_history_page, waiting_questions
from approval import ApprovalDecision, ApprovalRequest, ApprovalUnavailable
from approval.channel import ApprovalChannel
from llm.messages import model_visible_text
from runtime.session_message_store import SessionMessageStore
from runtime.persistence import RuntimeStore
from runtime.session_state import SessionStateStore
from runtime.stream_events import StreamEvent
from runtime.workspaces import WorkspaceStore
from schedules.occurrences import OccurrenceRecord, OccurrenceStore
from schedules.persistence import SCHEDULE_DISPATCH_LOCK, claim_file
from schedules.store import ScheduleStore

if TYPE_CHECKING:
    from app.background.service import BackgroundService


class ScheduledSession:
    """只连接已存在的定时执行，不建立第二个 SessionRuntime。"""

    def __init__(self, occurrence_id: str, service: BackgroundService) -> None:
        """引用原发生、停止通道和事件缓存；传参：发生及宿主；返回：无。"""
        self.identity, self.service = occurrence_id, service
        self.events = EventBuffer()
        self.store = OccurrenceStore(service.services.data_root)
        self.approvals = ApprovalChannel(self._no_approval)

    def emit_event(self, run_id: str, event: StreamEvent) -> None:
        """保存执行端已冻结身份的事件；参数：实际运行及事件；返回：无。"""
        self.events.emit(event, run_id=run_id)

    def submit(
        self,
        text: str,
        *,
        input_id: str,
        model_config: dict[str, object],
        api_key: str | None = None,
        task_id: str | None = None,
    ) -> str:
        """原运行接收直接用户输入，暂停时显式接续同一次发生；传参：正文与身份；返回：已保存输入编号。"""
        del model_config, api_key, task_id
        if not text.strip():
            raise ValueError("input must be non-empty")
        root = self.service.services.data_root
        with claim_file(root / SCHEDULE_DISPATCH_LOCK) as acquired:
            if not acquired:
                raise ValueError(
                    "schedule dispatch is in progress; retry using the same input identity"
                )
            # 1. 【定时工作】【接纳接续】用户输入和原发生的接续意图同批次发布，失败不能留下已确认输入
            with RuntimeStore(root).transaction():
                record = self._record()
                SessionMessageStore(root).accept_input(
                    record.session_id, text, input_id=input_id
                )
                if record.status == "settled":
                    self.store.request_resume(
                        self.identity,
                        request_id=input_id,
                        input_id=input_id,
                        allow_completed=True,
                    )
                    with closing(ScheduleStore(root)) as schedules:
                        schedules.update(record.schedule_id, enabled=True, paused=False)
        return input_id

    def cancel(self, *, expected_run_id: str | None = None) -> None:
        """停止本次发生并暂停未来唤醒；参数：可选所见运行；返回：无，状态变化需重新选择。"""
        record = self._record()
        if expected_run_id is not None and expected_run_id != record.run_id:
            raise ValueError("定时运行已变化，请刷新活动后再停止")
        with closing(ScheduleStore(self.service.services.data_root)) as schedules:
            schedules.update(record.schedule_id, paused=True)
        self.service.cancel_occurrence(self.identity)

    def confirm_completion(
        self,
        *,
        question_id: str,
        action_id: str,
        accepted: bool,
        api_key: str | None = None,
    ) -> str:
        """接纳真实用户确认但不扩大原定时权限；传参：明确动作及可选凭据；返回：输入身份。"""
        record = self._record()
        root = self.service.services.data_root
        WorkspaceStore(root).for_session(self._record().session_id).require_available()
        with (
            completion_service(root) as confirmations,
            RuntimeStore(root).transaction(),
        ):
            event = confirmations.decide(
                record.session_id, question_id, action_id=action_id, accepted=accepted
            )
            input_id = str(event.payload["source_input_id"])
            entry = next(
                row
                for row in SessionMessageStore(root).read_entries(record.session_id)
                if row.entry_id == input_id
            )
            assert entry.message is not None
            return self.submit(
                model_visible_text(entry.message),
                input_id=input_id,
                model_config={},
                api_key=api_key,
            )

    def snapshot(
        self, *, after: int | None = None, history: bool = False
    ) -> dict[str, Any]:
        """投影原发生及其唯一会话；传参：展示游标；返回：运行状态和已有正文。"""
        record = self._record()
        state = SessionStateStore(self.service.services.data_root).load(
            record.session_id
        )
        with closing(ScheduleStore(self.service.services.data_root)) as schedules:
            schedule = schedules.load_schedule(record.schedule_id)
        if schedule is None:
            raise FileNotFoundError(record.schedule_id)
        result = {
            "session_id": record.session_id,
            "current_task_id": state.focus_task_id if state else None,
            "compatibility_task_id": state.compatibility_task_id if state else None,
            "current_run_id": record.run_id,
            "root_run_id": record.budget_run_id or record.run_id,
            "status": record.result_status or record.status,
            "active": self.service.occurrence_active(self.identity),
            "stopped": schedule.paused or not schedule.enabled,
            "error": record.error,
            "approval": None,
            "occurrence_id": self.identity,
            **self.events.read(after, include_streams=history),
        }
        result.update(
            self.service.workspaces.for_session(record.session_id).snapshot(),
            data_space_id=self.service.workspaces.database.data_space_id,
            data_root=str(self.service.services.data_root),
        )
        if history or result["gap"]:
            result.update(
                session_history_page(self.service.services.data_root, record.session_id)
            )
            result["questions"] = waiting_questions(
                self.service.services.data_root, record.session_id, record.run_id
            )
        result["completion_requests"] = []
        if result["status"] in {"paused", "waiting_user"}:
            with completion_service(self.service.services.data_root) as confirmations:
                result["completion_requests"] = confirmations.pending(record.session_id)
        return result

    def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        """定时运行不能通过界面连接扩大原权限；传参：审批请求；返回：无，明确要求修正授权。"""
        raise ApprovalUnavailable(
            "定时工作沿用已保存的权限，请在用户会话中明确授权并重新安排"
        )

    def approval_control(self, command: str, action_id: str) -> str:
        """定时连接不提供用户运行模式提权；传参：界面命令和身份；返回：无，明确说明原权限不变。"""
        raise ValueError("定时工作沿用原权限；请在用户会话管理授权后重新安排")

    def _record(self) -> OccurrenceRecord:
        """要求原发生仍在持久主存；传参：无；返回：当前记录。"""
        record = self.store.load(self.identity)
        if record is None:
            raise FileNotFoundError(self.identity)
        return record

    def _no_approval(self, identity: str, request: ApprovalRequest) -> None:
        """拒绝与定时执行者无关联的审批；传参：编号和请求；返回：无。"""
        raise ApprovalUnavailable(
            "scheduled session has no interactive approval request"
        )
