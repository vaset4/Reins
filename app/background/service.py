"""后台组合根：会话执行、时间唤醒和通知各自持有真实资源。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

import logging
from builtins import ExceptionGroup
from contextlib import closing
from dataclasses import asdict
from functools import partial
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any

from app.background.sessions import (
    BackgroundSession,
    BackgroundSessionRecords,
    SessionRecord,
    SessionServices,
    load_session_records,
)
from app.background.scheduled_session import ScheduledSession
from app.scheduled_run import create_scheduler
from approval import ApprovalDecision, ApprovalRequest, ApprovalUnavailable
from approval.batch_types import ApprovalBatch, BatchDecision
from runtime.cancellation import CancellationToken
from runtime.cron import CronScheduler
from runtime.types import new_session_id
from runtime.workspaces import WorkspaceStore
from schedules.notifications import NotificationStore
from schedules.occurrences import OccurrenceStore
from schedules.store import ScheduleStore
from schedules.persistence import record_path
from schedules.windows_notifications import WindowsNotifications

PUMP_SECONDS = 0.5
_LOG = logging.getLogger(__name__)


class BackgroundService:
    """实际后台执行责任；关闭前台连接不调用本对象的停止入口。"""

    def __init__(self, services: SessionServices) -> None:
        """绑定生产依赖并载入已接纳的会话；传参：工厂与目录；返回：无。"""
        self.services = services
        self._lock = RLock()
        self.stopping = Event()
        self.workspaces = WorkspaceStore(services.data_root)
        self.records = BackgroundSessionRecords(services.data_root)
        self._sessions: dict[str, BackgroundSession] = {}
        self._scheduled_sessions: dict[str, ScheduledSession] = {}
        self._jobs: dict[str, tuple[str, Thread, CancellationToken]] = {}
        self._pumps: list[Thread] = []
        self._errors: dict[str, str] = {}
        from app.background.file_restore import FileRestoreJobs

        self.file_restore = FileRestoreJobs(self)

    def start(self) -> None:
        """恢复未结束会话并启动时间和通知接收者；传参：无；返回：无。"""
        self.file_restore.start()
        for row in load_session_records(self.services.data_root):
            if row.pending_notice is not None or (
                not row.stopped and row.status in {"queued", "running"}
            ):
                session = self.attach(row.session_id)
                assert isinstance(session, BackgroundSession)
                session.recover()
        self._pumps = [
            Thread(target=self._schedule_pump, name="reins-schedules"),
            Thread(target=self._notification_pump, name="reins-notifications"),
        ]
        for thread in self._pumps:
            thread.start()

    def attach(
        self, session_id: str | None = None, *, project_root: Path | None = None
    ) -> BackgroundSession | ScheduledSession:
        """接回指定或启动工作区最近会话；参数：会话及启动目录；返回：原归属会话，未知编号拒绝。"""
        with self._lock:
            if self.stopping.is_set():
                raise RuntimeError("background host is stopping")
            if session_id is None:
                root = (
                    project_root
                    if project_root is not None
                    else self.services.project_root
                )
                workspace = self.workspaces.register(root)
                with self.workspaces.database.index_connection() as connection:
                    row = connection.execute(
                        """
                        SELECT b.session_id FROM records b
                        WHERE b.kind='background_session' AND b.workspace_id=?
                        ORDER BY json_extract(b.payload,'$.last_input_at') DESC,
                            (SELECT MAX(e.sequence) FROM records e WHERE e.kind='session_entry'
                             AND e.session_id=b.session_id AND json_extract(e.payload,'$.type')='inbound') DESC,
                            b.sequence DESC LIMIT 1
                    """,
                        (workspace.workspace_id,),
                    ).fetchone()
                if row is None:
                    return self.create_session(root)
                session_id = row[0]
            identity = session_id
            if not isinstance(identity, str):
                raise ValueError("session identity must be a string")
            record_path(self.services.data_root / "sessions", identity)
            for occurrence in OccurrenceStore(self.services.data_root).list_all():
                if occurrence.session_id == identity:
                    if identity not in self._scheduled_sessions:
                        self._scheduled_sessions[identity] = ScheduledSession(
                            occurrence.occurrence_id, self
                        )
                    return self._scheduled_sessions[identity]
            if identity not in self._sessions:
                if not self.workspaces.messages.exists(identity):
                    raise FileNotFoundError(f"session does not exist: {identity}")
                workspace = self.workspaces.for_session(identity)
                services = self.services.for_workspace(workspace.project_root)
                self._sessions[identity] = BackgroundSession(
                    self.records.load(identity) or SessionRecord(identity), services
                )
            return self._sessions[identity]

    def create_session(self, project_root: Path) -> BackgroundSession:
        """在明确目录原子创建空交互会话；参数：当前工作区；返回：已持久绑定的会话，不调用模型。"""
        from tasks.ids import utc_now

        with self._lock:
            if self.stopping.is_set():
                raise RuntimeError("background host is stopping")
            identity = new_session_id()
            with self.workspaces.database.transaction():
                workspace = self.workspaces.bind_session(identity, project_root)
                record = SessionRecord(identity, updated_at=utc_now())
                self.records.save(record)
            session = BackgroundSession(
                record, self.services.for_workspace(workspace.project_root)
            )
            self._sessions[identity] = session
            return session

    def occurrence_active(self, identity: str) -> bool:
        """查询原定时发生是否有执行线程；传参：发生编号；返回：实际持有情况。"""
        with self._lock:
            return identity in self._jobs

    def cancel_occurrence(self, identity: str) -> None:
        """将停止传给持有原发生的执行者；传参：发生编号；返回：无。"""
        with self._lock:
            job = self._jobs.get(identity)
            if job is not None:
                job[2].cancel()

    def approve(self, request: ApprovalRequest) -> ApprovalDecision:
        """按所属根会话路由并发审批，不能由后发请求覆盖前一个；传参：实际审批；返回：决定。"""
        with self._lock:
            session = self._sessions.get(request.owner_session_id or request.session_id)
        if session is None:
            raise ApprovalUnavailable(
                "该后台工作没有交互审批通道；请打开对应会话后明确继续"
            )
        return session.approve(request)

    def status(self) -> dict[str, Any]:
        """展示后台自身状态、独立运行和通知回执；传参：无；返回：只读状态。"""
        with self._lock:
            sessions, jobs, errors = (
                tuple(self._sessions.values()),
                tuple(self._jobs),
                dict(self._errors),
            )
        notices = NotificationStore(self.services.data_root).list_all(unread_only=True)
        return {
            "status": "stopping" if self.stopping.is_set() else "running",
            "errors": errors,
            "sessions": [session.snapshot() for session in sessions],
            "running_occurrences": list(jobs),
            "unread_notifications": len(notices),
        }

    def approve_batch(self, batch: ApprovalBatch) -> BatchDecision:
        """路由真实运行的整批申请，不能由模型参数指定权限实例；传参：冻结批次；返回：逐项决定。"""
        request = batch.requests[0]
        with self._lock:
            session = self._sessions.get(request.owner_session_id or request.session_id)
        if session is None:
            raise ApprovalUnavailable("该后台工作没有交互审批通道")
        return session.approve_batch(batch)

    def stop(self) -> None:
        """发出明确停止并取消已接纳执行，进程释放锁后才算宿主退出；传参：无；返回：无。"""
        self.stopping.set()
        with self._lock:
            sessions, jobs = tuple(self._sessions.values()), tuple(self._jobs.values())
        errors: list[Exception] = []
        for session in sessions:
            try:
                session.cancel()
            except Exception as exc:
                errors.append(exc)
                self._component_failed(f"stop:{session.record.session_id}", exc)
        for _schedule, _thread, cancellation in jobs:
            cancellation.cancel("host_stop")
        if errors:
            raise ExceptionGroup(
                "work cancellation requested, but durable stop intent could not be saved",
                errors,
            )

    def close(self) -> None:
        """等候各执行责任释放资源；传参：无；返回：无，关闭错误保留日志。"""
        self.stop()
        self.file_restore.close()
        for thread in self._pumps:
            thread.join()
        with self._lock:
            jobs, sessions = tuple(self._jobs.values()), tuple(self._sessions.values())
        for _schedule, thread, _cancel in jobs:
            thread.join()
        for session in sessions:
            session.close()

    def notifications(self, action: str, identity: str | None = None) -> dict[str, Any]:
        """读取通知或记录界面确认/显式重发；传参：动作及编号；返回：实际持久状态。"""
        store = NotificationStore(self.services.data_root)
        if action == "list":
            return {
                "notifications": [
                    asdict(row) for row in store.list_all(unread_only=True)
                ]
            }
        if identity is None:
            raise ValueError("notification identity is required")
        if action == "get":
            record = store.load(identity)
            if record is None:
                raise FileNotFoundError(identity)
            return asdict(record)
        if action == "visible":
            return asdict(store.acknowledge(identity))
        if action == "read":
            return asdict(store.acknowledge(identity, read=True))
        if action == "retry":
            return asdict(store.retry(identity))
        raise ValueError(f"unknown notification action: {action}")

    def _scheduler(self, *, session: ScheduledSession | None = None) -> CronScheduler:
        """在当前工作线程创建调度依赖；传参：无；返回：需要关闭的调度器。"""
        services = self.services
        if session is not None:
            workspace = self.workspaces.for_session(session.snapshot()["session_id"])
            services = self.services.for_workspace(workspace.project_root)
        return create_scheduler(
            project_root=services.project_root,
            data_root=services.data_root,
            llm_factory=services.make_llm,
            registry_factory=services.make_registry,
            event_sink=session.emit_event if session else None,
        )

    def _schedule_pump(self) -> None:
        """接纳时间事件，同一计划的发生串行、不同计划可并行；传参：无；返回：无。"""
        try:
            with closing(self._scheduler()) as scheduler:
                while not self.stopping.is_set():
                    scheduler.accept_due()
                    scheduler.accept_replies()
                    self._dispatch_occurrences()
                    self.stopping.wait(PUMP_SECONDS)
        except Exception as exc:
            self._component_failed("schedules", exc)

    def _dispatch_occurrences(self) -> None:
        """从持久发生派发未完成工作并传播明确取消；传参：无；返回：无。"""
        root = self.services.data_root
        with closing(ScheduleStore(root)) as store, self._lock:
            maintenance_busy = False
            for _identity, (schedule_id, _thread, cancellation) in self._jobs.items():
                job = store.load_schedule(schedule_id)
                if job is not None and not job.enabled:
                    cancellation.cancel("schedule_cancelled")
                if job is not None and job.knowledge_origin is not None:
                    maintenance_busy |= (
                        job.knowledge_origin.get("work_kind") == "knowledge_maintenance"
                    )
            busy = {item[0] for item in self._jobs.values()}
            for row in sorted(
                OccurrenceStore(root).list_all(), key=lambda item: item.scheduled_at
            ):
                if self.stopping.is_set():
                    return
                if (
                    row.status
                    not in {"queued", "running", "result_ready", "resume_queued"}
                    or row.schedule_id in busy
                ):
                    continue
                job = store.load_schedule(row.schedule_id)
                if job is None or not job.enabled or job.paused:
                    continue
                maintenance = (
                    job.knowledge_origin is not None
                    and job.knowledge_origin.get("work_kind") == "knowledge_maintenance"
                )
                if maintenance and maintenance_busy:
                    continue
                cancellation = CancellationToken()
                thread = Thread(
                    target=partial(
                        self._run_occurrence, row.occurrence_id, cancellation
                    ),
                    name=f"scheduled-{row.occurrence_id}",
                )
                self._jobs[row.occurrence_id] = (row.schedule_id, thread, cancellation)
                busy.add(row.schedule_id)
                maintenance_busy |= maintenance
                thread.start()

    def _run_occurrence(self, identity: str, cancellation: CancellationToken) -> None:
        """用生产调度入口执行已认领发生；传参：发生编号与停止信号；返回：无。"""
        try:
            occurrence = OccurrenceStore(self.services.data_root).load(identity)
            assert occurrence is not None
            session = (
                self.attach(occurrence.session_id) if occurrence.session_id else None
            )
            assert session is None or isinstance(session, ScheduledSession)
            with closing(self._scheduler(session=session)) as scheduler:
                scheduler.run_occurrence(identity, cancellation=cancellation)
            with self._lock:
                self._errors.pop(f"occurrence:{identity}", None)
        except Exception as exc:
            self._component_failed(f"occurrence:{identity}", exc)
        finally:
            with self._lock:
                self._jobs.pop(identity, None)

    def _notification_pump(self) -> None:
        """在同一原生窗口线程提交通知并接收可见回执；传参：无；返回：无。"""
        try:
            with closing(WindowsNotifications(self.services.data_root)) as channel:
                store = NotificationStore(self.services.data_root)
                while not self.stopping.is_set():
                    for record in store.list_all():
                        if record.delivery_status in {"pending", "failed", "sending"}:
                            store.deliver(record.notification_id, channel.send)
                    channel.pump()
                    self.stopping.wait(PUMP_SECONDS)
        except Exception as exc:
            self._component_failed("notifications", exc)

    def _component_failed(self, component: str, error: Exception) -> None:
        """公开停止工作的组件及根因，避免后台静默失效；传参：组件和异常；返回：无。"""
        with self._lock:
            self._errors[component] = f"{type(error).__name__}: {error}"
        _LOG.exception("【本机后台】【执行失败】%s", component)
