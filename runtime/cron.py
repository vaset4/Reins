"""持久时间发生的调度与交接，实际工作仍进入统一AgentLoop。

作者：xxx
时间：2026-09-14 19:16:10
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from llm.messages import AssistantMessage, model_visible_text
from runtime.cancellation import CancellationToken
from runtime.persistence import RuntimeStore
from runtime.run_facts import RunFactStore, latest_lifecycle_from_facts
from runtime.session_message_store import SessionMessageStore
from runtime.workspaces import WorkspaceStore
from schedules.notifications import NotificationStore
from schedules.occurrences import OccurrenceRecord, OccurrenceStore
from schedules.persistence import SCHEDULE_DISPATCH_LOCK, claim_file
from schedules.store import ScheduleRecord, ScheduleStore
from schedules.timing import (
    TimeIntent,
    format_instant,
    interval_delta,
    parse_instant,
    require_aware,
)

PAUSE_ERROR_TYPES = frozenset(
    {"MissingConfigurationError", "PromptNotSelfContainedError"}
)
_PENDING_STATES = frozenset({"queued", "running", "result_ready", "resume_queued"})


@dataclass(frozen=True, slots=True)
class CronExecution:
    """交给生产入口的已认领工作，包含原始发生身份与实际取消信号。"""

    schedule: ScheduleRecord
    occurrence: OccurrenceRecord
    now: datetime
    cancellation: CancellationToken
    previous_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class CronOutcome:
    """统一运行的返回值；done只表示本次运行结束。"""

    status: str
    output: str = ""
    task_id: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CronRunResult:
    """提供给宿主的发生与执行结果，通知状态由独立回执查询。"""

    job_id: str
    status: str
    task_id: str | None = None
    error: str | None = None
    run_id: str | None = None
    session_id: str | None = None
    occurrence_id: str | None = None
    notification_id: str | None = None


class CronScheduler:
    """先持久接纳到期发生，再由可重启执行者认领；不创建缺省模型或把暂停算成功。"""

    def __init__(
        self,
        *,
        project_root: Path,
        data_root: Path,
        execute: Callable[[CronExecution], CronOutcome] | None = None,
    ) -> None:
        """绑定持久存储与生产执行入口；传参：目录及执行依赖；返回：无。"""
        self._data_root = data_root
        self._project_root = project_root.resolve()
        self._store = ScheduleStore(self._data_root)
        self.occurrences = OccurrenceStore(self._data_root)
        self.notifications = NotificationStore(self._data_root)
        self._execute = execute

    def close(self) -> None:
        """释放调度线程的索引连接；传参：无；返回：无。"""
        self._store.close()

    def register_job(
        self, *, name: str, task: str, interval_seconds: int, prompt: str
    ) -> ScheduleRecord:
        """保留已存在的立即首次执行间隔入口；传参：计划内容；返回：持久记录。"""
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")
        workspace = WorkspaceStore(self._data_root).register(self._project_root)
        return self._store.create_schedule(
            f"cron_{uuid4().hex}",
            f"interval:{interval_seconds}",
            workspace_id=workspace.workspace_id,
            target_task_id=task,
            next_run_at=_utc_now(),
            name=name,
            prompt=prompt,
        )

    def list_jobs(self) -> list[ScheduleRecord]:
        """列出启用计划，暂停状态保留在记录内；传参：无；返回：计划列表。"""
        return self._store.list_enabled_schedules()

    def accept_due(self, *, now: datetime | None = None) -> list[OccurrenceRecord]:
        """接纳到期意图后推进时间游标，重投使用同一发生；传参：时钟；返回：接纳记录。"""
        current = require_aware(now or datetime.now(timezone.utc))
        accepted: list[OccurrenceRecord] = []
        with claim_file(self._data_root / SCHEDULE_DISPATCH_LOCK) as acquired:
            if not acquired:
                return accepted
            for job in self._store.list_enabled_schedules():
                if job.paused or job.next_run_at is None:
                    continue
                if not self._maintenance_ready(job):
                    continue
                try:
                    due = parse_instant(job.next_run_at)
                    if due > current:
                        continue
                    next_at = TimeIntent(job.cron, job.timezone_name).after(
                        due, now=current
                    )
                except ValueError as exc:
                    self._store.update(
                        job.schedule_id, paused=True, last_error=str(exc)
                    )
                    raise ValueError(
                        f"invalid schedule {job.schedule_id}: {exc}"
                    ) from exc
                # 1. 【定时工作】【接纳发生】发生、会话归属和计划游标同批次发布
                with RuntimeStore(self._data_root).transaction():
                    record = self.occurrences.accept(
                        job.schedule_id,
                        format_instant(due),
                        next_run_at=format_instant(next_at)
                        if next_at is not None
                        else None,
                        schedule_snapshot=asdict(job),
                    )
                    self._store.update_next_run_at(
                        job.schedule_id, record.next_run_at, last_run_at=job.last_run_at
                    )
                accepted.append(record)
        return accepted

    def _maintenance_ready(self, job: ScheduleRecord) -> bool:
        """【知识维护】【同源接纳】交互空闲且原来源无在途工作才冻结下一发生；参数：计划；返回：能否派发。"""
        origin = job.knowledge_origin
        if origin is None or origin.get("work_kind") != "knowledge_maintenance":
            return True
        from runtime.model_dispatch import foreground_active

        if foreground_active(self._data_root):
            return False
        for occurrence in self.occurrences.list_all():
            if occurrence.schedule_id == job.schedule_id or occurrence.status not in {
                "queued",
                "running",
                "resume_queued",
            }:
                continue
            source = (occurrence.schedule_snapshot or {}).get("knowledge_origin")
            if (
                source
                and source.get("work_kind") == "knowledge_maintenance"
                and source["source_session_id"] == origin["source_session_id"]
            ):
                return False
        return True

    def run_due_jobs(self, *, now: datetime | None = None) -> list[CronRunResult]:
        """供同步调用方执行到期与中断工作；传参：时钟；返回：实际处理结果。"""
        current = require_aware(now or datetime.now(timezone.utc))
        self.accept_due(now=current)
        self.accept_replies()
        results: list[CronRunResult] = []
        for occurrence in self.occurrences.list_all():
            if occurrence.status in _PENDING_STATES:
                result = self.run_occurrence(occurrence.occurrence_id, now=current)
                if result is not None:
                    results.append(result)
        return results

    def accept_replies(self) -> None:
        """把终态交接边界尚未投影的真实用户输入交给原发生；传参：无；返回：无，不重试已交付输入。"""
        messages = SessionMessageStore(self._data_root)
        with claim_file(self._data_root / SCHEDULE_DISPATCH_LOCK) as acquired:
            if not acquired:
                return
            for record in self.occurrences.list_all():
                if record.status != "settled" or not record.session_id:
                    continue
                replies = [
                    row
                    for row in messages.pending_inputs(record.session_id)
                    if row.input_source != "agent"
                    and row.entry_id != record.resume_input_id
                ]
                if not replies:
                    continue
                job = self._store.load_schedule(record.schedule_id)
                boundary = latest_lifecycle_from_facts(
                    RunFactStore(self._data_root).read_run(record.run_id)
                )
                if (
                    job is None
                    or not job.enabled
                    or boundary.get("reason")
                    in {"cancelled", "manual_pause", "user_stop"}
                ):
                    continue
                self.occurrences.request_resume(
                    record.occurrence_id,
                    request_id=replies[0].entry_id,
                    input_id=replies[0].entry_id,
                    allow_completed=True,
                )
                self._store.update(job.schedule_id, paused=False)

    def run_occurrence(
        self,
        identity: str,
        *,
        now: datetime | None = None,
        cancellation: CancellationToken | None = None,
    ) -> CronRunResult | None:
        """持有真实进程锁执行一次发生，恢复先核对已落盘结果；传参：发生、时钟、取消；返回：结果。"""
        current = require_aware(now or datetime.now(timezone.utc))
        claims = ExitStack()
        try:
            record = claims.enter_context(self.occurrences.claim(identity))
            if record is None or record.status not in _PENDING_STATES:
                return None
            current_job = self._store.load_schedule(record.schedule_id)
            if current_job is None:
                raise FileNotFoundError(record.schedule_id)
            job = (
                ScheduleRecord.from_dict(record.schedule_snapshot)
                if record.schedule_snapshot
                else current_job
            )
            observed = self._observed_outcome(record)
            if observed is not None:
                return self._complete(job, record, outcome=observed, current=current)
            if not current_job.enabled or current_job.paused:
                return None
            origin = job.knowledge_origin
            if (
                origin is not None
                and origin.get("work_kind") == "knowledge_maintenance"
            ):
                from hashlib import sha256
                from runtime.model_dispatch import foreground_active

                if foreground_active(self._data_root):
                    return None
                source_key = sha256(
                    str(origin["source_session_id"]).encode()
                ).hexdigest()
                if not claims.enter_context(
                    claim_file(
                        self._data_root
                        / "runtime/locks/knowledge"
                        / f"{source_key}.lock"
                    )
                ):
                    return None
            if not job.prompt or not job.prompt.strip():
                return self._complete(
                    job,
                    record,
                    outcome=CronOutcome("failed", error="prompt_not_self_contained"),
                    current=current,
                )
            if job.kind == "reminder":
                # 【定时工作】【提醒交付】到期仅创建通知；发送和用户确认仍由通知回执证明
                record = replace(record, session_id="", run_id="", run_ids=())
                return self._complete(
                    job,
                    record,
                    outcome=CronOutcome("notification_queued", job.prompt),
                    current=current,
                )
            if self._execute is None:
                raise RuntimeError(
                    "cron execution requires the production runtime factory"
                )
            previous = (
                record.run_id if record.status in {"running", "resume_queued"} else None
            )
            started = self.occurrences.begin(
                record,
                resume=previous is not None,
                renew_budget=record.status == "resume_queued",
            )
            request = CronExecution(
                job, started, current, cancellation or CancellationToken(), previous
            )
            outcome = self._invoke(request)
            return self._complete(job, started, outcome=outcome, current=current)
        finally:
            # 【定时工作】【认领释放】统一释放发生和同源锁，执行异常保持向外传播
            claims.close()

    def _invoke(self, request: CronExecution) -> CronOutcome:
        """调用注入的共同执行入口并保留明确失败；传参：工作；返回：实际运行边界。"""
        assert self._execute is not None
        try:
            outcome = self._execute(request)
            _run_status(outcome.status)
            return outcome
        except TimeoutError as exc:
            return CronOutcome("timeout", error=str(exc))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if _should_pause(exc):
                self._store.update(request.schedule.schedule_id, paused=True)
            return CronOutcome("failed", error=error)

    def _observed_outcome(self, record: OccurrenceRecord) -> CronOutcome | None:
        """认领者先核对运行事实，已完成但未确认的工作不重做；传参：发生记录；返回：已有结果。"""
        status = record.result_status
        if status is None and record.status == "running":
            facts = RunFactStore(self._data_root).read_run(record.run_id)
            boundaries = [row for row in facts if row.get("event") == "run:lifecycle"]
            if boundaries and boundaries[-1].get("lifecycle") in {
                "done",
                "paused",
                "failed",
                "waiting_user",
                "waiting_approval",
            }:
                lifecycle = str(boundaries[-1]["lifecycle"])
                status = "paused" if lifecycle.startswith("waiting_") else lifecycle
        if status is None:
            return None
        messages = SessionMessageStore(self._data_root)
        entries = (
            messages.read_entries(record.session_id)
            if record.session_id and messages.exists(record.session_id)
            else ()
        )
        outputs = [
            model_visible_text(entry.message)
            for entry in entries
            if entry.run_id == record.run_id
            and isinstance(entry.message, AssistantMessage)
        ]
        output = outputs[-1] if outputs else ""
        error = record.error
        if status == "failed" and error is None:
            # 1. 【后台工作】【中断交接】失败答复已由会话保存，尚未提交发生结果时从原答复恢复原因
            error = output
        return CronOutcome(status, output, error=error)

    def _complete(
        self,
        job: ScheduleRecord,
        record: OccurrenceRecord,
        *,
        outcome: CronOutcome,
        current: datetime,
    ) -> CronRunResult:
        """持久结果、幂等通知与计划投影完成后确认交接；传参：计划、发生及结果；返回：回执。"""
        # 1. 【定时工作】【完成交接】结果、通知意图和计划回执同批次发布，外部通知发送另行执行
        with RuntimeStore(self._data_root).transaction():
            outcome = self._knowledge_outcome(job, outcome)
            pending = replace(
                record,
                status="result_ready",
                result_status=outcome.status,
                error=outcome.error,
            )
            self.occurrences.save(pending)
            notice_id = f"notice-{record.occurrence_id}-{record.run_id or 'reminder'}"
            title = job.name or job.schedule_id
            labels = {
                "done": "本次运行结束",
                "paused": "待继续",
                "failed": "失败",
                "timeout": "超时",
            }
            message = (
                job.prompt
                if job.kind == "reminder"
                else f"{labels[outcome.status]}\n{outcome.output or outcome.error or ''}"
            )
            assert message is not None
            if self.notifications.load(notice_id) is None:
                self.notifications.enqueue(
                    notice_id,
                    title=title,
                    message=message,
                    source={
                        "schedule_id": job.schedule_id,
                        "workspace_id": job.workspace_id,
                        "occurrence_id": record.occurrence_id,
                        "scheduled_at": record.scheduled_at,
                        "session_id": record.session_id,
                        "run_id": record.run_id,
                    },
                )
            status = _run_status(outcome.status)
            latest = self._store.load_schedule(job.schedule_id)
            assert latest is not None
            paused = (
                latest.paused
                or status == "paused"
                or outcome.error == "prompt_not_self_contained"
            )
            self._store.update_run_status(
                job.schedule_id,
                status=status,
                run_id=record.run_id,
                run_at=format_instant(current),
                next_run_at=latest.next_run_at,
                error=outcome.error,
                paused=paused,
                occurrence_id=record.occurrence_id,
                occurrence_count=sum(
                    item.schedule_id == job.schedule_id
                    and item.result_status is not None
                    for item in self.occurrences.list_all()
                ),
            )
            self.occurrences.save(
                replace(pending, status="settled", notification_id=notice_id)
            )
            return CronRunResult(
                job_id=job.schedule_id,
                status=status,
                task_id=outcome.task_id,
                error=outcome.error,
                run_id=record.run_id or None,
                session_id=record.session_id or None,
                occurrence_id=record.occurrence_id,
                notification_id=notice_id,
            )

    def _knowledge_outcome(
        self, job: ScheduleRecord, outcome: CronOutcome
    ) -> CronOutcome:
        """【知识维护】【恢复对账】运行终态不冒充来源核验完成；参数：原计划/真实运行结果；返回：知识交付结论。"""
        origin = job.knowledge_origin
        if origin is None or origin.get("work_kind") != "knowledge_maintenance":
            return outcome
        from runtime.knowledge_maintenance import FINISHED, KnowledgeMaintenance

        manager = KnowledgeMaintenance(self._data_root)
        row = manager.load(origin["work_id"])
        if row["state"] in FINISHED:
            return replace(
                outcome, status="done", output=str(row["reason"]), error=None
            )
        error = (
            outcome.error
            or outcome.output
            or "knowledge worker ended without verified source coverage"
        )
        if row["state"] in {"queued", "running", "cancelling"}:
            manager.update(
                row["work_id"],
                state="cancelled" if row.get("cancel_requested") else "failed",
                error=error,
            )
        if outcome.status == "done":
            return replace(
                outcome,
                status="failed",
                error="knowledge worker ended without verified source coverage",
            )
        return outcome


def _interval(cron: str) -> timedelta:
    """解析已声明的间隔；传参：表达；返回：时间间隔。"""
    return interval_delta(cron)


def _run_status(status: str) -> str:
    """转换明确的运行边界，拒绝未知成功标签；传参：状态；返回：计划展示状态。"""
    if status == "done":
        return "succeeded"
    if status in {"paused", "failed", "timeout", "notification_queued"}:
        return status
    raise ValueError(f"unknown cron run status: {status}")


def _utc_now() -> str:
    """提供默认UTC调度时钟；传参：无；返回：持久时间。"""
    return format_instant(datetime.now(timezone.utc))


def _should_pause(exc: Exception) -> bool:
    """对缺配置或缺完整提示的明确错误暂停重试；传参：异常；返回：是否需要修正后继续。"""
    return type(exc).__name__ in PAUSE_ERROR_TYPES
