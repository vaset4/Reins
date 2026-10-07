"""每次到期的持久身份和执行交接，不以计划索引代替运行事实。

作者：xxx
时间：2026-09-14 19:16:10
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from runtime.types import new_run_id, new_session_id
from schedules.persistence import claim_file, record_path
from runtime.persistence import RuntimeStore
from runtime.workspaces import WorkspaceStore
from schedules.timing import occurrence_identity


@dataclass(frozen=True, slots=True)
class OccurrenceRecord:
    """一次到期的投递状态；运行结果和正文仍由所引用的Session/Run负责。"""

    occurrence_id: str
    schedule_id: str
    scheduled_at: str
    next_run_at: str | None
    session_id: str
    run_id: str
    run_ids: tuple[str, ...]
    status: str = "queued"
    result_status: str | None = None
    error: str | None = None
    notification_id: str | None = None
    schedule_snapshot: dict[str, Any] = field(default_factory=dict)
    resume_input_id: str | None = None
    resume_request_id: str | None = None
    budget_run_id: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OccurrenceRecord:
        """恢复同一发生的身份与交接状态；传参：持久对象；返回：不可变记录。"""
        return cls(
            occurrence_id=value["occurrence_id"],
            schedule_id=value["schedule_id"],
            scheduled_at=value["scheduled_at"],
            next_run_at=value["next_run_at"],
            session_id=value["session_id"],
            run_id=value["run_id"],
            run_ids=tuple(value["run_ids"]),
            status=value["status"],
            result_status=value.get("result_status"),
            error=value.get("error"),
            notification_id=value.get("notification_id"),
            schedule_snapshot=dict(value.get("schedule_snapshot", {})),
            resume_input_id=value.get("resume_input_id"),
            resume_request_id=value.get("resume_request_id"),
            budget_run_id=value.get("budget_run_id"),
        )


class OccurrenceStore:
    """在文件主存内保存接纳、认领和结果引用，进程间以实际持锁状态决定执行责任。"""

    def __init__(self, data_root: Path | str) -> None:
        """绑定数据根；传参：数据目录；返回：无。"""
        self.root = Path(data_root) / "runtime" / "locks"
        self._data_root = Path(data_root)
        self._db = RuntimeStore(data_root)

    def accept(
        self,
        schedule_id: str,
        scheduled_at: str,
        *,
        next_run_at: str | None,
        schedule_snapshot: dict[str, Any] | None = None,
    ) -> OccurrenceRecord:
        """先保存发生再允许推进计划游标；传参：计划、原定时刻和后继；返回：新记录或同一已有记录。"""
        identity = occurrence_identity(schedule_id, scheduled_at)
        with self._db.transaction():
            existing = self.load(identity)
            if existing is not None:
                return existing
            run_id = new_run_id()
            record = OccurrenceRecord(
                identity,
                schedule_id,
                scheduled_at,
                next_run_at,
                new_session_id(),
                run_id,
                (run_id,),
                schedule_snapshot=dict(schedule_snapshot or {}),
                budget_run_id=run_id,
            )
            if schedule_snapshot is not None:
                # 1. 【定时工作】【接纳发生】工作和提醒的会话都与冻结计划归属同事务发布，后台才能按原目录交付
                workspaces = WorkspaceStore(self._data_root)
                workspace = workspaces.get(schedule_snapshot["workspace_id"])
                workspaces.bind_session(record.session_id, workspace.project_root)
            self.save(record)
            return record

    def load(self, identity: str) -> OccurrenceRecord | None:
        """读取一次发生并核对文件身份；传参：发生编号；返回：记录或不存在。"""
        with self._db.snapshot() as source:
            row = source.get("schedule_occurrence", identity)
            return OccurrenceRecord.from_dict(row) if row is not None else None

    def list_all(self) -> list[OccurrenceRecord]:
        """按接纳身份枚举发生；传参：无；返回：完整记录。"""
        with self._db.snapshot() as source:
            return [
                OccurrenceRecord.from_dict(row)
                for row in sorted(
                    source.list("schedule_occurrence"),
                    key=lambda row: row["occurrence_id"],
                )
            ]

    def save(self, record: OccurrenceRecord) -> None:
        """提交发生交接状态；传参：完整记录；返回：无。"""
        with self._db.transaction() as batch:
            batch.put(
                "schedule_occurrence",
                record.occurrence_id,
                asdict(record),
                session_id=record.session_id,
                workspace_id=record.schedule_snapshot.get("workspace_id"),
            )

    @contextmanager
    def claim(self, identity: str) -> Iterator[OccurrenceRecord | None]:
        """在执行期间持有进程锁，其他唤醒只能观察；传参：发生身份；返回：认领后的当前记录或未取得。"""
        with claim_file(
            record_path(self.root, identity, suffix=".run.lock")
        ) as acquired:
            yield self.load(identity) if acquired else None

    def begin(
        self,
        record: OccurrenceRecord,
        *,
        resume: bool = False,
        renew_budget: bool = False,
    ) -> OccurrenceRecord:
        """在真实派发前记录运行引用，恢复不改变发生身份；传参：已认领记录与恢复选择；返回：新状态。"""
        run_id = new_run_id() if resume else record.run_id
        run_ids = (*record.run_ids, run_id) if resume else record.run_ids
        updated = replace(
            record,
            run_id=run_id,
            run_ids=run_ids,
            status="running",
            result_status=None,
            error=None,
            budget_run_id=run_id
            if renew_budget
            else record.budget_run_id or record.run_ids[0],
        )
        self.save(updated)
        return updated

    def request_resume(
        self,
        identity: str,
        *,
        request_id: str,
        input_id: str,
        allow_completed: bool = False,
    ) -> OccurrenceRecord:
        """显式接续等待中的发生，重复交付不创建第二次执行；传参：发生与已保存输入引用；返回：排队记录。"""
        with self.claim(identity) as record:
            if record is None:
                raise ValueError(
                    "occurrence is missing or currently executing; inspect its status"
                )
            if record.resume_request_id == request_id:
                return record
            allowed = (
                {"paused", "failed", "timeout", "done"}
                if allow_completed
                else {"paused", "failed", "timeout"}
            )
            if record.status != "settled" or record.result_status not in allowed:
                raise ValueError(
                    "only a paused or failed occurrence can be explicitly resumed"
                )
            updated = replace(
                record,
                status="resume_queued",
                result_status=None,
                error=None,
                resume_request_id=request_id,
                resume_input_id=input_id,
            )
            self.save(updated)
            return updated
