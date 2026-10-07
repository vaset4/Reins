from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime.persistence import RuntimeStore
from runtime.workspaces import WorkspaceStore

from llm.public_config import PUBLIC_MODEL_FIELDS
from schedules.timing import TimeIntent, format_instant, parse_instant


MAX_ERROR_LENGTH = 200
DEFAULT_SCHEDULE_STEPS = 30
DEFAULT_SCHEDULE_TOKENS = 200000


@dataclass(frozen=True, slots=True)
class ScheduleRecord:
    schedule_id: str
    cron: str
    workspace_id: str
    target_task_id: str | None = None
    enabled: bool = True
    next_run_at: str | None = None
    last_run_at: str | None = None
    name: str | None = None
    prompt: str | None = None
    last_run_id: str | None = None
    last_status: str | None = None
    last_error: str | None = None
    run_count: int = 0
    paused: bool = False
    timezone_name: str = "UTC"
    kind: str = "work"
    source_session_id: str | None = None
    source_run_id: str | None = None
    source_entry_id: str | None = None
    model_config: dict[str, object] = field(default_factory=dict)
    capabilities: dict[str, object] = field(default_factory=dict)
    required_permanent_grants: tuple[dict[str, object], ...] = ()
    last_occurrence_id: str | None = None
    max_steps: int = DEFAULT_SCHEDULE_STEPS
    max_tokens: int = DEFAULT_SCHEDULE_TOKENS
    knowledge_origin: dict[str, Any] | None = None
    context_compaction_origin: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduleRecord":
        """恢复已保存的计划及原工作区；参数：持久记录；返回：计划，缺少归属明确失败。"""
        workspace_id = data.get("workspace_id")
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("schedule workspace_id is missing or invalid")
        return cls(
            schedule_id=str(data["schedule_id"]),
            cron=str(data["cron"]),
            workspace_id=workspace_id,
            target_task_id=(
                str(data["target_task_id"])
                if data.get("target_task_id") is not None
                else None
            ),
            enabled=bool(data.get("enabled", True)),
            next_run_at=(
                str(data["next_run_at"])
                if data.get("next_run_at") is not None
                else None
            ),
            last_run_at=(
                str(data["last_run_at"])
                if data.get("last_run_at") is not None
                else None
            ),
            name=data.get("name"),
            prompt=data.get("prompt"),
            last_run_id=data.get("last_run_id"),
            last_status=data.get("last_status"),
            last_error=data.get("last_error"),
            run_count=int(data.get("run_count", 0)),
            paused=bool(data.get("paused", False)),
            timezone_name=str(
                data.get(
                    "timezone_name",
                    "UTC" if str(data["cron"]).startswith("interval:") else "",
                )
            ),
            kind=str(data.get("kind", "work")),
            source_session_id=data.get("source_session_id"),
            source_run_id=data.get("source_run_id"),
            source_entry_id=data.get("source_entry_id"),
            model_config=dict(data.get("model_config", {})),
            capabilities=dict(data.get("capabilities", {})),
            required_permanent_grants=tuple(
                dict(item) for item in data.get("required_permanent_grants", [])
            ),
            last_occurrence_id=data.get("last_occurrence_id"),
            max_steps=_positive_budget(data.get("max_steps", DEFAULT_SCHEDULE_STEPS)),
            max_tokens=_positive_budget(
                data.get("max_tokens", DEFAULT_SCHEDULE_TOKENS)
            ),
            knowledge_origin=dict(data["knowledge_origin"])
            if data.get("knowledge_origin") is not None
            else None,
            context_compaction_origin=dict(data["context_compaction_origin"])
            if data.get("context_compaction_origin") is not None
            else None,
        )


class ScheduleStore:
    def __init__(self, data_root: Path | str) -> None:
        """绑定文件主存和派生索引；传参：数据根；返回：无。"""
        self._data_root = Path(data_root)
        self._db = RuntimeStore(data_root)
        self._db.ensure_space()
        self._closed = False

    def close(self) -> None:
        """结束当前计划存储实例的使用；传参：无；返回：无，不持有长期数据库连接。"""
        self._closed = True

    def create_schedule(
        self,
        schedule_id: str,
        cron: str,
        *,
        workspace_id: str,
        target_task_id: str | None = None,
        enabled: bool = True,
        next_run_at: str | None = None,
        name: str | None = None,
        prompt: str | None = None,
        timezone_name: str = "UTC",
        kind: str = "work",
        source_session_id: str | None = None,
        source_run_id: str | None = None,
        source_entry_id: str | None = None,
        model_config: dict[str, object] | None = None,
        capabilities: dict[str, object] | None = None,
        required_permanent_grants: tuple[dict[str, object], ...] = (),
        max_steps: int = DEFAULT_SCHEDULE_STEPS,
        max_tokens: int = DEFAULT_SCHEDULE_TOKENS,
        knowledge_origin: dict[str, Any] | None = None,
        context_compaction_origin: dict[str, Any] | None = None,
    ) -> ScheduleRecord:
        """持久接纳明确时间与原工作区；参数：计划、工作区身份及来源；返回：已保存记录。"""
        self._require_open()
        if set(model_config or {}) - PUBLIC_MODEL_FIELDS:
            raise ValueError(
                "schedule model config accepts public model fields only; credentials must remain in the secret store"
            )
        intent = TimeIntent(cron, timezone_name)
        intent.validate()
        if kind not in {"work", "reminder"}:
            raise ValueError("schedule kind must be work or reminder")
        if next_run_at is not None:
            parse_instant(next_run_at)
        if not isinstance(workspace_id, str) or not workspace_id.strip():
            raise ValueError("schedule workspace_id is missing or invalid")
        workspaces = WorkspaceStore(self._data_root)
        workspaces.get(workspace_id)
        if (
            source_session_id is not None
            and workspaces.for_session(source_session_id).workspace_id != workspace_id
        ):
            raise ValueError("schedule workspace differs from its source session")
        record = ScheduleRecord(
            schedule_id=schedule_id,
            cron=cron,
            workspace_id=workspace_id,
            target_task_id=target_task_id,
            enabled=enabled,
            next_run_at=next_run_at
            or format_instant(intent.first_at(datetime.now(timezone.utc))),
            name=name,
            prompt=prompt,
            timezone_name=timezone_name,
            kind=kind,
            source_session_id=source_session_id,
            source_run_id=source_run_id,
            source_entry_id=source_entry_id,
            model_config=dict(model_config or {}),
            capabilities=dict(capabilities or {}),
            required_permanent_grants=required_permanent_grants,
            max_steps=_positive_budget(max_steps),
            max_tokens=_positive_budget(max_tokens),
            knowledge_origin=knowledge_origin,
            context_compaction_origin=context_compaction_origin,
        )
        with self._db.transaction():
            if self.load_schedule(schedule_id) is not None:
                raise FileExistsError(schedule_id)
            self._write(record)
        return record

    def load_schedule(self, schedule_id: str) -> ScheduleRecord | None:
        """按编号读取唯一计划正文；传参：计划编号；返回：记录或不存在。"""
        self._require_open()
        with self._db.snapshot() as source:
            row = source.get("schedule", schedule_id)
            return ScheduleRecord.from_dict(row) if row is not None else None

    def list_enabled_schedules(self) -> list[ScheduleRecord]:
        """从文件主存枚举可用计划，索引丢失不能丢掉已接纳工作；传参：无；返回：计划列表。"""
        return sorted(
            (item for item in self.list_all_schedules() if item.enabled),
            key=lambda item: item.next_run_at or "",
        )

    def update_next_run_at(
        self,
        schedule_id: str,
        next_run_at: str | None,
        *,
        last_run_at: str | None = None,
    ) -> ScheduleRecord:
        """推进已持久交接的时间游标；传参：计划和时间；返回：新记录。"""
        return self.update(
            schedule_id, next_run_at=next_run_at, last_run_at=last_run_at
        )

    def update_run_status(
        self,
        schedule_id: str,
        *,
        status: str,
        run_id: str,
        run_at: str,
        next_run_at: str | None,
        error: str | None = None,
        paused: bool = False,
        occurrence_id: str | None = None,
        occurrence_count: int | None = None,
    ) -> ScheduleRecord:
        """按真实运行提交结果，相同回执不重复计数；传参：身份、状态和时间；返回：新记录。"""
        with self._db.transaction():
            record = self.load_schedule(schedule_id)
            if record is None:
                raise FileNotFoundError(schedule_id)
            updated = replace(
                record,
                last_status=status,
                last_run_id=run_id,
                last_run_at=run_at,
                next_run_at=next_run_at,
                run_count=occurrence_count
                if occurrence_count is not None
                else record.run_count + int(record.last_run_id != run_id),
                last_error=_truncate_error(error),
                paused=paused,
                last_occurrence_id=occurrence_id,
            )
            self._write(updated)
            return updated

    def update(self, schedule_id: str, **changes: Any) -> ScheduleRecord:
        """在独占窗口重读并更新计划元数据；传参：编号及明确变更；返回：新记录。"""
        with self._db.transaction():
            return self._update_locked(schedule_id, changes)

    def _update_locked(
        self, schedule_id: str, changes: dict[str, Any]
    ) -> ScheduleRecord:
        """在调用方持有认领锁时提交新快照；传参：编号和变更；返回：新记录。"""
        record = self.load_schedule(schedule_id)
        if record is None:
            raise FileNotFoundError(schedule_id)
        if "workspace_id" in changes and changes["workspace_id"] != record.workspace_id:
            raise ValueError("schedule workspace cannot be rebound")
        updated = replace(record, **changes)
        self._write(updated)
        return updated

    def list_all_schedules(self) -> list[ScheduleRecord]:
        """枚举包含暂停和禁用的计划；传参：无；返回：完整计划列表。"""
        self._require_open()
        with self._db.snapshot() as source:
            return [
                ScheduleRecord.from_dict(row)
                for row in sorted(
                    source.list("schedule"), key=lambda row: row["schedule_id"]
                )
            ]

    def _write(self, record: ScheduleRecord) -> None:
        """原子发布计划原件，索引由提交服务派生；传参：完整计划；返回：无。"""
        self._require_open()
        payload = asdict(record)
        with self._db.transaction() as batch:
            batch.put(
                "schedule",
                record.schedule_id,
                payload,
                workspace_id=record.workspace_id,
            )

    def _require_open(self) -> None:
        """阻止关闭后的存储继续写文件；传参：无；返回：无。"""
        if self._closed:
            raise RuntimeError("ScheduleStore is closed")


def _truncate_error(error: str | None) -> str | None:
    """限制列表中的错误摘要长度，完整错误保留在运行证据；传参：错误；返回：摘要。"""
    if error is None:
        return None
    if len(error) <= MAX_ERROR_LENGTH:
        return error
    return error[: MAX_ERROR_LENGTH - 3] + "..."


def _positive_budget(value: object) -> int:
    """拒绝损坏或非正整数的持久运行额度；传参：待保存值；返回：已校验整数。"""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("schedule run budgets must be positive integers")
    return value
