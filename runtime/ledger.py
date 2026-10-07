from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Self

from tasks.ids import utc_now
from runtime.persistence import RuntimeStore

LEDGER_TYPE = "ledger_event"
_REQUIRED_FIELDS = frozenset({"type", "event", "ts", "event_id", "source", "payload"})


@dataclass(frozen=True, slots=True)
class LedgerEvent:
    type: str
    event: str
    ts: str
    event_id: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    session_id: str | None = None
    run_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Self:
        _validate_required_fields(value)
        event_type = _required_text(value, "type")
        if event_type != LEDGER_TYPE:
            raise ValueError(f"ledger event type must be {LEDGER_TYPE}")
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise ValueError("ledger event payload must be an object")
        return cls(
            type=event_type,
            event=_required_text(value, "event"),
            ts=_required_text(value, "ts"),
            event_id=_required_text(value, "event_id"),
            source=_required_text(value, "source"),
            payload=dict(payload),
            task_id=_optional_text(value.get("task_id")),
            session_id=_optional_text(value.get("session_id")),
            run_id=_optional_text(value.get("run_id")),
        )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        return {key: value for key, value in row.items() if value is not None}


class LedgerStore:
    """按文件提交顺序保存授权与执行事件，重复身份只可重用同一决定。"""

    def __init__(self, data_root: Path | str) -> None:
        """准备 Ledger 实体与查询索引；参数：数据根；返回：无。"""
        self.database = RuntimeStore(data_root)

    def append(self, event: LedgerEvent | Mapping[str, object]) -> Path:
        """提交已校验事件；参数：事件；返回：所属日志原件位置。"""
        with self.database.transaction():
            return self._append_payload(_coerce_event(event).to_dict())

    def append_once(self, event: LedgerEvent | Mapping[str, object]) -> LedgerEvent:
        """同一事件身份只能提交相同决定；参数：候选；返回：已提交事件。"""
        candidate = _coerce_event(event)
        with self.database.transaction() as batch:
            row = batch.get("ledger", candidate.event_id)
            if row is not None:
                existing = LedgerEvent.from_mapping(row)
                before = {
                    key: value
                    for key, value in existing.to_dict().items()
                    if key != "ts"
                }
                after = {
                    key: value
                    for key, value in candidate.to_dict().items()
                    if key != "ts"
                }
                if before != after:
                    raise ValueError("ledger event identity has a different decision")
                return existing
            self._append_payload(candidate.to_dict())
            return candidate

    def _append_payload(self, payload: dict[str, Any]) -> Path:
        """在同一批次保存事件及正文引用；参数：规范载荷；返回：所属日志原件路径。"""
        with self.database.transaction() as batch:
            if batch.get("ledger", payload["event_id"]) is not None:
                raise FileExistsError(
                    f"ledger event already exists: {payload['event_id']}"
                )
            batch.put(
                "ledger",
                payload["event_id"],
                payload,
                session_id=payload.get("session_id"),
            )
        return self.database.source_path("ledger", payload["event_id"])

    def read_events(self) -> list[LedgerEvent]:
        """读取所有已提交事件；参数：无；返回：按提交次序排列的严格事件。"""
        return self._read_events()

    def read_event(self, event_id: str) -> LedgerEvent | None:
        """精确读取指定身份的已提交原件；参数：事件编号；返回：严格事件或不存在，损坏明确失败。"""
        if not event_id or not event_id.strip():
            raise ValueError("ledger lookup requires a valid event identity")
        with self.database.snapshot() as source:
            row = source.get("ledger", event_id)
        if row is None:
            return None
        event = LedgerEvent.from_mapping(row)
        if event.event_id != event_id:
            raise ValueError("ledger event identity does not match its source")
        return event

    def read_events_of_type(self, event: str) -> list[LedgerEvent]:
        """只展开指定事件名称的原件；参数：事件名称；返回：按提交顺序排列的匹配事件。"""
        return self._read_events("event", event)

    def _read_events(
        self, field: str | None = None, value: str | None = None
    ) -> list[LedgerEvent]:
        """按固定归属或事件名称查询；参数：内部字段与查询值；返回：事件，损坏明确失败。"""
        if field is not None and (
            field not in {"task_id", "run_id", "session_id", "event"}
            or not value
            or not value.strip()
        ):
            raise ValueError("ledger lookup requires a valid identity")
        with self.database.snapshot() as source:
            rows = source.list(
                "ledger",
                session_id=value if field == "session_id" else None,
                filters=None if field is None else {field: value},
            )
            return [LedgerEvent.from_mapping(row) for row in rows]

    def read_task_events(self, task_id: str) -> list[LedgerEvent]:
        """读取任务授权与执行事件；参数：任务编号；返回：事件序列。"""
        return self._read_events("task_id", task_id)

    def read_run_events(self, run_id: str) -> list[LedgerEvent]:
        """读取运行事件；参数：运行编号；返回：事件序列。"""
        return self._read_events("run_id", run_id)

    def read_session_events(self, session_id: str) -> list[LedgerEvent]:
        """读取会话事件；参数：会话编号；返回：事件序列。"""
        return self._read_events("session_id", session_id)


def new_ledger_event(
    event: str,
    event_id: str,
    source: str,
    payload: Mapping[str, object],
    *,
    task_id: str | None = None,
    session_id: str | None = None,
    run_id: str | None = None,
    ts: str | None = None,
) -> LedgerEvent:
    return LedgerEvent.from_mapping(
        {
            "type": LEDGER_TYPE,
            "event": event,
            "ts": ts or utc_now(),
            "event_id": event_id,
            "task_id": task_id,
            "session_id": session_id,
            "run_id": run_id,
            "source": source,
            "payload": dict(payload),
        }
    )


def _coerce_event(event: LedgerEvent | Mapping[str, object]) -> LedgerEvent:
    if isinstance(event, LedgerEvent):
        return event
    return LedgerEvent.from_mapping(event)


def _validate_required_fields(value: Mapping[str, object]) -> None:
    missing = sorted(
        field_name for field_name in _REQUIRED_FIELDS if field_name not in value
    )
    if missing:
        raise ValueError(f"ledger event missing required fields: {', '.join(missing)}")


def _required_text(value: Mapping[str, object], field_name: str) -> str:
    text = _optional_text(value.get(field_name))
    if text is None:
        raise ValueError(f"ledger event {field_name} must be non-empty")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["LEDGER_TYPE", "LedgerEvent", "LedgerStore", "new_ledger_event"]
