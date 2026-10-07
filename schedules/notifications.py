"""通知的持久接纳、渠道回执和用户确认。

作者：xxx
时间：2026-09-14 19:16:10
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from schedules.persistence import claim_file
from runtime.persistence import RuntimeStore
from schedules.timing import format_instant, parse_instant

NOTIFICATION_RETRY_SECONDS = 60


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    """渠道只报告实际知道的提交结果，不把提交或发送解释为用户已读。"""

    status: str
    detail: str
    channel: str = "desktop"


@dataclass(frozen=True, slots=True)
class NotificationRecord:
    """通知正文是一份交付内容；source引用原工作，渠道和界面证据分开保存。"""

    notification_id: str
    title: str
    message: str
    source: dict[str, object]
    created_at: str
    delivery_status: str = "pending"
    attempts: tuple[dict[str, object], ...] = ()
    retry_at: str | None = None
    visible_at: str | None = None
    read_at: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> NotificationRecord:
        """恢复通知和已有回执；传参：持久对象；返回：通知记录。"""
        return cls(
            notification_id=value["notification_id"],
            title=value["title"],
            message=value["message"],
            source=dict(value["source"]),
            created_at=value["created_at"],
            delivery_status=value["delivery_status"],
            attempts=tuple(dict(item) for item in value["attempts"]),
            retry_at=value["retry_at"],
            visible_at=value["visible_at"],
            read_at=value["read_at"],
        )


class NotificationStore:
    """先持久交付意图，再尝试外部渠道；界面重连可以读取同一未读通知。"""

    def __init__(self, data_root: Path | str) -> None:
        """绑定通知主存；传参：数据根；返回：无。"""
        self.root = Path(data_root) / "runtime" / "locks"
        self._db = RuntimeStore(data_root)

    def enqueue(
        self,
        identity: str,
        *,
        title: str,
        message: str,
        source: Mapping[str, object],
        now: datetime | None = None,
    ) -> NotificationRecord:
        """幂等接纳同一业务通知；传参：稳定身份、正文和来源；返回：持久记录。"""
        if not title.strip() or not message.strip():
            raise ValueError("notification title and message must be non-empty")
        with self._db.transaction():
            existing = self.load(identity)
            if existing is not None:
                if (existing.title, existing.message, existing.source) != (
                    title,
                    message,
                    dict(source),
                ):
                    raise ValueError(
                        "notification identity already refers to different content"
                    )
                return existing
            record = NotificationRecord(
                identity,
                title,
                message,
                dict(source),
                format_instant(now or datetime.now(timezone.utc)),
            )
            self._save(record)
            return record

    def load(self, identity: str) -> NotificationRecord | None:
        """读取通知，损坏和身份不匹配明确失败；传参：编号；返回：记录或不存在。"""
        with self._db.snapshot() as source:
            row = source.get("notification", identity)
            return NotificationRecord.from_dict(row) if row is not None else None

    def list_all(self, *, unread_only: bool = False) -> list[NotificationRecord]:
        """枚举已接纳通知；传参：是否只要未读；返回：通知记录。"""
        with self._db.snapshot() as source:
            rows = [
                row
                for row in source.list("notification")
                if not unread_only or row["read_at"] is None
            ]
            return [
                NotificationRecord.from_dict(row)
                for row in sorted(
                    rows, key=lambda row: (row["created_at"], row["notification_id"])
                )
            ]

    def deliver(
        self,
        identity: str,
        sender: Callable[[NotificationRecord], DeliveryReceipt],
        *,
        now: datetime | None = None,
    ) -> NotificationRecord | None:
        """记录发送尝试后调用渠道，未知结果不自动重复发送；传参：编号、渠道及时间；返回：新回执。"""
        current = now or datetime.now(timezone.utc)
        with claim_file(self.root / "notifications.lock") as acquired:
            if not acquired:
                return None
            record = self._require(identity)
            if record.delivery_status == "sending":
                lost_attempt = {
                    **record.attempts[-1],
                    "status": "unknown",
                    "observed_at": format_instant(current),
                    "detail": "host interrupted before receiving a channel acknowledgement",
                }
                interrupted = replace(
                    record,
                    delivery_status="unknown",
                    attempts=(*record.attempts[:-1], lost_attempt),
                    retry_at=None,
                )
                self._save(interrupted)
                return interrupted
            if record.delivery_status not in {"pending", "failed"}:
                return record
            if record.retry_at is not None and parse_instant(record.retry_at) > current:
                return record
            attempt: dict[str, object] = {
                "attempt_id": uuid4().hex,
                "started_at": format_instant(current),
                "status": "sending",
            }
            sending = replace(
                record,
                delivery_status="sending",
                attempts=(*record.attempts, attempt),
                retry_at=None,
            )
            self._save(sending)
            try:
                receipt = sender(sending)
            except Exception as exc:
                receipt = DeliveryReceipt("unknown", f"{type(exc).__name__}: {exc}")
            finished_at = datetime.now(timezone.utc) if now is None else current
            return self._finish_attempt(sending, receipt, finished_at)

    def acknowledge(
        self, identity: str, *, read: bool = False, now: datetime | None = None
    ) -> NotificationRecord:
        """界面展示成功后记可见，明确用户确认后才记已读；传参：编号及确认类型；返回：新记录。"""
        stamp = format_instant(now or datetime.now(timezone.utc))
        with self._db.transaction():
            record = self._require(identity)
            updated = replace(
                record,
                visible_at=record.visible_at or stamp,
                read_at=(record.read_at or stamp) if read else record.read_at,
            )
            self._save(updated)
            return updated

    def retry(self, identity: str) -> NotificationRecord:
        """用户显式重发未知或失败通知，保留全部旧尝试；传参：编号；返回：待发送记录。"""
        with self._db.transaction():
            record = self._require(identity)
            if record.delivery_status not in {"failed", "unknown"}:
                raise ValueError(
                    "only failed or unknown notification delivery can be retried"
                )
            updated = replace(record, delivery_status="pending", retry_at=None)
            self._save(updated)
            return updated

    def _finish_attempt(
        self, record: NotificationRecord, receipt: DeliveryReceipt, now: datetime
    ) -> NotificationRecord:
        """保存渠道实际回执与已知失败的重试时间；传参：通知、回执及时间；返回：新记录。"""
        if receipt.status not in {"submitted", "failed", "unknown"}:
            raise ValueError(f"unsupported notification receipt: {receipt.status}")
        attempt = {
            **record.attempts[-1],
            "status": receipt.status,
            "detail": receipt.detail,
            "channel": receipt.channel,
            "finished_at": format_instant(now),
        }
        retry_at = (
            format_instant(now + timedelta(seconds=NOTIFICATION_RETRY_SECONDS))
            if receipt.status == "failed"
            else None
        )
        with self._db.transaction():
            current = self._require(record.notification_id)
            updated = replace(
                current,
                delivery_status=receipt.status,
                attempts=(*current.attempts[:-1], attempt),
                retry_at=retry_at,
            )
            self._save(updated)
            return updated

    def _require(self, identity: str) -> NotificationRecord:
        """要求通知已经接纳；传参：编号；返回：现有记录。"""
        record = self.load(identity)
        if record is None:
            raise FileNotFoundError(identity)
        return record

    def _save(self, record: NotificationRecord) -> None:
        """原子提交调用方持锁的通知状态；传参：完整记录；返回：无。"""
        session_id = record.source.get("session_id")
        workspace_id = record.source.get("workspace_id")
        with self._db.transaction() as batch:
            batch.put(
                "notification",
                record.notification_id,
                asdict(record),
                session_id=str(session_id) if session_id is not None else None,
                workspace_id=str(workspace_id) if workspace_id is not None else None,
            )
