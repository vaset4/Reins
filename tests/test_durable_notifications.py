"""通知接纳、发送失败与用户确认的真实持久边界。

作者：xxx
时间：2026-09-14 19:16:10
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from schedules.notifications import (
    DeliveryReceipt,
    NOTIFICATION_RETRY_SECONDS,
    NotificationStore,
)


def test_known_failure_retries_without_claiming_read(tmp_path):
    """渠道失败保留回执，重试提交不等于用户已读；传参：隔离目录；返回：无。"""
    store = NotificationStore(tmp_path)
    now = datetime(2026, 9, 14, tzinfo=timezone.utc)
    record = store.enqueue(
        "notice-one",
        title="提醒",
        message="核对发票",
        source={"schedule_id": "one"},
        now=now,
    )
    assert (
        record.delivery_status == "pending"
        and record.visible_at is None
        and record.read_at is None
    )
    failed = store.deliver(
        record.notification_id,
        lambda _: DeliveryReceipt("failed", "桌面通知暂时不可用"),
        now=now,
    )
    assert failed.delivery_status == "failed"
    calls = []

    def send(_record):
        """记录渠道实际收到的发送请求；传参：通知；返回：提交回执。"""
        calls.append(_record.notification_id)
        return DeliveryReceipt("submitted", "渠道已接纳")

    assert (
        store.deliver(record.notification_id, send, now=now).delivery_status == "failed"
    )
    restarted = NotificationStore(tmp_path)
    sent = restarted.deliver(
        record.notification_id,
        send,
        now=now + timedelta(seconds=NOTIFICATION_RETRY_SECONDS),
    )
    assert calls == [record.notification_id]
    assert sent.delivery_status == "submitted" and len(sent.attempts) == 2
    assert sent.read_at is None
    visible = restarted.acknowledge(record.notification_id)
    assert visible.visible_at is not None and visible.read_at is None
    assert restarted.acknowledge(record.notification_id, read=True).read_at is not None


def test_unconfirmed_send_is_unknown_after_restart_and_is_not_repeated(tmp_path):
    """发送后失去回执不能自动再发，也不能标送达；传参：隔离目录；返回：无。"""
    store = NotificationStore(tmp_path)
    record = store.enqueue("notice-two", title="提醒", message="核对余额", source={})
    store._save(
        replace(
            record,
            delivery_status="sending",
            attempts=({"attempt_id": "interrupted", "status": "sending"},),
        )
    )
    calls = []
    restarted = NotificationStore(tmp_path)
    observed = restarted.deliver(record.notification_id, lambda _: calls.append(True))
    assert observed.delivery_status == "unknown" and observed.read_at is None
    assert observed.attempts[-1]["status"] == "unknown"
    restarted.deliver(record.notification_id, lambda _: calls.append(True))
    assert calls == []
    restarted.retry(record.notification_id)
    retried = restarted.deliver(
        record.notification_id, lambda _: DeliveryReceipt("submitted", "显式重发已提交")
    )
    assert retried.delivery_status == "submitted" and len(retried.attempts) == 2


def test_repeated_acceptance_does_not_duplicate_notifications(tmp_path):
    """发生交接重投复用已接纳通知；传参：隔离目录；返回：无。"""
    store = NotificationStore(tmp_path)
    first = store.enqueue(
        "same-occurrence", title="结果", message="已核对", source={"run_id": "run-one"}
    )
    repeated = NotificationStore(tmp_path).enqueue(
        "same-occurrence", title="结果", message="已核对", source={"run_id": "run-one"}
    )
    assert repeated == first
    assert len(store.list_all()) == 1
