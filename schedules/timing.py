"""持久时间意图的校验和下一次发生时间计算。

作者：xxx
时间：2026-09-14 19:16:10
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from typing import cast
from zoneinfo import ZoneInfo

from croniter import croniter  # type: ignore[import-untyped]

CALENDAR_FIELD_COUNT = 5


@dataclass(frozen=True, slots=True)
class TimeIntent:
    """保留原始表达和时区；只计算时间，不替模型决定工作内容。"""

    expression: str
    timezone_name: str

    def validate(self) -> None:
        """检查明确支持的时间语义；传参：无；返回：无，错误表达抛出异常。"""
        if not self.timezone_name:
            raise ValueError(
                "schedule timezone_name is required; set the intended timezone for a legacy calendar"
            )
        zone = ZoneInfo(self.timezone_name)
        if self.expression.startswith("at:"):
            parse_instant(self.expression.removeprefix("at:"), zone=zone)
            return
        if self.expression.startswith("interval:"):
            interval_delta(self.expression)
            return
        calendar = self.expression.removeprefix("cron:")
        if len(calendar.split()) != CALENDAR_FIELD_COUNT or not croniter.is_valid(
            calendar
        ):
            raise ValueError(f"unsupported schedule expression: {self.expression}")

    def first_at(self, now: datetime) -> datetime:
        """计算首次发生时刻，过期的一次提醒仍保留原定时刻；传参：当前时间；返回：UTC时间。"""
        self.validate()
        if self.expression.startswith("at:"):
            return parse_instant(
                self.expression.removeprefix("at:"), zone=ZoneInfo(self.timezone_name)
            )
        following = self.after(now)
        assert following is not None
        return following

    def after(
        self, occurrence: datetime, *, now: datetime | None = None
    ) -> datetime | None:
        """按原时间锚点计算下一次发生；传参：上次时刻和可选当前时间；返回：下一时刻或一次性结束。"""
        self.validate()
        occurrence = require_aware(occurrence)
        if self.expression.startswith("at:"):
            return None
        current = max(occurrence, require_aware(now)) if now is not None else occurrence
        if self.expression.startswith("interval:"):
            delta = interval_delta(self.expression)
            elapsed = (current - occurrence) // delta
            return occurrence + (elapsed + 1) * delta
        local = current.astimezone(ZoneInfo(self.timezone_name))
        iterator = croniter(self.expression.removeprefix("cron:"), local)
        return cast(datetime, iterator.get_next(datetime)).astimezone(timezone.utc)


def interval_delta(expression: str) -> timedelta:
    """校验正整数秒间隔；传参：interval表达；返回：时间间隔。"""
    if not expression.startswith("interval:"):
        raise ValueError(f"unsupported interval expression: {expression}")
    value = expression.removeprefix("interval:")
    if not value.isascii() or not value.isdigit() or int(value) <= 0:
        raise ValueError("interval must be a positive integer number of seconds")
    return timedelta(seconds=int(value))


def parse_instant(value: str, *, zone: ZoneInfo | None = None) -> datetime:
    """解析绝对时刻，夏令时歧义须由显式偏移消除；传参：ISO时间及本地时区；返回：UTC时间。"""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc)
    if zone is None:
        raise ValueError("timestamp requires an explicit timezone offset")
    local = parsed.replace(tzinfo=zone)
    if local.utcoffset() != local.replace(fold=1).utcoffset():
        raise ValueError(
            "ambiguous or nonexistent local time; supply an explicit UTC offset"
        )
    if local.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) != parsed:
        raise ValueError("nonexistent local time; supply a valid timestamp")
    return local.astimezone(timezone.utc)


def require_aware(value: datetime) -> datetime:
    """拒绝没有时区的调度时钟；传参：时间；返回：统一UTC时间。"""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("schedule clock requires a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def format_instant(value: datetime) -> str:
    """统一持久发生时刻的表示；传参：时间；返回：精确到微秒的UTC字符串。"""
    return require_aware(value).isoformat(timespec="microseconds")


def occurrence_identity(schedule_id: str, scheduled_at: str) -> str:
    """给同一次到期生成稳定身份；传参：计划编号和原定时刻；返回：发生编号。"""
    stamp = format_instant(parse_instant(scheduled_at))
    digest = sha256(f"{schedule_id}\0{stamp}".encode()).hexdigest()
    return f"occurrence-{digest}"
