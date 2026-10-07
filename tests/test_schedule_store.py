from __future__ import annotations

from pathlib import Path

import pytest

from schedules.store import (
    MAX_ERROR_LENGTH,
    ScheduleRecord,
    ScheduleStore,
    _truncate_error,
)
from runtime.workspaces import WorkspaceStore


@pytest.fixture()
def store(tmp_path: Path) -> ScheduleStore:
    """使用实际事务存储验证调度；传参：隔离目录；返回：计划存储。"""
    return ScheduleStore(tmp_path)


@pytest.fixture()
def workspace_id(tmp_path: Path) -> str:
    """为每个计划明确指定隔离工作区；参数：临时目录；返回：持久工作区编号。"""
    return WorkspaceStore(tmp_path).register(tmp_path).workspace_id


class TestScheduleRecordSerialization:
    def test_new_fields_round_trip(
        self, store: ScheduleStore, workspace_id: str
    ) -> None:
        """计划内容与工作区一同往返；参数：存储和工作区；返回：无。"""
        store.create_schedule(
            "sched-1",
            "interval:60",
            workspace_id=workspace_id,
            name="daily-check",
            prompt="run health check",
        )
        loaded = store.load_schedule("sched-1")
        assert loaded is not None
        assert loaded.name == "daily-check"
        assert loaded.prompt == "run health check"
        assert loaded.workspace_id == workspace_id
        assert loaded.last_run_id is None
        assert loaded.last_status is None
        assert loaded.run_count == 0
        assert loaded.paused is False

    def test_optional_fields_keep_documented_defaults(self) -> None:
        """已有明确归属时，可选显示字段保留默认值；参数：无；返回：无。"""
        data = {
            "schedule_id": "new-1",
            "cron": "interval:300",
            "enabled": True,
            "workspace_id": "workspace-1",
        }
        record = ScheduleRecord.from_dict(data)
        assert record.name is None
        assert record.prompt is None
        assert record.run_count == 0
        assert record.paused is False


class TestUpdateRunStatus:
    def test_success_updates_fields(
        self, store: ScheduleStore, workspace_id: str
    ) -> None:
        """成功回执更新状态但保留计划归属；参数：存储和工作区；返回：无。"""
        store.create_schedule(
            "sched-2", "interval:60", workspace_id=workspace_id, prompt="do stuff"
        )
        updated = store.update_run_status(
            "sched-2",
            status="succeeded",
            run_id="run-abc123",
            run_at="2026-05-20T10:00:00+00:00",
            next_run_at="2026-05-20T10:01:00+00:00",
        )
        assert updated.last_status == "succeeded"
        assert updated.last_run_id == "run-abc123"
        assert updated.run_count == 1
        assert updated.last_error is None
        assert updated.paused is False

    def test_failure_with_pause(self, store: ScheduleStore, workspace_id: str) -> None:
        """失败与暂停状态可以恢复查看；参数：存储和工作区；返回：无。"""
        store.create_schedule(
            "sched-3", "interval:60", workspace_id=workspace_id, prompt="do stuff"
        )
        updated = store.update_run_status(
            "sched-3",
            status="failed",
            run_id="run-def456",
            run_at="2026-05-20T10:00:00+00:00",
            next_run_at="2026-05-20T10:01:00+00:00",
            error="MissingConfigurationError: no API key",
            paused=True,
        )
        assert updated.last_status == "failed"
        assert updated.paused is True
        assert updated.last_error is not None

    def test_run_count_increments(
        self, store: ScheduleStore, workspace_id: str
    ) -> None:
        """独立运行回执各计一次；参数：存储和工作区；返回：无。"""
        store.create_schedule(
            "sched-4", "interval:60", workspace_id=workspace_id, prompt="do stuff"
        )
        store.update_run_status(
            "sched-4",
            status="succeeded",
            run_id="run-1",
            run_at="2026-05-20T10:00:00+00:00",
            next_run_at="2026-05-20T10:01:00+00:00",
        )
        store.update_run_status(
            "sched-4",
            status="succeeded",
            run_id="run-2",
            run_at="2026-05-20T10:01:00+00:00",
            next_run_at="2026-05-20T10:02:00+00:00",
        )
        loaded = store.load_schedule("sched-4")
        assert loaded is not None
        assert loaded.run_count == 2


class TestListAllSchedules:
    def test_returns_all_including_disabled(
        self, store: ScheduleStore, workspace_id: str
    ) -> None:
        """查询保留启用和禁用的完整计划；参数：存储和工作区；返回：无。"""
        store.create_schedule(
            "s1", "interval:60", workspace_id=workspace_id, enabled=True, prompt="a"
        )
        store.create_schedule(
            "s2", "interval:60", workspace_id=workspace_id, enabled=False, prompt="b"
        )
        all_schedules = store.list_all_schedules()
        ids = {s.schedule_id for s in all_schedules}
        assert ids == {"s1", "s2"}


class TestTruncateError:
    def test_none_returns_none(self) -> None:
        assert _truncate_error(None) is None

    def test_short_error_unchanged(self) -> None:
        assert _truncate_error("short") == "short"

    def test_long_error_truncated(self) -> None:
        long_msg = "x" * 300
        result = _truncate_error(long_msg)
        assert result is not None
        assert len(result) == MAX_ERROR_LENGTH
        assert result.endswith("...")
