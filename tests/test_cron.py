from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from runtime.cron import CronScheduler, _should_pause
from runtime.workspaces import WorkspaceStore


class MissingConfigurationError(Exception):
    pass


class PromptNotSelfContainedError(Exception):
    pass


@pytest.fixture()
def scheduler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> CronScheduler:
    """使用真实隔离存储和受控时钟验证调度；参数：目录/替换器；返回：调度器。"""
    monkeypatch.setattr("runtime.cron._utc_now", lambda: "2026-05-20T12:00:00+00:00")
    return CronScheduler(project_root=tmp_path, data_root=tmp_path / "data")


class TestRegisterJob:
    def test_requires_non_empty_prompt(self, scheduler: CronScheduler) -> None:
        with pytest.raises(ValueError, match="prompt must be non-empty"):
            scheduler.register_job(
                name="test", task="t1", interval_seconds=60, prompt=""
            )

    def test_whitespace_only_prompt_rejected(self, scheduler: CronScheduler) -> None:
        with pytest.raises(ValueError, match="prompt must be non-empty"):
            scheduler.register_job(
                name="test", task="t1", interval_seconds=60, prompt="   "
            )

    def test_saves_name_and_prompt(self, scheduler: CronScheduler) -> None:
        record = scheduler.register_job(
            name="health", task="t1", interval_seconds=300, prompt="check health"
        )
        assert record.name == "health"
        assert record.prompt == "check health"


@dataclass
class _FakeResponse:
    status: str = "done"
    task_id: str = "task-001"
    output: str = "已执行"
    error: str | None = None


class TestRunDueJobs:
    def test_success_writes_status(self, scheduler: CronScheduler) -> None:
        scheduler.register_job(
            name="j1", task="t1", interval_seconds=60, prompt="do it"
        )
        now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        with patch.object(scheduler, "_execute", return_value=_FakeResponse()):
            results = scheduler.run_due_jobs(now=now)
        assert len(results) == 1
        assert results[0].status == "succeeded"
        assert results[0].run_id is not None
        assert results[0].session_id is not None
        record = scheduler._store.load_schedule(results[0].job_id)
        assert record is not None
        assert record.last_status == "succeeded"
        assert record.run_count == 1

    def test_timeout_maps_to_timeout_status(self, scheduler: CronScheduler) -> None:
        scheduler.register_job(
            name="j2", task="t2", interval_seconds=60, prompt="slow task"
        )
        now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        with patch.object(scheduler, "_execute", side_effect=TimeoutError("timed out")):
            results = scheduler.run_due_jobs(now=now)
        assert results[0].status == "timeout"
        record = scheduler._store.load_schedule(results[0].job_id)
        assert record is not None
        assert record.last_status == "timeout"
        assert record.paused is False

    def test_pause_on_missing_config(self, scheduler: CronScheduler) -> None:
        scheduler.register_job(
            name="j3", task="t3", interval_seconds=60, prompt="needs config"
        )
        now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        with patch.object(
            scheduler,
            "_execute",
            side_effect=MissingConfigurationError("no key"),
        ):
            results = scheduler.run_due_jobs(now=now)
        assert results[0].status == "failed"
        record = scheduler._store.load_schedule(results[0].job_id)
        assert record is not None
        assert record.paused is True

    def test_skips_paused_jobs(self, scheduler: CronScheduler) -> None:
        scheduler.register_job(
            name="j4", task="t4", interval_seconds=60, prompt="paused"
        )
        now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        with patch.object(
            scheduler,
            "_execute",
            side_effect=MissingConfigurationError("no key"),
        ):
            scheduler.run_due_jobs(now=now)
        with patch.object(
            scheduler, "_execute", return_value=_FakeResponse()
        ) as mock_run:
            results = scheduler.run_due_jobs(now=now)
        assert len(results) == 0
        mock_run.assert_not_called()

    def test_empty_prompt_at_runtime_marks_failed_and_paused(
        self, scheduler: CronScheduler, tmp_path: Path
    ) -> None:
        """Defensive check: if a schedule record has empty prompt (e.g. old YAML),
        it should fail with prompt_not_self_contained and pause."""
        # Bypass register_job validation by writing directly to store
        scheduler._store.create_schedule(
            "sched-empty-prompt",
            "interval:60",
            workspace_id=WorkspaceStore(tmp_path / "data")
            .register(tmp_path)
            .workspace_id,
            name="broken",
            prompt="",
            next_run_at="2020-01-01T00:00:00+00:00",
        )
        now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        with patch.object(
            scheduler, "_execute", return_value=_FakeResponse()
        ) as mock_run:
            results = scheduler.run_due_jobs(now=now)
        mock_run.assert_not_called()
        assert len(results) == 1
        assert results[0].status == "failed"
        assert results[0].error == "prompt_not_self_contained"
        record = scheduler._store.load_schedule("sched-empty-prompt")
        assert record is not None
        assert record.paused is True
        assert record.last_status == "failed"


class TestShouldPause:
    def test_missing_config_pauses(self) -> None:
        assert _should_pause(MissingConfigurationError("x")) is True

    def test_prompt_not_self_contained_pauses(self) -> None:
        assert _should_pause(PromptNotSelfContainedError("x")) is True

    def test_generic_error_does_not_pause(self) -> None:
        assert _should_pause(RuntimeError("x")) is False


class TestRunIdFormat:
    def test_session_and_run_id_format(self, scheduler: CronScheduler) -> None:
        scheduler.register_job(
            name="j5", task="t5", interval_seconds=60, prompt="check"
        )
        now = datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        with patch.object(scheduler, "_execute", return_value=_FakeResponse()):
            results = scheduler.run_due_jobs(now=now)
        assert results[0].session_id is not None
        assert results[0].session_id.startswith("session-")
        assert results[0].run_id is not None
        assert results[0].run_id.startswith("run-")
