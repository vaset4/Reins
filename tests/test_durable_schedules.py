"""定时工作真实状态与持久交接的回归验证。

作者：xxx
时间：2026-09-14 19:16:10
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from runtime.cron import CronOutcome, CronScheduler, _interval
from runtime.persistence import RuntimeStore
from runtime.workspaces import WorkspaceStore
from schedules.occurrences import OccurrenceStore
from schedules.persistence import claim_file
from schedules.store import ScheduleStore
from schedules.timing import TimeIntent, format_instant, parse_instant


def test_paused_run_is_not_reported_as_success(tmp_path: Path) -> None:
    """暂停的定时工作仍待继续；传参：隔离目录；返回：无。"""
    scheduler = CronScheduler(project_root=tmp_path, data_root=tmp_path)
    scheduler.register_job(
        name="等待用户", task="task-1", interval_seconds=60, prompt="核对后再写入"
    )
    response = CronOutcome("paused", "需要补充信息", "task-1")
    with patch.object(scheduler, "_execute", return_value=response):
        results = scheduler.run_due_jobs(now=datetime(2099, 1, 1, tzinfo=timezone.utc))
    assert results[0].status == "paused"
    stored = scheduler._store.load_schedule(results[0].job_id)
    assert stored is not None
    assert stored.last_status == "paused"


def test_unknown_time_expression_is_rejected() -> None:
    """不认识的时间表达不能执行为每小时；传参：无；返回：无。"""
    with pytest.raises(ValueError, match="unsupported|不支持"):
        _interval("every sometime")


def test_time_intents_preserve_timezone_and_interval_anchor() -> None:
    """一次性、日历和延迟间隔使用明确时间；传参：无；返回：无。"""
    now = datetime(2026, 9, 14, 0, 0, tzinfo=timezone.utc)
    once = TimeIntent("at:2026-09-14T09:30:00", "Asia/Shanghai")
    assert once.first_at(now) == datetime(2026, 9, 14, 1, 30, tzinfo=timezone.utc)
    assert once.after(now) is None
    calendar = TimeIntent("0 9 * * *", "Asia/Shanghai")
    assert calendar.first_at(now) == datetime(2026, 9, 14, 1, tzinfo=timezone.utc)
    interval = TimeIntent("interval:60", "UTC")
    late = datetime(2026, 9, 14, 0, 3, 12, tzinfo=timezone.utc)
    assert interval.after(now, now=late) == datetime(
        2026, 9, 14, 0, 4, tzinfo=timezone.utc
    )


@pytest.mark.parametrize(
    "expression", ["interval:0", "interval:-1", "later", "0 0 0 * * *"]
)
def test_invalid_time_intents_fail_explicitly(expression: str) -> None:
    """非法表达在接纳前报错；传参：表达；返回：无。"""
    with pytest.raises(ValueError):
        TimeIntent(expression, "UTC").validate()


def test_ambiguous_local_time_requires_an_offset() -> None:
    """夏令时回拨不能默选其中一次；传参：无；返回：无。"""
    with pytest.raises(ValueError, match="ambiguous"):
        TimeIntent("at:2026-11-01T01:30:00", "America/New_York").validate()


def test_reaccepting_after_restart_reuses_occurrence_and_execution_ids(
    tmp_path: Path,
) -> None:
    """接纳后尚未推进游标就崩溃，重启仍只有一次发生；传参：隔离目录；返回：无。"""
    first = OccurrenceStore(tmp_path).accept(
        "daily", "2026-09-14T09:00:00+08:00", next_run_at=None
    )
    restarted = OccurrenceStore(tmp_path)
    repeated = restarted.accept("daily", "2026-09-14T01:00:00Z", next_run_at=None)
    assert repeated == first
    assert len(restarted.list_all()) == 1
    with restarted.claim(first.occurrence_id) as claimed:
        assert claimed is not None
        with OccurrenceStore(tmp_path).claim(first.occurrence_id) as competing:
            assert competing is None
    with restarted.claim(first.occurrence_id) as released:
        assert released is not None


def test_schedule_enumeration_does_not_depend_on_derived_index(tmp_path: Path) -> None:
    """派生索引丢行不能让已接纳提醒消失；传参：隔离目录；返回：无。"""
    store = ScheduleStore(tmp_path)
    try:
        store.create_schedule(
            "reminder",
            "at:2026-09-14T09:00:00+08:00",
            kind="reminder",
            prompt="核对材料",
            workspace_id=WorkspaceStore(tmp_path).register(tmp_path).workspace_id,
        )
        with RuntimeStore(tmp_path).index_connection() as connection:
            connection.execute("DELETE FROM records WHERE kind='schedule'")
        assert [item.schedule_id for item in store.list_enabled_schedules()] == [
            "reminder"
        ]
        assert (
            format_instant(parse_instant(store.list_enabled_schedules()[0].next_run_at))
            == "2026-09-14T01:00:00.000000+00:00"
        )
    finally:
        store.close()


def test_legacy_calendar_requires_explicit_data_cutover(tmp_path: Path) -> None:
    """旧文件资料库不得被新调度静默初始化，原件保持可读；参数：隔离目录；返回：无。"""
    root = tmp_path / "schedules"
    root.mkdir()
    (root / "legacy.yaml").write_text(
        "schedule_id: legacy\ncron: '0 9 * * *'\nenabled: true\n"
        "next_run_at: '2026-09-14T01:00:00Z'\nprompt: check\nlast_status: succeeded\n",
        encoding="utf-8",
    )
    before = (root / "legacy.yaml").read_bytes()
    with pytest.raises(ValueError, match="file data space identity missing"):
        CronScheduler(project_root=tmp_path, data_root=tmp_path)
    assert (root / "legacy.yaml").read_bytes() == before
    assert not (tmp_path / "reins.db").exists()


def test_schedule_does_not_persist_credentials(tmp_path: Path) -> None:
    """计划只存公开模型选择，密钥不能写入计划文件；传参：隔离目录；返回：无。"""
    store = ScheduleStore(tmp_path)
    try:
        with pytest.raises(ValueError, match="public model fields"):
            store.create_schedule(
                "secret",
                "interval:60",
                model_config={"api_key": "test-only-placeholder"},
                workspace_id=WorkspaceStore(tmp_path).register(tmp_path).workspace_id,
            )
        assert store.load_schedule("secret") is None
    finally:
        store.close()


def test_process_exit_releases_execution_claim(tmp_path: Path) -> None:
    """执行进程结束后Windows释放认领锁，无须删除持久锁文件；传参：隔离目录；返回：无。"""
    code = (
        "import sys\nfrom pathlib import Path\nfrom schedules.persistence import claim_file\n"
        "with claim_file(Path(sys.argv[1])) as acquired:\n"
        " print('claimed' if acquired else 'busy', flush=True)\n sys.stdin.read()\n"
    )
    path = tmp_path / "claim.lock"
    process_stop_seconds = 5
    with subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ) as process:
        try:
            assert process.stdout.readline().strip() == "claimed"
            with claim_file(path) as contested:
                assert contested is False
        finally:
            process.terminate()
            process.wait(timeout=process_stop_seconds)
    with claim_file(path) as recovered:
        assert recovered is True
