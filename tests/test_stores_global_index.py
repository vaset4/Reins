from __future__ import annotations

from pathlib import Path

from artifacts.store import ArtifactStore
from schedules.store import ScheduleStore
from runtime.persistence import RuntimeStore
from runtime.workspaces import WorkspaceStore
from tasks.index_sync import rebuild_full_index, sync_tasks_table
from tasks.store import TASK_STATUS_DONE, TaskStore


def test_task_store_persists_and_indexes_tasks(tmp_path: Path) -> None:
    """任务状态与查询索引一致，原件保存在全局日志；参数：隔离根；返回：无。"""
    store = TaskStore(tmp_path)
    task = store.create_task("ship phase 1", task_id="2026-05-04-01")
    task.tags.append("phase1")
    store.update_task_status(task.task_id, TASK_STATUS_DONE)

    assert not (tmp_path / "tasks" / task.task_id / "task.yaml").exists()
    loaded = store.load_task(task.task_id)
    assert loaded is not None
    assert loaded.status == TASK_STATUS_DONE
    assert [item.task_id for item in store.list_tasks(TASK_STATUS_DONE)] == [
        task.task_id
    ]


def test_task_store_reads_inbox_tasks(tmp_path: Path) -> None:
    """临时目标仍可按原语义查询；参数：隔离根；返回：无。"""
    store = TaskStore(tmp_path)
    store.create_task("quick chat", task_id="2026-05-04-02", is_inbox=True)

    inbox = store.get_inbox_tasks()
    assert [item.task_id for item in inbox] == ["2026-05-04-02"]
    assert not (tmp_path / "tasks" / "_inbox" / "2026-05-04-02" / "task.yaml").exists()


def test_indexes_rebuild_without_replacing_canonical_records(tmp_path: Path) -> None:
    """重建派生查询后原任务和产物仍完整存在；参数：隔离根；返回：无。"""
    task = TaskStore(tmp_path).create_task("searchable task", task_id="2026-05-04-03")
    ScheduleStore(tmp_path).create_schedule(
        "nightly",
        "0 1 * * *",
        workspace_id=WorkspaceStore(tmp_path).register(tmp_path).workspace_id,
    )
    (tmp_path / "out.txt").write_text("done", encoding="utf-8")
    ArtifactStore(tmp_path).create_artifact(
        "2026-05-04-03", "output", str(tmp_path / "out.txt"), "done", 4
    )

    counts = rebuild_full_index(tmp_path)
    assert counts == {"tasks": 1, "schedules": 1, "artifacts": 1}
    assert sync_tasks_table(tmp_path) == 1
    assert TaskStore(tmp_path).load_task(task.task_id) == task


def test_tasks_fts_matches_goal_text(tmp_path: Path) -> None:
    """全文索引定位文件原件中的真实目标；参数：隔离根；返回：无。"""
    TaskStore(tmp_path).create_task("finish alpha beta", task_id="2026-05-04-04")

    with RuntimeStore(tmp_path).index_connection() as connection:
        rows = connection.execute(
            "SELECT record_id FROM records_fts WHERE kind='task' AND records_fts MATCH ?",
            ("alpha",),
        ).fetchall()
    assert [row[0] for row in rows] == ["2026-05-04-04"]


def test_schedule_store_crud_and_enabled_listing(tmp_path: Path) -> None:
    """计划与后继时间经规范存储往返；参数：隔离根；返回：无。"""
    store = ScheduleStore(tmp_path)
    store.create_schedule(
        "daily-review",
        "0 3 * * *",
        workspace_id=WorkspaceStore(tmp_path).register(tmp_path).workspace_id,
        target_task_id="2026-05-04-01",
        next_run_at="2026-05-05T03:00:00+00:00",
    )

    loaded = store.load_schedule("daily-review")
    assert loaded is not None
    assert loaded.target_task_id == "2026-05-04-01"
    assert [item.schedule_id for item in store.list_enabled_schedules()] == [
        "daily-review"
    ]

    updated = store.update_next_run_at("daily-review", "2026-05-06T03:00:00+00:00")
    assert updated.next_run_at == "2026-05-06T03:00:00+00:00"


def test_artifact_store_crud_and_retention(tmp_path: Path) -> None:
    """实际产物保留原到期语义；参数：隔离根；返回：无。"""
    TaskStore(tmp_path).create_task("artifact task", task_id="2026-05-04-05")
    store = ArtifactStore(tmp_path)
    (tmp_path / "shot.png").write_bytes(b"x" * 12)
    (tmp_path / "out.txt").write_bytes(b"x" * 20)

    screenshot = store.create_artifact(
        "2026-05-04-05", "screenshot", str(tmp_path / "shot.png"), "screen", 12
    )
    output = store.create_artifact(
        "2026-05-04-05", "output", str(tmp_path / "out.txt"), "final", 20
    )

    loaded = store.load_artifact(screenshot.artifact_id)
    assert loaded is not None
    assert loaded.summary == "screen"
    assert output.retention_until is None
    assert screenshot.artifact_id in store.prune_expired_artifacts(
        "9999-01-01T00:00:00+00:00"
    )
