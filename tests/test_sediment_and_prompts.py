from __future__ import annotations

import json
from pathlib import Path


from memory.sediment import (
    SedimentInput,
    run_sediment,
    should_run_sediment,
)
from memory.store import MemoryStore
from runtime.run_facts import RunFactStore
from runtime.persistence import RuntimeStore
from tasks.store import TaskStore


def test_sediment_whitelist_allows_only_long_done_or_failed_tasks(
    tmp_path: Path,
) -> None:
    inbox_id = _task(tmp_path, "inbox", "done", steps=5, inbox=True)
    short_id = _task(tmp_path, "short", "done", steps=4)
    done_id = _task(tmp_path, "done", "done", steps=5)
    failed_id = _task(tmp_path, "failed", "failed", steps=5)
    active_id = _task(tmp_path, "active", "active", steps=5)

    assert should_run_sediment(tmp_path, inbox_id).reason == "skip_inbox"
    assert should_run_sediment(tmp_path, short_id).reason == "short_task"
    assert should_run_sediment(tmp_path, done_id).memory_type == "fact"
    assert should_run_sediment(tmp_path, failed_id).memory_type == "lesson"
    assert should_run_sediment(tmp_path, active_id).reason == "status_not_eligible"


def test_sediment_commit_protocol_writes_draft_memory_and_task_flag(
    tmp_path: Path,
) -> None:
    task_id = _task(tmp_path, "done", "done", steps=5)

    result = run_sediment(tmp_path, task_id, _proposal)

    assert result.status == "written"
    assert result.memory_id == "sediment-memory"
    with RuntimeStore(tmp_path).snapshot() as source:
        reflections = source.list("task_reflection", filters={"task_id": task_id})
    assert len(reflections) == 1
    assert (
        reflections[0]["payload"]["memory"]["content"]
        == "Keep focused tests before commit."
    )
    task = TaskStore(tmp_path).load_task_payload(task_id)
    assert task["sediment_done"] is True
    assert _indexed_sediment_done(tmp_path, task_id) is True
    memory = MemoryStore(tmp_path).load_memory("sediment-memory")
    assert memory.type == "fact"
    assert memory.content == "Keep focused tests before commit."


def test_failed_task_forces_lesson_and_failure_attempts_are_recorded(
    tmp_path: Path,
) -> None:
    task_id = _task(tmp_path, "failed", "failed", steps=5)

    result = run_sediment(tmp_path, task_id, _proposal)

    assert result.status == "written"
    assert MemoryStore(tmp_path).load_memory("sediment-memory").type == "lesson"

    failing_task = _task(tmp_path, "failing", "done", steps=5)
    for _index in range(4):
        failed = run_sediment(tmp_path, failing_task, lambda _draft: {"type": "fact"})
        assert failed.status == "failed"
    task = TaskStore(tmp_path).load_task_payload(failing_task)
    assert task["sediment_attempts"] == 4
    assert task["sediment_failed"] is True


def test_sediment_updates_preserve_unknown_task_fields(tmp_path: Path) -> None:
    task_id = _task(tmp_path, "custom", "done", steps=5)
    store = TaskStore(tmp_path)
    task = store.load_task_payload(task_id)
    with RuntimeStore(tmp_path).transaction() as batch:
        batch.put(
            "task",
            task_id,
            {**task, "external_note": {"owner": "legacy", "keep": True}},
        )

    result = run_sediment(tmp_path, task_id, _proposal)

    assert result.status == "written"
    updated = store.load_task_payload(task_id)
    assert updated["external_note"] == {"owner": "legacy", "keep": True}


def test_inbox_sediment_skip_does_not_write_sediment_state(tmp_path: Path) -> None:
    task_id = _task(tmp_path, "inbox-skip", "done", steps=5, inbox=True)

    result = run_sediment(tmp_path, task_id, _proposal)

    assert result.status == "skipped"
    assert result.reason == "skip_inbox"
    with RuntimeStore(tmp_path).snapshot() as source:
        assert not source.list("task_reflection", filters={"task_id": task_id})
    task = TaskStore(tmp_path).load_task_payload(task_id)
    assert task["sediment_done"] is False
    assert "sediment_attempts" not in task
    assert "sediment_failed" not in task


def _task(
    data_root: Path,
    task_id: str,
    status: str,
    *,
    steps: int,
    inbox: bool = False,
) -> str:
    store = TaskStore(data_root)
    store.create_task(task_id, task_id=task_id, is_inbox=inbox)
    store.update_task_status(task_id, status)
    fact_store = RunFactStore(data_root)
    for index in range(steps):
        fact_store.append(
            {
                "event": "step",
                "ts": f"2026-05-30T00:00:{index:02d}+00:00",
                "session_id": f"session-{task_id}",
                "run_id": f"run-{task_id}",
                "task_id": task_id,
                "index": index,
            }
        )
    store.update_summary(task_id, "summary")
    store.append_journal(task_id, "journal")
    return task_id


def _indexed_sediment_done(data_root: Path, task_id: str) -> bool:
    """从已追平原件的派生索引读取整理状态；参数：数据根和目标身份；返回：完成标记。"""
    with RuntimeStore(data_root).index_connection() as connection:
        row = connection.execute(
            "SELECT payload FROM records WHERE kind = ? AND record_id = ?",
            ("task", task_id),
        ).fetchone()
    assert row is not None
    return json.loads(row[0])["sediment_done"]


def _proposal(draft: SedimentInput) -> dict[str, object]:
    return {
        "memory_id": "sediment-memory",
        "type": draft.memory_type,
        "content": "Keep focused tests before commit.",
        "tags": ["git-commit"],
    }
