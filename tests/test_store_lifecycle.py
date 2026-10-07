from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from artifacts.store import ArtifactStore
from memory.store import MemoryStore
from runtime.persistence import RuntimeStore
from schedules.store import ScheduleStore
from tasks.store import TaskStore


@pytest.mark.parametrize(
    "factory, after_close, expected",
    [
        (
            lambda root: TaskStore(root),
            lambda store: (
                store.create_task("after close", task_id="task-closed").task_id
            ),
            "task-closed",
        ),
        (
            lambda root: ArtifactStore(root),
            lambda store: store.load_artifact("missing-artifact"),
            None,
        ),
    ],
)
def test_store_remains_usable_after_short_index_connection_closes(
    tmp_path: Path,
    factory,
    after_close,
    expected,
) -> None:
    """索引短连接释放后领域存储仍可读写原件；参数：目录、存储工厂、操作和预期；返回：无。"""
    store = factory(tmp_path)
    with RuntimeStore(tmp_path).index_connection() as connection:
        assert connection.execute("SELECT 1").fetchone()[0] == 1
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    store.close()
    assert after_close(store) == expected


def test_schedule_store_rejects_use_after_close_without_leaking_connection(
    tmp_path: Path,
) -> None:
    """调度门面关闭后明确拒绝复用，新实例仍可读取原件；参数：独立目录；返回：无。"""
    store = ScheduleStore(tmp_path)
    assert store.list_enabled_schedules() == []
    store.close()
    with pytest.raises(RuntimeError, match="ScheduleStore is closed"):
        store.list_enabled_schedules()
    reopened = ScheduleStore(tmp_path)
    try:
        assert reopened.list_enabled_schedules() == []
    finally:
        reopened.close()


def test_memory_originals_survive_the_index_connection_closing(tmp_path: Path) -> None:
    # 记忆的原文是文件、索引只是派生物：连接关掉后原文查询必须照常，
    # 索引不可用时也不该把整库判成不可读
    store = MemoryStore(tmp_path)
    store.create_memory("fact", "索引只是派生", ["memory"])
    store.close()

    assert [record.memory_id for record in store.list_memories()] != []
