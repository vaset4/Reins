"""【文件原件】【批量读取】验证同一原件的读取成本、完整性与并发快照。

作者：xxx
时间：2026-10-06 20:50:00
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import pytest

from runtime.file_records import SourceCorruptionError, StoredRecord
from runtime.persistence import RuntimeStore
from scripts.benchmark_session_metrics import measure_reads

LARGE_GROUP_SIZE = 100
WRITER_WAIT_SECONDS = 5


def _seed_facts(root: Path) -> RuntimeStore:
    """把大小两组事实发布到同一会话原件；参数：独立数据根；返回：实际文件存储。"""
    store = RuntimeStore(root)
    with store.transaction() as batch:
        for index in range(LARGE_GROUP_SIZE + 1):
            batch.put(
                "run_fact",
                f"fact-{index}",
                {
                    "event": "test:observed",
                    "session_id": "session",
                    "run_id": "small" if index == 0 else "large",
                    "value": f"value-{index:03d}",
                },
                session_id="session",
            )
    return store


def test_same_original_does_not_reopen_for_every_selected_fact(tmp_path: Path) -> None:
    """单一原件从一条扩为百条时，文件打开次数不随条数增长；参数：数据根；返回：无。"""
    store = _seed_facts(tmp_path)
    with measure_reads(tmp_path) as small_meter:
        with store.snapshot() as source:
            small = source.list("run_fact", filters={"run_id": "small"})
    with measure_reads(tmp_path) as large_meter:
        with store.snapshot() as source:
            large = source.list("run_fact", filters={"run_id": "large"})
    assert len(small) == 1 and len(large) == LARGE_GROUP_SIZE
    assert {row["value"] for row in large} == {
        f"value-{index:03d}" for index in range(1, LARGE_GROUP_SIZE + 1)
    }
    assert large_meter.snapshot().file_opens == small_meter.snapshot().file_opens


def test_bulk_read_rejects_same_size_corruption_with_restored_mtime(
    tmp_path: Path,
) -> None:
    """改回文件时间不能让批量读取信任受损原件；参数：独立数据根；返回：无。"""
    store = _seed_facts(tmp_path)
    path = store.source_path("run_fact", "fact-50")
    stat = path.stat()
    before = path.read_bytes()
    changed = before.replace(b"value-050", b"value-999")
    assert changed != before and len(changed) == len(before)
    path.write_bytes(changed)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(SourceCorruptionError):
        with store.snapshot() as source:
            source.list("run_fact", filters={"run_id": "large"})


def test_bulk_read_keeps_snapshot_when_another_thread_appends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """读取期间合法追加不破坏旧快照，下次查询可见新事实；参数：目录及并发探针；返回：无。"""
    from runtime import persistence

    store = _seed_facts(tmp_path)
    deepcopy = persistence.copy.deepcopy
    appended = False

    def append_fact() -> None:
        """通过正式事务追加同一原件；参数：无；返回：无。"""
        with RuntimeStore(tmp_path).transaction() as batch:
            batch.put(
                "run_fact",
                "new",
                {
                    "event": "test:observed",
                    "session_id": "session",
                    "run_id": "large",
                    "value": "new",
                },
                session_id="session",
            )

    with ThreadPoolExecutor(max_workers=1) as pool:

        def copy_with_concurrent_append(value, *args, **kwargs):
            """复制读取结果时协调另一写者，保持真实副本语义；参数：原对象及复制选项；返回：独立副本。"""
            nonlocal appended
            if isinstance(value, StoredRecord) and not appended:
                appended = True
                pool.submit(append_fact).result(WRITER_WAIT_SECONDS)
            return deepcopy(value, *args, **kwargs)

        monkeypatch.setattr(persistence.copy, "deepcopy", copy_with_concurrent_append)
        with store.snapshot() as source:
            records = source.list("run_fact", filters={"run_id": "large"})
    assert appended and len(records) == LARGE_GROUP_SIZE
    assert all(row["value"] != "new" for row in records)
    with store.snapshot() as source:
        assert (
            len(source.list("run_fact", filters={"run_id": "large"}))
            == LARGE_GROUP_SIZE + 1
        )


def test_bulk_read_rejects_original_corrupted_while_copying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """复制期间原件遭等长篡改且时间被恢复也必须报错；参数：目录与复制探针；返回：无。"""
    from runtime import persistence

    store = _seed_facts(tmp_path)
    path = store.source_path("run_fact", "fact-1")
    original_stat = path.stat()
    before = path.read_bytes()
    changed = before.replace(b"value-001", b"value-999")
    assert changed != before and len(changed) == len(before)
    deepcopy = persistence.copy.deepcopy
    corrupted = False

    def copy_with_corruption(value, *args, **kwargs):
        """在首条已验证记录复制时篡改原件；参数：复制对象与选项；返回：真实独立副本。"""
        nonlocal corrupted
        if isinstance(value, StoredRecord) and not corrupted:
            corrupted = True
            path.write_bytes(changed)
            os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        return deepcopy(value, *args, **kwargs)

    monkeypatch.setattr(persistence.copy, "deepcopy", copy_with_corruption)
    with pytest.raises(SourceCorruptionError):
        with store.snapshot() as source:
            source.list("run_fact", filters={"run_id": "large"})
    assert corrupted
