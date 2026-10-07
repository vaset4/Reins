"""【记忆】【并发发布】验证准备新版、发布原件与读取之间的实际边界。

作者：xxx
时间：2026-10-06 19:15:00
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from typing import cast

import pytest

from context.memory_recall import recall_memories
from memory.records import Memory, MemorySource
from memory.store import MemoryStore
from runtime.memory_actions import _maintenance_publication
from runtime.native_actions import NativeActionContext

WAIT_SECONDS = 5
PUBLICATION_PROBE_SECONDS = 0.1


def test_prepared_revision_does_not_block_recall_or_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实修订准备好但尚未发布时仍读旧版，发布后读新版；参数：目录及暂停器；返回：无。"""
    writer = MemoryStore(tmp_path)
    identity = writer.create_memory("fact", "budget is 620", ["budget"])
    old = writer.load_memory(identity)
    prepared, release = Event(), Event()
    publish = writer._publish

    def pause_prepared(memory: Memory, *, previous: Memory | None = None) -> None:
        """在新版准备完成后暂停实际发布；参数：新版及旧版；返回：无。"""
        prepared.set()
        assert release.wait(WAIT_SECONDS)
        publish(memory, previous=previous)

    monkeypatch.setattr(writer, "_publish", pause_prepared)
    with ThreadPoolExecutor(max_workers=2) as pool:
        revision = pool.submit(
            writer.revise_memory,
            identity,
            "budget is 800",
            expected_version=old.version,
            reason="用户更正预算",
            sources=(MemorySource("user_input", "input-2"),),
        )
        try:
            assert prepared.wait(WAIT_SECONDS)
            recalled = pool.submit(
                recall_memories, tmp_path, task_summary="budget", task_tags=[]
            ).result(WAIT_SECONDS)
            assert recalled[0].memory.content == old.content
            assert recalled[0].memory.version == old.version
            assert MemoryStore(tmp_path).load_memory(identity).last_used_at is not None
        finally:
            release.set()
        updated = revision.result(WAIT_SECONDS)
    current = recall_memories(tmp_path, task_summary="budget", task_tags=[])[0].memory
    assert current.content == "budget is 800" and current.version == updated.version
    assert writer.load_memory(identity, version=old.version).content == old.content


@pytest.mark.parametrize("damage", ["missing", "malformed"])
def test_preparing_writer_does_not_hide_original_damage(
    tmp_path: Path, damage: str
) -> None:
    """写者准备期间损坏原件仍及时报错，不能返回索引旧稿；参数：目录和损坏方式；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "budget is 620", ["budget"])
    path = store.current_path(identity)
    expected_error: type[Exception]
    if damage == "missing":
        path.unlink()
        expected_error = FileNotFoundError
    else:
        path.write_text("---\nbroken: [\n---\nnew budget", encoding="utf-8")
        expected_error = ValueError
    with ThreadPoolExecutor(max_workers=1) as pool, store.locked():
        pending = pool.submit(
            recall_memories, tmp_path, task_summary="budget", task_tags=[]
        )
        with pytest.raises(expected_error):
            pending.result(WAIT_SECONDS)


def test_reader_sees_complete_publication_and_matching_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """正文替换与登记之间不暴露半次发布，索引须对应实际新版；参数：目录和暂停器；返回：无。"""
    writer = MemoryStore(tmp_path)
    identity = writer.create_memory("fact", "budget is 620", ["budget"])
    old = writer.load_memory(identity)
    replaced, release, reading = Event(), Event(), Event()
    register = writer._files.register

    def pause_registration(memory: Memory, path: Path) -> None:
        """只在新版正文已替换而尚未登记时暂停；参数：记忆和原件路径；返回：无。"""
        if memory.content == "budget is 800":
            replaced.set()
            assert release.wait(WAIT_SECONDS)
        register(memory, path)

    def read_published() -> Memory:
        """读取当前原件并核对其检索命中；参数：无；返回：召回的版本。"""
        reading.set()
        return recall_memories(tmp_path, task_summary="800", task_tags=[])[0].memory

    monkeypatch.setattr(writer._files, "register", pause_registration)
    with ThreadPoolExecutor(max_workers=2) as pool:
        revision = pool.submit(
            writer.revise_memory,
            identity,
            "budget is 800",
            expected_version=old.version,
            reason="用户更正预算",
            sources=(MemorySource("user_input", "input-2"),),
        )
        try:
            assert replaced.wait(WAIT_SECONDS)
            result = pool.submit(read_published)
            assert reading.wait(WAIT_SECONDS)
            with pytest.raises(TimeoutError):
                result.result(PUBLICATION_PROBE_SECONDS)
        finally:
            release.set()
        updated = revision.result(WAIT_SECONDS)
        recalled = result.result(WAIT_SECONDS)
    assert recalled.content == "budget is 800" and recalled.version == updated.version
    assert MemoryStore(tmp_path).index_status().state == "current"


def test_maintenance_waits_for_reader_before_taking_fact_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """维护提交等待读快照时不能先占事实锁，读者对账须继续；参数：目录和时序探针；返回：无。"""
    reader = MemoryStore(tmp_path)
    identity = reader.create_memory("fact", "budget is 620", ["budget"])
    old = reader.load_memory(identity)
    reader_thread, attempted = get_ident(), Event()
    publication = MemoryStore.publication

    @contextmanager
    def observe_publication(store: MemoryStore) -> Iterator[None]:
        """标记维护线程开始等待发布锁，保持真实互斥行为；参数：存储；返回：原发布窗口。"""
        if get_ident() != reader_thread:
            attempted.set()
        with publication(store):
            yield

    def maintain() -> Memory:
        """沿维护实际提交边界修订当前记忆；参数：无；返回：真实新版。"""
        context = cast(NativeActionContext, SimpleNamespace(run=object()))
        with _maintenance_publication(context, data_root=tmp_path):
            return MemoryStore(tmp_path).revise_memory(
                identity,
                "budget is 800",
                expected_version=old.version,
                reason="维护更正预算",
                sources=(MemorySource("user_input", "input-2"),),
            )

    monkeypatch.setattr(
        "runtime.knowledge_maintenance.automatic_origin",
        lambda _run: {"source_session_id": "chat"},
    )
    monkeypatch.setattr(MemoryStore, "publication", observe_publication)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with reader.index_snapshot():
            pending = pool.submit(maintain)
            assert attempted.wait(WAIT_SECONDS)
            assert reader.load_memory(identity).version == old.version
        updated = pending.result(WAIT_SECONDS)
    assert reader.load_memory(identity).version == updated.version
