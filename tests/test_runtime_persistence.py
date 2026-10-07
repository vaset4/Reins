"""【存储】【文件原件验证】复合发布、恢复、正文与派生索引。

作者：xxx
时间：2026-09-30 15:30:00
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from llm.messages import model_visible_text
from runtime.file_records import ContentReference, SourceCorruptionError, record_key
from runtime.persistence import RuntimeStore
from runtime.session_message_store import SessionMessageStore
from runtime.workspaces import WorkspaceStore


def test_content_roundtrip_and_workspace_deduplication(tmp_path: Path) -> None:
    """用户标记不冒充内部引用，同工作区正文复用；参数：根；返回：无。"""
    store = RuntimeStore(tmp_path)
    text = "中文原文🙂" * 10000
    value = {
        "$ref": {"text": text},
        "shape": ["text", text],
        "data:": "data:image/png;ordinary",
        "b": [None, 7, False],
    }
    with store.transaction() as batch:
        batch.put("ledger", "one", value)
        batch.put("ledger", "two", {"body": text})
    with store.snapshot() as source:
        assert source.get("ledger", "one") == value
        assert source.get("ledger", "two") == {"body": text}
        assert source.raw("ledger", "one").payload["$ref"]["text"] is None
    assert len(tuple((tmp_path / "global" / "objects").rglob("*.txt"))) == 1
    assert text not in (tmp_path / "commits.jsonl").read_text(encoding="utf-8")


def test_compound_failure_rolls_back_all_domains(tmp_path: Path) -> None:
    """跨Store失败不发布输入或当前叶；参数：隔离根；返回：无。"""
    messages = SessionMessageStore(tmp_path)
    messages.accept_input("session-a", "原始", input_id="first")
    with pytest.raises(OSError, match="publication failed"):
        with RuntimeStore(tmp_path).transaction() as batch:
            messages.accept_input("session-a", "新内容" * 1000, input_id="second")
            batch.put(
                "background_input",
                "second",
                {"input_id": "second"},
                session_id="session-a",
            )
            raise OSError("publication failed")
    assert [entry.entry_id for entry in messages.read_entries("session-a")] == ["first"]
    with messages.database.snapshot() as source:
        assert source.get("background_input", "second") is None


def _append_input(root: Path, number: int) -> str:
    """独立Store接纳输入；参数：根与编号；返回：已提交身份。"""
    return (
        SessionMessageStore(root)
        .accept_input("session-parallel", f"输入 {number}", input_id=f"input-{number}")
        .entry_id
    )


def test_concurrent_writers_keep_one_connected_tree(tmp_path: Path) -> None:
    """多个实例竞争同一叶仍完整提交；参数：根；返回：无。"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        submitted = tuple(
            executor.map(lambda number: _append_input(tmp_path, number), range(12))
        )
    entries = SessionMessageStore(tmp_path).read_entries("session-parallel")
    assert {entry.entry_id for entry in entries} == set(submitted)
    assert all(
        right.parent_id == left.entry_id for left, right in zip(entries, entries[1:])
    )


def test_process_exit_keeps_committed_message_only(tmp_path: Path) -> None:
    """强杀不发布未确认输入，OS释放锁；参数：根；返回：无。"""
    code = """
import os,sys
from runtime.persistence import RuntimeStore
from runtime.session_message_store import SessionMessageStore
store=SessionMessageStore(sys.argv[1])
store.accept_input('session-crash','已提交',input_id='committed')
with RuntimeStore(sys.argv[1]).transaction():
    store.accept_input('session-crash','未提交',input_id='uncommitted')
    os._exit(9)
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code, str(tmp_path)],
        timeout=15,
        check=False,
    )
    assert result.returncode == 9
    messages = SessionMessageStore(tmp_path)
    entries = messages.read_entries("session-crash")
    assert [entry.entry_id for entry in entries] == ["committed"]
    assert model_visible_text(entries[0].message) == "已提交"
    messages.accept_input("session-crash", "恢复后新输入")


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_corrupt_content_is_not_partial_history(tmp_path: Path, mutation: str) -> None:
    """正文坏源明确失败；参数：根与损坏方式；返回：无。"""
    messages = SessionMessageStore(tmp_path)
    messages.accept_input("session-a", "重要原文" * 1000)
    path = next((tmp_path / "global" / "objects").rglob("*.txt"))
    if mutation == "missing":
        path.unlink()
    else:
        path.write_text("篡改", encoding="utf-8")
    with pytest.raises(SourceCorruptionError, match="content missing or corrupt"):
        messages.read_entries("session-a")


def test_uncommitted_tail_is_invisible_and_diagnosed_before_next_write(
    tmp_path: Path,
) -> None:
    """未提交日志尾部只留诊断不变输入；参数：根；返回：无。"""
    messages = SessionMessageStore(tmp_path)
    messages.accept_input("session-a", "已提交", input_id="first")
    path = messages.database.source_path("session", "session-a")
    with path.open("ab") as handle:
        handle.write(b"not committed\n")
    assert len(messages.read_entries("session-a")) == 1
    messages.accept_input("session-a", "下一条", input_id="second")
    assert len(messages.read_entries("session-a")) == 2
    reports = tuple((tmp_path / "runtime" / "recovery").glob("*.json"))
    assert len(reports) == 1
    assert (
        bytes.fromhex(json.loads(reports[0].read_bytes())["tail_hex"])
        == b"not committed\n"
    )


def test_committed_event_corruption_fails_without_empty_reinitialization(
    tmp_path: Path,
) -> None:
    """已提交中段坏行不能静默跳过；参数：根；返回：无。"""
    messages = SessionMessageStore(tmp_path)
    messages.accept_input("session-a", "原文")
    path = messages.database.source_path("session", "session-a")
    path.write_bytes(
        path.read_bytes().replace(b'"kind":"session"', b'"kind":"sessioX"', 1)
    )
    with pytest.raises(SourceCorruptionError, match="committed source corrupt"):
        # 当前session索引指向后来修订，检查首次事件通过完整源重建
        restored = tmp_path.parent / (tmp_path.name + "-copy")
        shutil.copytree(tmp_path, restored)
        SessionMessageStore(restored).read_entries("session-a")
    (tmp_path / "index.sqlite").unlink()
    with pytest.raises(SourceCorruptionError, match="committed source corrupt"):
        messages.database.rebuild_index()


def test_missing_index_rebuild_and_stopped_source_backup(tmp_path: Path) -> None:
    """复制完整源而不带索引仍保留身份与消息；参数：根；返回：无。"""
    root = tmp_path / "source"
    workspaces = WorkspaceStore(root)
    workspace = workspaces.bind_session("session-a", tmp_path / "project")
    workspaces.messages.accept_input("session-a", "故障排查正文", input_id="first")
    identity = workspaces.database.data_space_id
    copy_root = tmp_path / "restored"
    shutil.copytree(root, copy_root, ignore=shutil.ignore_patterns("index.sqlite"))
    restored = WorkspaceStore(copy_root)
    assert restored.database.data_space_id == identity
    assert restored.for_session("session-a").workspace_id == workspace.workspace_id
    assert len(restored.messages.read_entries("session-a")) == 1
    with restored.database.index_connection() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM records WHERE kind='session_entry'"
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM records_fts WHERE records_fts MATCH ?",
                ('"故障"',),
            ).fetchone()[0]
            == 1
        )


def test_index_failure_does_not_reject_committed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """索引失败后回执仍对应已提交输入；参数：根和注入；返回：无。"""
    from runtime.file_index import FileIndex

    def fail_index(self: FileIndex, state: object, space_id: str) -> None:
        """模拟派生构建故障；参数：索引、状态、身份；返回：无。"""
        raise OSError("index unavailable")

    messages = SessionMessageStore(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(FileIndex, "synchronize", fail_index)
        accepted = messages.accept_input("session-a", "持久正文", input_id="once")
    assert messages.database.index_status["state"] == "failed"
    assert messages.accept_input("session-a", "持久正文", input_id="once") == accepted
    assert len(messages.read_entries("session-a")) == 1
    messages.database.rebuild_index()
    assert messages.database.index_status["state"] == "ready"


def test_historical_snapshot_and_raw_content_paging(tmp_path: Path) -> None:
    """旧截止点不被后续修订污染且正文可分块；参数：根；返回：无。"""
    store = RuntimeStore(tmp_path)
    text = "记录内容" * 10000
    with store.transaction() as batch:
        batch.put("run_fact", "run", {"text": text, "state": "running"})
    with store.snapshot() as source:
        cutoff = source.sequence
        raw = source.raw("run_fact", "run")
        reference = ContentReference.from_mapping(raw.references[0]["content"])
    with store.transaction() as batch:
        batch.put("run_fact", "run", {"text": "finished", "state": "done"})
    with store.snapshot(sequence=cutoff) as source:
        assert source.get("run_fact", "run")["state"] == "running"
    encoded = text.encode("utf-8")
    assert store.read_content(reference, offset=10, limit=32) == encoded[10:42]
    assert b"".join(store.iter_content(reference, chunk_size=1024)) == encoded


def test_nested_savepoint_and_expected_revision(tmp_path: Path) -> None:
    """内层失败只撤销自身修订，过期版本明确冲突；参数：根；返回：无。"""
    store = RuntimeStore(tmp_path)
    with store.transaction() as batch:
        batch.put("task", "task", {"goal": "first"})
        with pytest.raises(ValueError, match="stop"):
            with RuntimeStore(tmp_path).transaction() as nested:
                nested.put("task", "task", {"goal": "second"})
                raise ValueError("stop")
        assert batch.get("task", "task") == {"goal": "first"}
    with pytest.raises(ValueError, match="revision changed"):
        with store.transaction() as batch:
            batch.put("task", "task", {"goal": "third"}, expected_revision=0)


def test_fsync_failure_before_commit_never_acknowledges_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """源追加同步失败不产生提交记录；参数：根和注入；返回：无。"""
    import runtime.file_journal as journal

    messages = SessionMessageStore(tmp_path)
    messages.accept_input("session-a", "first", input_id="first")
    real_append = journal.append_synced

    def fail_commit(path: Path, data: bytes) -> int:
        """模拟提交前文件同步失败；参数：路径和候选；返回：无。"""
        if path.name == "commits.jsonl":
            raise OSError("commit sync failed")
        return real_append(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(journal, "append_synced", fail_commit)
        with pytest.raises(OSError, match="commit sync failed"):
            messages.accept_input("session-a", "second", input_id="second")
    assert [entry.entry_id for entry in messages.read_entries("session-a")] == ["first"]
    messages.accept_input("session-a", "second", input_id="second")
    with messages.database.snapshot() as source:
        assert (
            source.get("session_entry", record_key("session-a", "second")) is not None
        )


@pytest.mark.parametrize("kill_after_commit", [False, True])
def test_killed_writer_at_real_publication_boundary(
    tmp_path: Path, kill_after_commit: bool
) -> None:
    """在实际落盘窗口强杀进程只恢复完整提交；参数：根与停止点；返回：无。"""
    root = tmp_path / "data"
    SessionMessageStore(root).accept_input("session-a", "first", input_id="first")
    signal = tmp_path / "writer-ready"
    code = """
import sys,time
from pathlib import Path
import runtime.file_journal as journal
from runtime.session_message_store import SessionMessageStore
root,signal,after=sys.argv[1],Path(sys.argv[2]),sys.argv[3]=='True'
append=journal.append_synced
def boundary(path,data):
    result=append(path,data)
    if (after and path.name=='commits.jsonl') or (not after and path.name=='events.jsonl'):
        signal.write_text('ready')
        time.sleep(120)
    return result
journal.append_synced=boundary
SessionMessageStore(root).accept_input('session-a','second',input_id='second')
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-X",
            "utf8",
            "-c",
            code,
            str(root),
            str(signal),
            str(kill_after_commit),
        ]
    )
    try:
        deadline = time.monotonic() + 10
        while (
            not signal.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert signal.exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
    messages = SessionMessageStore(root)
    expected = ["first", "second"] if kill_after_commit else ["first"]
    assert [entry.entry_id for entry in messages.read_entries("session-a")] == expected
    messages.accept_input("session-a", "third", input_id="third")
    assert messages.current_leaf("session-a") == "third"


def test_space_identity_tampering_rejects_existing_commits(tmp_path: Path) -> None:
    """提交链不接受换空间身份；参数：根；返回：无。"""
    SessionMessageStore(tmp_path).accept_input("session-a", "first")
    marker = tmp_path / "space.json"
    metadata = json.loads(marker.read_bytes())
    marker.write_text(json.dumps({**metadata, "space_id": "0" * 32}), encoding="utf-8")
    with pytest.raises(SourceCorruptionError, match="identity changed"):
        SessionMessageStore(tmp_path).read_entries("session-a")


def test_commit_hash_tampering_is_detected_after_local_write(tmp_path: Path) -> None:
    """本进程刚提交后也不得忽略提交日志篡改；参数：根；返回：无。"""
    messages = SessionMessageStore(tmp_path)
    messages.accept_input("session-a", "first")
    journal = tmp_path / "commits.jsonl"
    payload = json.loads(journal.read_bytes())
    payload["batch_id"] = "0" * len(payload["batch_id"])
    journal.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SourceCorruptionError, match="hash or sequence chain"):
        messages.read_entries("session-a")


def test_existing_snapshot_revalidates_changed_source(tmp_path: Path) -> None:
    """同一快照内再读也不能用缓存隐藏新损坏；参数：根；返回：无。"""
    import os

    store = RuntimeStore(tmp_path)
    with store.transaction() as batch:
        batch.put("task", "one", {"body": "trusted"})
    path = store.source_path("task", "one")
    with store.snapshot() as source:
        assert source.get("task", "one") == {"body": "trusted"}
        metadata = path.stat()
        path.write_bytes(path.read_bytes().replace(b"trusted", b"changed"))
        os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        with pytest.raises(SourceCorruptionError):
            source.get("task", "one")


def test_deferred_index_query_failure_keeps_source_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """索引延迟追平失败不撤销原件且后续写入保留错误状态；参数：根和注入；返回：无。"""
    from runtime.file_index import FileIndex

    store = RuntimeStore(tmp_path)
    with store.transaction() as batch:
        batch.put("task", "one", {"state": "first"})

    def fail_query(
        self: FileIndex, state: object, identity: str, *, force: bool = False
    ) -> None:
        """模拟真正查询时派生同步失败；参数：索引、源、身份和策略；返回：无。"""
        raise OSError("query index failed")

    with monkeypatch.context() as patch:
        patch.setattr(FileIndex, "synchronize", fail_query)
        with store.transaction() as batch:
            batch.put("task", "one", {"state": "second"})
        assert store.index_status["state"] == "stale"
        with pytest.raises(OSError, match="query index failed"):
            with store.index_connection():
                pass
        with store.transaction() as batch:
            batch.put("task", "one", {"state": "third"})
        assert store.index_status["state"] == "failed"
        with store.snapshot() as source:
            assert source.get("task", "one") == {"state": "third"}
    with store.index_connection() as connection:
        assert (
            json.loads(
                connection.execute(
                    "SELECT payload FROM records WHERE kind='task'"
                ).fetchone()[0]
            )["state"]
            == "third"
        )


def test_newer_same_space_index_is_rebuilt_from_older_complete_sources(
    tmp_path: Path,
) -> None:
    """较新索引不能替较旧完整源备份制造不存在的事实；参数：根；返回：无。"""
    root, backup = tmp_path / "current", tmp_path / "backup"
    store = RuntimeStore(root)
    with store.transaction() as batch:
        batch.put("task", "one", {"state": "original"})
    shutil.copytree(root, backup)
    with store.transaction() as batch:
        batch.put("task", "two", {"state": "later"})
    store.rebuild_index()
    shutil.copyfile(root / "index.sqlite", backup / "index.sqlite")
    restored = RuntimeStore(backup)
    with restored.index_connection() as connection:
        assert {
            row[0]
            for row in connection.execute(
                "SELECT record_id FROM records WHERE kind='task'"
            )
        } == {"one"}
    with restored.snapshot() as source:
        assert source.get("task", "two") is None


def test_event_fsync_failure_keeps_complete_previous_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实事件fsync调用失败时不发布下一条输入；参数：根和系统同步注入；返回：无。"""
    store = SessionMessageStore(tmp_path)
    store.accept_input("session-a", "first", input_id="first")
    committed = (tmp_path / "commits.jsonl").read_bytes()

    def fail_fsync(descriptor: int) -> None:
        """模拟当前文件句柄同步失败；参数：描述符；返回：无，真实错误抛出。"""
        raise OSError("event fsync failed")

    with monkeypatch.context() as patch:
        patch.setattr("runtime.file_journal.os.fsync", fail_fsync)
        with pytest.raises(OSError, match="event fsync failed"):
            store.accept_input("session-a", "second", input_id="second")
    assert (tmp_path / "commits.jsonl").read_bytes() == committed
    assert [entry.entry_id for entry in store.read_entries("session-a")] == ["first"]
    store.accept_input("session-a", "second", input_id="second")
    assert [entry.entry_id for entry in store.read_entries("session-a")] == [
        "first",
        "second",
    ]
