"""【文件存储】【独立检查】从公开源读取和导出边界验证原件完整性。

作者：xxx
时间：2026-09-30 22:00:00
"""

from __future__ import annotations

import json
import os
import shutil
from threading import Event
from pathlib import Path

import pytest

from runtime.persistence import ContentReference, RuntimeStore, SourceCorruptionError
from runtime.run_evidence import RunEvidenceStore
from runtime.request_export import freeze_export, write_export
from runtime.tool_operations import ToolOperationStore
from tests.test_request_inspection_export import _attempt, _detail, _operation, _store


def _replace_preserving_times(path: Path, before: bytes, after: bytes) -> None:
    """模拟保留时间戳的文件恢复或外部改写；参数：路径及替换字节；返回：无。"""
    original = path.read_bytes()
    metadata = path.stat()
    assert before in original and len(before) == len(after)
    path.write_bytes(original.replace(before, after, 1))
    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))


@pytest.mark.parametrize("filename", ["input.json", "output.json"])
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_actual_attempt_original_damage_is_not_recovered_from_index(
    tmp_path: Path, filename: str, damage: str
) -> None:
    """已发送请求或返回原件损坏后不能借索引或副本继续成功；参数：隔离根和损坏方式；返回：无。"""
    inspection, writer, context = _store(tmp_path)
    _attempt(writer, context)
    section = "request" if filename == "input.json" else "response"
    _detail(inspection, request_id="request-1", attempt_id="attempt-1", section=section)
    paths = list(tmp_path.rglob(filename))
    assert len(paths) == 1
    if damage == "missing":
        paths[0].unlink()
    else:
        raw = paths[0].read_bytes()
        paths[0].write_bytes(raw.replace(b'"body"', b'"b0dy"', 1))
    with pytest.raises(SourceCorruptionError):
        _detail(
            inspection, request_id="request-1", attempt_id="attempt-1", section=section
        )
    (tmp_path / "index.sqlite").unlink()
    with pytest.raises(SourceCorruptionError):
        RunEvidenceStore(tmp_path).attempt_references("request-1")


def test_cached_content_revalidates_bytes_when_mtime_is_restored(
    tmp_path: Path,
) -> None:
    """保持长度和mtime的外部修改也不能作为原哈希内容返回；参数：隔离根；返回：无。"""
    store = RuntimeStore(tmp_path)
    reference = store.prepare_content(b"trusted payload")
    assert store.read_content(reference) == b"trusted payload"
    _replace_preserving_times(tmp_path / reference.path, b"trusted", b"changed")
    with pytest.raises(SourceCorruptionError):
        store.read_content(reference)


def test_cached_commit_revalidates_restored_timestamp(tmp_path: Path) -> None:
    """提交日志前缀同长度改写不能被缓存隐藏；参数：隔离根；返回：无。"""
    store = RuntimeStore(tmp_path)
    with store.transaction() as batch:
        batch.put("task", "one", {"state": "active"})
    raw = (tmp_path / "commits.jsonl").read_bytes()
    commit = json.loads(raw)
    old_hash = commit["sha256"].encode()
    changed_hash = (b"0" if old_hash[:1] != b"0" else b"1") + old_hash[1:]
    _replace_preserving_times(tmp_path / "commits.jsonl", old_hash, changed_hash)
    with pytest.raises(SourceCorruptionError):
        with store.snapshot() as snapshot:
            snapshot.get("task", "one")


def test_cached_space_identity_revalidates_restored_timestamp(tmp_path: Path) -> None:
    """空间身份同长度改写不能继续用旧缓存确认原空间；参数：隔离根；返回：无。"""
    store = RuntimeStore(tmp_path)
    old = store.data_space_id.encode()
    changed = (b"0" if old[:1] != b"0" else b"1") + old[1:]
    _replace_preserving_times(tmp_path / "space.json", old, changed)
    with pytest.raises(SourceCorruptionError):
        assert store.data_space_id


def test_entire_source_copy_without_index_preserves_actual_request(
    tmp_path: Path,
) -> None:
    """复制完整源空间而不复制索引后，实际请求和返回仍相同；参数：隔离根；返回：无。"""
    original, copied = tmp_path / "original", tmp_path / "copied"
    inspection, writer, context = _store(original)
    _attempt(
        writer,
        context,
        request={"messages": [{"role": "user", "content": "冻结中文原件" * 100}]},
    )
    expected = _detail(inspection, request_id="request-1", attempt_id="attempt-1")
    shutil.copytree(original, copied, ignore=shutil.ignore_patterns("index.sqlite*"))
    from runtime.request_inspection import RequestInspection

    restored = RequestInspection(copied)
    assert json.loads(
        _detail(restored, request_id="request-1", attempt_id="attempt-1")
    ) == json.loads(expected)
    assert restored.db.data_space_id == inspection.db.data_space_id


def test_explicit_content_reference_cannot_escape_data_space(tmp_path: Path) -> None:
    """显式正文引用不能读取data以外的文件；参数：隔离根；返回：无。"""
    root = tmp_path / "data"
    store = RuntimeStore(root)
    reference = store.prepare_content(b"original")
    (tmp_path / "outside.bin").write_bytes(b"original")
    escaped = ContentReference("../outside.bin", reference.sha256, reference.size)
    with pytest.raises(SourceCorruptionError, match="escapes"):
        store.read_content(escaped)


def test_missing_committed_tombstone_does_not_look_like_absence(tmp_path: Path) -> None:
    """已提交删除事件丢失不能返回不存在掩盖主数据损坏；参数：隔离根；返回：无。"""
    store = RuntimeStore(tmp_path)
    with store.transaction() as batch:
        batch.put("task", "one", {"state": "active"})
    path = store.source_path("task", "one")
    with store.transaction() as batch:
        batch.delete("task", "one")
    path.unlink()
    with pytest.raises(SourceCorruptionError):
        with store.snapshot() as snapshot:
            assert snapshot.get("task", "one") is None


def test_commit_prefix_damage_is_not_hidden_by_an_uncommitted_tail(
    tmp_path: Path,
) -> None:
    """提交日志增长不证明旧前缀完整，坏旧提交不能被当作合法缓存；参数：隔离根；返回：无。"""
    store = RuntimeStore(tmp_path)
    with store.transaction() as batch:
        batch.put("task", "one", {"state": "active"})
    path = tmp_path / "commits.jsonl"
    raw = path.read_bytes()
    old_hash = json.loads(raw)["sha256"].encode()
    changed_hash = (b"0" if old_hash[:1] != b"0" else b"1") + old_hash[1:]
    path.write_bytes(raw.replace(old_hash, changed_hash, 1) + b" ")
    with pytest.raises(SourceCorruptionError):
        with store.snapshot() as snapshot:
            snapshot.get("task", "one")


def test_export_keeps_tool_version_after_late_same_identity_update(
    tmp_path: Path,
) -> None:
    """冻结后的工具同身份更新不得污染成品中的旧结果；参数：隔离根；返回：无。"""
    root = tmp_path / "data"
    inspection, writer, context = _store(root)
    _attempt(writer, context)
    identity = _operation(root, 1)
    manifest = freeze_export(inspection, {"session_id": "s", "run_id": "r"})
    ToolOperationStore(root).write(
        identity,
        {
            "state": "late_completed",
            "result": {"output": "迟到的新结果" * 1000, "status": "ok"},
        },
    )
    (root / "index.sqlite").unlink()
    target = tmp_path / "export"
    target.mkdir()
    write_export(inspection, manifest, target, Event())
    saved = json.loads((target / "requests.json").read_text(encoding="utf-8"))
    assert saved["tools"][0]["result"]["output"] == "完整的小结果"
    assert saved["tools"][0]["state"] == "completed"
