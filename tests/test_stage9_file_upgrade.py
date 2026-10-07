"""【存储】【阶段九升级】v3/v4保留升级及旧读者拒绝新记录门禁。

作者：xxx
时间：2026-10-01 12:00:00
"""

from __future__ import annotations

import json
import shutil

import pytest

from runtime.file_records import FORMAT_VERSION, SourceCorruptionError, json_bytes
from runtime.file_upgrade import upgrade_file_space
from runtime.persistence import RuntimeStore
from runtime.session_message_store import SessionMessageStore


@pytest.mark.parametrize("old_version", [3, 4])
def test_upgrade_retains_identity_commits_originals_and_backup(
    tmp_path, monkeypatch, old_version
):
    """格式门禁升级不改空间或原件，旧读者明确拒绝；参数：隔离根、原格式；返回：无。"""
    root, backup = tmp_path / "source", tmp_path / "backup"
    owner = SessionMessageStore(root)
    owner.accept_input("kept", "原始用户条件：不得上传", input_id="input")
    database = RuntimeStore(root)
    identity = database.data_space_id
    marker = root / "space.json"
    marker.write_bytes(
        json_bytes({"space_id": identity, "format_version": old_version})
    )
    prefix = (root / "commits.jsonl").read_bytes()
    originals = owner.read_entries("kept")
    shutil.copytree(root, backup)
    result = upgrade_file_space(root, backup)
    assert (
        result["previous_format"] == old_version
        and result["format_version"] == FORMAT_VERSION == 5
    )
    assert database.data_space_id == identity
    assert (root / "commits.jsonl").read_bytes() == prefix
    assert (
        json.loads((backup / "space.json").read_bytes())["format_version"]
        == old_version
    )
    assert SessionMessageStore(root).read_entries("kept") == originals
    with database.transaction() as batch:
        batch.put(
            "context_compaction_job", "new-job", {"status": "queued"}, session_id="kept"
        )
    assert (root / "commits.jsonl").read_bytes().startswith(prefix)
    monkeypatch.setattr(
        "runtime.persistence.READABLE_FORMAT_VERSIONS", frozenset({3, 4})
    )
    with pytest.raises(SourceCorruptionError, match="unsupported file storage format"):
        RuntimeStore(root).ensure_space()
