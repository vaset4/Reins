"""【存储】【保留升级验收】格式门禁、提交前缀和完整副本验证。

作者：xxx
时间：2026-09-30 22:00:00
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from runtime.file_records import FORMAT_VERSION, SourceCorruptionError, json_bytes
from runtime.file_upgrade import upgrade_file_space, verify_complete_source_backup
from runtime.persistence import RuntimeStore
from runtime.schema_meta import UnsupportedSchemaError, ensure_current_schema
from runtime.session_message_store import SessionMessageStore


def legacy_copy(tmp_path: Path) -> tuple[Path, Path, bytes]:
    """以旧编码事件构造完整v3空间和独立副本；参数：根；返回：源、备份和原提交前缀。"""
    root, backup = tmp_path / "source", tmp_path / "backup"
    messages = SessionMessageStore(root)
    messages.accept_input("kept-session", "kept-history" * 50, input_id="kept-input")
    marker = root / "space.json"
    metadata = json.loads(marker.read_bytes())
    marker.write_bytes(json_bytes({**metadata, "format_version": 3}))
    shutil.copytree(root, backup)
    return root, backup, (root / "commits.jsonl").read_bytes()


def test_v3_read_only_until_explicit_upgrade_preserves_history(tmp_path: Path) -> None:
    """v3可验证但不能新增写入，升级保持身份和原始提交；参数：根；返回：无。"""
    root, backup, prefix = legacy_copy(tmp_path)
    identity = RuntimeStore(root).data_space_id
    assert (
        SessionMessageStore(root).read_entries("kept-session")[0].entry_id
        == "kept-input"
    )
    with pytest.raises(
        UnsupportedSchemaError, match="explicit stopped-runtime upgrade"
    ):
        ensure_current_schema(root)
    with pytest.raises(ValueError, match="explicit stopped-runtime upgrade"):
        with RuntimeStore(root).transaction() as batch:
            batch.put("file_restore_point", "forbidden", {})
    result = upgrade_file_space(root, backup)
    assert result["space_id"] == identity and result["format_version"] == FORMAT_VERSION
    assert (root / "commits.jsonl").read_bytes() == prefix
    assert json.loads((backup / "space.json").read_bytes())["format_version"] == 3
    assert ensure_current_schema(root).status == "current"
    with RuntimeStore(root).transaction() as batch:
        batch.put("file_restore_point", "enabled", {"entries": []})
    assert (root / "commits.jsonl").read_bytes().startswith(prefix)
    assert (
        SessionMessageStore(root).read_entries("kept-session")[0].entry_id
        == "kept-input"
    )
    RuntimeStore(root).rebuild_index(force=True)


def test_incomplete_backup_never_changes_format_gate(tmp_path: Path) -> None:
    """正文缺失的备份拒绝升级且原空间不变；参数：根；返回：无。"""
    root, backup, prefix = legacy_copy(tmp_path)
    next(backup.rglob("*.txt")).unlink()
    with pytest.raises((ValueError, SourceCorruptionError)):
        upgrade_file_space(root, backup)
    assert json.loads((root / "space.json").read_bytes())["format_version"] == 3
    assert (root / "commits.jsonl").read_bytes() == prefix


def test_failure_before_gate_publication_keeps_legacy_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """门禁发布前中断保持完整旧格式；参数：根和IO故障；返回：无。"""
    root, backup, prefix = legacy_copy(tmp_path)

    def fail_publication(path: Path, original: bytes, updated: bytes) -> None:
        """模拟实际门禁发布前IO失败；参数：发布参数；返回：不返回。"""
        raise OSError("gate publication interrupted")

    monkeypatch.setattr("runtime.file_upgrade.publish_file", fail_publication)
    with pytest.raises(OSError, match="interrupted"):
        upgrade_file_space(root, backup)
    assert json.loads((root / "space.json").read_bytes())["format_version"] == 3
    assert (root / "commits.jsonl").read_bytes() == prefix


def test_unarchived_target_volume_staging_blocks_complete_backup_claim(
    tmp_path: Path,
) -> None:
    """仅复制data根不能遗漏已登记的外卷原件；参数：根；返回：无。"""
    root, backup = tmp_path / "data", tmp_path / "copy"
    displaced = tmp_path / "target-volume-original"
    displaced.write_bytes(b"actual displaced original")
    with RuntimeStore(root).transaction() as batch:
        batch.put(
            "file_restore_operation",
            "interrupted",
            {"entries": [{"backup_path": str(displaced)}]},
        )
    shutil.copytree(root, backup)
    with pytest.raises(ValueError, match="retained target-volume staging"):
        verify_complete_source_backup(root, backup)


def test_ciphertext_backup_requires_retained_acl_and_account_protection(
    tmp_path: Path,
) -> None:
    """字节一致但权限丢失的密文副本不能声称可用；参数：根；返回：无。"""
    from tools.restore_protection import (
        ProtectedContentStore,
        apply_security,
        read_security,
    )

    root, backup = tmp_path / "data", tmp_path / "backup"
    reference = ProtectedContentStore(root).freeze(
        b"TOKEN=synthetic-backup-secret", workspace_id=None
    )
    shutil.copytree(root, backup)
    with pytest.raises(PermissionError, match="another account"):
        verify_complete_source_backup(root, backup)
    apply_security(backup / reference.path, read_security(root / reference.path))
    verified = verify_complete_source_backup(root, backup)
    assert verified["space_id"] == RuntimeStore(root).data_space_id
