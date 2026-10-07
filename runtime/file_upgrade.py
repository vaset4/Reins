"""【存储】【保留升级】离线核验完整副本后原子发布格式门禁。

作者：xxx
时间：2026-09-30 22:00:00
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from runtime.file_journal import FileJournal, file_digest
from runtime.file_records import ContentReference, FORMAT_VERSION, json_bytes
from runtime.persistence import RuntimeStore
from tools.file_persistence import file_edit_lock, publish_file

_LEGACY_FORMATS = frozenset({3, 4})
_STAGING_FIELDS = frozenset(
    {"backup_path", "temporary_path", "staging_path", "staging_directory"}
)
_INDEX_FILES = frozenset({"index.sqlite", "index.sqlite-wal", "index.sqlite-shm"})


def retained_staging_paths(database: RuntimeStore) -> list[Path]:
    """定位数据根外尚存的恢复暂存；参数：源空间；返回：完整备份必须另行保留的精确位置。"""
    paths: set[Path] = set()
    with database.snapshot() as source:
        for kind in ("file_restore_point", "file_restore_operation"):
            for record in source.list(kind):
                for path in _staging_paths(record):
                    resolved = path.resolve()
                    if resolved.exists() and not resolved.is_relative_to(
                        database.data_root
                    ):
                        paths.add(resolved)
    return sorted(paths)


def _staging_paths(value: Any) -> Iterator[Path]:
    """只解析领域约定的暂存定位字段；参数：恢复记录；返回：不由任意正文推测的路径。"""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _STAGING_FIELDS and isinstance(item, str):
                yield Path(item)
            elif isinstance(item, (Mapping, list)):
                yield from _staging_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _staging_paths(item)


def _source_manifest(root: Path) -> dict[str, tuple[int, str]]:
    """流式核验全部保留文件；参数：完整空间；返回：相对路径及内容证据，索引可重建文件除外。"""
    manifest = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink() or getattr(path.lstat(), "st_reparse_tag", False):
            raise ValueError(f"backup source contains a reparse point: {relative}")
        if path.is_file() and relative.as_posix() not in _INDEX_FILES:
            manifest[relative.as_posix()] = file_digest(path)
    return manifest


def _verify_protected_originals(
    root: Path, manifest: Mapping[str, tuple[int, str]]
) -> None:
    """验证备份仍能读取账户保护原件；参数：空间及已核验清单；返回：无，丢失私有ACL或解密失败直抛。"""
    from tools.restore_protection import PROTECTED_MEDIA_TYPE, ProtectedContentStore

    protected = ProtectedContentStore(root)
    for relative, (size, digest) in manifest.items():
        path = Path(relative)
        if "protected" in path.parts and path.suffix == ".dpapi":
            protected.read(
                ContentReference(relative, digest, size, PROTECTED_MEDIA_TYPE)
            )


def verify_complete_source_backup(
    data_root: Path | str, backup_root: Path | str
) -> dict[str, Any]:
    """核对停机空间的完整原件副本；参数：源和独立备份；返回：空间、提交与文件数，缺项明确失败。"""
    database, backup = RuntimeStore(data_root), Path(backup_root).resolve()
    if (
        backup == database.data_root
        or backup.is_relative_to(database.data_root)
        or database.data_root.is_relative_to(backup)
    ):
        raise ValueError("backup must be independent from the data root")
    pending = retained_staging_paths(database)
    if pending:
        raise ValueError(
            "complete backup requires reconciliation of retained target-volume staging: "
            + ", ".join(str(path) for path in pending)
        )
    state = FileJournal(database.data_root).load()
    copied = FileJournal(backup).load()
    original, retained = _source_manifest(database.data_root), _source_manifest(backup)
    if original != retained:
        changed = sorted(
            key
            for key in original.keys() | retained.keys()
            if original.get(key) != retained.get(key)
        )
        raise ValueError("backup originals differ: " + ", ".join(changed))
    if (state.sequence, state.commit_hash) != (copied.sequence, copied.commit_hash):
        raise ValueError("backup commit prefix differs")
    _verify_protected_originals(database.data_root, original)
    _verify_protected_originals(backup, retained)
    return {
        "space_id": database.data_space_id,
        "sequence": state.sequence,
        "commit_hash": state.commit_hash,
        "files": len(original),
    }


def upgrade_file_space(
    data_root: Path | str, backup_root: Path | str
) -> dict[str, Any]:
    """在所有旧后台已停止后保留升级；参数：原空间与完整独立备份；返回：同一身份的新门禁证据。"""
    database = RuntimeStore(data_root)
    database.ensure_space()
    marker = database.data_root / "space.json"
    # 1. 【存储】【升级准入】旧程序须由宿主先停止；两个资源锁阻止本次核验窗口的新初始化及提交
    with file_edit_lock(database.data_root / "runtime" / "space.lock", wait=False):
        with file_edit_lock(database.data_root / "runtime" / "commit.lock", wait=False):
            original = marker.read_bytes()
            metadata = json.loads(original)
            if metadata["format_version"] not in _LEGACY_FORMATS:
                raise ValueError("upgrade requires an unchanged v3 or v4 source space")
            verified = verify_complete_source_backup(database.data_root, backup_root)
            # 2. 【存储】【升级发布】唯一变化是格式门禁；事件编码、提交前缀、正文和稳定身份全部保留
            publish_file(
                marker,
                original,
                json_bytes({**metadata, "format_version": FORMAT_VERSION}),
            )
    return {
        **verified,
        "previous_format": metadata["format_version"],
        "format_version": FORMAT_VERSION,
    }
