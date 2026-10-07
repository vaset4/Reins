"""【存储】【批次提交】记录文件范围、耐久提交和未提交尾部恢复。

作者：xxx
时间：2026-09-30 15:00:00
"""

from __future__ import annotations

import json
import hashlib
import os
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from runtime.file_content import CONTENT_CHUNK_BYTES, confined_path
from runtime.file_records import (
    PreparedContent,
    RecordLocation,
    SourceCorruptionError,
    StoredRecord,
    digest_bytes,
    json_bytes,
)

_LOG = logging.getLogger(__name__)


@dataclass
class JournalState:
    """已验证提交前缀的进程内投影，删除后可完整重读。"""

    sequence: int = 0
    commit_hash: str = ""
    commit_size: int = 0
    records: dict[tuple[str, str], StoredRecord] = field(default_factory=dict)
    history: list[StoredRecord] = field(default_factory=list)
    boundaries: dict[str, int] = field(default_factory=dict)
    immutable: dict[str, tuple[int, str]] = field(default_factory=dict)
    signature: tuple[int, int] | None = None
    commit_lines: list[bytes] = field(default_factory=list)
    file_versions: dict[str, tuple[int, int]] = field(default_factory=dict)


def read_range(root: Path, location: RecordLocation) -> bytes:
    """核对已发布事件范围；参数：空间及位置；返回：完整原件行。"""
    try:
        with confined_path(root, location.path).open("rb") as handle:
            handle.seek(location.offset)
            data = handle.read(location.length)
    except FileNotFoundError as exc:
        raise SourceCorruptionError(
            f"committed source missing: {location.path}"
        ) from exc
    if len(data) != location.length or digest_bytes(data) != location.sha256:
        raise SourceCorruptionError(
            f"committed source corrupt: {location.path}@{location.offset}"
        )
    return data


def file_digest(path: Path) -> tuple[int, str]:
    """常量内存校验不可变文件；参数：原件路径；返回：长度与摘要。"""
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(CONTENT_CHUNK_BYTES):
                digest.update(chunk)
                size += len(chunk)
    except FileNotFoundError as exc:
        raise SourceCorruptionError(f"committed source missing: {path}") from exc
    return size, digest.hexdigest()


def append_synced(path: Path, data: bytes) -> int:
    """追加后同步文件；参数：目标与完整字节；返回：追加起点，异常直接暴露。"""
    from tools.file_persistence import publish_file

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        publish_file(path, None, data)
        return 0
    with path.open("ab") as handle:
        offset = handle.tell()
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        return offset


class FileJournal:
    """以提交日志发布多个领域原件；不执行任何业务恢复动作。"""

    def __init__(self, root: Path) -> None:
        """建立日志定位；参数：根；返回：初始未加载缓存。"""
        self.root, self.path = root, root / "commits.jsonl"
        self.state = JournalState()
        self._needs_recovery = True

    def load(self) -> JournalState:
        """增量核对完整提交；参数：无；返回：源投影，不把残缺末行当提交。"""
        from tools.file_persistence import file_revision

        try:
            signature = file_revision(self.path)
        except FileNotFoundError as exc:
            raise SourceCorruptionError(
                "commits.jsonl is missing from initialized data space"
            ) from exc
        if self.state.signature == signature:
            return self.state
        if signature[0] < self.state.commit_size:
            raise SourceCorruptionError("committed journal was truncated")
        state = replace(
            self.state,
            records=dict(self.state.records),
            history=list(self.state.history),
            boundaries=dict(self.state.boundaries),
            immutable=dict(self.state.immutable),
            commit_lines=list(self.state.commit_lines),
            file_versions=dict(self.state.file_versions),
        )
        with self.path.open("rb") as handle:
            # 1. 【存储】【提交链校验】外部文件增长也必须保留完整旧前缀，不能用尾部增长掩盖中段损坏
            for expected in state.commit_lines:
                if handle.read(len(expected)) != expected:
                    raise SourceCorruptionError(
                        "commit hash or sequence chain is broken: committed prefix changed"
                    )
            while line := handle.readline():
                if not line.endswith(b"\n"):
                    break
                self._apply_commit(state, line)
                state.commit_size = handle.tell()
        state.signature = signature
        self.state = state
        return state

    def _apply_commit(self, state: JournalState, line: bytes) -> None:
        """验证提交链和参与文件；参数：当前前缀及提交行；返回：推进内存投影。"""
        try:
            commit = json.loads(line)
            digest = commit.pop("sha256")
            if (
                digest_bytes(json_bytes(commit)) != digest
                or commit["sequence"] != state.sequence + 1
                or commit["previous"] != state.commit_hash
                or commit["space_id"]
                != json.loads((self.root / "space.json").read_bytes())["space_id"]
            ):
                raise SourceCorruptionError("commit hash or sequence chain is broken")
            for ordinal, item in enumerate(commit["files"]):
                self._apply_file(state, item, commit["sequence"], ordinal)
            state.sequence = commit["sequence"]
            state.commit_hash = digest
            state.commit_lines.append(line)
        except (KeyError, TypeError, json.JSONDecodeError, UnicodeError) as exc:
            raise SourceCorruptionError(
                f"invalid commit at byte {state.commit_size}"
            ) from exc

    def _apply_file(
        self, state: JournalState, item: dict[str, Any], sequence: int, ordinal: int
    ) -> None:
        """验证原件后投影领域修订；参数：前缀、文件范围、批次序号和序位；返回：无。"""
        from tools.file_persistence import file_revision

        location = RecordLocation(
            item["path"],
            item["offset"],
            item["length"],
            item["sha256"],
            sequence,
            ordinal,
        )
        if item["type"] == "immutable":
            if location.offset != 0 or file_digest(
                confined_path(self.root, location.path)
            ) != (location.length, location.sha256):
                raise SourceCorruptionError(
                    f"immutable source corrupt: {location.path}"
                )
            state.immutable[location.path] = (location.length, location.sha256)
            return
        try:
            version = file_revision(confined_path(self.root, location.path))
        except FileNotFoundError as exc:
            raise SourceCorruptionError(
                f"committed source missing: {location.path}"
            ) from exc
        # 1. 【存储】【增量核验】只给已核验的文件更新可信版本，无关提交不能认证别处的旧字节
        if (
            state.boundaries.get(location.path, 0)
            and state.file_versions.get(location.path) != version
        ):
            self._verify_source_prefix(state, location.path)
        data = read_range(self.root, location)
        if item["type"] != "event" or location.offset != state.boundaries.get(
            location.path, 0
        ):
            raise SourceCorruptionError(
                f"source range gap or invalid file type: {location.path}"
            )
        record = StoredRecord.from_mapping(json.loads(data), location)
        key = (record.kind, record.record_id)
        previous = state.records.get(key)
        expected = 1 if previous is None else previous.revision + 1
        if record.revision != expected:
            raise SourceCorruptionError(f"record revision chain is broken: {key}")
        state.records[key] = record
        state.history.append(record)
        state.boundaries[location.path] = location.offset + location.length
        state.file_versions[location.path] = version

    def _recovery_boundaries(
        self, state: JournalState, paths: set[str]
    ) -> dict[str, int]:
        """定位本次写入与中断恢复涉及的日志；参数：有效前缀和本批路径；返回：已提交字节边界。"""
        boundaries = {path: state.boundaries.get(path, 0) for path in paths}
        boundaries["commits.jsonl"] = state.commit_size
        if not self._needs_recovery:
            return boundaries
        boundaries.update(state.boundaries)
        roots = (self.root / "workspaces", self.root / "global", self.root / "skills")
        for base in roots:
            if base.exists():
                for path in base.rglob("events.jsonl"):
                    boundaries.setdefault(path.relative_to(self.root).as_posix(), 0)
        workspace_root = self.root / "workspaces"
        if workspace_root.exists():
            for path in workspace_root.glob("*/workspace.json"):
                boundaries.setdefault(path.relative_to(self.root).as_posix(), 0)
        owners = [
            self.root / "global",
            *(workspace_root.iterdir() if workspace_root.exists() else ()),
        ]
        for owner in owners:
            for path in owner.glob("sessions/*/requests/*/*/*.json"):
                boundaries.setdefault(path.relative_to(self.root).as_posix(), 0)
        return boundaries

    def recover_tail(self, state: JournalState, paths: set[str]) -> None:
        """写入前保留并移除未提交尾部；参数：有效前缀与本批路径；返回：无，诊断保留原字节。"""
        from tools.file_persistence import file_revision

        for relative, end in self._recovery_boundaries(state, paths).items():
            path = confined_path(self.root, relative)
            try:
                version = file_revision(path)
                size = version[0]
            except FileNotFoundError:
                if end:
                    raise SourceCorruptionError(
                        f"committed source missing: {relative}"
                    ) from None
                continue
            if size < end:
                raise SourceCorruptionError(f"committed source truncated: {relative}")
            if (
                relative != "commits.jsonl"
                and end
                and state.file_versions.get(relative) != version
            ):
                self._verify_source_prefix(state, relative)
            if size == end:
                continue
            with path.open("r+b") as handle:
                handle.seek(end)
                tail = handle.read()
                diagnostic = self.root / "runtime" / "recovery" / f"{uuid4().hex}.json"
                append_synced(
                    diagnostic,
                    json_bytes(
                        {"path": relative, "offset": end, "tail_hex": tail.hex()}
                    )
                    + b"\n",
                )
                handle.truncate(end)
                handle.flush()
                os.fsync(handle.fileno())
        state.signature = None
        self._needs_recovery = False

    def _verify_source_prefix(self, state: JournalState, relative: str) -> None:
        """外部改写后重验已有事件范围；参数：已提交前缀和日志路径；返回：无，坏源拒绝继续追加。"""
        for record in state.history:
            if record.location is not None and record.location.path == relative:
                read_range(self.root, record.location)

    def commit(
        self,
        rows: list[tuple[StoredRecord, str]],
        documents: list[Path | PreparedContent],
    ) -> JournalState:
        """先同步全部原件再发布批次；参数：领域记录与不可变材料；返回：已提交源投影。"""
        from tools.file_persistence import file_revision

        state = self.load()
        self.recover_tail(state, {path for _record, path in rows})
        self._needs_recovery = True
        files: list[dict[str, Any]] = []
        for record, relative in rows:
            data = json_bytes(record.to_mapping()) + b"\n"
            offset = append_synced(confined_path(self.root, relative), data)
            files.append(
                {
                    "type": "event",
                    "path": relative,
                    "offset": offset,
                    "length": len(data),
                    "sha256": digest_bytes(data),
                }
            )
        for document in dict.fromkeys(documents):
            path = document.path if isinstance(document, PreparedContent) else document
            if isinstance(document, PreparedContent):
                if document.handle.closed:
                    raise SourceCorruptionError(
                        f"prepared content handle released before commit: {path}"
                    )
                status = path.stat()
                if (
                    file_revision(path) != document.revision
                    or (status.st_dev, status.st_ino) != document.identity
                ):
                    raise SourceCorruptionError(
                        f"prepared content changed before commit: {path}"
                    )
                size, digest = document.size, document.sha256
            else:
                size, digest = file_digest(path)
            relative = path.relative_to(self.root).as_posix()
            if state.immutable.get(relative) == (size, digest):
                continue
            files.append(
                {
                    "type": "immutable",
                    "path": path.relative_to(self.root).as_posix(),
                    "offset": 0,
                    "length": size,
                    "sha256": digest,
                }
            )
        versions = {
            path: file_revision(confined_path(self.root, path))
            for path in {relative for _record, relative in rows}
        }
        commit = {
            "sequence": state.sequence + 1,
            "batch_id": uuid4().hex,
            "space_id": json.loads((self.root / "space.json").read_bytes())["space_id"],
            "previous": state.commit_hash,
            "files": files,
        }
        # 1. 【存储】【确认接纳】只有这一行同步完成才允许业务调用者确认接收
        envelope = {**commit, "sha256": digest_bytes(json_bytes(commit))}
        commit_bytes = json_bytes(envelope) + b"\n"
        offset = append_synced(self.path, commit_bytes)
        # 2. 【存储】【提交回执】耐久屏障后仅更新内存，不让再次读取失败伪装成未接纳
        updated = replace(
            state,
            records=dict(state.records),
            history=list(state.history),
            boundaries=dict(state.boundaries),
            immutable=dict(state.immutable),
            commit_lines=[*state.commit_lines, commit_bytes],
            file_versions={**state.file_versions, **versions},
        )
        for ordinal, ((record, relative), item) in enumerate(zip(rows, files)):
            location = RecordLocation(
                relative,
                item["offset"],
                item["length"],
                item["sha256"],
                commit["sequence"],
                ordinal,
            )
            committed = replace(record, location=location)
            updated.records[(record.kind, record.record_id)] = committed
            updated.history.append(committed)
            updated.boundaries[relative] = item["offset"] + item["length"]
        for item in files[len(rows) :]:
            updated.immutable[item["path"]] = (item["length"], item["sha256"])
        updated.sequence, updated.commit_hash = commit["sequence"], envelope["sha256"]
        updated.commit_size, updated.signature = offset + len(commit_bytes), None
        try:
            updated.signature = file_revision(self.path)
        except OSError as exc:
            _LOG.error(
                "【存储】【提交回执】原件已同步，提交位置缓存读取失败，下次读取重新校验: %s",
                exc,
            )
        self.state = updated
        self._needs_recovery = False
        return updated
