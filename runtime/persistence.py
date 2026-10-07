"""【Reins】【文件持久化】领域批次、源快照与可重建索引。

作者：xxx
时间：2026-09-30 15:00:00
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast
from uuid import uuid4
from urllib.parse import quote

from runtime.cancellation import CancellationToken
from runtime.file_content import CONTENT_CHUNK_BYTES, ContentFiles, confined_path
from runtime.file_index import FileIndex, open_index
from runtime.file_journal import FileJournal, JournalState, append_synced, read_range
from runtime.file_records import (
    FORMAT_VERSION,
    READABLE_FORMAT_VERSIONS,
    RECORD_KINDS,
    SCHEDULE_KINDS,
    ContentReference,
    RecordConflictError,
    SourceCorruptionError,
    StoredRecord,
    PreparedContent,
    PreparedPayload,
    json_bytes,
    record_key,
)

SPACE_ID_FILE = "space.json"
INDEX_NAME = "index.sqlite"
_LOG = logging.getLogger(__name__)
_ACTIVE: ContextVar[tuple[tuple[str, int, SourceSnapshot], ...]] = ContextVar(
    "reins_file_source", default=()
)
_SHARED: dict[str, _SpaceServices] = {}
_SHARED_LOCK = threading.RLock()
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass
class _SpaceServices:
    """共享非权威缓存，删除后可从文件重建。"""

    journal: FileJournal
    index: FileIndex
    lock: threading.RLock
    identity: str | None = None


class SourceSnapshot:
    """固定提交位置的读取视图，不执行业务恢复动作。"""

    def __init__(self, store: RuntimeStore, state: JournalState) -> None:
        """固定源前缀；参数：存储与状态；返回：独立映射。"""
        self.store = store
        self.sequence = state.sequence
        self._records = state.records
        self._source_versions = state.file_versions
        self._verified_records: dict[tuple[str, int, str], tuple[int, int]] = {}

    def raw(self, kind: str, record_id: str) -> StoredRecord | None:
        """读取未展开正文的原件；参数：领域及身份；返回：校验后记录或不存在。"""
        _validate_kind(kind)
        record = self._records.get((kind, record_id))
        if record is None:
            return None
        if record.location is not None:
            self._verify_record(record, self._source_version(record.location.path))
        if record.deleted:
            return None
        return copy.deepcopy(record)

    def _source_version(self, path: str) -> tuple[int, int]:
        """读取原件真实变更代次，缺失必须显式失败；参数：空间内路径；返回：长度与Windows变更时间。"""
        from tools.file_persistence import file_revision

        try:
            return file_revision(confined_path(self.store.data_root, path))
        except FileNotFoundError as exc:
            raise SourceCorruptionError(f"committed source missing: {path}") from exc

    def _verify_record(self, record: StoredRecord, version: tuple[int, int]) -> None:
        """原件代次变化时核对选定已提交范围；参数：原件记录与实际代次；返回：无，损坏直接失败。"""
        location = record.location
        assert location is not None
        key = (location.path, location.offset, location.sha256)
        if (
            self._source_versions.get(location.path) != version
            and self._verified_records.get(key) != version
        ):
            read_range(self.store.data_root, location)
            self._verified_records[key] = version

    def get(self, kind: str, record_id: str) -> dict[str, Any] | None:
        """展开目标原件；参数：领域及身份；返回：独立JSON，损坏明确失败。"""
        record = self.raw(kind, record_id)
        return (
            None
            if record is None
            else self.store._content.unpack(record.payload, record.references)
        )

    def list_raw(
        self,
        kind: str,
        *,
        session_id: str | None = None,
        workspace_id: str | None = None,
        filters: Mapping[str, object] | None = None,
    ) -> tuple[StoredRecord, ...]:
        """读取轻量原件元信息；参数：类型及归属；返回：未展开正文的记录。"""
        _validate_kind(kind)
        records = []
        for (stored_kind, _), record in self._records.items():
            if (
                stored_kind != kind
                or (session_id is not None and record.session_id != session_id)
                or (workspace_id is not None and record.workspace_id != workspace_id)
            ):
                continue
            if filters is not None and any(
                record.payload.get(key) != value for key, value in filters.items()
            ):
                continue
            records.append(record)
        return self._read_records(records)

    def _read_records(self, records: list[StoredRecord]) -> tuple[StoredRecord, ...]:
        """批量核验选中原件并返回独立副本，允许合法并发追加；参数：冻结的记录集合；返回：未删除的原件。"""
        # 1. 【文件原件】【批量核验】同一物理原件只取一次代次，不为每一条记录反复打开相同文件
        versions: dict[str, tuple[int, int]] = {}
        for record in records:
            if record.location is None:
                continue
            path = record.location.path
            if path not in versions:
                versions[path] = self._source_version(path)
            self._verify_record(record, versions[path])
        result = tuple(
            copy.deepcopy(record) for record in records if not record.deleted
        )
        # 2. 【文件原件】【并发核验】读取期间变更须重验所选范围；合法尾部追加保留旧快照，改坏原件则报错
        changed = {}
        for path, previous in versions.items():
            current = self._source_version(path)
            if current != previous:
                changed[path] = current
        for record in records:
            location = record.location
            if location is not None and location.path in changed:
                read_range(self.store.data_root, location)
                self._verified_records[
                    (location.path, location.offset, location.sha256)
                ] = changed[location.path]
        return result

    def list(
        self,
        kind: str,
        *,
        session_id: str | None = None,
        workspace_id: str | None = None,
        filters: Mapping[str, object] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """读取领域记录；参数：类型及可选归属；返回：首次创建顺序的完整记录。"""
        records = self.list_raw(
            kind, session_id=session_id, workspace_id=workspace_id, filters=filters
        )
        return tuple(
            self.store._content.unpack(record.payload, record.references)
            for record in records
        )

    def revision(self, kind: str, record_id: str) -> int:
        """取得包含删除墓碑的修订；参数：领域及身份；返回：版本，不存在为零。"""
        _validate_kind(kind)
        record = self._records.get((kind, record_id))
        return 0 if record is None else record.revision


class FileTransaction(SourceSnapshot):
    """共享短写入边界的领域事件批次。"""

    def __init__(self, store: RuntimeStore, state: JournalState) -> None:
        """建立批次；参数：存储及状态；返回：空变更队列。"""
        super().__init__(store, state)
        self._records = dict(state.records)
        self._pending: list[tuple[StoredRecord, str]] = []
        self._documents: list[Path | PreparedContent] = []

    def put(
        self,
        kind: str,
        record_id: str,
        payload: Mapping[str, Any] | PreparedPayload,
        *,
        session_id: str | None = None,
        workspace_id: str | None = None,
        expected_revision: int | None = None,
    ) -> int:
        """登记领域修订；参数：类型、身份、载荷、归属和期望版；返回：新版本。"""
        _validate_kind(kind)
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record identity must be non-empty text")
        current = self._records.get((kind, record_id))
        revision = self.revision(kind, record_id)
        if expected_revision is not None and revision != expected_revision:
            raise RecordConflictError(f"record revision changed: {kind}/{record_id}")
        session_id, workspace_id = self._record_owner(current, session_id, workspace_id)
        directory = self.store._owner_directory(workspace_id)
        if isinstance(payload, PreparedPayload):
            if payload.owner != directory.relative_to(self.store.data_root).as_posix():
                raise RecordConflictError(
                    "prepared content belongs to another workspace"
                )
            packed, references = (
                copy.deepcopy(payload.payload),
                copy.deepcopy(payload.references),
            )
        else:
            packed, references = (
                (dict(payload), ())
                if kind == "workspace"
                else self.store._content.pack(payload, directory)
            )
        record = StoredRecord(
            kind, record_id, revision + 1, packed, references, workspace_id, session_id
        )
        relative = self.store._record_path(record)
        json_bytes(record.to_mapping())
        self._records[(kind, record_id)] = record
        self._pending.append((record, relative))
        self._documents.extend(
            confined_path(self.store.data_root, item["content"]["path"])
            for item in references
        )
        return record.revision

    def _record_owner(
        self,
        current: StoredRecord | None,
        session_id: str | None,
        workspace_id: str | None,
    ) -> tuple[str | None, str | None]:
        """解析已有身份或会话的原归属；参数：旧记录和明确归属；返回：会话、工作区身份。"""
        if current is not None:
            session_id = session_id if session_id is not None else current.session_id
            workspace_id = (
                workspace_id if workspace_id is not None else current.workspace_id
            )
        if workspace_id is None and session_id is not None:
            binding = self.get("session_workspace", session_id)
            workspace_id = None if binding is None else binding["workspace_id"]
        return session_id, workspace_id

    def delete(
        self, kind: str, record_id: str, *, expected_revision: int | None = None
    ) -> None:
        """登记删除墓碑；参数：身份及期望版；返回：无，保留旧原件。"""
        record = self.raw(kind, record_id)
        if record is None:
            return
        if expected_revision is not None and record.revision != expected_revision:
            raise RecordConflictError(f"record revision changed: {kind}/{record_id}")
        deleted = StoredRecord(
            kind,
            record_id,
            record.revision + 1,
            {},
            (),
            record.workspace_id,
            record.session_id,
            True,
        )
        self._records[(kind, record_id)] = deleted
        relative = (
            record.location.path if record.location else self.store._record_path(record)
        )
        self._pending.append((deleted, relative))

    def publish_document(self, relative_path: str, content: bytes) -> Path:
        """发布不可变请求或修订材料；参数：空间相对路径及字节；返回：原件路径。"""
        from tools.file_persistence import publish_file

        path = confined_path(self.store.data_root, relative_path)
        if not Path(relative_path).parts or Path(relative_path).parts[0] not in {
            "workspaces",
            "global",
            "skills",
        }:
            raise ValueError(
                "immutable document must belong to workspace, global or skills"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != content:
                raise RecordConflictError(
                    f"immutable document already has different content: {relative_path}"
                )
        else:
            publish_file(path, None, content)
        self._documents.append(path)
        return path

    def reference_content(self, reference: ContentReference) -> None:
        """把冻结内容纳入提交证据；参数：明确引用；返回：无，缺失或坏内容拒绝提交。"""
        with self.store._content._open_verified(reference):
            pass
        self._documents.append(confined_path(self.store.data_root, reference.path))

    def reference_prepared_content(self, prepared: PreparedContent) -> None:
        """登记锁外已固定原件；参数：仍持有只读句柄的证据；返回：无，已释放的证据禁止复用。"""
        if not prepared.path.is_relative_to(self.store.data_root):
            raise ValueError("prepared original belongs to another data space")
        if prepared.handle.closed:
            raise SourceCorruptionError(
                "prepared original handle was released before commit"
            )
        self._documents.append(prepared)


class RuntimeStore:
    """文件拥有事实，SQLite只持有可删除重建的投影。"""

    def __init__(self, data_root: Path | str) -> None:
        """绑定空间；参数：数据根；返回：无长期连接的门面。"""
        self.data_root = Path(data_root).resolve()
        self.path = self.data_root / INDEX_NAME
        self.commit_path = self.data_root / "commits.jsonl"
        self._key = str(self.data_root).casefold()
        self._content = ContentFiles(self.data_root)
        with _SHARED_LOCK:
            if self._key not in _SHARED:
                _SHARED[self._key] = _SpaceServices(
                    FileJournal(self.data_root),
                    FileIndex(self.data_root),
                    threading.RLock(),
                )
            self._shared = _SHARED[self._key]

    def _active(self) -> SourceSnapshot | None:
        """取得本线程同根范围；参数：无；返回：当前快照或事务。"""
        for key, thread, source in reversed(_ACTIVE.get()):
            if key == self._key and thread == threading.get_ident():
                return source
        return None

    def ensure_space(self) -> str:
        """初始化或核对空间；参数：无；返回：稳定空间编号。"""
        from tools.file_persistence import file_edit_lock

        marker = self.data_root / SPACE_ID_FILE
        if self._shared.identity is not None:
            return self._read_space_identity(marker)
        if self.data_root.exists() and not self.data_root.is_dir():
            raise ValueError("runtime data root is not a directory")
        self.data_root.mkdir(parents=True, exist_ok=True)
        with (
            self._shared.lock,
            file_edit_lock(self.data_root / "runtime" / "space.lock", wait=True),
        ):
            if not marker.exists():
                self._initialize_space(marker)
            identity = self._read_space_identity(marker)
            if not self.commit_path.is_file():
                raise SourceCorruptionError(
                    "commits.jsonl is missing from initialized data space"
                )
            self._shared.identity = identity
            return identity

    def _initialize_space(self, marker: Path) -> None:
        """仅在空白资料根发布新身份；参数：身份文件；返回：无，旧原件不自动迁移或删除。"""
        from tools.file_persistence import publish_file

        for child in self.data_root.iterdir():
            if child.name != "runtime" and (
                child.is_file() or any(p.is_file() for p in child.rglob("*"))
            ):
                raise ValueError(
                    "file data space identity missing: old or incomplete runtime data"
                )
        append_synced(self.commit_path, b"")
        publish_file(
            marker,
            None,
            json_bytes({"format_version": FORMAT_VERSION, "space_id": uuid4().hex}),
        )

    def _read_space_identity(self, marker: Path) -> str:
        """读取小型身份原件并核对现存空间；参数：身份路径；返回：完整空间编号。"""
        try:
            metadata = json.loads(marker.read_bytes())
            if (
                type(metadata["format_version"]) is not int
                or metadata["format_version"] not in READABLE_FORMAT_VERSIONS
            ):
                raise SourceCorruptionError("unsupported file storage format")
            identity = metadata["space_id"]
            if not isinstance(identity, str) or not re.fullmatch(
                r"[0-9a-f]{32}", identity
            ):
                raise SourceCorruptionError("invalid data space identity")
        except FileNotFoundError as exc:
            raise SourceCorruptionError("data space identity is missing") from exc
        except (KeyError, TypeError, json.JSONDecodeError, UnicodeError) as exc:
            raise SourceCorruptionError("invalid space.json identity") from exc
        if self._shared.identity is not None and self._shared.identity != identity:
            raise SourceCorruptionError(
                "data space identity changed while the store is open"
            )
        return identity

    @property
    def data_space_id(self) -> str:
        """读取跨重启身份；参数：无；返回：空间编号。"""
        return self.ensure_space()

    def require_current_format(self) -> None:
        """禁止旧空间接收新格式写入；参数：无；返回：无，旧空间须停止运行并验证备份后显式升级。"""
        self.ensure_space()
        version = json.loads((self.data_root / SPACE_ID_FILE).read_bytes())[
            "format_version"
        ]
        if version != FORMAT_VERSION:
            raise ValueError(
                f"file storage format v{version} requires an explicit stopped-runtime upgrade to v{FORMAT_VERSION}"
            )

    @property
    def index_status(self) -> dict[str, Any]:
        """读取索引状态；参数：无；返回：状态与真实故障。"""
        return dict(self._shared.index.status)

    @contextmanager
    def snapshot(self, sequence: int | None = None) -> Iterator[SourceSnapshot]:
        """捕获源前缀后释放提交锁；参数：无；返回：一致读取视图。"""
        from tools.file_persistence import file_edit_lock

        active = self._active()
        if active is not None and (sequence is None or sequence == active.sequence):
            yield active
            return
        self.ensure_space()
        with (
            self._shared.lock,
            file_edit_lock(self.data_root / "runtime" / "commit.lock", wait=True),
        ):
            state = self._shared.journal.load()
            source = SourceSnapshot(self, state)
            if sequence is not None:
                if (
                    type(sequence) is not int
                    or sequence < 0
                    or sequence > state.sequence
                ):
                    raise ValueError("snapshot sequence is outside committed history")
                source.sequence = sequence
                source._records = {
                    (row.kind, row.record_id): row
                    for row in state.history
                    if row.location is not None and row.location.sequence <= sequence
                }
        token = _ACTIVE.set(
            (*_ACTIVE.get(), (self._key, threading.get_ident(), source))
        )
        try:
            yield source
        finally:
            _ACTIVE.reset(token)

    @contextmanager
    def transaction(self) -> Iterator[FileTransaction]:
        """发布跨Store复合变更；参数：无；返回：可嵌套批次，异常不发布。"""
        from tools.file_persistence import file_edit_lock

        active = self._active()
        if isinstance(active, FileTransaction):
            saved = (
                dict(active._records),
                len(active._pending),
                len(active._documents),
            )
            try:
                yield active
            except BaseException:
                active._records = saved[0]
                del active._pending[saved[1] :]
                del active._documents[saved[2] :]
                raise
            return
        if active is not None:
            raise RuntimeError("cannot write within a read-only source snapshot")
        identity = self.ensure_space()
        state: JournalState | None = None
        with (
            self._shared.lock,
            file_edit_lock(self.data_root / "runtime" / "commit.lock", wait=True),
        ):
            self.require_current_format()
            batch = FileTransaction(self, self._shared.journal.load())
            token = _ACTIVE.set(
                (*_ACTIVE.get(), (self._key, threading.get_ident(), batch))
            )
            try:
                yield batch
                if batch._pending or batch._documents:
                    state = self._shared.journal.commit(
                        batch._pending, batch._documents
                    )
            finally:
                _ACTIVE.reset(token)
        if state is not None:
            if self.path.is_file():
                previous_status = self._shared.index.status
                self._shared.index.status = (
                    {**previous_status, "source_sequence": state.sequence}
                    if previous_status["state"] == "failed"
                    else {
                        "state": "stale",
                        "source_sequence": state.sequence,
                        "error": None,
                    }
                )
                return
            try:
                self._shared.index.synchronize(state, identity)
            except (OSError, ValueError, sqlite3.Error) as exc:
                self._shared.index.status = {
                    "state": "failed",
                    "sequence": state.sequence,
                    "error": str(exc),
                }
                _LOG.error("【存储】【索引更新】原件批次已提交，派生索引失败: %s", exc)

    @contextmanager
    def connection_scope(self) -> Iterator[None]:
        """保持执行生命周期入口，不跨模型等待持锁；参数：无；返回：无。"""
        self.ensure_space()
        yield None

    def open_index_connection(self) -> sqlite3.Connection:
        """准备索引并打开自有连接；参数：无；返回：调用者关闭的连接。"""
        self.rebuild_index()
        return open_index(self.path)

    @contextmanager
    def index_connection(self) -> Iterator[sqlite3.Connection]:
        """获取派生连接；参数：无；返回：退出关闭的短连接。"""
        connection = self.open_index_connection()
        try:
            yield connection
        finally:
            connection.close()

    def initialize_index(self, sql: str) -> None:
        """建立领域派生表；参数：静态DDL；返回：无，领域负责从源填充。"""
        with self.index_connection() as connection:
            connection.executescript(sql)

    def rebuild_index(self, *, force: bool = False) -> None:
        """同步或重建索引；参数：无；返回：无，失败明确传播。"""
        from tools.file_persistence import file_edit_lock

        identity = self.ensure_space()
        with (
            self._shared.lock,
            file_edit_lock(self.data_root / "runtime" / "commit.lock", wait=True),
        ):
            state = self._shared.journal.load()
        try:
            self._shared.index.synchronize(state, identity, force=force)
        except (OSError, ValueError, sqlite3.Error) as exc:
            self._shared.index.status = {
                "state": "failed",
                "sequence": state.sequence,
                "error": str(exc),
            }
            raise

    def workspace_directory(self, workspace_id: str) -> Path:
        """定位工作区原件目录；参数：稳定身份；返回：可识别路径。"""
        with self.snapshot() as source:
            workspace = source.get("workspace", workspace_id)
            if workspace is None:
                raise ValueError(f"workspace identity not found: {workspace_id}")
            return confined_path(self.data_root, "workspaces/" + workspace["directory"])

    def session_directory(self, session_id: str) -> Path:
        """按原归属定位会话；参数：会话编号；返回：源目录。"""
        _safe_identity(session_id)
        with self.snapshot() as source:
            binding = source.get("session_workspace", session_id)
            owner = self._owner_directory(
                None if binding is None else binding["workspace_id"]
            )
            return owner / "sessions" / session_id

    def _owner_directory(self, workspace_id: str | None) -> Path:
        """选择所属目录；参数：可选工作区；返回：全局或工作区位置。"""
        return (
            self.data_root / "global"
            if workspace_id is None
            else self.workspace_directory(workspace_id)
        )

    def _record_path(self, record: StoredRecord) -> str:
        """按固定领域路由原件；参数：领域记录；返回：相对日志路径。"""
        owner = self._owner_directory(record.workspace_id)
        path: Path
        if record.kind == "workspace":
            path = (
                self.data_root
                / "workspaces"
                / record.payload["directory"]
                / "workspace.json"
            )
        elif record.kind in SCHEDULE_KINDS:
            path = owner / "schedules" / "events.jsonl"
        elif record.kind.startswith("skill_"):
            skill_id = record.payload.get("skill_id")
            _safe_identity(skill_id)
            path = self.data_root / "skills" / cast(str, skill_id) / "events.jsonl"
        elif record.kind == "run_evidence" and record.payload.get("kind") in {
            "attempt_request",
            "attempt_response",
        }:
            if record.session_id is None:
                raise ValueError("model attempt original requires session ownership")
            _safe_identity(record.session_id)
            attempt = record.payload["attempt"]
            directory = owner / "sessions" / record.session_id / "requests"
            directory = (
                directory
                / _request_component(attempt["request_id"])
                / _request_component(attempt["attempt_id"])
            )
            path = directory / (
                "input.json"
                if record.payload["kind"] == "attempt_request"
                else "output.json"
            )
        elif record.session_id is not None:
            _safe_identity(record.session_id)
            path = owner / "sessions" / record.session_id / "events.jsonl"
        elif record.kind == "artifact_records" and record.workspace_id is not None:
            path = owner / "events.jsonl"
        else:
            path = self.data_root / "global" / "events.jsonl"
        return path.relative_to(self.data_root).as_posix()

    def source_path(self, kind: str, record_id: str) -> Path:
        """定位领域原件；参数：领域及身份；返回：日志路径，未知身份明确失败。"""
        with self.snapshot() as source:
            record = source.raw(kind, record_id)
            if record is None:
                raise ValueError(f"record not found: {kind}/{record_id}")
            relative = (
                self._record_path(record)
                if record.location is None
                else record.location.path
            )
            return confined_path(self.data_root, relative)

    def prepare_content(
        self,
        content: bytes,
        *,
        workspace_id: str | None = None,
        session_id: str | None = None,
        media_type: str = "application/octet-stream",
    ) -> ContentReference:
        """锁外准备冻结字节；参数：正文、归属、类型；返回：明确内容引用。"""
        self.require_current_format()
        if workspace_id is None and session_id is not None:
            with self.snapshot() as source:
                binding = source.get("session_workspace", session_id)
                workspace_id = None if binding is None else binding["workspace_id"]
        return self._content.prepare(
            content, self._owner_directory(workspace_id), media_type
        )

    def prepare_payload(
        self,
        payload: Mapping[str, Any],
        *,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> PreparedPayload:
        """在事务外冻结大正文；参数：载荷和所属范围；返回：可由批次直接发布的内容候选。"""
        self.require_current_format()
        if workspace_id is None and session_id is not None:
            with self.snapshot() as source:
                binding = source.get("session_workspace", session_id)
                workspace_id = None if binding is None else binding["workspace_id"]
        directory = self._owner_directory(workspace_id)
        packed, references = self._content.pack(payload, directory)
        return PreparedPayload(
            packed, references, directory.relative_to(self.data_root).as_posix()
        )

    def read_content(
        self, reference: ContentReference, *, offset: int = 0, limit: int | None = None
    ) -> bytes:
        """校验并读取冻结正文；参数：引用与字节分页；返回：原件字节。"""
        return self._content.read(reference, offset=offset, limit=limit)

    def prepare_file(
        self,
        path: Path | str,
        *,
        workspace_id: str | None = None,
        session_id: str | None = None,
        media_type: str = "application/octet-stream",
    ) -> ContentReference:
        """锁外流式冻结文件；参数：路径、归属、类型；返回：内容引用。"""
        self.require_current_format()
        if workspace_id is None and session_id is not None:
            with self.snapshot() as source:
                binding = source.get("session_workspace", session_id)
                workspace_id = None if binding is None else binding["workspace_id"]
        return self._content.prepare_path(
            Path(path), self._owner_directory(workspace_id), media_type
        )

    def prepare_stream(
        self,
        source: BinaryIO,
        *,
        workspace_id: str,
        cancellation: CancellationToken | None = None,
    ) -> ContentReference:
        """锁外冻结已固定句柄；参数：完整文件流、工作区、取消信号；返回：原件引用。"""
        self.require_current_format()
        return self._content.prepare_stream(
            source,
            self._owner_directory(workspace_id),
            "application/octet-stream",
            cancellation=cancellation,
        )

    @contextmanager
    def prepare_reference(
        self,
        reference: ContentReference,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[PreparedContent]:
        """锁外核验并固定原件直到提交结束；参数：引用、取消信号；返回：只在上下文内有效的字节证据。"""
        from tools.file_persistence import file_change_time

        with self._content._open_verified(
            reference, cancellation=cancellation
        ) as source:
            revision = (reference.size, file_change_time(source.fileno()))
            status = os.fstat(source.fileno())
            yield PreparedContent(
                confined_path(self.data_root, reference.path),
                reference.size,
                reference.sha256,
                revision,
                (status.st_dev, status.st_ino),
                source,
            )

    @contextmanager
    def prepare_references(
        self,
        references: Iterable[ContentReference],
        *,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[tuple[PreparedContent, ...]]:
        """按内容路径去重并短期固定一批原件；参数：引用与取消信号；返回：提交外校验且退出全释放的证据。"""
        unique: dict[str, ContentReference] = {}
        for reference in references:
            previous = unique.get(reference.path)
            if previous is not None and previous != reference:
                raise SourceCorruptionError(
                    "same original path has inconsistent references"
                )
            unique[reference.path] = reference
        with ExitStack() as access:
            originals = tuple(
                access.enter_context(
                    self.prepare_reference(reference, cancellation=cancellation)
                )
                for reference in unique.values()
            )
            yield originals

    def iter_content(
        self,
        reference: ContentReference,
        *,
        chunk_size: int = CONTENT_CHUNK_BYTES,
        cancellation: CancellationToken | None = None,
    ) -> Iterator[bytes]:
        """流式读取冻结内容；参数：引用、字节块大小、取消信号；返回：常量内存迭代器。"""
        return self._content.iterate(
            reference, chunk_size=chunk_size, cancellation=cancellation
        )


def _validate_kind(kind: str) -> None:
    """拒绝未定义领域；参数：类型名；返回：无。"""
    if kind not in RECORD_KINDS:
        raise ValueError(f"unknown runtime record kind: {kind}")


def _safe_identity(identity: Any) -> None:
    """检查路径身份；参数：编号；返回：无，路径穿越明确失败。"""
    if (
        not isinstance(identity, str)
        or not _SAFE_ID.fullmatch(identity)
        or identity in {".", ".."}
    ):
        raise ValueError(f"invalid source path identity: {identity!r}")


def _request_component(identity: Any) -> str:
    """编码请求目录身份；参数：协议编号；返回：不能穿越目录的可定位片段。"""
    if not isinstance(identity, str) or not identity or identity in {".", ".."}:
        raise ValueError("invalid request source identity")
    encoded = quote(identity, safe="-_.")
    if encoded.endswith("."):
        encoded = encoded[:-1] + "%2E"
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if encoded.split(".", 1)[0].upper() in reserved:
        encoded = f"%{ord(encoded[0]):02X}" + encoded[1:]
    return encoded


__all__ = [
    "RuntimeStore",
    "SourceSnapshot",
    "FileTransaction",
    "ContentReference",
    "PreparedPayload",
    "record_key",
    "SourceCorruptionError",
    "RecordConflictError",
    "FORMAT_VERSION",
    "SPACE_ID_FILE",
    "INDEX_NAME",
]
