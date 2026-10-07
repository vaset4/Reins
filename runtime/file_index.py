"""【存储】【派生索引】从已提交文件建立目录、来源定位和全文查询。

作者：xxx
时间：2026-09-30 15:00:00
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from runtime.file_content import ContentFiles
from runtime.file_journal import FileJournal, JournalState
from runtime.file_records import INDEX_VERSION, StoredRecord

INDEX_SCHEMA = """
CREATE TABLE runtime_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE records(kind TEXT NOT NULL,record_id TEXT NOT NULL,workspace_id TEXT,session_id TEXT,
 revision INTEGER NOT NULL,sequence INTEGER NOT NULL,source_path TEXT NOT NULL,
 source_offset INTEGER NOT NULL,source_length INTEGER NOT NULL,payload TEXT NOT NULL,
 PRIMARY KEY(kind,record_id));
CREATE INDEX records_session ON records(kind,session_id,sequence);
CREATE INDEX records_workspace ON records(kind,workspace_id,sequence);
CREATE TABLE search_documents(rowid INTEGER PRIMARY KEY,kind TEXT NOT NULL,record_id TEXT NOT NULL,UNIQUE(kind,record_id));
CREATE VIRTUAL TABLE records_fts USING fts5(kind UNINDEXED,record_id UNINDEXED,body);
"""


def open_index(path: Path) -> sqlite3.Connection:
    """打开短期派生连接；参数：索引文件；返回：调用者关闭的连接。"""
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
    except BaseException:
        connection.close()
        raise


def search_tokens(text: str) -> str:
    """加入中文单双字使中文词可查询；参数：原文；返回：FTS词串。"""
    chars = [char for char in text if "\u3400" <= char <= "\u9fff"]
    return " ".join(
        (text, *chars, *(left + right for left, right in zip(chars, chars[1:])))
    )


def _strings(value: Any) -> list[str]:
    """提取检索用正文字符串；参数：JSON值；返回：扁平文字列表。"""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return []


class FileIndex:
    """索引失败不撤销文件提交，所有派生数据都能从源重建。"""

    def __init__(self, root: Path) -> None:
        """绑定索引路径；参数：空间根；返回：初始状态。"""
        self.root, self.path = root, root / "index.sqlite"
        self.status: dict[str, Any] = {"state": "stale", "error": None}

    def synchronize(
        self, state: JournalState, space_id: str, *, force: bool = False
    ) -> None:
        """增量同步或重建损坏索引；参数：有效源前缀与空间身份；返回：无，失败保留原因。"""
        from tools.file_persistence import file_edit_lock

        with file_edit_lock(self.root / "runtime" / "index.lock", wait=True):
            applied = self._applied(space_id)
            if applied is not None and applied > state.sequence:
                state = FileJournal(self.root).load()
            if force or applied is None or applied > state.sequence:
                self._rebuild(state, space_id)
                return
            if applied == state.sequence:
                self.status = {"state": "ready", "sequence": applied, "error": None}
                return
            self.status = {"state": "building", "sequence": applied, "error": None}
            connection = open_index(self.path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                for record in state.records.values():
                    if (
                        record.location is not None
                        and record.location.sequence > applied
                    ):
                        self._apply(connection, record)
                connection.execute(
                    "UPDATE runtime_meta SET value=? WHERE key='sequence'",
                    (str(state.sequence),),
                )
                connection.execute("COMMIT")
                self.status = {
                    "state": "ready",
                    "sequence": state.sequence,
                    "error": None,
                }
            finally:
                connection.close()

    def _applied(self, space_id: str) -> int | None:
        """检查索引元信息；参数：源空间身份；返回：序号，缺失或损坏要求重建。"""
        if not self.path.is_file():
            return None
        connection: sqlite3.Connection | None = None
        try:
            connection = open_index(self.path)
            metadata = dict(connection.execute("SELECT key,value FROM runtime_meta"))
            if metadata.get("space_id") != space_id or metadata.get(
                "index_version"
            ) != str(INDEX_VERSION):
                return None
            return int(metadata["sequence"])
        except (sqlite3.DatabaseError, KeyError, ValueError):
            return None
        finally:
            if connection is not None:
                connection.close()

    def _rebuild(self, state: JournalState, space_id: str) -> None:
        """完整构建新代次后发布；参数：有效源和空间身份；返回：无，不展示半张空表。"""
        self.status = {"state": "building", "sequence": state.sequence, "error": None}
        verified = FileJournal(self.root).load()
        if verified.sequence < state.sequence:
            raise ValueError("source commit prefix disappeared during index rebuild")
        state = verified
        temporary = self.root / f".index-{uuid4().hex}.sqlite"
        connection = open_index(temporary)
        try:
            connection.executescript(INDEX_SCHEMA)
            connection.execute("BEGIN IMMEDIATE")
            for record in state.records.values():
                self._apply(connection, record)
            connection.executemany(
                "INSERT INTO runtime_meta(key,value) VALUES (?,?)",
                (
                    ("space_id", space_id),
                    ("sequence", str(state.sequence)),
                    ("index_version", str(INDEX_VERSION)),
                ),
            )
            connection.execute(f"PRAGMA user_version={INDEX_VERSION}")
            connection.execute("COMMIT")
        except BaseException:
            connection.close()
            temporary.unlink(missing_ok=True)
            raise
        connection.close()
        temporary.replace(self.path)
        self.status = {"state": "ready", "sequence": state.sequence, "error": None}

    def _apply(self, connection: sqlite3.Connection, record: StoredRecord) -> None:
        """应用领域事件最新版本；参数：派生连接和原件；返回：无，不修改源文件。"""
        key = (record.kind, record.record_id)
        connection.execute("DELETE FROM records WHERE kind=? AND record_id=?", key)
        searchable_kind = record.kind in {"session_entry", "tool_operation", "task"}
        if searchable_kind:
            old = connection.execute(
                "SELECT rowid FROM search_documents WHERE kind=? AND record_id=?", key
            ).fetchone()
            if old is not None:
                connection.execute("DELETE FROM records_fts WHERE rowid=?", (old[0],))
                connection.execute(
                    "DELETE FROM search_documents WHERE rowid=?", (old[0],)
                )
        if record.deleted:
            return
        location = record.location
        if location is None:
            raise ValueError("index cannot publish an uncommitted record")
        connection.execute(
            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                *key,
                record.workspace_id,
                record.session_id,
                record.revision,
                location.sequence,
                location.path,
                location.offset,
                location.length,
                json.dumps(record.payload, ensure_ascii=False),
            ),
        )
        if not searchable_kind:
            return
        payload = ContentFiles(self.root).unpack(record.payload, record.references)
        searchable = (
            payload.get("message", {}) if record.kind == "session_entry" else payload
        )
        if record.kind == "tool_operation":
            searchable = payload.get("result", {})
        body = search_tokens("\n".join(_strings(searchable)))
        cursor = connection.execute(
            "INSERT INTO search_documents(kind,record_id) VALUES (?,?)", key
        )
        connection.execute(
            "INSERT INTO records_fts(rowid,kind,record_id,body) VALUES (?,?,?,?)",
            (cursor.lastrowid, *key, body),
        )
