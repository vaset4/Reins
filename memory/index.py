"""可对账、可重建的记忆检索索引，原文不依赖索引存活。

作者：xxx
时间：2026-09-15 05:40:00
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from memory.records import Memory, effective_memory_states
from runtime.persistence import RuntimeStore

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_CJK_PART_RE = re.compile(
    r"[\u3400-\u9fff\U00020000-\U0002ffff]+|[^\u3400-\u9fff\U00020000-\U0002ffff]+"
)
TOKENIZER_VERSION = 2


@dataclass(frozen=True, slots=True)
class MemoryIndexState:
    """分别表达索引可用性与它是否对应当前原文，不以数据库可打开代替一致。"""

    state: str
    missing: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    error: str | None = None
    tokenizer_version: int | None = None
    expected_tokenizer_version: int = TOKENIZER_VERSION


class MemoryIndexUpdateError(RuntimeError):
    """原文已经提交、索引尚未跟上的可定位失败，调用者不能误报为原文未保存。"""

    def __init__(self, memory: Memory, error: Exception) -> None:
        """保留已提交版本和索引原因；传参：已保存记录与异常；返回：无。"""
        self.memory_id = memory.memory_id
        self.version = memory.version
        super().__init__(
            f"memory committed: {memory.memory_id}@{memory.version}; index update failed: {error}; rebuild index explicitly"
        )


def connect_memory_index(data_root: Path | str) -> sqlite3.Connection:
    """打开派生索引并补充可验证的版本列；传参：数据目录；返回：数据库连接。"""
    conn = RuntimeStore(data_root).open_index_connection()
    try:
        ensure_memory_index_schema(conn)
    except sqlite3.Error:
        conn.close()
        raise
    return conn


def ensure_memory_index_schema(
    conn: sqlite3.Connection, *, commit: bool = True
) -> None:
    """只建立派生结构，未对账的旧行保持空版本；传参：连接；返回：无。"""
    new_index = (
        conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'memories'"
        ).fetchone()
        is None
    )
    conn.execute("""CREATE TABLE IF NOT EXISTS memories (
        memory_id TEXT PRIMARY KEY, type TEXT NOT NULL, state TEXT NOT NULL, tags TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL, last_used_at TEXT,
        last_verified_at TEXT, applicable_task_tags TEXT, version TEXT NOT NULL DEFAULT '')""")
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memories)")}
    if "version" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN version TEXT NOT NULL DEFAULT ''")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_state ON memories(state)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(memory_id UNINDEXED, content, tags)"
    )
    # 【记忆】【索引版本】旧库只能通过显式重建升级，打开连接不能冒充完成分词迁移
    if new_index:
        conn.execute(
            "INSERT OR REPLACE INTO runtime_meta(key,value) VALUES(?,?)",
            ("memory_tokenizer_version", str(TOKENIZER_VERSION)),
        )
    if commit:
        conn.commit()


def upsert_memory_index(
    conn: sqlite3.Connection,
    memory: Memory,
    *,
    commit: bool = True,
    effective_state: str | None = None,
) -> None:
    """将一条已提交内容及版本写入同一事务；传参：连接、原文及提交控制；返回：无。"""
    conn.execute(
        """INSERT INTO memories (
        memory_id, type, state, tags, created_at, updated_at, last_used_at,
        last_verified_at, applicable_task_tags, version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(memory_id) DO UPDATE SET type=excluded.type, state=excluded.state,
        tags=excluded.tags, created_at=excluded.created_at, updated_at=excluded.updated_at,
        last_used_at=excluded.last_used_at, last_verified_at=excluded.last_verified_at,
        applicable_task_tags=excluded.applicable_task_tags, version=excluded.version""",
        (
            memory.memory_id,
            memory.type,
            effective_state or memory.state,
            json.dumps(memory.tags, ensure_ascii=False),
            memory.created_at,
            memory.updated_at,
            memory.last_used_at,
            memory.last_verified_at,
            json.dumps(memory.applicable_task_tags, ensure_ascii=False),
            memory.version,
        ),
    )
    conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory.memory_id,))
    conn.execute(
        "INSERT INTO memories_fts(memory_id, content, tags) VALUES (?, ?, ?)",
        (
            memory.memory_id,
            _search_text(memory.content),
            _search_text(" ".join(memory.tags)),
        ),
    )
    if commit:
        conn.commit()


def inspect_index(conn: sqlite3.Connection, records: list[Memory]) -> MemoryIndexState:
    """逐条核对当前版本和实际检索正文；传参：连接与原文快照；返回：一致性状态。"""
    try:
        rows = conn.execute(
            "SELECT memory_id, version, type, state FROM memories"
        ).fetchall()
        texts = conn.execute(
            "SELECT memory_id, content, tags FROM memories_fts"
        ).fetchall()
        tokenizer_row = conn.execute(
            "SELECT value FROM runtime_meta WHERE key=?", ("memory_tokenizer_version",)
        ).fetchone()
        tokenizer_version = int(tokenizer_row[0]) if tokenizer_row else None
    except sqlite3.Error as exc:
        return MemoryIndexState("unavailable", error=str(exc))
    expected = {record.memory_id: record for record in records}
    states = effective_memory_states(records)
    indexed = {row[0]: tuple(row[1:]) for row in rows}
    search = {row[0]: tuple(row[1:]) for row in texts}
    present = set(indexed) & set(search)
    missing = tuple(sorted(set(expected) - present))
    unexpected = tuple(sorted((set(indexed) | set(search)) - set(expected)))
    stale = {
        identity
        for identity in set(expected) & present
        if indexed[identity]
        != (
            expected[identity].version,
            expected[identity].type,
            states[identity]["effective_state"],
        )
        or search[identity]
        != (
            _search_text(expected[identity].content),
            _search_text(" ".join(expected[identity].tags)),
        )
    }
    if len(search) != len(texts):
        stale.update(row[0] for row in texts)
    state = (
        "stale"
        if missing or unexpected or stale or tokenizer_version != TOKENIZER_VERSION
        else "current"
    )
    return MemoryIndexState(
        state,
        missing,
        tuple(sorted(stale)),
        unexpected,
        tokenizer_version=tokenizer_version,
    )


def tokenize_search_text(text: str) -> tuple[str, ...]:
    """把中英文正文切成 FTS 和技能召回共用的检索词。

    参数：text 为待检索正文或查询；返回：保留英文/数字原词、中文单字和相邻双字的词序列。
    中文使用单字与双字同时保留，保证两字查询可以命中旧正文；其他文字沿用连续词行为。
    """
    tokens: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        for part in _CJK_PART_RE.findall(raw):
            if part and _is_cjk(part[0]):
                tokens.extend(part)
                tokens.extend(part[index : index + 2] for index in range(len(part) - 1))
            elif part:
                tokens.append(part)
    return tuple(tokens)


def _is_cjk(character: str) -> bool:
    """判断字符是否属于支持预切词的汉字区间；传参：单个字符；返回：是否为汉字。"""
    codepoint = ord(character)
    return 0x3400 <= codepoint <= 0x9FFF or 0x20000 <= codepoint <= 0x2FFFF


def _search_text(text: str) -> str:
    """生成与查询一致的 FTS 正文；传参：原始文本；返回：空格分隔的词序列。"""
    return " ".join(tokenize_search_text(text))
