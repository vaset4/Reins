"""以Markdown保存记忆正文与修订，SQLite仅保留可重建索引。

作者：xxx
时间：2026-09-15 05:40:00
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from memory.index import (
    TOKENIZER_VERSION,
    MemoryIndexState,
    MemoryIndexUpdateError,
    connect_memory_index,
    ensure_memory_index_schema,
    inspect_index,
    upsert_memory_index,
)
from memory.records import (
    MEMORY_FORMAT,
    MEMORY_STATE_ACTIVE,
    MEMORY_STATE_ARCHIVED,
    MEMORY_STATE_DRAFT,
    MEMORY_STATE_WITHDRAWN,
    MEMORY_STATES,
    MEMORY_TYPES,
    Memory,
    MemoryDetails,
    MemorySource,
    effective_memory_states,
    memory_view,
    seal_memory,
    validate_details,
    validate_sources,
    validate_time,
)
from memory.files import MemoryFiles
from runtime.persistence import RuntimeStore
from tools.file_persistence import file_edit_lock
from tasks.ids import new_ulid, utc_now

STATE_TRANSITIONS: dict[str, set[str]] = {
    MEMORY_STATE_DRAFT: {MEMORY_STATE_ACTIVE, MEMORY_STATE_WITHDRAWN},
    MEMORY_STATE_ACTIVE: {MEMORY_STATE_ARCHIVED, MEMORY_STATE_WITHDRAWN},
    MEMORY_STATE_ARCHIVED: {MEMORY_STATE_ACTIVE, MEMORY_STATE_WITHDRAWN},
    MEMORY_STATE_WITHDRAWN: set(),
}


class MemoryStore:
    """统一原文发布、历史读取和索引对账，不根据文本相似度覆盖内容。"""

    def __init__(self, data_root: Path | str, *, maintenance: bool = False) -> None:
        """绑定记忆版本事实和同库索引；传参：数据目录及维护方式；返回：无。"""
        self._data_root = Path(data_root).resolve()
        self._maintenance = maintenance
        self._db = RuntimeStore(data_root)
        self._db.ensure_space()
        self._files = MemoryFiles(self._data_root, self._db)

    @contextmanager
    def locked(self) -> Iterator[None]:
        """串行准备内容修订，允许读者继续核对已发布原件；传参：无；返回：写者窗口。"""
        self._require_runtime_access()
        with file_edit_lock(
            self._data_root / "global" / "memory" / ".write", wait=True
        ):
            yield

    @contextmanager
    def publication(self) -> Iterator[None]:
        """使当前稿、登记和索引在短发布区间内一致；传参：无；返回：先于事实事务取得的发布窗口。"""
        self._require_runtime_access()
        with file_edit_lock(
            self._data_root / "global" / "memory" / ".publish", wait=True
        ):
            yield

    @contextmanager
    def index_snapshot(self) -> Iterator[sqlite3.Connection | None]:
        """固定正文和FTS的同一个读取快照；传参：无；返回：只读连接。"""
        self._require_runtime_access()
        with self.publication():
            self._synchronize()
            with self._db.index_connection() as conn:
                conn.execute("BEGIN")
                try:
                    yield conn
                finally:
                    conn.rollback()

    def create_memory(
        self,
        type: str,
        content: str,
        tags: list[str],
        *,
        applicable_task_tags: list[str] | None = None,
        memory_id: str | None = None,
        state: str = MEMORY_STATE_ACTIVE,
        details: MemoryDetails | None = None,
        change_id: str | None = None,
    ) -> str:
        """提交新记录，同 ID 只接受相同内容的重投；传参：类型、正文、标签与出处；返回：记忆 ID。"""
        if type not in MEMORY_TYPES or state not in MEMORY_STATES:
            raise ValueError("invalid memory type or state")
        context = details or MemoryDetails()
        validate_details(context)
        if not content.strip():
            raise ValueError("memory content is empty")
        now = utc_now()
        memory = seal_memory(
            Memory(
                memory_id=memory_id or new_ulid(),
                type=type,
                state=state,
                content=content.strip(),
                tags=list(tags),
                applicable_task_tags=list(applicable_task_tags or []),
                created_at=now,
                updated_at=now,
                details=context,
                change_id=change_id,
            )
        )
        with self.locked():
            if self._exists(memory.memory_id):
                old = self.load_memory(memory.memory_id)
                if _same_creation(old, memory):
                    return old.memory_id
                raise ValueError(
                    "memory id already exists; revise it with expected_version and a reason"
                )
            self._publish(memory)
        return memory.memory_id

    def load_memory(self, memory_id: str, *, version: str | None = None) -> Memory:
        """按 ID 精确读取当前或可追溯历史原文；传参：身份和可选版本；返回：记忆。"""
        with self.publication():
            records = self._synchronize()
            memory = next(
                (item for item in records if item.memory_id == memory_id), None
            )
            if memory is None:
                raise FileNotFoundError(memory_id)
            if version is not None and version != memory.version:
                return self._historical_memory(memory, version)
            return memory

    def list_memories(
        self,
        state: str | None = None,
        type: str | None = None,
        tags: list[str] | None = None,
        *,
        scope: str | None = None,
        subject: str | None = None,
        fact_key: str | None = None,
        exact_fields: dict[str, str] | None = None,
    ) -> list[Memory]:
        """从原文枚举并做精确过滤；传参：状态、类别、范围或档案字段；返回：真实记录。"""
        if state is not None and state not in MEMORY_STATES | {"superseded"}:
            raise ValueError(f"invalid memory state: {state}")
        if type is not None and type not in MEMORY_TYPES:
            raise ValueError(f"invalid memory type: {type}")
        with self.publication():
            records = self._synchronize()
        states = effective_memory_states(records)
        filters = {"scope": scope, "subject": subject, "fact_key": fact_key}
        selected = [
            record
            for record in records
            if (state is None or states[record.memory_id]["effective_state"] == state)
            and (type is None or record.type == type)
            and set(tags or []).issubset(record.tags)
            and all(
                value is None or getattr(record.details, key) == value
                for key, value in filters.items()
            )
            and all(
                record.details.exact_fields.get(key) == value
                for key, value in (exact_fields or {}).items()
            )
        ]
        return sorted(
            selected,
            key=lambda record: (record.updated_at, record.memory_id),
            reverse=True,
        )

    def record_view(self, memory: Memory) -> dict[str, object]:
        """把历史原文与当前有效身份一同交给读者；传参：当前或历史版本；返回：内容与状态投影。"""
        return self.record_views([memory])[0]

    def record_views(self, memories: list[Memory]) -> list[dict[str, object]]:
        """用一次原文快照解释查询结果的状态；传参：当前或历史记录；返回：来源明确的查询视图。"""
        records = self.list_memories()
        current = {item.memory_id: item for item in records}
        states = effective_memory_states(records)
        return [
            {
                **memory_view(memory),
                **states[memory.memory_id],
                "current_version": current[memory.memory_id].version,
                "historical_version": current[memory.memory_id].version
                != memory.version,
            }
            for memory in memories
        ]

    def revise_memory(
        self,
        memory_id: str,
        content: str,
        *,
        expected_version: str,
        reason: str,
        sources: tuple[MemorySource, ...],
        observed_at: str | None = None,
        expires_at: str | None = None,
        change_id: str | None = None,
        details: MemoryDetails | None = None,
        tags: list[str] | None = None,
    ) -> Memory:
        """更正同一主体和范围的事实并保留旧版；传参：新正文、原版本、理由、真实来源；返回：新版。"""
        if not reason.strip() or not content.strip() or not sources:
            raise ValueError("memory correction requires content, reason, and sources")
        validate_sources(sources)
        with self.locked():
            old = self.load_memory(memory_id)
            context = replace(details or old.details, sources=sources)
            if details is None:
                context = replace(
                    context, observed_at=observed_at, expires_at=expires_at
                )
            validate_details(context)
            if change_id is not None and old.change_id == change_id:
                if (
                    old.content != content.strip()
                    or old.reason != reason
                    or old.details != context
                    or (tags is not None and old.tags != tags)
                ):
                    raise ValueError(
                        "memory change identity reused with different content"
                    )
                return old
            _require_version(old, expected_version)
            memory = _next_revision(
                old,
                content=content.strip(),
                details=context,
                reason=reason,
                state=MEMORY_STATE_ACTIVE,
                change_id=change_id,
                last_verified_at=None,
                verification=(),
                tags=list(old.tags if tags is None else tags),
            )
            self._publish(memory, previous=old)
        return memory

    def update_memory_state(
        self,
        memory_id: str,
        new_state: str,
        *,
        expected_version: str | None = None,
        reason: str = "state changed",
        sources: tuple[MemorySource, ...] = (),
        change_id: str | None = None,
    ) -> Memory:
        """归档或撤回形成可追溯版本；传参：身份、状态和改变依据；返回：更新后的记录。"""
        if new_state not in MEMORY_STATES:
            raise ValueError(f"invalid memory state: {new_state}")
        validate_sources(sources)
        with self.locked():
            old = self.load_memory(memory_id)
            if old.state == new_state:
                return old
            if expected_version is not None:
                _require_version(old, expected_version)
            if new_state not in STATE_TRANSITIONS.get(old.state, set()):
                raise ValueError(
                    f"invalid memory state transition: {old.state}->{new_state}"
                )
            details = replace(old.details, sources=sources or old.details.sources)
            memory = _next_revision(
                old,
                state=new_state,
                reason=reason,
                details=details,
                change_id=change_id,
            )
            self._publish(memory, previous=old)
        return memory

    def archive_memory(self, memory_id: str) -> Memory:
        """保留原文并退出自动召回；传参：身份；返回：归档版本。"""
        return self.update_memory_state(
            memory_id, MEMORY_STATE_ARCHIVED, reason="archived"
        )

    def restore_memory(self, memory_id: str) -> Memory:
        """恢复已归档内容；传参：身份；返回：重新启用的版本。"""
        return self.update_memory_state(
            memory_id, MEMORY_STATE_ACTIVE, reason="restored"
        )

    def touch_memory(self, memory_id: str, used_at: str | None = None) -> Memory:
        """只保存使用时间，不改原文、版本或核验；传参：身份和使用时刻；返回：带统计的记忆。"""
        self._require_runtime_access()
        timestamp = used_at or utc_now()
        validate_time(timestamp)
        with self.publication():
            memory = self.load_memory(memory_id)
            with self._db.transaction() as batch:
                batch.put(
                    "memory_usage",
                    memory_id,
                    {"memory_id": memory_id, "last_used_at": timestamp},
                )
        return replace(memory, last_used_at=timestamp)

    def verify_memory(
        self,
        memory_id: str,
        verified_at: str | None = None,
        *,
        evidence: tuple[MemorySource, ...],
        expected_version: str | None = None,
    ) -> Memory:
        """对指定内容记录独立核验证据；传参：身份、时刻、依据与原版本；返回：核验修订。"""
        validate_sources(evidence, verification=True)
        timestamp = verified_at or utc_now()
        validate_time(timestamp)
        with self.locked():
            old = self.load_memory(memory_id)
            if expected_version is not None:
                _require_version(old, expected_version)
            memory = _next_revision(
                old,
                last_verified_at=timestamp,
                verification=evidence,
                reason="verified with evidence",
            )
            self._publish(memory, previous=old)
        return memory

    def index_status(self, records: list[Memory] | None = None) -> MemoryIndexState:
        """报告索引与原文的一致性；传参：可复用原文快照；返回：状态及具体差异。"""
        with self.publication():
            originals = (
                self._files.synchronize(self._validate_external_revision)[0]
                if records is None
                else records
            )
            with self._db.index_connection() as conn:
                return inspect_index(conn, originals)

    def validate_current(self, memories: list[Memory]) -> None:
        """召回结果进入模型前核对原件；参数：已选择版本；返回：无，变化明确失败。"""
        with self.publication():
            for memory in memories:
                self._files.require_unchanged(memory)

    def current_path(self, memory_id: str) -> Path:
        """返回用户可编辑的当前Markdown；参数：记忆身份；返回：已核对的原件路径。"""
        with self.publication():
            files = self._files.scan()
            if memory_id not in files:
                raise FileNotFoundError(memory_id)
            return files[memory_id][0]

    def close(self) -> None:
        """短连接自动释放；传参：无；返回：无。"""

    def _require_runtime_access(self) -> None:
        """正常运行不能进入已声明的离线转换窗口；传参：无；返回：无。"""
        if not self._maintenance and (self._data_root / ".message_owner.lock").exists():
            raise RuntimeError("memory is locked for maintenance")

    def _publish(self, memory: Memory, *, previous: Memory | None = None) -> None:
        """先保留原件，再原子发布，最后更新索引；传参：新版与旧版；返回：无。"""
        with self.publication():
            self._validate_replacements(memory, previous=previous)
            path = self.current_path(memory.memory_id) if previous is not None else None
            original = path.read_bytes() if path is not None else None
            if previous is not None:
                self._files.require_unchanged(previous)
            # 1. 【记忆】【原件发布】短发布窗口同时保护当前稿、身份与索引，准备新版不占此锁
            if (
                previous is not None
                and path is not None
                and path.parent != self._files.directory(memory)
            ):
                assert original is not None
                self._files.relocate(memory, previous, path, original)
            else:
                self._files.publish(memory, path=path, original=original)
            self._synchronize(published=memory)

    def _synchronize(self, *, published: Memory | None = None) -> list[Memory]:
        """从原件同步编辑并事务更新派生视图；参数：无；返回：已核对正文与使用统计。"""
        records, committed = self._files.synchronize(self._validate_external_revision)
        with self._db.snapshot() as source:
            usage = {
                item["memory_id"]: item["last_used_at"]
                for item in source.list("memory_usage")
            }
        records = [
            replace(item, last_used_at=usage.get(item.memory_id)) for item in records
        ]
        try:
            with self._db.index_connection() as conn:
                ensure_memory_index_schema(conn)
                if inspect_index(conn, records).state != "current":
                    _replace_index(conn, records)
        except (OSError, sqlite3.Error, RuntimeError) as exc:
            saved = committed[-1] if committed else published
            if saved is not None:
                raise MemoryIndexUpdateError(saved, exc) from exc
            raise
        return records

    def _validate_external_revision(
        self, memory: Memory, previous: Memory | None, records: list[Memory]
    ) -> None:
        """外部编辑复用状态和替代规则；参数：候选、旧版、已读原件；返回：无。"""
        if previous is not None and memory.state != previous.state:
            if memory.state not in STATE_TRANSITIONS.get(previous.state, set()):
                raise ValueError(
                    f"invalid memory state transition: {previous.state}->{memory.state}"
                )
        self._validate_replacements(
            memory, previous=previous, records=records, external_edit=True
        )

    def _validate_replacements(
        self,
        memory: Memory,
        *,
        previous: Memory | None,
        records: list[Memory] | None = None,
        external_edit: bool = False,
    ) -> None:
        """在正文提交前校验替代关系及要求来源；传参：候选、前版；返回：无，失败保留旧有效状态。"""
        records = self.list_memories() if records is None else records
        states = effective_memory_states(records)
        if previous is not None:
            if not set(previous.details.supersedes).issubset(memory.details.supersedes):
                raise ValueError(
                    "published memory replacements cannot be removed or rewritten"
                )
            if states[previous.memory_id]["effective_state"] == "superseded":
                raise ValueError(
                    "superseded memory cannot be reactivated; create an explicit new decision"
                )
            if (
                previous.type in {"rule", "preference"}
                and memory.details.sources != previous.details.sources
                and not external_edit
            ):
                _require_user_change(memory)
        inherited = set(previous.details.supersedes) if previous else set()
        by_id = {item.memory_id: item for item in records}
        for target in memory.details.supersedes:
            if target in inherited:
                continue
            if target.memory_id == memory.memory_id or target.memory_id not in by_id:
                raise ValueError(
                    "memory replacement target is missing or self-referential"
                )
            old = by_id[target.memory_id]
            _require_version(old, target.version)
            if (
                states[old.memory_id]["effective_state"] != "active"
                or memory.state != "active"
            ):
                raise ValueError(
                    "memory replacement requires active source and target records"
                )
            changes = [
                key
                for key in ("scope", "subject", "fact_key")
                if getattr(old.details, key) != getattr(memory.details, key)
            ]
            if changes:
                raise ValueError(
                    f"memory replacement cannot change {', '.join(changes)}"
                )
            if not memory.details.sources:
                raise ValueError("memory replacement requires new evidence")
            if old.type in {"rule", "preference"}:
                _require_user_change(memory)

    def _exists(self, memory_id: str) -> bool:
        """精确判断已发布身份；传参：记忆编号；返回：是否存在。"""
        return any(item.memory_id == memory_id for item in self.list_memories())

    def _historical_memory(self, current: Memory, version: str) -> Memory:
        """仅沿已提交版本链补读，孤立的未发布文件不可见；传参：当前记录和版本；返回：旧版。"""
        seen = {current.version}
        while current.previous_version is not None:
            previous = current.previous_version
            if previous in seen:
                raise ValueError("memory revision chain contains a cycle")
            seen.add(previous)
            current = self._files.read_revision(
                self.current_path(current.memory_id), current.memory_id, previous
            )
            if current.version != previous:
                raise ValueError("memory revision filename and content differ")
            if current.version == version:
                return current
        raise FileNotFoundError(
            f"memory revision not found: {current.memory_id}@{version}"
        )


def _same_creation(old: Memory, new: Memory) -> bool:
    """相同 ID 的重投不得覆盖事实；传参：原记录和候选；返回：是否完全等价的创建。"""
    fields = (
        "type",
        "state",
        "content",
        "tags",
        "applicable_task_tags",
        "details",
        "change_id",
    )
    return all(getattr(old, key) == getattr(new, key) for key in fields)


def _next_revision(old: Memory, **changes: Any) -> Memory:
    """生成一个有前驱的完整修订；传参：原记录和字段变化；返回：固定版本的新版。"""
    return seal_memory(
        replace(
            old,
            **changes,
            format_version=MEMORY_FORMAT,
            revision=old.revision + 1,
            previous_version=old.version,
            updated_at=utc_now(),
            last_used_at=None,
        )
    )


def _require_version(memory: Memory, expected: str) -> None:
    """拒绝基于过时内容的修改；传参：当前记录和预期版本；返回：无。"""
    if memory.version != expected:
        raise ValueError(f"memory version conflict: current={memory.version}")


def _require_user_change(memory: Memory) -> None:
    """要求或偏好变化必须引用用户表达，事实可由适用观察纠正；传参：候选；返回：无。"""
    if not any(source.kind == "user_input" for source in memory.details.sources):
        raise ValueError(
            "changing a user requirement or preference requires user input evidence"
        )


def rebuild_memory_index(data_root: Path | str, *, maintenance: bool = False) -> int:
    """同一事务重建记忆索引，保留所有正文修订；传参：数据根及维护模式；返回：记忆数。"""
    store = MemoryStore(data_root, maintenance=maintenance)
    with store.publication():
        records, committed = store._files.synchronize(store._validate_external_revision)
        with store._db.snapshot() as source:
            usage = {
                item["memory_id"]: item["last_used_at"]
                for item in source.list("memory_usage")
            }
        records = [
            replace(item, last_used_at=usage.get(item.memory_id)) for item in records
        ]
        try:
            with store._db.index_connection() as conn:
                ensure_memory_index_schema(conn)
                _replace_index(conn, records)
        except (OSError, sqlite3.Error, RuntimeError) as exc:
            if committed:
                raise MemoryIndexUpdateError(committed[-1], exc) from exc
            raise
        return len(records)


def _replace_index(conn: sqlite3.Connection, records: list[Memory]) -> None:
    """在单一SQLite事务替换记忆投影，失败保持上一完整代次；参数：连接与原件；返回：无。"""
    states = effective_memory_states(records)
    conn.execute("BEGIN IMMEDIATE")
    with conn:
        conn.execute("DELETE FROM memories_fts")
        conn.execute("DELETE FROM memories")
        for memory in records:
            upsert_memory_index(
                conn,
                memory,
                commit=False,
                effective_state=str(states[memory.memory_id]["effective_state"]),
            )
        conn.execute(
            "INSERT OR REPLACE INTO runtime_meta(key,value) VALUES(?,?)",
            ("memory_tokenizer_version", str(TOKENIZER_VERSION)),
        )
        if inspect_index(conn, records).state != "current":
            raise RuntimeError("rebuilt memory index does not match Markdown originals")


__all__ = [
    "MEMORY_STATE_ACTIVE",
    "MEMORY_STATE_ARCHIVED",
    "MEMORY_STATE_DRAFT",
    "MEMORY_STATE_WITHDRAWN",
    "MEMORY_TYPES",
    "Memory",
    "MemoryDetails",
    "MemorySource",
    "MemoryStore",
    "MemoryIndexUpdateError",
    "connect_memory_index",
    "ensure_memory_index_schema",
    "rebuild_memory_index",
    "upsert_memory_index",
]
