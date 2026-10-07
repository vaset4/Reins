from __future__ import annotations

import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from memory.safety_scan import scan as safety_scan
from memory.index import MemoryIndexState, tokenize_search_text
from memory.store import MEMORY_STATE_ACTIVE, Memory, MemoryStore
from memory.records import effective_memory_states

logger = logging.getLogger(__name__)

RECALL_TYPES = {"lesson", "rule", "fact", "preference"}
TYPE_LIMITS = {"lesson": 3, "rule": 5, "fact": 3, "preference": 3}
LESSON_BOOST = 0.5


@dataclass(frozen=True, slots=True)
class MemoryRecallResult:
    memory: Memory
    score: float
    bm25: float
    tag_jaccard: float
    recency_bonus: float
    stale_penalty: float
    required: bool = False


@dataclass(frozen=True, slots=True)
class SkippedMemory:
    memory_id: str
    type: str
    reason: str


@dataclass(frozen=True, slots=True)
class RecallOutcome:
    selected: list[MemoryRecallResult]
    skipped: list[SkippedMemory]
    index: MemoryIndexState = field(default_factory=lambda: MemoryIndexState("current"))


def recall_memories(
    data_root: Path | str,
    *,
    task_summary: str,
    task_tags: list[str],
    now: datetime | None = None,
    states: Sequence[str] | None = None,
    scopes: Sequence[str] | None = None,
) -> list[MemoryRecallResult]:
    return recall_memories_with_outcome(
        data_root,
        task_summary=task_summary,
        task_tags=task_tags,
        now=now,
        states=states,
        scopes=scopes,
    ).selected


def recall_memories_with_outcome(
    data_root: Path | str,
    *,
    task_summary: str,
    task_tags: list[str],
    now: datetime | None = None,
    include_experience: bool = False,
    states: Sequence[str] | None = None,
    scopes: Sequence[str] | None = None,
    recent_user_text: str = "",
    include_global: bool = True,
    source_session_id: str | None = None,
) -> RecallOutcome:
    """只使用当前索引和有效范围的事实；传参：查询、标签、范围和时刻；返回：召回及未选原因。"""
    allowed_types = RECALL_TYPES | ({"experience"} if include_experience else set())
    store = MemoryStore(data_root)
    skipped: list[SkippedMemory] = []
    try:
        with store.index_snapshot() as connection:
            records = store.list_memories()
            index = store.index_status(records)
            bm25_scores = (
                _bm25_scores(connection, task_summary)
                if connection is not None and index.state == "current"
                else {}
            )
            current_scores = (
                _bm25_scores(connection, recent_user_text)
                if connection is not None
                and index.state == "current"
                and recent_user_text.strip()
                else {}
            )
        candidates = _collect_candidates(records, states, allowed_types)
        if not include_global:
            candidates = [item for item in candidates if item.details.scope != "global"]
        safe_candidates, unsafe_skipped = _filter_safe_candidates(
            candidates, now, scopes=scopes
        )
        skipped.extend(unsafe_skipped)
        required_ids = {
            item.memory_id
            for item in safe_candidates
            if item.state == MEMORY_STATE_ACTIVE
            and item.type in {"rule", "preference"}
            and any(source.kind == "user_input" for source in item.details.sources)
        }
        if index.state != "current":
            safe_candidates = [
                item for item in safe_candidates if item.memory_id in required_ids
            ]

        scored = [
            _score_memory(
                memory, bm25_scores.get(memory.memory_id, 0.0), task_tags, now
            )
            for memory in safe_candidates
        ]
        # 【记忆】【当前问题召回】先保留命中当前问题的原件，同层内继续使用既有评分，避免旧目标占满名额
        sorted_scored = sorted(
            scored,
            key=lambda item: (
                current_scores.get(item.memory.memory_id, 0.0) > 0,
                item.score,
                item.memory.memory_id,
            ),
            reverse=True,
        )
        selected, limited_skipped = _apply_type_limits_with_skipped(
            [
                item
                for item in sorted_scored
                if item.memory.memory_id not in required_ids
            ]
        )
        selected_ids = required_ids | {item.memory.memory_id for item in selected}
        selected = [
            replace(item, required=item.memory.memory_id in required_ids)
            for item in sorted_scored
            if item.memory.memory_id in selected_ids
        ]
        skipped.extend(limited_skipped)
        _touch_active(store, selected)
        store.validate_current([item.memory for item in selected])
        selected = _project_source_validity(
            data_root, selected, scopes=scopes, session_id=source_session_id
        )
        return RecallOutcome(selected=selected, skipped=skipped, index=index)
    finally:
        store.close()


def _project_source_validity(
    data_root: Path | str,
    selected: list[MemoryRecallResult],
    *,
    scopes: Sequence[str] | None,
    session_id: str | None = None,
) -> list[MemoryRecallResult]:
    """相关文件变化时把旧知识标为待核验，原记忆正文和版本不倒改；参数：入选资料/范围；返回：模型投影。"""
    from runtime.knowledge_validity import memory_source_validity

    session_id = session_id or next(
        (
            scope.split(":", 1)[1]
            for scope in scopes or ()
            if scope.startswith("session:")
        ),
        None,
    )
    result = []
    for item in selected:
        validity = memory_source_validity(data_root, item.memory, session_id=session_id)
        if validity:
            text = f"{validity['notice']}\n{item.memory.content}"
            item = replace(item, memory=replace(item.memory, content=text))
        result.append(item)
    return result


def _collect_candidates(
    records: list[Memory], states: Sequence[str] | None, allowed_types: set[str]
) -> list[Memory]:
    """按状态取出候选记忆

    参数:
        records: 当前原文快照
        states: 要纳入的状态；None 表示只取 active（自动召回走这条）
        allowed_types: 允许参与召回的记忆类型

    返回:
        候选记忆列表，未打分未限流

    说明:
        逐个状态各查一次再拼接，不去重：一条记忆只有一个状态，
        两次查不同状态不可能返回同一条，故无需为不会发生的重复兜底。
        非法状态由 list_memories 内的校验抛出，这里不另造一道
    """
    target_states = (MEMORY_STATE_ACTIVE,) if states is None else states
    effective = effective_memory_states(records)
    return [
        memory
        for memory in records
        if effective[memory.memory_id]["effective_state"] in target_states
        and memory.type in allowed_types
    ]


def _filter_safe_candidates(
    candidates: list[Memory],
    now: datetime | None,
    *,
    scopes: Sequence[str] | None,
) -> tuple[list[Memory], list[SkippedMemory]]:
    """过滤明确失效、范围不符及不安全内容；传参：候选、时刻和适用范围；返回：有效记录及原因。"""
    safe_candidates: list[Memory] = []
    skipped: list[SkippedMemory] = []
    for memory in candidates:
        reason = _inapplicable_reason(memory, now, scopes=scopes)
        if reason is not None:
            skipped.append(SkippedMemory(memory.memory_id, memory.type, reason))
            continue
        try:
            scan_result = safety_scan(memory.content)
        except Exception as exc:
            logger.error("memory safety scan failed for %s: %s", memory.memory_id, exc)
            skipped.append(
                SkippedMemory(memory.memory_id, memory.type, "safety_scan_error")
            )
            continue
        if not scan_result.is_safe:
            skipped.append(
                SkippedMemory(memory.memory_id, memory.type, "blocked_by_safety_scan")
            )
            continue
        safe_candidates.append(memory)
    return safe_candidates, skipped


def _inapplicable_reason(
    memory: Memory, now: datetime | None, *, scopes: Sequence[str] | None
) -> str | None:
    """仅按明确作用域和失效时间排除，不从年龄推断失效；传参：记忆与查询边界；返回：原因或空。"""
    if memory.details.scope == "unspecified":
        return "scope_unknown"
    if memory.details.scope not in {"global", *(scopes or ())}:
        return "scope_mismatch"
    expires = memory.details.expires_at
    if expires is not None and datetime.fromisoformat(expires) <= (
        now or datetime.now(timezone.utc)
    ):
        return "expired_explicitly"
    return None


def _touch_active(store: MemoryStore, selected: list[MemoryRecallResult]) -> None:
    """只给启用中的记忆盖使用时间戳

    参数:
        store: 已打开的记忆库
        selected: 本次入选的记忆

    说明:
        归档记忆命中后不写盘：写盘会刷新 last_used_at 抬高时新度得分，
        形成「越搜越靠前」把已作废结论顶回模型眼前；updated_at 又是
        列表查询的排序键，一并会把归档记忆推到列表最前面
    """
    for item in selected:
        if item.memory.state == MEMORY_STATE_ACTIVE:
            store.touch_memory(item.memory.memory_id)


def jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def recency_bonus(last_used_at: str | None, now: datetime | None = None) -> float:
    if last_used_at is None:
        return 0.0
    days = _days_since(last_used_at, now)
    return 1.0 / (1.0 + days / 7.0)


def stale_penalty(last_verified_at: str | None, now: datetime | None = None) -> float:
    if last_verified_at is None:
        return 1.0
    days = _days_since(last_verified_at, now)
    if days <= 30:
        return 0.0
    return min((days - 30.0) / 30.0, 1.0)


def _score_memory(
    memory: Memory,
    bm25: float,
    task_tags: list[str],
    now: datetime | None,
) -> MemoryRecallResult:
    tag_score = jaccard(task_tags, memory.tags)
    recency = recency_bonus(memory.last_used_at, now)
    stale = stale_penalty(memory.last_verified_at, now)
    score = 0.5 * bm25 + 0.3 * tag_score + 0.2 * recency - 0.5 * stale
    if memory.type == "lesson":
        score += LESSON_BOOST
    return MemoryRecallResult(memory, score, bm25, tag_score, recency, stale)


def _bm25_scores(conn: sqlite3.Connection, query: str) -> dict[str, float]:
    """只在已经核对的同一索引事务内打分；传参：只读快照和查询；返回：分数，IO错误直接暴露。"""
    match_query = _fts_query(query)
    if not match_query:
        return {}
    try:
        rows = conn.execute(
            """
            SELECT memory_id, bm25(memories_fts) AS rank
            FROM memories_fts
            WHERE memories_fts MATCH ?
            """,
            (match_query,),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise RuntimeError("memory FTS query failed") from exc
    return {str(memory_id): max(0.0, -float(rank)) for memory_id, rank in rows}


def _apply_type_limits_with_skipped(
    results: list[MemoryRecallResult],
) -> tuple[list[MemoryRecallResult], list[SkippedMemory]]:
    counts = {type_name: 0 for type_name in TYPE_LIMITS}
    selected: list[MemoryRecallResult] = []
    skipped: list[SkippedMemory] = []
    for item in results:
        type_name = item.memory.type
        if type_name not in counts:
            selected.append(item)
            continue
        if counts[type_name] >= TYPE_LIMITS[type_name]:
            skipped.append(
                SkippedMemory(item.memory.memory_id, type_name, "type_limit")
            )
            continue
        counts[type_name] += 1
        selected.append(item)
    return selected, skipped


def _fts_query(text: str) -> str:
    tokens = tokenize_search_text(text)
    return " OR ".join(f'"{token}"' for token in tokens[:20])


def _days_since(value: str, now: datetime | None = None) -> float:
    current = now or datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max((current - parsed).total_seconds() / 86400.0, 0.0)


__all__ = [
    "MemoryRecallResult",
    "RecallOutcome",
    "SkippedMemory",
    "jaccard",
    "recall_memories",
    "recall_memories_with_outcome",
    "recency_bonus",
    "stale_penalty",
]
