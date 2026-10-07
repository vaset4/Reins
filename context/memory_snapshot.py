from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from context.memory_recall import MemoryRecallResult, RecallOutcome, SkippedMemory


@dataclass(frozen=True, slots=True)
class InjectedExplain:
    memory_id: str
    type: str
    reason: str
    score: float
    is_stale: bool
    has_conflict: bool
    version: str
    scope: str
    sources: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """记忆召回快照

    conflicts: 有内容近重复的 memory_id 列表（归一化后 content 完全相同）
    """

    round_id: str
    taken_at: str
    entries: tuple[MemoryRecallResult, ...]
    skipped: tuple[SkippedMemory, ...]
    warnings: tuple[str, ...]
    conflicts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryInjectionExplain:
    round_id: str
    taken_at: str
    injected: tuple[InjectedExplain, ...]
    skipped: tuple[SkippedMemory, ...]
    warnings: tuple[str, ...]


def take_snapshot(
    outcome: RecallOutcome,
    *,
    round_id: str,
    now: datetime,
) -> MemorySnapshot:
    entries = tuple(outcome.selected)
    skipped = tuple(outcome.skipped)
    warnings: list[str] = []
    conflicts: set[str] = set()

    # 【记忆】【冲突证据】只有明确标注同一主体、范围和字段的不同结论才构成待解释冲突
    content_groups: dict[tuple[str, str, str], list[MemoryRecallResult]] = defaultdict(
        list
    )
    for entry in entries:
        details = entry.memory.details
        if details.subject and details.fact_key:
            content_groups[(details.scope, details.subject, details.fact_key)].append(
                entry
            )

    for group in content_groups.values():
        if len({entry.memory.content for entry in group}) >= 2:
            warnings.append(
                f"conflict_detected: {len(group)} different claims for the same scoped subject and field"
            )
            conflicts.update(entry.memory.memory_id for entry in group)
    if outcome.index.state != "current":
        warnings.append(f"memory_index_{outcome.index.state}: {outcome.index}")

    return MemorySnapshot(
        round_id=round_id,
        taken_at=now.isoformat(),
        entries=entries,
        skipped=skipped,
        warnings=tuple(warnings),
        conflicts=tuple(sorted(conflicts)),
    )


def to_explain(snapshot: MemorySnapshot) -> MemoryInjectionExplain:
    conflicts = set(snapshot.conflicts)
    injected = tuple(
        InjectedExplain(
            memory_id=entry.memory.memory_id,
            type=entry.memory.type,
            reason=_injection_reason(entry),
            score=entry.score,
            is_stale=entry.stale_penalty > 0.0,
            has_conflict=entry.memory.memory_id in conflicts,
            version=entry.memory.version,
            scope=entry.memory.details.scope,
            sources=tuple(
                {
                    "kind": source.kind,
                    "reference": source.reference,
                    "session_id": source.session_id,
                    "run_id": source.run_id,
                }
                for source in entry.memory.details.sources
            ),
        )
        for entry in snapshot.entries
    )
    return MemoryInjectionExplain(
        round_id=snapshot.round_id,
        taken_at=snapshot.taken_at,
        injected=injected,
        skipped=snapshot.skipped,
        warnings=snapshot.warnings,
    )


def _injection_reason(entry: MemoryRecallResult) -> str:
    if entry.bm25 > 0.0 and entry.tag_jaccard > 0.0:
        return "match_query_and_tag"
    if entry.bm25 > 0.0:
        return "match_query"
    if entry.tag_jaccard > 0.0:
        return "match_tag"
    if entry.recency_bonus > 0.0:
        return "recency_bonus"
    return "type_limit_kept"


__all__ = [
    "InjectedExplain",
    "MemoryInjectionExplain",
    "MemorySnapshot",
    "take_snapshot",
    "to_explain",
]
