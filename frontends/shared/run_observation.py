from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from frontends.shared.run_lifecycle import (
    LifecycleObservation,
    build_lifecycle_observation,
    row_pauses_run,
)


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    checkpoint_id: str
    state: str
    reason: str
    ts: str
    pending_tool_call: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ContextObservation:
    build_count: int
    segments: list[dict[str, Any]]
    total_tokens_est: int
    last_tool_history_count: object = 0


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    key: str
    path: str
    status: str
    gap: str = ""


@dataclass(frozen=True, slots=True)
class MemoryObservation:
    injections: list[dict[str, Any]] = field(default_factory=list)
    score_breakdowns: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CompressionObservation:
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WatchdogObservation:
    llm_failures: int
    tool_failures: int
    unique_failure_patterns: int
    paused: bool


@dataclass(frozen=True, slots=True)
class RunObservation:
    lifecycle: LifecycleObservation
    checkpoints: list[CheckpointRef]
    context: ContextObservation
    raw_evidence: list[EvidenceItem]
    memory: MemoryObservation
    compression: CompressionObservation
    watchdog: WatchdogObservation


def build_run_observation(facts: list[dict[str, Any]]) -> RunObservation:
    return RunObservation(
        lifecycle=build_lifecycle_observation(facts),
        checkpoints=_checkpoint_refs(facts),
        context=_context_observation(facts),
        raw_evidence=_raw_evidence(_latest_evidence(facts)),
        memory=_memory_observation(facts),
        compression=_compression_observation(facts),
        watchdog=build_watchdog_observation(facts),
    )


def build_watchdog_observation(rows: list[dict[str, Any]]) -> WatchdogObservation:
    llm_failures = 0
    tool_failures = 0
    patterns: set[str] = set()
    paused = False
    for row in rows:
        event = str(row.get("event", ""))
        if _llm_failed(event, row):
            llm_failures += 1
        if event == "tool:response":
            tool_failures += _tool_failure_count(row, patterns)
        if row_pauses_run(event, row):
            paused = True
    return WatchdogObservation(llm_failures, tool_failures, len(patterns), paused)


def _checkpoint_refs(facts: list[dict[str, Any]]) -> list[CheckpointRef]:
    refs: list[CheckpointRef] = []
    for fact in facts:
        if fact.get("event") != "checkpoint:saved":
            continue
        checkpoint = fact.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            continue
        pending = checkpoint.get("pending_tool_call")
        refs.append(
            CheckpointRef(
                checkpoint_id=str(checkpoint.get("checkpoint_id", "")),
                state=str(checkpoint.get("state", "")),
                reason=str(checkpoint.get("reason", "")),
                ts=str(fact.get("ts", "")),
                pending_tool_call=pending if isinstance(pending, Mapping) else None,
            )
        )
    return refs


def _context_observation(facts: list[dict[str, Any]]) -> ContextObservation:
    builds = [fact for fact in facts if fact.get("event") == "context:built"]
    segment_events = [fact for fact in facts if fact.get("event") == "context:segments"]
    segments = _latest_segments(segment_events)
    return ContextObservation(
        build_count=len(builds),
        segments=segments,
        total_tokens_est=sum(_int(item.get("tokens_est")) for item in segments),
        last_tool_history_count=_last_tool_history_count(builds),
    )


def _latest_segments(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not facts:
        return []
    value = facts[-1].get("segments")
    return (
        [dict(item) for item in value if isinstance(item, Mapping)]
        if isinstance(value, list)
        else []
    )


def _raw_evidence(evidence: Mapping[str, object]) -> list[EvidenceItem]:
    return [
        _evidence_item(evidence, "model_request"),
        _evidence_item(evidence, "model_response"),
        _evidence_item(evidence, "parsed_plan"),
        _evidence_item(evidence, "errors"),
    ]


def _evidence_item(evidence: Mapping[str, object], key: str) -> EvidenceItem:
    path = str(evidence.get(key, "") or "")
    if not path:
        return EvidenceItem(key, "", "missing", "no evidence path")
    if path.startswith("("):
        return EvidenceItem(key, path, "not_available", path)
    return EvidenceItem(key, path, "path_recorded")


def _memory_observation(facts: list[dict[str, Any]]) -> MemoryObservation:
    injections: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for fact in facts:
        if fact.get("event") == "memory:injection_explain":
            payload = _explain_payload(fact)
            injections.append({"ts": fact.get("ts", ""), **payload})
        if fact.get("event") == "memory:score_breakdown":
            scores.append(
                {
                    "ts": fact.get("ts", ""),
                    "round_id": fact.get("round_id", ""),
                    "injected": fact.get("injected", []),
                    "skipped": fact.get("skipped", []),
                }
            )
    return MemoryObservation(injections, scores)


def _compression_observation(facts: list[dict[str, Any]]) -> CompressionObservation:
    events = []
    for fact in facts:
        if fact.get("event") != "trim:delta":
            continue
        events.append(
            {
                "ts": fact.get("ts", ""),
                "reason": fact.get("reason", ""),
                "tokens_before": fact.get("tokens_before"),
                "tokens_after": fact.get("tokens_after"),
                "removed_sections": fact.get("removed_sections", []),
            }
        )
    return CompressionObservation(events)


def _latest_evidence(facts: list[dict[str, Any]]) -> Mapping[str, object]:
    for fact in reversed(facts):
        if fact.get("event") != "llm:response":
            continue
        summary = fact.get("summary")
        evidence = summary.get("evidence") if isinstance(summary, Mapping) else None
        if isinstance(evidence, Mapping):
            return evidence
    return {}


def _llm_failed(event: str, row: Mapping[str, Any]) -> bool:
    summary = row.get("summary")
    return (
        event == "llm:response"
        and isinstance(summary, Mapping)
        and bool(summary.get("error"))
    )


def _tool_failure_count(row: Mapping[str, Any], patterns: set[str]) -> int:
    tool = row.get("tool")
    if not isinstance(tool, Mapping) or tool.get("status") == "ok":
        return 0
    category = tool.get("error_category")
    if isinstance(category, str) and category:
        patterns.add(category)
    return 1


def _last_tool_history_count(rows: list[dict[str, Any]]) -> object:
    if not rows:
        return 0
    summary = rows[-1].get("summary")
    if isinstance(summary, Mapping):
        return summary.get("tool_history_count", 0)
    return rows[-1].get("tool_history_count", 0)


def _explain_payload(fact: Mapping[str, Any]) -> dict[str, Any]:
    explain = fact.get("explain")
    if isinstance(explain, Mapping):
        return dict(explain)
    detail = fact.get("detail")
    if isinstance(detail, Mapping):
        return dict(detail)
    return {}


def _int(value: object) -> int:
    if not isinstance(value, (str, int, float)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = [
    "CheckpointRef",
    "CompressionObservation",
    "ContextObservation",
    "EvidenceItem",
    "LifecycleObservation",
    "MemoryObservation",
    "RunObservation",
    "WatchdogObservation",
    "build_lifecycle_observation",
    "build_run_observation",
    "build_watchdog_observation",
]
