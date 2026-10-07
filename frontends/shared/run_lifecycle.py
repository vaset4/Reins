from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from runtime.run_facts import latest_lifecycle_from_facts

TERMINAL_STATES = frozenset({"DONE", "PAUSED", "FAILED"})
LEGACY_WAITING_APPROVAL_STATES = frozenset({"AWAITING_APPROVAL", "WAITING_APPROVAL"})
LEGACY_PAUSE_REASONS = frozenset({"paused", "failed", "interrupted"})


@dataclass(frozen=True, slots=True)
class LifecycleObservation:
    lifecycle: str
    reason: str
    ts: str
    source: str
    checkpoint_id: str = ""
    resumable: bool | None = None


def build_lifecycle_observation(facts: list[dict[str, Any]]) -> LifecycleObservation:
    lifecycle = latest_lifecycle_from_facts(facts)
    if lifecycle:
        return LifecycleObservation(
            lifecycle=str(lifecycle.get("lifecycle", "")),
            reason=str(lifecycle.get("reason", "")),
            ts=str(lifecycle.get("ts", "")),
            source="run:lifecycle",
            checkpoint_id=str(lifecycle.get("checkpoint_id", "")),
            resumable=_optional_bool(lifecycle.get("resumable")),
        )
    legacy = _latest_legacy_lifecycle(facts)
    if legacy:
        return LifecycleObservation(
            lifecycle=_legacy_lifecycle_name(legacy),
            reason=str(
                legacy.get("close_reason") or legacy.get("terminal_reason") or ""
            ),
            ts=str(legacy.get("ts", "")),
            source="legacy_state_transition",
            checkpoint_id=str(legacy.get("checkpoint_id", "")),
        )
    return LifecycleObservation("", "", "", "none")


def row_pauses_run(event: str, row: Mapping[str, Any]) -> bool:
    if event == "run:lifecycle":
        return str(row.get("lifecycle", "")).lower() in {"paused", "failed"}
    return (
        event == "state:transition" and row.get("close_reason") in LEGACY_PAUSE_REASONS
    )


def _latest_legacy_lifecycle(facts: list[dict[str, Any]]) -> dict[str, Any]:
    for fact in reversed(facts):
        if fact.get("event") != "state:transition":
            continue
        state = str(fact.get("to_state", "")).upper()
        if state in TERMINAL_STATES or state in LEGACY_WAITING_APPROVAL_STATES:
            return fact
    return {}


def _legacy_lifecycle_name(fact: Mapping[str, Any]) -> str:
    state = str(fact.get("to_state", "")).upper()
    if state in LEGACY_WAITING_APPROVAL_STATES:
        return "waiting_approval"
    return state.lower()


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


__all__ = [
    "LifecycleObservation",
    "build_lifecycle_observation",
    "row_pauses_run",
]
