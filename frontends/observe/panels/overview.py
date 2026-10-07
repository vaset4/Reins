from __future__ import annotations

from typing import Any

from frontends.shared.run_lifecycle import build_lifecycle_observation
from frontends.observe.panels.base import Panel, RunContext


class OverviewPanel(Panel):
    id = "overview"
    title = "Overview"
    section = "overview"
    phase = "A"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        facts = ctx.facts()
        summary = ctx.summary()
        state = ctx.state()
        return {
            "session_id": ctx.session_id,
            "run_id": ctx.run_id,
            "status": summary.status if summary else _status_from_facts(facts),
            "task_id": summary.task_id if summary else _first_field(facts, "task_id"),
            "focus_task_id": (
                summary.focus_task_id
                if summary
                else _first_field(facts, "focus_task_id")
            ),
            "started_at": summary.started_at if summary else _first_ts(facts),
            "updated_at": summary.updated_at if summary else _last_ts(facts),
            "last_event": summary.last_event if summary else _last_event(facts),
            "total_facts": len(facts),
            "llm_calls": sum(1 for f in facts if f.get("event") == "llm:response"),
            "tool_calls": sum(1 for f in facts if f.get("event") == "tool:request"),
            "lifecycle_events": _event_count(facts, "run:lifecycle"),
            "legacy_state_transitions": _event_count(facts, "state:transition"),
            "session_summary": state.summary if state else "",
            "errors_count": len(ctx.errors()),
        }


def _status_from_facts(facts: list[dict[str, Any]]) -> str:
    lifecycle = build_lifecycle_observation(facts)
    if lifecycle.lifecycle:
        return lifecycle.lifecycle
    for fact in reversed(facts):
        if fact.get("event") == "state:transition":
            to_state = str(fact.get("to_state", ""))
            if to_state in {"DONE", "PAUSED", "FAILED"}:
                return to_state.lower()
    return "running"


def _first_field(facts: list[dict[str, Any]], key: str) -> str:
    for fact in facts:
        value = fact.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _first_ts(facts: list[dict[str, Any]]) -> str:
    return str(facts[0].get("ts", "")) if facts else ""


def _last_ts(facts: list[dict[str, Any]]) -> str:
    return str(facts[-1].get("ts", "")) if facts else ""


def _last_event(facts: list[dict[str, Any]]) -> str:
    return str(facts[-1].get("event", "")) if facts else ""


def _event_count(facts: list[dict[str, Any]], event: str) -> int:
    return sum(1 for fact in facts if fact.get("event") == event)


__all__ = ["OverviewPanel"]
