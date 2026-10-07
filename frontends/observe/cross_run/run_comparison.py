from __future__ import annotations

from typing import Any

from frontends.shared.run_lifecycle import build_lifecycle_observation
from frontends.observe.readers.fact_reader import FactReader


def run_comparison(fact_reader: FactReader, run_a: str, run_b: str) -> dict[str, Any]:
    facts_a = fact_reader.read_facts(run_a)
    facts_b = fact_reader.read_facts(run_b)
    return {
        "run_a": _summarize_run(run_a, facts_a),
        "run_b": _summarize_run(run_b, facts_b),
    }


def _summarize_run(run_id: str, facts: list[dict[str, Any]]) -> dict[str, Any]:
    if not facts:
        return {"run_id": run_id, "status": "not_found"}
    llm_calls = [f for f in facts if f.get("event") == "llm:response"]
    tool_requests = [f for f in facts if f.get("event") == "tool:request"]
    total_tokens = 0
    for f in llm_calls:
        obs = (f.get("summary") or {}).get("observation") or {}
        if isinstance(obs, dict):
            total_tokens += _int(obs.get("prompt_tokens", 0)) + _int(
                obs.get("completion_tokens", 0)
            )
    status = _terminal_status(facts)
    return {
        "run_id": run_id,
        "session_id": str(facts[0].get("session_id", "")),
        "status": status,
        "total_facts": len(facts),
        "llm_calls": len(llm_calls),
        "tool_requests": len(tool_requests),
        "lifecycle_events": _count_event(facts, "run:lifecycle"),
        "legacy_state_transitions": _count_event(facts, "state:transition"),
        "total_tokens": total_tokens,
        "started_at": str(facts[0].get("ts", "")),
        "ended_at": str(facts[-1].get("ts", "")),
    }


def _terminal_status(facts: list[dict[str, Any]]) -> str:
    lifecycle = build_lifecycle_observation(facts)
    if lifecycle.lifecycle:
        return lifecycle.lifecycle
    for fact in reversed(facts):
        to_state = str(fact.get("to_state", ""))
        if to_state in {"DONE", "PAUSED", "FAILED"}:
            return to_state.lower()
    return "running"


def _count_event(facts: list[dict[str, Any]], event: str) -> int:
    return sum(1 for fact in facts if fact.get("event") == event)


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["run_comparison"]
