from __future__ import annotations

from typing import Any

from frontends.observe.panels.base import Panel, RunContext


class ContextInspectorPanel(Panel):
    id = "context_inspector"
    title = "Context Inspector"
    section = "context"
    phase = "C"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        facts = ctx.facts()
        segments = _extract_segments(facts)
        dropped = _extract_dropped(facts)
        return {"segments": segments, "dropped": dropped}


def _extract_segments(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for fact in reversed(facts):
        if fact.get("event") == "context:segments":
            segments = fact.get("segments", [])
            if isinstance(segments, list):
                return [row for row in segments if isinstance(row, dict)]
    return []


def _extract_dropped(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for fact in reversed(facts):
        if fact.get("event") == "context:dropped":
            dropped = fact.get("dropped", [])
            if isinstance(dropped, list):
                return [row for row in dropped if isinstance(row, dict)]
    return []


__all__ = ["ContextInspectorPanel"]
