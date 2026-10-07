from __future__ import annotations

from typing import Any

from frontends.observe.panels.base import Panel, RunContext


class CompressionLanePanel(Panel):
    id = "compression_lane"
    title = "Compression Lane"
    section = "context"
    phase = "B"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        facts = ctx.facts()
        events = _extract_compression_events(facts)
        return {"events": events, "count": len(events)}


def _extract_compression_events(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for fact in facts:
        event = fact.get("event", "")
        if event != "trim:delta":
            continue
        results.append(
            {
                "ts": fact.get("ts", ""),
                "event": event,
                "reason": fact.get("reason", ""),
                "tokens_before": fact.get("tokens_before"),
                "tokens_after": fact.get("tokens_after"),
                "removed_sections": fact.get("removed_sections", []),
            }
        )
    return results


__all__ = ["CompressionLanePanel"]
