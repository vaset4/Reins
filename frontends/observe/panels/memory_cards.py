from __future__ import annotations

from typing import Any

from frontends.observe.panels.base import Panel, RunContext


class MemoryCardsPanel(Panel):
    id = "memory_cards"
    title = "Memory Cards"
    section = "context"
    phase = "B"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        facts = ctx.facts()
        injections = _extract_injections(facts)
        return {"injections": injections, "count": len(injections)}


def _extract_injections(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for fact in facts:
        if fact.get("event") != "memory:injection_explain":
            continue
        detail = fact.get("explain")
        if not isinstance(detail, dict):
            continue
        results.append(
            {
                "ts": fact.get("ts", ""),
                "injected": detail.get("injected", []),
                "skipped": detail.get("skipped", []),
                "round_id": detail.get("round_id", ""),
            }
        )
    return results


__all__ = ["MemoryCardsPanel"]
