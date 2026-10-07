from __future__ import annotations

from typing import Any

from frontends.observe.panels.base import Panel, RunContext


class ErrorCardsPanel(Panel):
    id = "error_cards"
    title = "Error Cards"
    section = "errors"
    phase = "B"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        errors = ctx.errors()
        facts = ctx.facts()
        fact_errors = _extract_fact_errors(facts)
        grouped = _group_by_category(errors + fact_errors)
        return {"groups": grouped, "total": len(errors) + len(fact_errors)}


def _extract_fact_errors(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for fact in facts:
        if fact.get("event") != "llm:response":
            continue
        summary = fact.get("summary", {})
        if not isinstance(summary, dict):
            continue
        error = summary.get("error")
        if not error or not isinstance(error, dict):
            continue
        results.append(
            {
                "ts": fact.get("ts", ""),
                "category": error.get("category", "unknown"),
                "message": error.get("message", ""),
                "recoverable": error.get("recoverable", True),
            }
        )
    return results


def _group_by_category(errors: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for error in errors:
        cat = error.get("category", "unknown") or "unknown"
        groups.setdefault(cat, []).append(error)
    return groups


__all__ = ["ErrorCardsPanel"]
