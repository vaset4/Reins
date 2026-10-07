from __future__ import annotations

from typing import Any

from frontends.observe.panels.base import Panel, RunContext


class StateHeatmapPanel(Panel):
    id = "state_heatmap"
    title = "Lifecycle Heatmap"
    section = "state"
    phase = "B"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        facts = ctx.facts()
        lifecycles = _extract_lifecycles(facts)
        legacy = _extract_legacy_transitions(facts)
        dwells = _compute_dwells(lifecycles)
        return {
            "lifecycles": lifecycles,
            "legacy_transitions": legacy,
            "dwells": dwells,
        }


def _extract_lifecycles(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for fact in facts:
        if fact.get("event") != "run:lifecycle":
            continue
        results.append(
            {
                "ts": fact.get("ts", ""),
                "lifecycle": fact.get("lifecycle", ""),
                "reason": fact.get("reason", ""),
                "checkpoint_id": fact.get("checkpoint_id", ""),
                "resumable": fact.get("resumable"),
            }
        )
    return results


def _extract_legacy_transitions(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for fact in facts:
        if fact.get("event") != "state:transition":
            continue
        results.append(
            {
                "ts": fact.get("ts", ""),
                "from_state": fact.get("from_state", ""),
                "to_state": fact.get("to_state", ""),
                "close_reason": fact.get("close_reason", ""),
                "terminal_reason": fact.get("terminal_reason", ""),
                "approval_decision": fact.get("approval_decision", ""),
            }
        )
    return results


def _compute_dwells(lifecycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not lifecycles:
        return []
    dwells = []
    for index, lifecycle in enumerate(lifecycles):
        name = lifecycle["lifecycle"]
        start_ts = lifecycle["ts"]
        end_ts = lifecycles[index + 1]["ts"] if index + 1 < len(lifecycles) else ""
        dwells.append({"lifecycle": name, "start": start_ts, "end": end_ts})
    return dwells


__all__ = ["StateHeatmapPanel"]
