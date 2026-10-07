from __future__ import annotations

from typing import Any

from frontends.shared.run_lifecycle import build_lifecycle_observation
from frontends.observe.panels.base import Panel, RunContext


class DemoStoryPanel(Panel):
    """Five-step story for Demo Mode presentation.

    Steps: User Intent -> Context Assembly -> Model Decision ->
    Tool Execution -> Outcome & Recovery.
    """

    id = "demo_story"
    title = "Demo Story"
    section = "demo"
    phase = "B"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        facts = ctx.facts()
        return {
            "steps": [
                _user_intent(facts),
                _context_assembly(facts),
                _model_decision(facts),
                _tool_execution(facts),
                _outcome_recovery(facts),
            ]
        }


def _user_intent(facts: list[dict[str, Any]]) -> dict[str, Any]:
    for fact in facts:
        if fact.get("event") == "run:start":
            return {
                "step": 1,
                "title": "User Intent",
                "trigger": fact.get("trigger", ""),
                "task_id": fact.get("task_id", ""),
                "focus_task_id": fact.get("focus_task_id", ""),
                "ts": fact.get("ts", ""),
            }
    return {"step": 1, "title": "User Intent", "trigger": "(not found)"}


def _context_assembly(facts: list[dict[str, Any]]) -> dict[str, Any]:
    context_events = [f for f in facts if f.get("event") == "context:built"]
    return {
        "step": 2,
        "title": "Context Assembly",
        "context_builds": len(context_events),
        "last_tool_history": _last_tool_history(context_events),
    }


def _model_decision(facts: list[dict[str, Any]]) -> dict[str, Any]:
    llm_events = [f for f in facts if f.get("event") == "llm:response"]
    if not llm_events:
        return {"step": 3, "title": "Model Decision", "calls": 0}
    last = llm_events[-1]
    summary = last.get("summary", {})
    obs = summary.get("observation", {}) if isinstance(summary, dict) else {}
    return {
        "step": 3,
        "title": "Model Decision",
        "calls": len(llm_events),
        "provider": obs.get("provider", "") if isinstance(obs, dict) else "",
        "model": obs.get("model", "") if isinstance(obs, dict) else "",
        "has_final": summary.get("has_final") if isinstance(summary, dict) else None,
        "has_run_tools": (
            summary.get("has_run_tools") if isinstance(summary, dict) else None
        ),
    }


def _tool_execution(facts: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [f for f in facts if f.get("event") == "tool:request"]
    responses = [f for f in facts if f.get("event") == "tool:response"]
    tools_used = set()
    for f in requests:
        tool = f.get("tool", {})
        if isinstance(tool, dict):
            name = tool.get("name", "")
            if name:
                tools_used.add(name)
    return {
        "step": 4,
        "title": "Tool Execution",
        "requests": len(requests),
        "responses": len(responses),
        "tools_used": sorted(tools_used),
    }


def _outcome_recovery(facts: list[dict[str, Any]]) -> dict[str, Any]:
    lifecycle = build_lifecycle_observation(facts)
    if lifecycle.lifecycle:
        return {
            "step": 5,
            "title": "Outcome & Recovery",
            "status": lifecycle.lifecycle,
            "reason": lifecycle.reason,
            "checkpoint_id": lifecycle.checkpoint_id,
            "source": lifecycle.source,
            "ts": lifecycle.ts,
        }
    terminal = None
    for fact in reversed(facts):
        if fact.get("event") == "state:transition":
            to_state = fact.get("to_state", "")
            if to_state in {"DONE", "PAUSED", "FAILED"}:
                terminal = fact
                break
    if terminal is None:
        return {"step": 5, "title": "Outcome & Recovery", "status": "running"}
    return {
        "step": 5,
        "title": "Outcome & Recovery",
        "status": terminal.get("to_state", "").lower(),
        "close_reason": terminal.get("close_reason", ""),
        "terminal_reason": terminal.get("terminal_reason", ""),
        "source": "legacy_state_transition",
        "ts": terminal.get("ts", ""),
    }


def _last_tool_history(context_events: list[dict[str, Any]]) -> int:
    if not context_events:
        return 0
    last = context_events[-1]
    summary = last.get("summary", {})
    if isinstance(summary, dict):
        count = summary.get("tool_history_count", 0)
        if isinstance(count, int):
            return count
    return 0


__all__ = ["DemoStoryPanel"]
