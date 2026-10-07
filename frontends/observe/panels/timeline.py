from __future__ import annotations

from typing import Any

from frontends.observe.panels.base import Panel, RunContext


class TimelinePanel(Panel):
    id = "timeline"
    title = "Timeline"
    section = "timeline"
    phase = "B"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        facts = ctx.facts()
        events = []
        for fact in facts:
            events.append(
                {
                    "ts": fact.get("ts", ""),
                    "event": fact.get("event", ""),
                    "segment_id": fact.get("segment_id", ""),
                    "detail": _event_detail(fact),
                }
            )
        return {"events": events, "count": len(events)}


def _event_detail(fact: dict[str, Any]) -> dict[str, Any]:
    event = fact.get("event", "")
    if event == "run:lifecycle":
        return {
            "lifecycle": fact.get("lifecycle", ""),
            "reason": fact.get("reason", ""),
            "checkpoint_id": fact.get("checkpoint_id", ""),
            "resumable": fact.get("resumable"),
        }
    if event == "state:transition":
        return {
            "source": "legacy_state_transition",
            "from_state": fact.get("from_state", ""),
            "to_state": fact.get("to_state", ""),
            "close_reason": fact.get("close_reason", ""),
        }
    if event == "llm:response":
        summary = fact.get("summary", {})
        obs = summary.get("observation", {}) if isinstance(summary, dict) else {}
        return {
            "provider": obs.get("provider", "") if isinstance(obs, dict) else "",
            "model": obs.get("model", "") if isinstance(obs, dict) else "",
            "tokens": _token_summary(obs),
            "has_final": (
                summary.get("has_final") if isinstance(summary, dict) else None
            ),
            "has_run_tools": (
                summary.get("has_run_tools") if isinstance(summary, dict) else None
            ),
        }
    if event == "tool:request":
        tool = fact.get("tool", {})
        return {
            "call_id": tool.get("call_id", "") if isinstance(tool, dict) else "",
            "name": tool.get("name", "") if isinstance(tool, dict) else "",
            "risk": tool.get("risk", "") if isinstance(tool, dict) else "",
        }
    if event == "tool:response":
        tool = fact.get("tool", {})
        return {
            "call_id": tool.get("call_id", "") if isinstance(tool, dict) else "",
            "name": tool.get("name", "") if isinstance(tool, dict) else "",
            "status": tool.get("status", "") if isinstance(tool, dict) else "",
            "error": tool.get("error", "") if isinstance(tool, dict) else "",
        }
    if event == "context:built":
        summary = fact.get("summary", {})
        return {
            "tool_history_count": summary.get("tool_history_count", 0)
            if isinstance(summary, dict)
            else 0,
        }
    if event == "checkpoint:saved":
        cp = fact.get("checkpoint", {})
        return {
            "checkpoint_id": (
                cp.get("checkpoint_id", "") if isinstance(cp, dict) else ""
            ),
            "state": cp.get("state", "") if isinstance(cp, dict) else "",
            "reason": cp.get("reason", "") if isinstance(cp, dict) else "",
        }
    return {}


def _token_summary(obs: Any) -> str:
    if not isinstance(obs, dict):
        return ""
    prompt = obs.get("prompt_tokens", 0) or 0
    completion = obs.get("completion_tokens", 0) or 0
    return f"{prompt}/{completion}"


__all__ = ["TimelinePanel"]
