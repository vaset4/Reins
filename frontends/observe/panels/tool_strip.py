from __future__ import annotations

from typing import Any

from frontends.observe.panels.base import Panel, RunContext


class ToolStripPanel(Panel):
    id = "tool_strip"
    title = "Tool Strip"
    section = "tools"
    phase = "B"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        facts = ctx.facts()
        calls = _build_call_pairs(facts)
        return {"calls": calls, "count": len(calls)}


def _build_call_pairs(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requests: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    for fact in facts:
        event = fact.get("event", "")
        tool = fact.get("tool", {})
        if not isinstance(tool, dict):
            continue
        call_id = str(tool.get("call_id", ""))
        if event == "tool:request":
            request_entry = {
                "call_id": call_id,
                "name": tool.get("name", ""),
                "risk": tool.get("risk", ""),
                "args_summary": tool.get("args_summary", {}),
                "request_ts": fact.get("ts", ""),
                "response_ts": "",
                "status": "pending",
                "error": "",
                "output_summary": "",
                "artifact_refs": [],
            }
            requests[call_id] = request_entry
            pairs.append(request_entry)
        elif event == "tool:response":
            entry = requests.get(call_id)
            if entry is None:
                entry = {
                    "call_id": call_id,
                    "name": tool.get("name", ""),
                    "risk": "",
                    "args_summary": {},
                    "request_ts": "",
                    "response_ts": fact.get("ts", ""),
                    "status": tool.get("status", ""),
                    "error": tool.get("error", ""),
                    "output_summary": tool.get("output_summary", ""),
                    "artifact_refs": tool.get("artifact_refs", []),
                }
                pairs.append(entry)
            else:
                entry["response_ts"] = fact.get("ts", "")
                entry["status"] = tool.get("status", "")
                entry["error"] = tool.get("error", "")
                entry["output_summary"] = tool.get("output_summary", "")
                entry["artifact_refs"] = tool.get("artifact_refs", [])
    return pairs


__all__ = ["ToolStripPanel"]
