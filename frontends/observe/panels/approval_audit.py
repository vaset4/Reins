from __future__ import annotations

from typing import Any

from frontends.observe.panels.base import Panel, RunContext


class ApprovalAuditPanel(Panel):
    id = "approval_audit"
    title = "Approval Audit"
    section = "tools"
    phase = "B"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        facts = ctx.facts()
        decisions = _extract_approval_decisions(facts)
        return {"decisions": decisions, "count": len(decisions)}


def _extract_approval_decisions(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for fact in facts:
        event = fact.get("event")
        if event == "run:lifecycle" and fact.get("lifecycle") == "waiting_approval":
            results.append(
                {
                    "ts": fact.get("ts", ""),
                    "source": "run:lifecycle",
                    "lifecycle": fact.get("lifecycle", ""),
                    "reason": fact.get("reason", ""),
                    "checkpoint_id": fact.get("checkpoint_id", ""),
                }
            )
            continue
        if event == "approval:required":
            results.append(
                {
                    "ts": fact.get("ts", ""),
                    "source": "approval:required",
                    "tool": fact.get("tool", {}),
                }
            )
            continue
        if event != "state:transition":
            continue
        decision = fact.get("approval_decision")
        if not decision:
            continue
        results.append(
            {
                "ts": fact.get("ts", ""),
                "source": "legacy_state_transition",
                "from_state": fact.get("from_state", ""),
                "to_state": fact.get("to_state", ""),
                "decision": decision,
            }
        )
    return results


__all__ = ["ApprovalAuditPanel"]
