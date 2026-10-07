from __future__ import annotations

from typing import Any

from frontends.observe.panels.base import Panel, RunContext


class SkillActivationPanel(Panel):
    id = "skill_activation"
    title = "Skill Activation"
    section = "context"
    phase = "C"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        facts = ctx.facts()
        activations = _extract_activations(facts)
        return {"activations": activations, "count": len(activations)}


def _extract_activations(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for fact in facts:
        if fact.get("event") != "skill:activation":
            continue
        skills = fact.get("skills", [])
        results.append(
            {
                "ts": fact.get("ts", ""),
                "round_id": fact.get("round_id", ""),
                "skills": skills if isinstance(skills, list) else [],
            }
        )
    return results


__all__ = ["SkillActivationPanel"]
