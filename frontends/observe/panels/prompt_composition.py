from __future__ import annotations

from typing import Any, Mapping

from frontends.observe.panels.base import Panel, RunContext


class PromptCompositionPanel(Panel):
    id = "prompt_composition"
    title = "Prompt Composition"
    section = "model"
    phase = "B"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        evidence = ctx.evidence("model_request")
        if evidence is None:
            return {"status": "not_available", "messages": [], "context": {}}
        return _summarize_request(evidence)


def _summarize_request(evidence: Mapping[str, Any]) -> dict[str, Any]:
    request = evidence.get("request", {})
    messages = request.get("messages", []) if isinstance(request, dict) else []
    role_counts: dict[str, int] = {}
    for msg in messages if isinstance(messages, list) else []:
        role = msg.get("role", "unknown") if isinstance(msg, dict) else "unknown"
        role_counts[role] = role_counts.get(role, 0) + 1
    context = evidence.get("prompt_context", {})
    return {
        "status": "ok",
        "protocol_mode": evidence.get("protocol_mode", ""),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "role_counts": role_counts,
        "context": {
            "stage": (context.get("stage", "") if isinstance(context, dict) else ""),
            "system_reminder": (
                context.get("system_reminder", "") if isinstance(context, dict) else ""
            ),
        },
        "render_text_chars": len(str(evidence.get("render_text_to_model", ""))),
    }


__all__ = ["PromptCompositionPanel"]
