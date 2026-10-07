from __future__ import annotations

from typing import Any, Mapping

from frontends.observe.panels.base import Panel, RunContext


class ResponseAnatomyPanel(Panel):
    id = "response_anatomy"
    title = "Response Anatomy"
    section = "model"
    phase = "B"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        response_ev = ctx.evidence("model_response")
        parsed_ev = ctx.evidence("parsed_plan")
        return {
            "response": _summarize_response(response_ev),
            "parsed_plan": _summarize_parsed(parsed_ev),
        }


def _summarize_response(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    if evidence is None:
        return {"status": "not_available"}
    response = evidence.get("response", {})
    if not isinstance(response, dict):
        return {"status": "malformed"}
    tool_calls_raw = response.get("tool_calls", [])
    tool_calls_list = tool_calls_raw if isinstance(tool_calls_raw, list) else []
    return {
        "status": "ok",
        "ok": response.get("ok"),
        "text_chars": len(str(response.get("text", "") or "")),
        "tool_calls": len(tool_calls_list),
        "error_message": response.get("error_message") or None,
        "finish_reason": response.get("finish_reason", ""),
    }


def _summarize_parsed(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    if evidence is None:
        return {"status": "not_available"}
    return {
        "status": "ok",
        "success": evidence.get("success"),
        "has_final": evidence.get("has_final"),
        "has_run_tools": evidence.get("has_run_tools"),
        "tool_name": _extract_tool_name(evidence),
    }


def _extract_tool_name(evidence: Mapping[str, Any]) -> str:
    request = evidence.get("run_tools_request", {})
    if not isinstance(request, dict):
        return ""
    return str(request.get("tool_name") or request.get("action") or "")


__all__ = ["ResponseAnatomyPanel"]
