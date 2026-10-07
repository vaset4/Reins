from __future__ import annotations

from typing import Any

from frontends.observe.panels.base import Panel, RunContext


class TokenAccountantPanel(Panel):
    id = "token_accountant"
    title = "Token Accountant"
    section = "model"
    phase = "B"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        facts = ctx.facts()
        calls = _extract_token_calls(facts)
        budget = _extract_budget(facts)
        totals = _compute_totals(calls)
        return {
            "calls": calls,
            "totals": totals,
            "budget": budget,
        }


def _extract_token_calls(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for fact in facts:
        if fact.get("event") != "llm:response":
            continue
        summary = fact.get("summary", {})
        if not isinstance(summary, dict):
            continue
        obs = summary.get("observation", {})
        if not isinstance(obs, dict):
            continue
        calls.append(
            {
                "ts": fact.get("ts", ""),
                "provider": obs.get("provider", ""),
                "model": obs.get("model", ""),
                "prompt_tokens": _int(obs.get("prompt_tokens", 0)),
                "completion_tokens": _int(obs.get("completion_tokens", 0)),
                "total_tokens": (
                    _int(obs.get("prompt_tokens", 0))
                    + _int(obs.get("completion_tokens", 0))
                ),
                "elapsed_ms": _int(obs.get("elapsed_ms", 0)),
                "attempt_count": _int(obs.get("attempt_count", 1)),
                "stage": obs.get("stage", ""),
            }
        )
    return calls


def _extract_budget(facts: list[dict[str, Any]]) -> dict[str, Any]:
    for fact in facts:
        if fact.get("event") != "run:start":
            continue
        lease = fact.get("lease_summary", {})
        if isinstance(lease, dict):
            return {
                "max_tokens": lease.get("max_tokens"),
                "max_steps": lease.get("max_steps"),
            }
    return {}


def _compute_totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = sum(_int(c.get("prompt_tokens", 0)) for c in calls)
    completion = sum(_int(c.get("completion_tokens", 0)) for c in calls)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "call_count": len(calls),
    }


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


__all__ = ["TokenAccountantPanel"]
