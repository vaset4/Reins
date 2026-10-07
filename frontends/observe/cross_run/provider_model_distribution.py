from __future__ import annotations

from typing import Any

from frontends.observe.readers.fact_reader import FactReader


def provider_model_distribution(fact_reader: FactReader) -> dict[str, Any]:
    recent_runs = fact_reader.list_recent_runs(limit=50)
    distribution: dict[str, int] = {}
    for run in recent_runs:
        facts = fact_reader.read_facts(run.run_id)
        for fact in facts:
            if fact.get("event") != "llm:response":
                continue
            obs = (fact.get("summary") or {}).get("observation") or {}
            if not isinstance(obs, dict):
                continue
            provider = obs.get("provider", "unknown")
            model = obs.get("model", "unknown")
            key = f"{provider}/{model}"
            distribution[key] = distribution.get(key, 0) + 1
    sorted_dist = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    return {
        "distribution": [{"model": k, "calls": v} for k, v in sorted_dist],
        "total_calls": sum(distribution.values()),
    }


__all__ = ["provider_model_distribution"]
