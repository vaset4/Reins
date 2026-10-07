from __future__ import annotations

from pathlib import Path
from typing import Any

from frontends.observe.readers.fact_reader import FactReader
from runtime.run_evidence import RunEvidenceStore


def failure_pivot(fact_reader: FactReader, data_root: Path) -> dict[str, Any]:
    """按失败运行统计真实错误；参数：事实读者与数据根；返回：错误类别和阶段计数。"""
    recent_runs = fact_reader.list_recent_runs(limit=50)
    pivot: dict[str, dict[str, int]] = {}
    for run in recent_runs:
        if run.status != "failed":
            continue
        errors = _read_errors(data_root, run.session_id, run.run_id)
        facts = fact_reader.read_facts(run.run_id)
        stage = _last_stage(facts)
        for error in errors:
            category = error.get("category", "unknown")
            pivot.setdefault(category, {})
            pivot[category][stage] = pivot[category].get(stage, 0) + 1
    rows = []
    for category, stages in sorted(pivot.items()):
        for stage, count in sorted(stages.items()):
            rows.append({"category": category, "stage": stage, "count": count})
    return {"pivot": rows, "total_failures": sum(r["count"] for r in rows)}


def _read_errors(data_root: Path, session_id: str, run_id: str) -> list[dict[str, Any]]:
    """读取同一规范错误源；参数：运行归属；返回：已提交错误，存储故障直接报告。"""
    return [
        row["payload"]
        for row in RunEvidenceStore(data_root).list_records(
            session_id=session_id, run_id=run_id, kind="error"
        )
    ]


def _last_stage(facts: list[dict[str, Any]]) -> str:
    """读取最后模型调用阶段；参数：有序事实；返回：已记录阶段或明确未知。"""
    for fact in reversed(facts):
        if fact.get("event") == "llm:response":
            obs = (fact.get("summary") or {}).get("observation") or {}
            if isinstance(obs, dict):
                return str(obs.get("stage", "unknown"))
    return "unknown"


__all__ = ["failure_pivot"]
