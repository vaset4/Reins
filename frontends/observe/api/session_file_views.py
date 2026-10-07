from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runtime.run_evidence import RunEvidenceStore
from runtime.run_facts import RunFactStore

TRACKED_EVENTS = {
    "checkpoint:saved": "Checkpoint",
    "context:built": "上下文构建",
    "context:segments": "上下文分段",
    "run:lifecycle": "生命周期边界",
    "state:transition": "历史状态迁移",
}

STATE_GROUPS = [
    ("identity", "会话身份", ("session_id", "original_user_goal")),
    (
        "latest",
        "最近运行",
        (
            "last_run_id",
            "last_run_status",
            "last_run_event",
            "updated_at",
            "recent_run_ids",
        ),
    ),
    (
        "checkpoint",
        "Checkpoint",
        (
            "last_checkpoint_id",
            "last_checkpoint_state",
            "last_checkpoint_reason",
            "last_checkpoint_at",
        ),
    ),
    ("writeback", "写回目标", ("writeback_targets",)),
    ("debug", "调试计数", ("consecutive_readonly_count", "hint_injection_count")),
    (
        "protocol",
        "协议与任务",
        ("schema_version", "compatibility_task_id", "focus_task_id"),
    ),
]


def build_state_view(
    state: dict[str, Any], coverage: dict[str, tuple[str, str]]
) -> dict[str, Any]:
    groups = [
        _state_group(key, title, fields, state, coverage)
        for key, title, fields in STATE_GROUPS
    ]
    known = {field for _, _, fields in STATE_GROUPS for field in fields}
    extra = sorted(name for name in state if name not in known)
    if extra:
        groups.append(_state_group("other", "其他字段", tuple(extra), state, coverage))
    return {"groups": [group for group in groups if group["fields"]], "raw": state}


def event_totals(runs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        for event, count in run.get("event_counts", {}).items():
            counts[event] = counts.get(event, 0) + int(count)
    return {
        "checkpoint_events": counts.get("checkpoint:saved", 0),
        "lifecycle_events": counts.get("run:lifecycle", 0),
        "legacy_state_transitions": counts.get("state:transition", 0),
        "context_built": counts.get("context:built", 0),
        "context_segments": counts.get("context:segments", 0),
    }


def build_run_files(
    data_root: Path, session_id: str, run_id: str
) -> list[dict[str, Any]]:
    """构建事实与诊断引用列表；传参：根和运行归属；返回：可打开的记录列表。"""
    facts = RunFactStore(data_root).read_session_run(session_id, run_id)
    rows = [
        {
            "name": "运行事实",
            "kind": "事实流",
            "path": f"run:{run_id}",
            "present": bool(facts),
            "bytes": len(json.dumps(facts, ensure_ascii=False).encode()),
        }
    ]
    for item in RunEvidenceStore(data_root).list_records(
        session_id=session_id, run_id=run_id
    ):
        rows.append(
            {
                "name": f"{item['kind']} {item['source_id']}",
                "kind": item["kind"],
                "path": item["reference"],
                "present": True,
                "bytes": len(json.dumps(item["payload"], ensure_ascii=False).encode()),
            }
        )
    return rows


def build_event_track(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_event_row(fact) for fact in facts if fact.get("event") in TRACKED_EVENTS]


def _state_group(
    key: str,
    title: str,
    fields: tuple[str, ...],
    state: dict[str, Any],
    coverage: dict[str, tuple[str, str]],
) -> dict[str, Any]:
    rows = [
        _state_field(name, state[name], coverage) for name in fields if name in state
    ]
    return {"key": key, "title": title, "fields": rows}


def _state_field(
    name: str, value: Any, coverage: dict[str, tuple[str, str]]
) -> dict[str, Any]:
    state_coverage, note = coverage.get(name, ("未分类", "新字段，尚未纳入覆盖说明。"))
    return {
        "name": name,
        "value": value,
        "preview": _preview(value),
        "coverage": state_coverage,
        "note": note,
    }


def _event_row(fact: dict[str, Any]) -> dict[str, Any]:
    event = str(fact.get("event", ""))
    return {
        "event": event,
        "timestamp": str(fact.get("ts", "")),
        "title": TRACKED_EVENTS[event],
        "detail": _event_detail(event, fact),
        "raw": fact,
    }


def _event_detail(event: str, fact: dict[str, Any]) -> str:
    if event == "run:lifecycle":
        return f"{fact.get('lifecycle', '—')} / {fact.get('reason', '—')}"
    if event == "state:transition":
        return f"{fact.get('from_state', '—')} -> {fact.get('to_state', '—')}"
    if event == "checkpoint:saved":
        checkpoint = fact.get("checkpoint")
        if not isinstance(checkpoint, dict):
            return "— / —"
        return f"{checkpoint.get('state', '—')} / {checkpoint.get('reason', '—')}"
    if event == "context:segments":
        raw_segments = fact.get("segments")
        segments = raw_segments if isinstance(raw_segments, list) else []
        return f"{len(segments)} 个分段，约 {_segment_tokens(segments)} tokens"
    if event == "context:built":
        return _preview(fact.get("summary", {}))
    return event


def _segment_tokens(segments: list[Any]) -> int:
    return sum(
        int(seg.get("tokens_est", 0)) for seg in segments if isinstance(seg, dict)
    )


def _preview(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


__all__ = [
    "TRACKED_EVENTS",
    "build_event_track",
    "build_run_files",
    "build_state_view",
    "event_totals",
]
