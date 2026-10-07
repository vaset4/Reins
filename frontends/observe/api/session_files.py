from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from frontends.observe.api.session_file_views import (
    build_event_track,
    build_run_files,
    build_state_view,
    event_totals,
)
from runtime.run_facts import RunFactStore
from runtime.persistence import RuntimeStore
from runtime.run_evidence import RunEvidenceStore
from runtime.session_state import SessionStateStore

SESSION_STATE_COVERAGE = {
    "session_id": ("部分展示", "用于会话选择与故事元信息。"),
    "summary": ("部分展示", "作为最近写回摘要展示，但不是完整会话转录。"),
    "last_run_id": ("部分展示", "通过运行列表和默认选中运行间接体现。"),
    "last_run_status": ("部分展示", "作为会话状态展示。"),
    "updated_at": ("部分展示", "作为更新时间展示。"),
    "recent_run_ids": ("部分展示", "用于恢复运行顺序，但没有作为状态字段单独解释。"),
    "original_user_goal": (
        "部分展示",
        "每个 run 会显示自己的用户问题，会话级目标未单独成块。",
    ),
    "compatibility_task_id": ("未展示", "兼容任务关联没有在故事页中解释。"),
    "focus_task_id": ("未展示", "焦点任务字段进入 API，但前端没有一眼展示。"),
    "writeback_targets": ("未展示", "写回目标只留在原始状态证据中。"),
    "consecutive_readonly_count": ("未展示", "只适合调试只读循环。"),
    "hint_injection_count": ("未展示", "只适合诊断提示注入。"),
    "last_checkpoint_at": ("一等展示", "Checkpoint 卡片展示最近恢复点时间。"),
    "last_checkpoint_id": ("一等展示", "Checkpoint 卡片展示最近恢复点 ID。"),
    "last_checkpoint_reason": ("一等展示", "Checkpoint 卡片展示最近恢复原因。"),
    "last_checkpoint_state": ("一等展示", "Checkpoint 卡片展示最近恢复状态。"),
    "last_run_event": ("未展示", "最后事件只在原始时间线里出现。"),
    "schema_version": ("未展示", "属于存储协议元信息。"),
}

EVENT_COVERAGE = {
    "checkpoint:saved": "在运行事实流和 story.checkpoints 中展示",
    "context:built": "统计 context build 次数并进入 story.context",
    "context:segments": "展示当前 run 的最新分段快照并进入 story.context",
    "run:lifecycle": "作为当前公共生命周期边界展示",
    "state:transition": "仅作为历史状态迁移兼容展示",
}


def build_session_file_inventory(data_root: Path, session_id: str) -> dict[str, Any]:
    """查询会话结构化记录及诊断引用；传参：根与会话；返回：证据清单。"""
    if not session_id or any(char in session_id for char in "/\\:"):
        return {"status": "rejected", "session_id": session_id}
    state_record = SessionStateStore(data_root).load(session_id)
    facts = RunFactStore(data_root)
    summaries = facts.list_runs_for_session(session_id)
    if state_record is None and not summaries:
        return {"status": "missing", "session_id": session_id}
    state = asdict(state_record) if state_record else {"session_id": session_id}
    runs = [_run_inventory(data_root, session_id, row.run_id) for row in summaries]
    summary = {
        "top_level_files": 0,
        "runs": len(runs),
        "facts": sum(row["facts"]["lines"] for row in runs),
        "raw_files": sum(row["raw"]["count"] for row in runs),
        "error_files": sum(row["errors"]["lines"] for row in runs),
    }
    summary.update(event_totals(runs))
    store = RuntimeStore(data_root)
    source_root = (
        store.session_directory(session_id).relative_to(store.data_root).as_posix()
    )
    return {
        "status": "ok",
        "session_id": session_id,
        "root": source_root,
        "storage": "files",
        "summary": summary,
        "summary_file": {
            "path": f"session:{session_id}",
            "present": bool(state.get("summary")),
            "text": state.get("summary", ""),
            "bytes": len(str(state.get("summary", "")).encode()),
        },
        "top_level": [],
        "state": build_state_view(state, SESSION_STATE_COVERAGE),
        "state_fields": [
            {
                "name": key,
                "coverage": SESSION_STATE_COVERAGE.get(key, ("可查询", ""))[0],
                "note": SESSION_STATE_COVERAGE.get(key, ("", ""))[1],
            }
            for key in sorted(state)
        ],
        "runs": runs,
        "not_displayed": [],
    }


def _run_inventory(data_root: Path, session_id: str, run_id: str) -> dict[str, Any]:
    """聚合一个运行的事实与证据；传参：根及归属；返回：清单视图。"""
    facts = RunFactStore(data_root).read_session_run(session_id, run_id)
    records = RunEvidenceStore(data_root).list_records(
        session_id=session_id, run_id=run_id
    )
    errors = [row for row in records if row["kind"] == "error"]
    counts: dict[str, int] = {}
    for fact in facts:
        event = str(fact.get("event", ""))
        counts[event] = counts.get(event, 0) + 1
    files = build_run_files(data_root, session_id, run_id)
    return {
        "run_id": run_id,
        "facts": {
            "path": f"run:{run_id}",
            "present": bool(facts),
            "lines": len(facts),
            "coverage": "运行区展示",
            "note": "文件中的运行事实，按提交顺序读取",
        },
        "errors": {
            "path": errors[-1]["reference"] if errors else "",
            "present": bool(errors),
            "lines": len(errors),
            "coverage": "调试展示",
            "note": "运行错误记录",
        },
        "raw": {
            "count": len(records),
            "coverage": "运行区展示",
            "note": "诊断实体引用，完整正文按内容复用",
            "files": files[1:],
        },
        "files": files,
        "event_counts": counts,
        "event_track": build_event_track(facts),
        "not_first_class": [],
    }


__all__ = ["build_session_file_inventory"]
