"""从既有运行与工作记录投影活动，不维护另一份任务状态。

作者：xxx
时间：2026-09-30 12:00:00
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from runtime.collaboration_store import CollaborationStore
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from schedules.occurrences import OccurrenceStore
from tasks.store import TaskStore
from tools.todo_tool import list_todos

if TYPE_CHECKING:
    from app.background.service import BackgroundService


def activity_snapshot(service: BackgroundService, session_id: str) -> dict[str, Any]:
    """读取所选会话工作及后台执行概况；参数：真实宿主与会话；返回：只读活动视图。"""
    root = service.services.data_root
    session = service.attach(session_id)
    current = session.snapshot()
    status = service.status()
    root_run_id = current["root_run_id"]
    # 【后台活动】【来源隔离】1. 只展示所选运行所属协作，历史运行与其他会话的成员不混入
    members = _members(root, session_id, {current["current_run_id"], root_run_id})
    task_id = current["current_task_id"] or current["compatibility_task_id"]
    task = None
    if task_id:
        with closing(TaskStore(root)) as tasks:
            record = tasks.require_task(task_id)
        task = {
            "task_id": task_id,
            "goal": record.goal,
            "status": record.status,
            "todos": [asdict(item) for item in list_todos(task_id, data_root=root)],
        }
    # 【后台活动】【执行概况】2. 列出实际持有者与尚待执行的发生，不生成新的周期任务面板
    work = [
        _run_view(row)
        for row in status["sessions"]
        if row["active"] or row["session_id"] == session_id
    ]
    running = set(status["running_occurrences"])
    for occurrence in OccurrenceStore(root).list_all():
        if occurrence.occurrence_id not in running and occurrence.status not in {
            "queued",
            "resume_queued",
        }:
            continue
        if not occurrence.session_id:
            continue
        work.append(
            {
                "session_id": occurrence.session_id,
                "run_id": occurrence.run_id,
                "status": occurrence.result_status or occurrence.status,
                "active": occurrence.occurrence_id in running,
                "scheduled": True,
                "title": occurrence.schedule_snapshot.get("name")
                or occurrence.schedule_id,
            }
        )
    return {
        "session_id": session_id,
        "current": _run_view(current),
        "task": task,
        "members": members,
        "work": work,
        "host_status": status["status"],
        "errors": status["errors"],
        "unread_notifications": status["unread_notifications"],
    }


def _members(root: Path, session_id: str, run_ids: set[str]) -> list[dict[str, Any]]:
    """按共同运行归属读取子执行成果；参数：数据根、整合会话与运行集合；返回：成员及真实结果。"""
    records = CollaborationStore(SessionMessageStore(root), session_id).read()[
        "members"
    ]
    facts = RunFactStore(root)
    result = []
    for member in records.values():
        if member.get("budget_run_id") not in run_ids:
            continue
        rows = (
            facts.read_session_run(member["session_id"], member["run_id"])
            if member["run_id"]
            else []
        )
        finished = next(
            (row for row in reversed(rows) if row.get("event") == "agent:finished"),
            None,
        )
        result.append(
            {
                key: member[key]
                for key in (
                    "agent_id",
                    "name",
                    "session_id",
                    "run_id",
                    "task",
                    "backend",
                )
            }
            | {
                "status": str(finished["status"])
                if finished
                else "unknown"
                if member["run_id"]
                else "accepted",
                "output": str(finished.get("output", "")) if finished else "",
                "cancel_requested": bool(member.get("cancel_operation_id")),
            }
        )
    return result


def _run_view(snapshot: dict[str, Any]) -> dict[str, Any]:
    """保留显示与控制需要的运行身份；参数：宿主快照；返回：无正文的状态。"""
    return {
        "session_id": snapshot["session_id"],
        "run_id": snapshot["current_run_id"],
        "status": snapshot["status"],
        "active": snapshot["active"],
        "stopped": snapshot["stopped"],
        "error": snapshot["error"],
        "scheduled": bool(snapshot.get("occurrence_id")),
        "title": snapshot["session_id"],
    }
