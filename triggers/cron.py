from __future__ import annotations

from pathlib import Path
from typing import Any
from collections.abc import Mapping
from contextlib import closing
from dataclasses import asdict

from runtime.lease import Lease, from_trigger
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from schedules.store import (
    DEFAULT_SCHEDULE_STEPS,
    DEFAULT_SCHEDULE_TOKENS,
    ScheduleStore,
)
from tasks.ids import new_ulid


def make_run_context(
    schedule_id: str,
    *,
    data_root: Path | str,
    session_id: str = "",
    run_id: str = "",
    compatibility_task_id: str | None = None,
    schedule_data: dict[str, Any] | None = None,
) -> RunContext:
    """从持久计划构造CRON身份和原授权范围；传参：计划、存储和运行身份；返回：统一上下文。"""
    data = schedule_data
    if data is None:
        with closing(ScheduleStore(data_root)) as store:
            record = store.load_schedule(schedule_id)
        if record is None:
            raise FileNotFoundError(schedule_id)
        data = asdict(record)
    if not isinstance(data, dict):
        raise ValueError(f"invalid schedule: {schedule_id}")
    if not bool(data.get("enabled", True)):
        raise ValueError(f"disabled schedule: {schedule_id}")
    if "required_permanent_grants" not in data:
        raise ValueError("required_permanent_grants missing")
    task_id = data.get("target_task_id")
    if task_id is None and compatibility_task_id is None:
        raise ValueError("target_task_id missing")
    task_id_text = str(task_id) if task_id is not None else None
    workspace_id = data.get("workspace_id")
    if not isinstance(workspace_id, str) or not workspace_id:
        raise ValueError("schedule workspace_id is missing or invalid")
    workspaces = WorkspaceStore(data_root)
    workspace = workspaces.get(workspace_id)
    lease = _cron_lease(task_id_text or compatibility_task_id or "", data)
    if session_id:
        workspaces.bind_session(session_id, workspace.project_root)
    return RunContext(
        session_id=session_id,
        run_id=run_id,
        task_id=task_id_text,
        focus_task_id=task_id_text,
        focus_task={"task_id": task_id_text} if task_id_text else {},
        compatibility_task_id=compatibility_task_id,
        trigger=Trigger.CRON,
        payload={
            "schedule_id": schedule_id,
            "cron": str(data["cron"]),
            "workspace_id": workspace_id,
        },
        capability_lease=lease,
        segment_id=f"cron-{new_ulid()}",
    )


def _cron_lease(task_id: str, data: Mapping[str, Any]) -> Lease:
    """沿用计划接纳时的能力，不把定时触发转换成用户授权；传参：目标与计划；返回：本次租约。"""
    raw = data.get("capabilities")
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("schedule capabilities must be an object")
    max_steps, max_tokens = (
        data.get("max_steps", DEFAULT_SCHEDULE_STEPS),
        data.get("max_tokens", DEFAULT_SCHEDULE_TOKENS),
    )
    if (
        not isinstance(max_steps, int)
        or isinstance(max_steps, bool)
        or max_steps <= 0
        or not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens <= 0
    ):
        raise ValueError("schedule run budgets must be positive integers")
    lease = from_trigger(
        "cron",
        task_id=task_id,
        capabilities=raw,
        max_steps=max_steps,
        max_tokens=max_tokens,
    )
    capabilities = dict(lease.capabilities)
    capabilities["schedule"] = {
        "required_permanent_grants": _grant_list(data.get("required_permanent_grants"))
    }
    return from_trigger(
        "cron",
        task_id=task_id,
        capabilities=capabilities,
        max_steps=max_steps,
        max_tokens=max_tokens,
    )


def _grant_list(value: object) -> list[dict[str, object]]:
    """严格读取预声明授权，不把损坏授权悄悄变为空列表；传参：持久授权；返回：独立副本。"""
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, dict) for item in value
    ):
        raise ValueError("required_permanent_grants must be a list of objects")
    return [dict(item) for item in value]
