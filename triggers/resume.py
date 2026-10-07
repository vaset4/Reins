from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping

from runtime.checkpoint import (
    Checkpoint,
    load_latest_checkpoint,
    load_latest_checkpoint_for_run,
    load_latest_checkpoint_for_session,
)
from runtime.lease import from_trigger
from runtime.types import RunContext, Trigger
from tasks.ids import new_ulid


def make_run_context(
    task_id: str | None = None,
    *,
    data_root: Path | str,
    session_id: str = "",
    run_id: str = "",
    source_run_id: str = "",
    checkpoint: Checkpoint | None = None,
) -> RunContext:
    """从任务身份或已解析 checkpoint 构造统一 resume 上下文

    参数：task_id/source_run_id/session_id 为兼容查找入口；checkpoint 为精确恢复点；其余参数指定运行身份和数据根
    返回：携带 fresh resume lease 与来源 parent segment 的 RunContext
    """
    resolved = checkpoint
    if resolved is None and task_id is not None:
        resolved = load_latest_checkpoint(task_id, data_root=data_root)
    elif resolved is None and source_run_id:
        resolved = load_latest_checkpoint_for_run(source_run_id, data_root=data_root)
    elif resolved is None:
        resolved = load_latest_checkpoint_for_session(session_id, data_root=data_root)
    if resolved is None:
        raise FileNotFoundError(task_id or source_run_id or session_id)
    compatibility_task_id = resolved.compatibility_task_id
    is_compatibility_checkpoint = compatibility_task_id == resolved.task_id
    focus_task_id = resolved.focus_task_id
    if focus_task_id is None and not is_compatibility_checkpoint:
        focus_task_id = resolved.task_id
    formal_task_id = task_id
    if formal_task_id is None and not is_compatibility_checkpoint:
        formal_task_id = resolved.task_id
    lease_task_id = formal_task_id or focus_task_id or compatibility_task_id or ""
    capabilities = (
        resolved.lease_snapshot.get("capabilities") if resolved.lease_snapshot else None
    )
    if capabilities is not None and not isinstance(capabilities, Mapping):
        raise ValueError("checkpoint capabilities must be an object")
    return RunContext(
        session_id=session_id or resolved.session_id,
        run_id=run_id,
        task_id=formal_task_id,
        focus_task_id=focus_task_id,
        focus_task={"task_id": focus_task_id} if focus_task_id is not None else {},
        terminal_focus_policy=resolved.terminal_focus_policy,
        compatibility_task_id=compatibility_task_id,
        trigger=Trigger.RESUME,
        payload={
            "checkpoint_id": resolved.checkpoint_id,
            "checkpoint_state": resolved.state,
            "pending_tool_call": resolved.pending_tool_call,
            "terminal_focus_policy": resolved.terminal_focus_policy.value,
            "previous_run_id": resolved.run_id,
            "compatibility_task_id": compatibility_task_id,
        },
        # 【运行恢复】【授权范围】沿用已记录的能力范围，预算与到期时间属于这次新运行
        capability_lease=from_trigger(
            "resume", task_id=lease_task_id, capabilities=capabilities
        ),
        segment_id=f"resume-{new_ulid()}",
        parent_segment_id=resolved.segment_id,
    )
