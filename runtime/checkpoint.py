from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping

from runtime.checkpoint_view import (
    CheckpointProjection,
    CheckpointView,
    build_checkpoint_view,
)
from runtime.ledger import LedgerStore
from runtime.types import TerminalFocusPolicy

Idempotent = Literal["yes", "conditional", "no"]


@dataclass(slots=True, kw_only=True)
class Checkpoint:
    # checkpoint 用来恢复 agent loop，不负责回滚文件系统或外部工具副作用。
    # pending_tool_call 必须显式保存，后续 replay/skip/ask 才有依据。
    # working_memory_snapshot 是诊断快照；当前 resume 不会把它恢复成运行时内存态。
    task_id: str
    segment_id: str
    # state 是落盘时的 lifecycle 快照串，值域是多值而非两值：工具子路径写
    # pre_tool/post_tool，_record_lifecycle_boundary 主写入口还会写
    # done/paused/failed/waiting_approval/waiting_user。保持 str 不收窄成
    # Literal，因为生产写入遍布上述 lifecycle 值。
    state: str
    session_id: str = ""
    run_id: str = ""
    focus_task_id: str | None = None
    terminal_focus_policy: TerminalFocusPolicy = TerminalFocusPolicy.CLEAR
    compatibility_task_id: str | None = None
    working_memory_snapshot: dict[str, object] = field(default_factory=dict)
    pending_tool_call: dict[str, object] | None = None
    lease_snapshot: dict[str, object] = field(default_factory=dict)
    reason: str = ""
    saved_at: str = ""
    checkpoint_id: str = ""


@dataclass(frozen=True, slots=True)
class RecoveryPointSummary:
    checkpoint_id: str
    task_id: str
    session_id: str = ""
    run_id: str = ""
    focus_task_id: str | None = None
    compatibility_task_id: str | None = None
    state: str = ""
    reason: str = ""
    saved_at: str = ""


def save_checkpoint(ckpt: Checkpoint) -> Checkpoint:
    """为 Checkpoint 补齐保存身份，不触碰磁盘。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：ckpt 为待保存的 checkpoint 对象
    返回：已补齐 saved_at 和 checkpoint_id 的 checkpoint
    """
    if not ckpt.saved_at:
        ckpt.saved_at = _utc_now()
    if not ckpt.checkpoint_id:
        ckpt.checkpoint_id = _new_checkpoint_id(ckpt.saved_at)
    return ckpt


def list_checkpoints(
    task_id: str,
    *,
    data_root: Path | str | None = None,
) -> list[Checkpoint]:
    return [
        _checkpoint_from_projection(item)
        for item in _checkpoint_view(data_root).for_task(task_id).checkpoints
    ]


def load_latest_checkpoint(
    task_id: str,
    *,
    data_root: Path | str | None = None,
) -> Checkpoint | None:
    checkpoints = list_checkpoints(task_id, data_root=data_root)
    return checkpoints[-1] if checkpoints else None


def load_checkpoint(
    task_id: str,
    checkpoint_id: str,
    *,
    data_root: Path | str | None = None,
) -> Checkpoint | None:
    for checkpoint in list_checkpoints(task_id, data_root=data_root):
        if checkpoint.checkpoint_id == checkpoint_id:
            return checkpoint
    return None


def load_checkpoint_by_id(
    checkpoint_id: str,
    *,
    data_root: Path | str | None = None,
) -> Checkpoint | None:
    projection = _checkpoint_view(data_root).by_checkpoint_id(checkpoint_id)
    return _checkpoint_from_projection(projection) if projection is not None else None


def list_recent_checkpoints(
    *,
    data_root: Path | str | None = None,
    limit: int | None = None,
) -> list[RecoveryPointSummary]:
    checkpoints = [
        _checkpoint_from_projection(item)
        for item in reversed(_checkpoint_view(data_root).checkpoints)
    ]
    summaries = [summarize_checkpoint(item) for item in checkpoints]
    return summaries[:limit] if limit is not None else summaries


def summarize_checkpoint(checkpoint: Checkpoint) -> RecoveryPointSummary:
    return RecoveryPointSummary(
        checkpoint_id=checkpoint.checkpoint_id,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
        focus_task_id=checkpoint.focus_task_id,
        compatibility_task_id=checkpoint.compatibility_task_id,
        state=checkpoint.state,
        reason=checkpoint.reason,
        saved_at=checkpoint.saved_at,
    )


def load_latest_checkpoint_for_session(
    session_id: str,
    *,
    data_root: Path | str | None = None,
) -> Checkpoint | None:
    if not session_id:
        return None
    view = build_checkpoint_view(
        LedgerStore(_data_root(data_root)).read_session_events(session_id)
    )
    latest = view.latest()
    return _checkpoint_from_projection(latest) if latest is not None else None


def load_latest_checkpoint_for_run(
    run_id: str,
    *,
    data_root: Path | str | None = None,
) -> Checkpoint | None:
    if not run_id:
        return None
    view = build_checkpoint_view(
        LedgerStore(_data_root(data_root)).read_run_events(run_id)
    )
    latest = view.latest()
    return _checkpoint_from_projection(latest) if latest is not None else None


def save_pre_tool_checkpoint(
    segment_id: str,
    tool_name: str,
    args: Mapping[str, object],
    call_id: str,
    working_memory: Mapping[str, object],
    *,
    operation_task_id: str | None = None,
    request_id: str = "",
    operation_id: str = "",
    task_id: str,
    session_id: str = "",
    run_id: str = "",
    focus_task_id: str | None = None,
    terminal_focus_policy: TerminalFocusPolicy = TerminalFocusPolicy.CLEAR,
    compatibility_task_id: str | None = None,
    lease_snapshot: Mapping[str, object] | None = None,
) -> Checkpoint:
    """保存执行前检查点与原操作身份；传参：运行、工具及归属；返回：已保存检查点。"""
    pending: dict[str, object] = {
        "tool_name": tool_name,
        "args": dict(args),
        "call_id": call_id,
    }
    if operation_task_id is not None:
        pending["operation_task_id"] = operation_task_id
    if request_id:
        pending["request_id"] = request_id
    if operation_id:
        pending["operation_id"] = operation_id
    return save_checkpoint(
        Checkpoint(
            task_id=task_id,
            segment_id=segment_id,
            state="pre_tool",
            session_id=session_id,
            run_id=run_id,
            focus_task_id=focus_task_id,
            terminal_focus_policy=terminal_focus_policy,
            compatibility_task_id=compatibility_task_id,
            working_memory_snapshot=dict(working_memory),
            pending_tool_call=pending,
            lease_snapshot=dict(lease_snapshot or {}),
            reason="pre_tool",
        )
    )


def save_post_tool_checkpoint(
    segment_id: str,
    working_memory: Mapping[str, object],
    *,
    task_id: str,
    session_id: str = "",
    run_id: str = "",
    focus_task_id: str | None = None,
    terminal_focus_policy: TerminalFocusPolicy = TerminalFocusPolicy.CLEAR,
    compatibility_task_id: str | None = None,
    lease_snapshot: Mapping[str, object] | None = None,
) -> Checkpoint:
    return save_checkpoint(
        Checkpoint(
            task_id=task_id,
            segment_id=segment_id,
            state="post_tool",
            session_id=session_id,
            run_id=run_id,
            focus_task_id=focus_task_id,
            terminal_focus_policy=terminal_focus_policy,
            compatibility_task_id=compatibility_task_id,
            working_memory_snapshot=dict(working_memory),
            lease_snapshot=dict(lease_snapshot or {}),
            reason="post_tool",
        )
    )


def normalize_idempotency(value: object) -> Idempotent | None:
    text = str(getattr(value, "value", value))
    if text == "yes":
        return "yes"
    if text == "conditional":
        return "conditional"
    if text == "no":
        return "no"
    return None


def checkpoint_to_ledger_state(checkpoint: Checkpoint) -> dict[str, object]:
    """把 checkpoint 转成 Ledger state 结构。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：checkpoint 为已保存身份的 checkpoint
    返回：可写入 checkpoint.saved payload 的 state 结构
    """
    return {
        "task_id": checkpoint.task_id,
        "segment_id": checkpoint.segment_id,
        "state": checkpoint.state,
        "session_id": checkpoint.session_id,
        "run_id": checkpoint.run_id,
        "focus_task_id": checkpoint.focus_task_id,
        "terminal_focus_policy": checkpoint.terminal_focus_policy.value,
        "compatibility_task_id": checkpoint.compatibility_task_id,
        "working_memory_snapshot": dict(checkpoint.working_memory_snapshot),
        "pending_tool_call": checkpoint.pending_tool_call,
        "lease_snapshot": dict(checkpoint.lease_snapshot),
        "reason": checkpoint.reason,
        "saved_at": checkpoint.saved_at,
    }


def _checkpoint_view(data_root: Path | str | None) -> CheckpointView:
    return build_checkpoint_view(LedgerStore(_data_root(data_root)).read_events())


def _checkpoint_from_projection(projection: CheckpointProjection) -> Checkpoint:
    return Checkpoint(
        task_id=projection.task_id,
        segment_id=projection.segment_id,
        state=projection.state,
        session_id=projection.session_id,
        run_id=projection.run_id,
        focus_task_id=projection.focus_task_id,
        terminal_focus_policy=projection.terminal_focus_policy,
        compatibility_task_id=projection.compatibility_task_id,
        working_memory_snapshot=dict(projection.working_memory_snapshot),
        pending_tool_call=dict(projection.pending_tool_call)
        if projection.pending_tool_call is not None
        else None,
        lease_snapshot=dict(projection.lease_snapshot),
        reason=projection.reason,
        saved_at=projection.saved_at,
        checkpoint_id=projection.checkpoint_id,
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _new_checkpoint_id(saved_at: str) -> str:
    safe_stamp = saved_at.replace(":", "-")
    return f"{safe_stamp}-{secrets.token_hex(4)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _data_root(data_root: Path | str | None) -> Path:
    return Path(data_root) if data_root is not None else Path.home() / ".reins" / "data"
