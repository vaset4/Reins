from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from runtime.ledger import LedgerEvent
from runtime.ledger_writer import CHECKPOINT_SAVED
from runtime.types import TerminalFocusPolicy


@dataclass(frozen=True, slots=True)
class CheckpointProjection:
    """Ledger 派生的 checkpoint 投影。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：各字段为恢复所需 checkpoint 事实
    返回：不可变 checkpoint 投影
    """

    checkpoint_id: str
    task_id: str
    segment_id: str
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

    def matches_task(self, task_id: str) -> bool:
        """判断 checkpoint 是否归属指定任务。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：task_id 为待匹配任务 id
        返回：匹配返回 True，否则返回 False
        """
        return task_id in {
            self.task_id,
            self.focus_task_id,
            self.compatibility_task_id,
        }


@dataclass(frozen=True, slots=True)
class CheckpointView:
    """Ledger 派生的 checkpoint 视图。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：checkpoints 为按保存时间排序的 checkpoint 投影
    返回：不可变 checkpoint 视图
    """

    checkpoints: tuple[CheckpointProjection, ...] = ()

    def latest(self) -> CheckpointProjection | None:
        """读取最新 checkpoint。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：无
        返回：最新 checkpoint 投影，不存在时返回 None
        """
        return self.checkpoints[-1] if self.checkpoints else None

    def for_task(self, task_id: str) -> "CheckpointView":
        """筛选指定任务的 checkpoint。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：task_id 为任务 id
        返回：只包含该任务相关 checkpoint 的新视图
        """
        return CheckpointView(
            tuple(item for item in self.checkpoints if item.matches_task(task_id))
        )

    def by_checkpoint_id(self, checkpoint_id: str) -> CheckpointProjection | None:
        """按 checkpoint id 查找投影。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：checkpoint_id 为 checkpoint id
        返回：匹配的 checkpoint 投影，不存在时返回 None
        """
        matches = [
            item for item in self.checkpoints if item.checkpoint_id == checkpoint_id
        ]
        return matches[-1] if matches else None


def build_checkpoint_view(
    events: Iterable[LedgerEvent | Mapping[str, object]],
) -> CheckpointView:
    """从 Ledger 事件重建 checkpoint 视图。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：events 为 LedgerEvent 或可转换 mapping
    返回：CheckpointView，不读取旧 checkpoint 文件、run facts 或 session state
    """
    checkpoints = [
        projection
        for event in (_coerce_event(raw_event) for raw_event in events)
        if (projection := _projection_from_event(event)) is not None
    ]
    checkpoints.sort(key=lambda item: (item.saved_at, item.checkpoint_id))
    return CheckpointView(tuple(checkpoints))


def _projection_from_event(event: LedgerEvent) -> CheckpointProjection | None:
    """把 checkpoint.saved 事件转成 checkpoint 投影。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：event 为 Ledger 事件
    返回：checkpoint 投影，非 checkpoint 事件返回 None
    """
    if event.event != CHECKPOINT_SAVED:
        return None
    payload = event.payload
    state_payload = payload.get("state", {})
    state_map = state_payload if isinstance(state_payload, Mapping) else {}
    pending = state_map.get("pending_tool_call")
    return CheckpointProjection(
        checkpoint_id=str(payload.get("checkpoint_id", "")),
        task_id=_text(state_map.get("task_id")) or _text(event.task_id),
        segment_id=_text(state_map.get("segment_id")),
        state=_text(state_map.get("state")) or _text(state_payload),
        session_id=_text(state_map.get("session_id")) or _text(event.session_id),
        run_id=_text(state_map.get("run_id")) or _text(event.run_id),
        focus_task_id=_optional_text(state_map.get("focus_task_id")),
        terminal_focus_policy=_terminal_focus_policy(
            state_map.get("terminal_focus_policy")
        ),
        compatibility_task_id=_optional_text(state_map.get("compatibility_task_id")),
        working_memory_snapshot=_dict_field(state_map.get("working_memory_snapshot")),
        pending_tool_call=dict(pending) if isinstance(pending, Mapping) else None,
        lease_snapshot=_dict_field(state_map.get("lease_snapshot")),
        reason=_text(payload.get("reason")),
        saved_at=_text(state_map.get("saved_at")) or event.ts,
    )


def _coerce_event(event: LedgerEvent | Mapping[str, object]) -> LedgerEvent:
    """把 mapping 转换成 LedgerEvent。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：event 为 LedgerEvent 或 mapping
    返回：LedgerEvent
    """
    if isinstance(event, LedgerEvent):
        return event
    return LedgerEvent.from_mapping(event)


def _dict_field(value: object) -> dict[str, object]:
    """读取 dict 字段。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：value 为原始字段值
    返回：dict 字段，非 mapping 返回空 dict
    """
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_text(value: object) -> str | None:
    """读取可空文本字段。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：value 为原始字段值
    返回：非空文本或 None
    """
    text = _text(value)
    return text or None


def _terminal_focus_policy(value: object) -> TerminalFocusPolicy:
    """读取 checkpoint 终态 focus 策略。

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：value 为 Ledger state 中的策略值
    返回：显式策略；旧 checkpoint 缺字段时按 CLEAR 解释
    """
    text = str(value).strip() if value is not None else ""
    return TerminalFocusPolicy(text) if text else TerminalFocusPolicy.CLEAR


def _text(value: object) -> str:
    """读取文本字段。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：value 为原始字段值
    返回：去除首尾空白后的文本
    """
    return "" if value is None else str(value).strip()


__all__ = [
    "CheckpointProjection",
    "CheckpointView",
    "build_checkpoint_view",
]
