from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from runtime.ledger import LedgerEvent
from runtime.ledger_writer import (
    CONTEXT_SEGMENTS_RECORDED,
    MODEL_REQUESTED,
    RUN_LIFECYCLE_CHANGED,
    TOOL_COMPLETED,
    TOOL_REQUESTED,
)


@dataclass(frozen=True, slots=True)
class RunEvidenceView:
    """Ledger 派生的运行证据视图。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：各字段为从 Ledger 事件重建出的运行证据
    返回：不可变运行证据视图

    **别因为「生产上没人调它」就删它**（2026-08-07 核实留注）。生产读取面确实
    都走 `RunFactStore`——observe/TUI、REPL dashboard、slash commands 读旧链是
    `docs/context-management/04-phase3-ledger-view-closure.md:75` 明确允许保留的
    形态，不是漏接线。但本视图有两个测试消费方，且是当前唯一的两道守卫：

    - `tests/test_run_evidence.py` 跑完整循环后从 Ledger 重建，逐类断言
      model_requests / context_segments / tool_events / lifecycle_events
      都真写进了 Ledger —— 「Ledger 写入正确」的唯一守卫。
    - `tests/test_audit_runtime_repro_harnesses.py` 拿本视图的 context_segments
      与 `RunFactStore` 同名字段逐字段比对 —— 「Ledger 与 run facts 双轨没漂」
      的唯一守卫。后者在两条轨道并行双写的当下尤其要紧。

    `MODEL_REQUESTED` / `CONTEXT_SEGMENTS_RECORDED` / `RUN_LIFECYCLE_CHANGED`
    三类事件目前只被本视图投影，故生产运行时无人读回；删掉本视图会连带废掉
    上面两道断言。
    """

    run_id: str = ""
    model_requests: tuple[dict[str, object], ...] = field(default_factory=tuple)
    tool_events: tuple[dict[str, object], ...] = field(default_factory=tuple)
    context_segments: tuple[dict[str, object], ...] = field(default_factory=tuple)
    lifecycle_events: tuple[dict[str, object], ...] = field(default_factory=tuple)
    event_count: int = 0
    seen_event_ids: tuple[str, ...] = field(default_factory=tuple)


def build_run_evidence_view(
    events: Iterable[LedgerEvent | Mapping[str, object]],
) -> RunEvidenceView:
    """从 Ledger 事件重建运行证据视图。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：events 为 LedgerEvent 或可转换 mapping
    返回：RunEvidenceView，不读取旧 run facts
    """
    builder = _RunEvidenceBuilder()
    for raw_event in events:
        builder.apply(_coerce_event(raw_event))
    return builder.to_view()


@dataclass(slots=True)
class _RunEvidenceBuilder:
    run_id: str = ""
    model_requests: list[dict[str, object]] = field(default_factory=list)
    tool_events: list[dict[str, object]] = field(default_factory=list)
    context_segments: list[dict[str, object]] = field(default_factory=list)
    lifecycle_events: list[dict[str, object]] = field(default_factory=list)
    event_count: int = 0
    seen_event_ids: list[str] = field(default_factory=list)

    def apply(self, event: LedgerEvent) -> None:
        """应用单个 Ledger 事件。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：event 为 LedgerEvent
        返回：无
        """
        self.event_count += 1
        self.seen_event_ids.append(event.event_id)
        if self.run_id == "" and event.run_id is not None:
            self.run_id = event.run_id
        row = _event_row(event)
        if event.event == MODEL_REQUESTED:
            self.model_requests.append(row)
        elif event.event in {TOOL_REQUESTED, TOOL_COMPLETED}:
            self.tool_events.append(row)
        elif event.event == CONTEXT_SEGMENTS_RECORDED:
            self.context_segments.append(row)
        elif event.event == RUN_LIFECYCLE_CHANGED:
            self.lifecycle_events.append(row)

    def to_view(self) -> RunEvidenceView:
        """生成不可变运行证据视图。

        作者：LKX
        时间：2026-07-02 00:00:00
        传参：无
        返回：RunEvidenceView
        """
        return RunEvidenceView(
            run_id=self.run_id,
            model_requests=tuple(self.model_requests),
            tool_events=tuple(self.tool_events),
            context_segments=tuple(self.context_segments),
            lifecycle_events=tuple(self.lifecycle_events),
            event_count=self.event_count,
            seen_event_ids=tuple(self.seen_event_ids),
        )


def _event_row(event: LedgerEvent) -> dict[str, object]:
    """把 LedgerEvent 转成视图行。

    作者：LKX
    时间：2026-07-02 00:00:00
    传参：event 为 LedgerEvent
    返回：包含事件元数据和 payload 的字典
    """
    return {
        "event": event.event,
        "event_id": event.event_id,
        "ts": event.ts,
        "task_id": event.task_id,
        "session_id": event.session_id,
        "run_id": event.run_id,
        "payload": dict(event.payload),
    }


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


__all__ = ["RunEvidenceView", "build_run_evidence_view"]
