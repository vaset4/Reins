from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from runtime.ledger import LedgerEvent


@dataclass(frozen=True, slots=True)
class TaskStateView:
    task_id: str | None = None
    intent: str = ""
    progress: str = ""
    resume_hint: str = ""
    summary: str = ""
    summary_kind: str = ""
    summary_event_id: str = ""
    focused_task_id: str | None = None
    previous_task_id: str | None = None
    focus_reason: str = ""
    focus_event_id: str = ""
    event_count: int = 0
    seen_event_ids: tuple[str, ...] = field(default_factory=tuple)


def build_task_state_view(
    events: Iterable[LedgerEvent | Mapping[str, object]],
) -> TaskStateView:
    state = _TaskStateAccumulator()
    for raw_event in events:
        event = _coerce_event(raw_event)
        state.apply(event)
    return state.to_view()


@dataclass(slots=True)
class _TaskStateAccumulator:
    task_id: str | None = None
    intent: str = ""
    progress: str = ""
    resume_hint: str = ""
    summary: str = ""
    summary_kind: str = ""
    summary_event_id: str = ""
    focused_task_id: str | None = None
    previous_task_id: str | None = None
    focus_reason: str = ""
    focus_event_id: str = ""
    event_count: int = 0
    seen_event_ids: list[str] = field(default_factory=list)

    def apply(self, event: LedgerEvent) -> None:
        self.event_count += 1
        self.seen_event_ids.append(event.event_id)
        if self.task_id is None and event.task_id is not None:
            self.task_id = event.task_id
        if event.event == "summary.updated":
            self._apply_summary_updated(event)
        elif event.event == "task.focus_changed":
            self._apply_task_focus_changed(event)

    def to_view(self) -> TaskStateView:
        return TaskStateView(
            task_id=self.task_id,
            intent=self.intent,
            progress=self.progress,
            resume_hint=self.resume_hint,
            summary=self.summary,
            summary_kind=self.summary_kind,
            summary_event_id=self.summary_event_id,
            focused_task_id=self.focused_task_id,
            previous_task_id=self.previous_task_id,
            focus_reason=self.focus_reason,
            focus_event_id=self.focus_event_id,
            event_count=self.event_count,
            seen_event_ids=tuple(self.seen_event_ids),
        )

    def _apply_summary_updated(self, event: LedgerEvent) -> None:
        content = str(event.payload.get("content", ""))
        kind = str(event.payload.get("summary_kind", ""))
        if kind == "intent":
            self.intent = content
        elif kind == "progress":
            self.progress = content
        elif kind == "resume_hint":
            self.resume_hint = content
        self.summary = content
        self.summary_kind = kind
        self.summary_event_id = event.event_id

    def _apply_task_focus_changed(self, event: LedgerEvent) -> None:
        previous = event.payload.get("previous_task_id")
        next_task = event.payload.get("next_task_id")
        self.previous_task_id = _optional_text(previous)
        self.focused_task_id = _optional_text(next_task)
        self.focus_reason = str(event.payload.get("reason", ""))
        self.focus_event_id = event.event_id
        if self.focused_task_id is not None:
            self.task_id = self.focused_task_id


def _coerce_event(event: LedgerEvent | Mapping[str, object]) -> LedgerEvent:
    if isinstance(event, LedgerEvent):
        return event
    return LedgerEvent.from_mapping(event)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["TaskStateView", "build_task_state_view"]
