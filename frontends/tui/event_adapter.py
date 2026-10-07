from __future__ import annotations

import json
from typing import Any, Literal

from runtime.stream_events import (
    AssistantReasoningDelta,
    AssistantTextDelta,
    AssistantTurnComplete,
    LeaseSnapshot,
    LifecycleChanged,
    ModelRetryScheduled,
    SegmentPaused,
    StateTransition,
    StreamEvent,
    ToolApprovalRequested,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)

from frontends.tui.transcript import TranscriptBuffer, truncate_text
from app.repl.status import format_model_retry

TOOL_OUTPUT_LIMIT = 1200


class TuiEventAdapter:
    def __init__(self, transcript: TranscriptBuffer, *, trace_on: bool = False) -> None:
        self._transcript = transcript
        self._trace_on = trace_on

    def render(self, event: StreamEvent) -> None:
        if isinstance(event, AssistantReasoningDelta):
            # 流式思考链一轮几十上百片，每片一行会把 transcript 刷爆；并进同一条里增长
            self._transcript.append_to_last("status", "reasoning", event.text)
            return
        if isinstance(event, AssistantTextDelta):
            self._transcript.append_to_last_assistant(event.text)
            return
        if isinstance(event, ModelRetryScheduled):
            self._transcript.append("status", "等待重试", format_model_retry(event))
            return
        if isinstance(event, AssistantTurnComplete):
            self._render_turn_complete(event)
            return
        if isinstance(event, ToolExecutionStarted):
            self._render_tool_started(event)
            return
        if isinstance(event, ToolExecutionCompleted):
            self._render_tool_completed(event)
            return
        if isinstance(event, ToolApprovalRequested):
            self._render_approval_requested(event)
            return
        if isinstance(event, StateTransition):
            self._render_state_transition(event)
            return
        if isinstance(event, LifecycleChanged):
            self._render_lifecycle_changed(event)
            return
        if isinstance(event, SegmentPaused):
            self._render_segment_paused(event)
            return
        if isinstance(event, LeaseSnapshot):
            self._render_lease_snapshot(event)
            return
        self._transcript.append("status", "unknown event", type(event).__name__)

    def _render_turn_complete(self, event: AssistantTurnComplete) -> None:
        failed = bool(
            event.stop_reason and event.stop_reason.startswith("model_error:")
        )
        if event.content and (failed or not _last_assistant_has_text(self._transcript)):
            self._transcript.append("assistant", "assistant", event.content)
        usage = _format_usage(event.usage, event.stop_reason)
        if usage:
            self._transcript.append("status", "turn complete", usage, state="success")

    def _render_tool_started(self, event: ToolExecutionStarted) -> None:
        body = "\n".join(
            [
                f"call: {event.call_id}",
                f"risk: {event.risk}",
                f"args: {_format_json(event.args)}",
            ]
        )
        self._transcript.append("tool", event.tool_name, body, state="pending")

    def _render_tool_completed(self, event: ToolExecutionCompleted) -> None:
        state: Literal["error", "success"] = "error" if event.is_error else "success"
        category = f" [{event.error_category}]" if event.error_category else ""
        body = "\n".join(
            [
                f"call: {event.call_id}",
                f"output: {truncate_text(event.output, TOOL_OUTPUT_LIMIT)}",
            ]
        )
        self._transcript.append(
            "tool",
            f"{event.tool_name}{category}",
            body,
            state=state,
        )

    def _render_approval_requested(self, event: ToolApprovalRequested) -> None:
        body = "\n".join(
            [
                event.reason,
                f"tool: {event.tool_name}",
                f"risk: {event.risk}",
                f"args: {_format_json(event.args)}",
                "type /approve once, /approve task, or /deny",
            ]
        )
        self._transcript.append("approval", "approval required", body, state="pending")

    def _render_state_transition(self, event: StateTransition) -> None:
        if not self._trace_on:
            return
        body = f"{event.from_state} -> {event.to_state}"
        self._transcript.append("status", "state", body)

    def _render_lifecycle_changed(self, event: LifecycleChanged) -> None:
        body = "\n".join(
            [
                f"lifecycle: {event.lifecycle}",
                f"reason: {event.reason}",
                f"checkpoint: {event.checkpoint_id or '(none)'}",
            ]
        )
        self._transcript.append("status", "lifecycle", body)

    def _render_segment_paused(self, event: SegmentPaused) -> None:
        suffix = "Use /resume to continue." if event.resumable else "Not resumable."
        body = f"{event.reason}\ntask: {event.task_id}\n{suffix}"
        self._transcript.append("status", "segment paused", body)

    def _render_lease_snapshot(self, event: LeaseSnapshot) -> None:
        body = (
            f"trigger={event.trigger} max_steps={event.max_steps} "
            f"max_tokens={event.max_tokens} expires={event.expires_at}"
        )
        self._transcript.append("status", "lease", body)


def _last_assistant_has_text(transcript: TranscriptBuffer) -> bool:
    items = transcript.snapshot()
    return bool(items and items[-1].role == "assistant" and items[-1].body.strip())


def _format_usage(usage: dict[str, int], stop_reason: str | None) -> str:
    parts: list[str] = []
    if usage:
        in_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        out_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        if in_tokens or out_tokens:
            parts.append(f"tokens={in_tokens}/{out_tokens}")
    if stop_reason:
        parts.append(f"stop={stop_reason}")
    return " ".join(parts)


def _format_json(value: dict[str, Any]) -> str:
    try:
        return truncate_text(json.dumps(value, ensure_ascii=False, sort_keys=True), 320)
    except Exception:
        return truncate_text(repr(value), 320)


__all__ = ["TuiEventAdapter"]
