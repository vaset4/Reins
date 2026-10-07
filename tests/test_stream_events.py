"""Stability tests for the V2.1 stream event protocol.

The renderer (CLI REPL today, Textual TUI later) and the agent loop generator
both depend on these dataclass shapes. A field rename or removal here is a
breaking wire change and must be intentional, not accidental.
"""

from __future__ import annotations

import dataclasses

import pytest

from runtime.stream_events import (
    AssistantReasoningDelta,
    AssistantTextDelta,
    AssistantTurnComplete,
    AssistantStreamClosed,
    LeaseSnapshot,
    LifecycleChanged,
    ModelRetryScheduled,
    ModelRequestStarted,
    SegmentPaused,
    StateTransition,
    StreamEvent,
    ToolApprovalRequested,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)


def test_assistant_text_delta_is_frozen_with_text_field() -> None:
    event = AssistantTextDelta(text="hi")
    assert event.text == "hi"
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.text = "no"  # type: ignore[misc]


def test_assistant_reasoning_delta_is_frozen_with_text_field() -> None:
    event = AssistantReasoningDelta(text="thinking")
    assert event.text == "thinking"
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.text = "no"  # type: ignore[misc]


def test_assistant_turn_complete_defaults() -> None:
    event = AssistantTurnComplete(content="answer")
    assert event.content == "answer"
    assert event.usage == {}
    assert event.stop_reason is None


def test_tool_execution_started_required_fields() -> None:
    event = ToolExecutionStarted(
        tool_name="list",
        args={"path": "tools"},
        call_id="call-1",
        risk="safe",
    )
    assert event.tool_name == "list"
    assert event.args == {"path": "tools"}
    assert event.risk == "safe"


def test_tool_execution_completed_error_flags() -> None:
    ok = ToolExecutionCompleted(tool_name="list", output="...", call_id="call-1")
    assert ok.is_error is False
    assert ok.error_category is None

    err = ToolExecutionCompleted(
        tool_name="terminal",
        output="boom",
        call_id="call-2",
        is_error=True,
        error_category="permission",
    )
    assert err.is_error is True
    assert err.error_category == "permission"


def test_tool_approval_requested_carries_call_id() -> None:
    event = ToolApprovalRequested(
        tool_name="file_write",
        args={"path": "out.txt"},
        risk="confirm",
        reason="write requires approval",
        call_id="call-7",
    )
    assert event.call_id == "call-7"


def test_state_transition_carries_segment_id() -> None:
    event = StateTransition(
        from_state="AWAITING_MODEL",
        to_state="PARSING",
        segment_id="user-01HK",
    )
    assert event.from_state == "AWAITING_MODEL"
    assert event.to_state == "PARSING"


def test_lifecycle_changed_carries_public_boundary() -> None:
    event = LifecycleChanged(
        lifecycle="waiting_approval",
        reason="tool requires approval",
        segment_id="user-01HK",
        checkpoint_id="ck-1",
    )
    assert event.lifecycle == "waiting_approval"
    assert event.reason == "tool requires approval"
    assert event.checkpoint_id == "ck-1"


def test_segment_paused_resumable_default() -> None:
    event = SegmentPaused(
        reason="lease step limit reached",
        task_id="2026-05-06-01HK",
        segment_id="user-01HK",
    )
    assert event.resumable is True


def test_lease_snapshot_defaults_capabilities_summary() -> None:
    event = LeaseSnapshot(
        trigger="user",
        task_id="2026-05-06-01HK",
        segment_id="user-01HK",
        max_steps=30,
        max_tokens=200000,
        expires_at="2026-05-06T13:00:00Z",
    )
    assert event.capabilities_summary == {}


def test_stream_event_union_includes_all_variants() -> None:
    members = StreamEvent.__args__  # type: ignore[attr-defined]
    expected = {
        AssistantTextDelta,
        AssistantReasoningDelta,
        ModelRetryScheduled,
        ModelRequestStarted,
        AssistantTurnComplete,
        AssistantStreamClosed,
        ToolExecutionStarted,
        ToolExecutionCompleted,
        ToolApprovalRequested,
        StateTransition,
        LifecycleChanged,
        SegmentPaused,
        LeaseSnapshot,
    }
    assert set(members) == expected
