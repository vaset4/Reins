"""V2.1 stream events for the real chat REPL.

These dataclasses are the wire protocol between the agent loop generator
(`AgentLoop.run_stream`) and any front-end renderer (CLI REPL today, Textual
TUI later). Names align with the stable trajectory event names in
`.trellis/spec/backend/logging-guidelines.md` so consumers can reuse the same
taxonomy.

Frozen dataclasses keep events trivially hashable and safe to forward through
queues, generators, or test capture helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class AssistantTextDelta:
    """Incremental assistant text. MVP-0 emits this as a single chunk per turn;
    a future LLM streaming adapter may split into multiple deltas."""

    text: str
    message_id: str = ""


@dataclass(frozen=True, slots=True)
class AssistantReasoningDelta:
    """Assistant reasoning text surfaced by the provider or <think> tags."""

    text: str
    message_id: str = ""


@dataclass(frozen=True, slots=True)
class ModelRequestStarted:
    """请求尝试即将发送；参数：请求身份与尝试次数；返回：等待响应事件。"""

    message_id: str
    attempt_index: int
    max_attempts: int


@dataclass(frozen=True, slots=True)
class ModelRetryScheduled:
    """模型请求失败后的等待状态，不属于回答正文。

    作者：xxx
    时间：2026-09-29 10:40:00
    参数：attempt_index为下一次尝试，max_attempts为总次数，wait_seconds为等待秒数，error_category为原因
    返回：可通过后台事件通道传递的不可变状态
    """

    attempt_index: int
    max_attempts: int
    wait_seconds: float
    error_category: str


@dataclass(frozen=True, slots=True)
class AssistantStreamClosed:
    """未提交输出因真实边界结束；参数：请求身份、原因及是否将重试；返回：不可变控制事件。"""

    message_id: str
    reason: str
    retrying: bool = False


@dataclass(frozen=True, slots=True)
class AssistantTurnComplete:
    """One LLM turn finished. `content` is the final answer when present;
    otherwise the model returned a tool request (see ToolExecutionStarted)."""

    content: str | None
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str | None = None
    message_id: str = ""
    entry_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolExecutionStarted:
    """A tool call is about to run through ToolRegistry.execute_tool."""

    tool_name: str
    args: dict[str, Any]
    call_id: str
    risk: str  # "safe" | "confirm" | "deny"


@dataclass(frozen=True, slots=True)
class ToolExecutionCompleted:
    """A tool call finished. `is_error=True` means the registry returned a
    ToolError; `error_category` carries ToolErrorCategory.value when present."""

    tool_name: str
    output: str
    call_id: str
    is_error: bool = False
    error_category: str | None = None
    execution: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolApprovalRequested:
    """A confirm-risk tool needs user approval.

    The renderer only paints the request; approval is resolved through the
    registered approval backend rather than `generator.send(...)`.
    """

    tool_name: str
    args: dict[str, Any]
    risk: str
    reason: str
    call_id: str


@dataclass(frozen=True, slots=True)
class StateTransition:
    """Historical AgentLoop state transition stream variant.

    Current-facing status consumers should read `LifecycleChanged`.
    Renderers usually hide this except in `/trace on` mode for old runs.
    """

    from_state: str
    to_state: str
    segment_id: str


@dataclass(frozen=True, slots=True)
class LifecycleChanged:
    """Public run lifecycle boundary for current-facing status consumers."""

    lifecycle: str
    reason: str
    segment_id: str
    checkpoint_id: str | None = None


@dataclass(frozen=True, slots=True)
class SegmentPaused:
    """Segment paused by lease budget, watchdog, or user `/pause`. `resumable`
    is True when a checkpoint has been written that `agent_loop.resume()`
    can pick up."""

    reason: str
    task_id: str
    segment_id: str
    resumable: bool = True


@dataclass(frozen=True, slots=True)
class LeaseSnapshot:
    """Emitted at segment start. Mirrors the trajectory `lease_snapshot` row
    so the REPL can show what capabilities are in scope."""

    trigger: str
    task_id: str
    segment_id: str
    max_steps: int
    max_tokens: int
    expires_at: str
    capabilities_summary: dict[str, Any] = field(default_factory=dict)


StreamEvent = (
    AssistantTextDelta
    | AssistantReasoningDelta
    | ModelRequestStarted
    | ModelRetryScheduled
    | AssistantTurnComplete
    | AssistantStreamClosed
    | ToolExecutionStarted
    | ToolExecutionCompleted
    | ToolApprovalRequested
    | StateTransition
    | LifecycleChanged
    | SegmentPaused
    | LeaseSnapshot
)


__all__ = [
    "AssistantTextDelta",
    "AssistantReasoningDelta",
    "ModelRetryScheduled",
    "ModelRequestStarted",
    "AssistantTurnComplete",
    "AssistantStreamClosed",
    "ToolExecutionStarted",
    "ToolExecutionCompleted",
    "ToolApprovalRequested",
    "StateTransition",
    "LifecycleChanged",
    "SegmentPaused",
    "LeaseSnapshot",
    "StreamEvent",
]
