"""Rich-based renderer for `runtime.stream_events.StreamEvent`."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from time import perf_counter
from typing import Any

from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from app.repl.console import get_console
from app.repl.status import (
    explain_model_stop,
    explain_tool_error,
    format_status_hint,
    format_model_retry,
)
from app.repl.render_format import (
    compact_error_output,
    convergence_pause_detail as _convergence_pause_detail,
    format_elapsed as _format_elapsed,
    format_normal_status_hint,
    format_process_line,
    format_tool_args as _format_tool_args,
    pop_elapsed as _pop_elapsed,
    tool_activity_explanation as _tool_activity_explanation,
    tool_panel_target_summary as _tool_panel_target_summary,
    truncate_tool_output as _truncate_tool_output,
)
from runtime.stream_events import (
    AssistantReasoningDelta,
    AssistantTextDelta,
    AssistantTurnComplete,
    AssistantStreamClosed,
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


_RISK_BORDER = {
    "safe": "green",
    "confirm": "yellow",
    "deny": "red",
}

_NORMAL_VISIBLE_LIFECYCLES = {
    "failed",
    "error",
    "aborted",
    "paused",
    "waiting_user",
    "waiting_approval",
}


class EventRenderer:
    """Stateful stream-event renderer."""

    def __init__(
        self,
        *,
        trace_on: bool = False,
        on_first_output: Callable[[], None] | None = None,
    ) -> None:
        self.trace_on = trace_on
        self._run_started_at: float | None = None
        self._tool_started_at: dict[str, float] = {}
        self._tool_args_by_call_id: dict[str, dict[str, Any]] = {}
        self._process_step_count = 0
        self._reasoning_open = False
        self._assistant_open = False
        self._streamed_text = False
        self._on_first_output = on_first_output

    def render(self, event: StreamEvent) -> None:
        console = get_console()
        if isinstance(event, (AssistantReasoningDelta, AssistantTextDelta)):
            self._announce_first_output()
        # 思考区与正文区互斥：一类增量到达就把另一片区域收尾换行；
        # 工具、轮末这些别的事件到达则两片一起收尾
        if not isinstance(event, AssistantReasoningDelta):
            self._close_reasoning()
        if not isinstance(event, AssistantTextDelta):
            self._close_assistant()
        if isinstance(event, AssistantReasoningDelta):
            self._render_reasoning_delta(event)
            return
        if isinstance(event, AssistantTextDelta):
            self._render_text_delta(event)
            return
        if isinstance(event, ModelRetryScheduled):
            self._announce_first_output()
            self._streamed_text = False
            console.print(Text(format_model_retry(event), style="yellow"))
            return
        if isinstance(event, AssistantStreamClosed):
            self._streamed_text = False
            console.print(Text(f"输出未保存为回答：{event.reason}", style="yellow"))
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
            self._render_approval_panel(event)
            return
        if isinstance(event, StateTransition):
            if self.trace_on:
                console.print(
                    Rule(
                        f"{event.from_state} → {event.to_state}",
                        style="dim cyan",
                    )
                )
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
        # Unknown event — render a defensive line so the REPL never silently
        # drops events when a future variant gets added.
        console.print(f"[dim]<unknown event: {type(event).__name__}>[/dim]")

    def _render_turn_complete(self, event: AssistantTurnComplete) -> None:
        console = get_console()
        # 正文已经逐字打完了，这里再 Markdown 渲染一遍就是同一段答案出现两次；
        # 收尾换行由正文区负责。流式正文是纯文本，没有 Markdown 排版
        # 【模型调用】【失败提示】已显示的半截回答不能挡住重试耗尽后的真实错误
        failed = bool(
            event.stop_reason and event.stop_reason.startswith("model_error:")
        )
        if (not self._streamed_text or failed) and event.content:
            console.print(Markdown(event.content))
        self._streamed_text = False
        status_hint = explain_model_stop(event.stop_reason, event.content)
        if status_hint is not None:
            hint = (
                format_status_hint(status_hint)
                if self.trace_on
                else format_normal_status_hint(status_hint)
            )
            console.print(hint, style="yellow")
        if self.trace_on:
            usage_line = self._format_usage(event.usage, event.stop_reason)
            if usage_line:
                console.print(usage_line, style="dim")

    def _render_reasoning_delta(self, event: AssistantReasoningDelta) -> None:
        console = get_console()
        # 流式思考链一轮会来几十上百条，每条一个带框 Panel 会把屏幕刷爆。
        # 首条打一次标题，之后所有增量都追加进同一片可增长区域
        if not self._reasoning_open:
            console.print("reasoning", style="dim yellow")
            self._reasoning_open = True
        console.print(event.text, end="", soft_wrap=True, highlight=False, style="dim")

    def _render_text_delta(self, event: AssistantTextDelta) -> None:
        """正文增量逐字上屏，每片连续正文区首片先打一次 assistant 标题。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：event 为一片答案正文增量
        返回：无

        模型会把答案通道当草稿纸：2026-09-05 实测一轮 10 次工具调用里有 4 轮先吐出
        "I'll read the rest of the paper" 这类自言自语再发工具调用。这些话和真正的
        最终答案走同一个通道，不带标题就在屏幕上长得一模一样，用户分不清哪段是答案、
        哪段只是模型在念叨。配合思考区那个 reasoning 标题，两个通道各自可辨。
        """
        console = get_console()
        if not self._assistant_open:
            console.print("assistant", style="dim cyan")
            self._assistant_open = True
        # 记一笔，轮末就不能再把整段 Markdown 打第二遍
        self._streamed_text = True
        console.print(event.text, end="", soft_wrap=True, highlight=False)

    def _announce_first_output(self) -> None:
        """首片模型输出到达时通知一次调用方，之后不再通知。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：无
        返回：无

        终端入口用它停掉那个 Working... 转圈：转圈是 Rich 的活动区，只能管理完整的行，
        而流式正文故意不换行，两者同时占屏会把转圈糊到句子中间。模型一开口，文字本身
        就是"有事在发生"的信号，转圈没用了，停掉且不再恢复。
        """
        if self._on_first_output is None:
            return
        callback = self._on_first_output
        self._on_first_output = None
        callback()

    def _close_reasoning(self) -> None:
        """思考链区域收尾：补一个换行，让后面的内容从新行开始。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：无
        返回：无
        """
        if not self._reasoning_open:
            return
        get_console().print()
        self._reasoning_open = False

    def _close_assistant(self) -> None:
        """正文区域收尾：补一个换行，让后面的内容从新行开始。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：无
        返回：无
        """
        if not self._assistant_open:
            return
        get_console().print()
        self._assistant_open = False

    def _render_tool_started(self, event: ToolExecutionStarted) -> None:
        self._tool_started_at[event.call_id] = perf_counter()
        self._tool_args_by_call_id[event.call_id] = dict(event.args)
        if not self.trace_on:
            return
        console = get_console()
        border = _RISK_BORDER.get(event.risk, "blue")
        target = _tool_panel_target_summary(event.tool_name, event.args)
        started = datetime.now().strftime("%H:%M:%S")
        body = (
            f"{_format_tool_args(event.args)}\n\n"
            f"{_tool_activity_explanation(event.tool_name)}"
        )
        title = (
            f"tool: {event.tool_name} {target}  risk={event.risk}  "
            f"started={started}  {event.call_id}"
        )
        console.print(Panel(body, title=title, border_style=border, expand=False))

    def _render_tool_completed(self, event: ToolExecutionCompleted) -> None:
        console = get_console()
        args = self._tool_args_by_call_id.pop(event.call_id, {})
        elapsed = _format_elapsed(_pop_elapsed(self._tool_started_at, event.call_id))
        if not self.trace_on:
            self._render_process_line(event, args)
            return
        if event.is_error:
            mark = "✗"
            border = "red"
            cat = f"  [{event.error_category}]" if event.error_category else ""
            title = f"{mark} {event.tool_name}{cat}  {elapsed}  {event.call_id}"
        else:
            mark = "✓"
            border = "green"
            title = f"{mark} {event.tool_name}  {elapsed}  {event.call_id}"
        body = _truncate_tool_output(event.output)
        if event.is_error:
            status_hint = explain_tool_error(
                event.error_category,
                event.output,
                tool_name=event.tool_name,
            )
            body = f"{body}\n\n{format_status_hint(status_hint)}"
        console.print(Panel(body, title=title, border_style=border, expand=False))

    def _render_process_line(
        self,
        event: ToolExecutionCompleted,
        args: dict[str, Any],
    ) -> None:
        console = get_console()
        self._process_step_count += 1
        line = format_process_line(
            step=self._process_step_count,
            tool_name=event.tool_name,
            args=args,
            is_error=event.is_error,
            error_category=event.error_category,
        )
        console.print(line, style="red" if event.is_error else "dim")
        if not event.is_error:
            return
        error = compact_error_output(event.output)
        if error:
            console.print(f"错误: {error}", style="red")
        status_hint = explain_tool_error(
            event.error_category,
            event.output,
            tool_name=event.tool_name,
        )
        console.print(format_normal_status_hint(status_hint), style="yellow")

    def _render_approval_panel(self, event: ToolApprovalRequested) -> None:
        console = get_console()
        body = Text.assemble(
            (f"reason: {event.reason}\n", "bold"),
            (f"tool:   {event.tool_name}\n", ""),
            (f"args:   {_format_tool_args(event.args)}\n", ""),
            (f"risk:   {event.risk}", "yellow"),
        )
        console.print(
            Panel(body, title="approval required", border_style="red", expand=False)
        )

    def _render_segment_paused(self, event: SegmentPaused) -> None:
        console = get_console()
        suffix = " — use /resume to continue" if event.resumable else ""
        convergence = _convergence_pause_detail(event.reason)
        body = Text.assemble(
            (f"reason: {event.reason}\n", ""),
            (convergence, "yellow") if convergence else ("", ""),
            (f"task:   {event.task_id}\n", "dim"),
            (f"segment: {event.segment_id}{suffix}", "dim"),
        )
        console.print(
            Panel(body, title="segment paused", border_style="yellow", expand=False)
        )

    def _render_lease_snapshot(self, event: LeaseSnapshot) -> None:
        if not self.trace_on:
            return
        console = get_console()
        now = perf_counter()
        if self._run_started_at is None:
            self._run_started_at = now
        elapsed = _format_elapsed(now - self._run_started_at)
        # One dim line — the dashboard view shows the full capability table.
        console.print(
            f"[dim]lease  trigger={event.trigger}  "
            f"max_steps={event.max_steps}  max_tokens={event.max_tokens}  "
            f"elapsed={elapsed}  expires={event.expires_at}[/dim]"
        )

    def _render_lifecycle_changed(self, event: LifecycleChanged) -> None:
        console = get_console()
        failed = event.lifecycle in {"failed", "aborted", "error"}
        if not self.trace_on:
            if event.lifecycle not in _NORMAL_VISIBLE_LIFECYCLES:
                return
            reason = f" - {event.reason}" if event.reason else ""
            style = "red" if failed else "yellow"
            console.print(f"状态: {event.lifecycle}{reason}", style=style)
            return
        style = "red" if failed else "dim"
        checkpoint = (
            f"  checkpoint={event.checkpoint_id}" if event.checkpoint_id else ""
        )
        reason = f"  reason={event.reason}" if event.reason else ""
        console.print(
            f"[{style}]lifecycle {event.lifecycle}{reason}{checkpoint}[/{style}]"
        )

    def _format_usage(self, usage: dict[str, int], stop_reason: str | None) -> str:
        """区分实际零消耗与未采集用量；传参：已知用量/结束原因；返回：计量说明。"""
        in_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        out_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        parts = [
            f"tokens={in_tokens if in_tokens is not None else 'unknown'}/{out_tokens if out_tokens is not None else 'unknown'}"
        ]
        if stop_reason:
            parts.append(f"stop={stop_reason}")
        return "  ".join(parts)


__all__ = [
    "EventRenderer",
]
