"""Renderer tests using a captured `rich.Console`.

Each test forces a fresh recording console via `reset_console_for_tests` so the
renderer's output can be inspected without leaking ANSI escape codes back into
pytest's terminal capture.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from app.repl.console import get_console, reset_console_for_tests
from app.repl.render import EventRenderer
from app.repl.render_format import format_tool_args, truncate_tool_output
from runtime.stream_events import (
    AssistantReasoningDelta,
    AssistantTextDelta,
    AssistantTurnComplete,
    SegmentPaused,
    StateTransition,
    ToolApprovalRequested,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)


@pytest.fixture()
def recorded() -> Console:
    console = Console(record=True, width=120, force_terminal=False, color_system=None)
    reset_console_for_tests(console)
    yield console
    reset_console_for_tests(None)


def text_of(console: Console) -> str:
    return console.export_text()


def test_text_delta_prints_inline_without_newline(recorded: Console) -> None:
    """连续正文增量接在同一行后面，首片之前只打一次 assistant 标题。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：recorded 为录制型控制台
    返回：无

    精确相等同时钉住三件事：标题只出现一次、增量之间不插换行、最后一片后面不补换行。
    """
    renderer = EventRenderer()
    renderer.render(AssistantTextDelta(text="streaming "))
    renderer.render(AssistantTextDelta(text="token"))
    assert text_of(recorded) == "assistant\nstreaming token"


def test_interstitial_text_gets_its_own_label_between_tool_calls(
    recorded: Console,
) -> None:
    """工具调用之间的自言自语各自带一次标题，不与最终答案混成一段。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：recorded 为录制型控制台
    返回：无

    2026-09-05 实测一轮 10 次工具调用里 4 轮先吐出 "I'll read the rest of the paper"
    这类话再发工具调用。它们走答案通道，不分区就和真正的答案长得一模一样——用户正是
    因此把整轮读成"只有思考"。
    """
    renderer = EventRenderer()
    renderer.render(AssistantTextDelta(text="我先读一下文件"))
    renderer.render(
        ToolExecutionCompleted(
            tool_name="file_read",
            output="paper text",
            call_id="call-1",
        )
    )
    renderer.render(AssistantTextDelta(text="读完了，结论是"))

    rendered = text_of(recorded)
    assert rendered.count("assistant") == 2
    assert "我先读一下文件\n" in rendered


def test_reasoning_delta_renders_distinct_panel(recorded: Console) -> None:
    EventRenderer().render(AssistantReasoningDelta(text="check the workspace"))
    rendered = text_of(recorded)
    assert "reasoning" in rendered
    assert "check the workspace" in rendered


def test_reasoning_deltas_share_one_growing_region(recorded: Console) -> None:
    """多条思考增量只打一次标题，全部追加进同一片区域。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：recorded 为录制型控制台
    返回：无

    流式一轮会来几十上百条增量；每条一个带框 Panel 会把屏幕刷爆，用户什么也读不到。
    """
    renderer = EventRenderer()
    for fragment in ("先看清", "用户要什么"):
        renderer.render(AssistantReasoningDelta(text=fragment))

    rendered = text_of(recorded)
    assert rendered.count("reasoning") == 1
    assert "先看清用户要什么" in rendered


def test_streamed_answer_is_not_reprinted_at_turn_complete(recorded: Console) -> None:
    """答案已经逐字打过，轮末不得再把整段渲染一遍。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：recorded 为录制型控制台
    返回：无

    生产开始 yield 正文增量之后，原先那句无条件 print(Markdown(content)) 会让同一段
    答案在屏幕上出现两次。
    """
    renderer = EventRenderer()
    renderer.render(AssistantTextDelta(text="做完了"))
    renderer.render(AssistantTurnComplete(content="做完了", usage={}, stop_reason=None))

    assert text_of(recorded).count("做完了") == 1


def test_first_output_delta_stops_the_spinner_exactly_once(recorded: Console) -> None:
    """首片模型输出到达时通知调用方一次，后续增量不再通知。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：recorded 为录制型控制台
    返回：无

    终端入口拿这个回调停 Working... 转圈。转圈是 Rich 活动区、只管理完整的行，
    而流式正文故意不换行，两者同抢屏幕会把 Working... 糊进句子中间。
    """
    calls: list[str] = []
    renderer = EventRenderer(on_first_output=lambda: calls.append("stop"))

    renderer.render(
        StateTransition(from_state="planning", to_state="acting", segment_id="seg-1")
    )
    assert not calls, "非模型输出的事件不该停转圈"

    renderer.render(AssistantReasoningDelta(text="想"))
    renderer.render(AssistantReasoningDelta(text="想完了"))
    renderer.render(AssistantTextDelta(text="答案"))

    assert calls == ["stop"]


def test_turn_complete_renders_markdown(recorded: Console) -> None:
    EventRenderer().render(
        AssistantTurnComplete(
            content="# Heading\nbody text", usage={}, stop_reason=None
        )
    )
    rendered = text_of(recorded)
    assert "Heading" in rendered
    assert "body text" in rendered


def test_turn_complete_hides_usage_line_by_default(recorded: Console) -> None:
    EventRenderer().render(
        AssistantTurnComplete(
            content="ok",
            usage={"input_tokens": 100, "output_tokens": 50},
            stop_reason="end_turn",
        )
    )
    rendered = text_of(recorded)
    assert "ok" in rendered
    assert "tokens=100/50" not in rendered
    assert "stop=end_turn" not in rendered


def test_turn_complete_shows_usage_line_with_trace_on(recorded: Console) -> None:
    EventRenderer(trace_on=True).render(
        AssistantTurnComplete(
            content="ok",
            usage={"input_tokens": 100, "output_tokens": 50},
            stop_reason="end_turn",
        )
    )
    rendered = text_of(recorded)
    assert "tokens=100/50" in rendered
    assert "stop=end_turn" in rendered


def test_tool_started_hidden_by_default(recorded: Console) -> None:
    EventRenderer().render(
        ToolExecutionStarted(
            tool_name="list",
            args={"path": "tools"},
            call_id="call-1",
            risk="safe",
        )
    )
    assert text_of(recorded) == ""


def test_tool_started_panel_shows_name_args_risk_with_trace_on(
    recorded: Console,
) -> None:
    EventRenderer(trace_on=True).render(
        ToolExecutionStarted(
            tool_name="list",
            args={"path": "tools"},
            call_id="call-1",
            risk="safe",
        )
    )
    rendered = text_of(recorded)
    assert "list" in rendered
    assert "safe" in rendered
    assert "call-1" in rendered
    assert "path" in rendered  # arg key visible


def test_tool_completed_success_panel_with_trace_on(recorded: Console) -> None:
    EventRenderer(trace_on=True).render(
        ToolExecutionCompleted(
            tool_name="list",
            output="tools/\nbrowser/",
            call_id="call-1",
        )
    )
    rendered = text_of(recorded)
    assert "✓" in rendered
    assert "tools/" in rendered


def test_tool_completed_error_panel_shows_category_with_trace_on(
    recorded: Console,
) -> None:
    EventRenderer(trace_on=True).render(
        ToolExecutionCompleted(
            tool_name="terminal",
            output="permission denied",
            call_id="call-2",
            is_error=True,
            error_category="permission",
        )
    )
    rendered = text_of(recorded)
    assert "✗" in rendered
    assert "permission" in rendered
    assert "Status hint" in rendered
    assert "hard_blocked" in rendered


def test_turn_complete_protocol_error_shows_recovery_hint(recorded: Console) -> None:
    EventRenderer().render(
        AssistantTurnComplete(
            content="MODEL_PROTOCOL_ERROR: invalid response type",
            usage={},
            stop_reason="protocol_error",
        )
    )

    rendered = text_of(recorded)
    assert "MODEL_PROTOCOL_ERROR" in rendered
    assert "状态提示" in rendered
    assert "recoverable_failure" in rendered


def test_long_tool_output_truncated() -> None:
    long_output = "x" * 4096
    truncated = truncate_tool_output(long_output)
    assert len(truncated) < len(long_output)
    assert "more chars" in truncated
    assert "artifacts/" in truncated


def test_short_tool_output_passes_through() -> None:
    assert truncate_tool_output("short") == "short"


def test_format_tool_args_compact_json() -> None:
    rendered = format_tool_args({"path": "tools", "mode": "list"})
    assert rendered == '{"mode": "list", "path": "tools"}'


def test_format_tool_args_truncates_long_payload() -> None:
    big = {"data": "x" * 500}
    rendered = format_tool_args(big)
    assert rendered.endswith(" …")
    assert len(rendered) <= 205


def test_approval_panel_carries_reason_and_args(recorded: Console) -> None:
    EventRenderer().render(
        ToolApprovalRequested(
            tool_name="file_write",
            args={"path": "out.txt"},
            risk="confirm",
            reason="write requires user approval",
            call_id="call-7",
        )
    )
    rendered = text_of(recorded)
    assert "approval required" in rendered
    assert "file_write" in rendered
    assert "write requires user approval" in rendered


def test_state_transition_hidden_unless_trace_on(recorded: Console) -> None:
    EventRenderer(trace_on=False).render(
        StateTransition(
            from_state="AWAITING_MODEL",
            to_state="PARSING",
            segment_id="user-1",
        )
    )
    assert text_of(recorded) == ""


def test_state_transition_visible_with_trace_on(recorded: Console) -> None:
    EventRenderer(trace_on=True).render(
        StateTransition(
            from_state="AWAITING_MODEL",
            to_state="PARSING",
            segment_id="user-1",
        )
    )
    rendered = text_of(recorded)
    assert "AWAITING_MODEL" in rendered
    assert "PARSING" in rendered


def test_segment_paused_panel_includes_reason(recorded: Console) -> None:
    EventRenderer().render(
        SegmentPaused(
            reason="lease step limit reached",
            task_id="2026-05-06-01HK",
            segment_id="user-01HK",
        )
    )
    rendered = text_of(recorded)
    assert "segment paused" in rendered
    assert "lease step limit reached" in rendered
    assert "/resume" in rendered


def test_unknown_event_renders_defensive_message(recorded: Console) -> None:
    class FakeEvent:
        pass

    EventRenderer().render(FakeEvent())  # type: ignore[arg-type]
    rendered = text_of(recorded)
    assert "unknown event" in rendered
    assert "FakeEvent" in rendered


def test_console_singleton_reused() -> None:
    reset_console_for_tests(None)
    first = get_console()
    second = get_console()
    assert first is second
    reset_console_for_tests(None)
