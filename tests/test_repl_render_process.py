"""Compact process timeline tests for the REPL renderer."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from rich.console import Console

from app.repl.console import reset_console_for_tests
from app.repl.render import EventRenderer
from runtime.stream_events import (
    AssistantTurnComplete,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)


@pytest.fixture()
def recorded() -> Generator[Console, None, None]:
    console = Console(record=True, width=120, force_terminal=False, color_system=None)
    reset_console_for_tests(console)
    yield console
    reset_console_for_tests(None)


def text_of(console: Console) -> str:
    return console.export_text()


def test_process_timeline_renders_recovery_path_in_real_time(
    recorded: Console,
) -> None:
    renderer = EventRenderer()

    _render_tool_pair(
        renderer,
        tool_name="file_read",
        args={"path": "AGENT_CAPABILITIES.md"},
        call_id="call-1",
        output="not a file: AGENT_CAPABILITIES.md",
        is_error=True,
        error_category="invalid_input",
    )
    _render_tool_pair(
        renderer,
        tool_name="grep",
        args={"path": ".", "query": "AGENT_CAPABILITIES.md"},
        call_id="call-2",
        output="docs/AGENT_CAPABILITIES.md",
    )
    _render_tool_pair(
        renderer,
        tool_name="file_read",
        args={"path": "docs/AGENT_CAPABILITIES.md"},
        call_id="call-3",
        output="raw document content should stay hidden",
    )
    renderer.render(
        AssistantTurnComplete(
            content="已找到目标章节。", usage={}, stop_reason="end_turn"
        )
    )

    rendered = text_of(recorded)
    assert "过程 1. file_read AGENT_CAPABILITIES.md -> error invalid_input" in rendered
    assert '过程 2. grep . "AGENT_CAPABILITIES.md" -> ok' in rendered
    assert "过程 3. file_read docs/AGENT_CAPABILITIES.md -> ok" in rendered
    assert "raw document content should stay hidden" not in rendered
    assert "已找到目标章节" in rendered


def test_normal_process_lines_omit_diagnostic_fields(recorded: Console) -> None:
    renderer = EventRenderer()
    _render_tool_pair(
        renderer,
        tool_name="terminal",
        args={"command": "python -m pytest"},
        call_id="call-diagnostic",
        output="passed",
    )

    rendered = text_of(recorded)
    assert "过程 1. terminal python -m pytest -> ok" in rendered
    assert "call-diagnostic" not in rendered
    assert "risk" not in rendered
    assert "elapsed" not in rendered


def test_failed_tool_renders_compact_error_and_chinese_status(
    recorded: Console,
) -> None:
    renderer = EventRenderer()
    _render_tool_pair(
        renderer,
        tool_name="file_read",
        args={"path": "missing.md"},
        call_id="call-error",
        output="invalid_input: not a file: missing.md\nfull stack should stay hidden",
        is_error=True,
        error_category="invalid_input",
    )

    rendered = text_of(recorded)
    assert "过程 1. file_read missing.md -> error invalid_input" in rendered
    assert "错误: invalid_input: not a file: missing.md" in rendered
    assert "状态提示" in rendered
    assert "strategy_available" in rendered
    assert "full stack should stay hidden" not in rendered
    assert "╭" not in rendered


def test_trace_mode_keeps_raw_tool_panel_without_process_duplication(
    recorded: Console,
) -> None:
    renderer = EventRenderer(trace_on=True)
    _render_tool_pair(
        renderer,
        tool_name="grep",
        args={"path": ".", "query": "AGENT_CAPABILITIES.md"},
        call_id="call-trace",
        output="docs/AGENT_CAPABILITIES.md",
    )

    rendered = text_of(recorded)
    assert "tool: grep" in rendered
    assert "call-trace" in rendered
    assert "过程 1." not in rendered


def _render_tool_pair(
    renderer: EventRenderer,
    *,
    tool_name: str,
    args: dict[str, object],
    call_id: str,
    output: str,
    is_error: bool = False,
    error_category: str | None = None,
) -> None:
    renderer.render(
        ToolExecutionStarted(
            tool_name=tool_name,
            args=args,
            call_id=call_id,
            risk="safe",
        )
    )
    renderer.render(
        ToolExecutionCompleted(
            tool_name=tool_name,
            output=output,
            call_id=call_id,
            is_error=is_error,
            error_category=error_category,
        )
    )
