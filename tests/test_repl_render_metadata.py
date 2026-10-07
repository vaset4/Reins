"""Runtime metadata visibility tests for the REPL renderer."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from rich.console import Console

from app.repl.console import reset_console_for_tests
from app.repl.render import EventRenderer
from runtime.stream_events import LeaseSnapshot, LifecycleChanged


@pytest.fixture()
def recorded() -> Generator[Console, None, None]:
    console = Console(record=True, width=120, force_terminal=False, color_system=None)
    reset_console_for_tests(console)
    yield console
    reset_console_for_tests(None)


def text_of(console: Console) -> str:
    return console.export_text()


def test_lease_snapshot_hidden_by_default(recorded: Console) -> None:
    EventRenderer().render(_lease_snapshot())
    assert text_of(recorded) == ""


def test_lease_snapshot_dim_summary_line_with_trace_on(recorded: Console) -> None:
    EventRenderer(trace_on=True).render(_lease_snapshot())
    rendered = text_of(recorded)
    assert "trigger=user" in rendered
    assert "max_steps=30" in rendered
    assert "expires" in rendered


def test_lifecycle_done_hidden_by_default(recorded: Console) -> None:
    EventRenderer().render(
        LifecycleChanged(
            lifecycle="done",
            reason="final_output",
            segment_id="segment-1",
            checkpoint_id="ck-1",
        )
    )
    assert text_of(recorded) == ""


def test_lifecycle_failed_renders_status_without_checkpoint_by_default(
    recorded: Console,
) -> None:
    EventRenderer().render(
        LifecycleChanged(
            lifecycle="failed",
            reason="timeout",
            segment_id="segment-1",
            checkpoint_id="ck-1",
        )
    )
    rendered = text_of(recorded)
    assert "状态: failed" in rendered
    assert "timeout" in rendered
    assert "ck-1" not in rendered


def test_lifecycle_changed_renders_checkpoint_with_trace_on(
    recorded: Console,
) -> None:
    EventRenderer(trace_on=True).render(
        LifecycleChanged(
            lifecycle="failed",
            reason="timeout",
            segment_id="segment-1",
            checkpoint_id="ck-1",
        )
    )
    rendered = text_of(recorded)
    assert "lifecycle failed" in rendered
    assert "timeout" in rendered
    assert "ck-1" in rendered


def _lease_snapshot() -> LeaseSnapshot:
    return LeaseSnapshot(
        trigger="user",
        task_id="2026-05-06-01HK",
        segment_id="user-01HK",
        max_steps=30,
        max_tokens=200000,
        expires_at="2026-05-06T13:00:00Z",
        capabilities_summary={},
    )
