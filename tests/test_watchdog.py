from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest

from runtime.lease import from_trigger
from runtime.watchdog import Watchdog
from tools.types import ToolError, ToolErrorCategory


def test_watchdog_records_steps_past_limit_without_pausing() -> None:
    """步数超过 lease 上限只继续计数，不再暂停运行"""
    watchdog = Watchdog(from_trigger("user", task_id="task-1", max_steps=2))

    assert not watchdog.tick(steps_taken=2).paused
    result = watchdog.tick(steps_taken=3)

    assert not result.paused
    assert watchdog.steps_taken == 3


def test_reserve_tool_step_increments_before_each_tool_call() -> None:
    """验证工具 step 预占只有 Watchdog 一个计数 owner

    作者：LKX
    时间：2026-08-16 00:00:00
    传参：无
    返回：无；断言每次预占都递增计数，超出上限也不再拒绝
    """
    watchdog = Watchdog(from_trigger("user", task_id="task-1", max_steps=1))

    assert watchdog.reserve_tool_step().paused is False
    assert watchdog.steps_taken == 1
    decision = watchdog.reserve_tool_step()

    assert decision.paused is False
    assert watchdog.steps_taken == 2


def test_watchdog_records_tokens_past_limit_without_pausing() -> None:
    """token 超过 lease 上限只继续计数，不再暂停运行"""
    watchdog = Watchdog(from_trigger("user", task_id="task-1", max_tokens=10))

    result = watchdog.tick(tokens_used=11)

    assert not result.paused
    assert watchdog.tokens_used == 11


def test_watchdog_tool_timeout_returns_timeout_error() -> None:
    watchdog = Watchdog(
        from_trigger("user", task_id="task-1"), tool_timeout_seconds=0.01
    )

    result = watchdog.run_tool_with_timeout(lambda: time.sleep(0.2))

    assert isinstance(result, ToolError)
    assert result.category is ToolErrorCategory.TIMEOUT
    assert result.partial_state == "stop not confirmed; side effects unknown"
    assert result.details["execution_state"] == "unknown"


def test_watchdog_pause_hotkey_sets_next_tick_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks: dict[str, object] = {}

    fake_keyboard = SimpleNamespace(
        add_hotkey=lambda shortcut, callback: callbacks.setdefault(shortcut, callback)
    )
    monkeypatch.setitem(sys.modules, "keyboard", fake_keyboard)
    watchdog = Watchdog(
        from_trigger("user", task_id="task-1"),
        task_id="task-1",
        segment_id="user-1",
    )

    assert watchdog.register_pause_hotkey()
    callback = callbacks["ctrl+alt+p"]
    assert callable(callback)
    callback()

    result = watchdog.tick()
    assert result.paused
    assert result.reason == "manual_pause"


def test_watchdog_pause_hotkey_failure_is_logged(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "keyboard", SimpleNamespace())
    watchdog = Watchdog(from_trigger("user", task_id="task-1"))

    with caplog.at_level("ERROR", logger="runtime.watchdog"):
        registered = watchdog.register_pause_hotkey()

    assert registered is False
    assert "pause hotkey unavailable" in caplog.text
