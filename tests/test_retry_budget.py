from __future__ import annotations

from runtime.lease import from_trigger
from runtime.watchdog import Watchdog, retry_with_budget


def test_retry_with_budget_allows_three_retries() -> None:
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 4:
            raise TimeoutError("temporary")
        return "ok"

    assert retry_with_budget(operation, sleep=lambda _delay: None) == "ok"
    assert attempts == 4


def test_tool_failure_count_does_not_replace_dispatch_budget() -> None:
    watchdog = Watchdog(from_trigger("user", task_id="task-1"))

    for index in range(9):
        assert not watchdog.record_tool_failure(
            "file_read", {"path": str(index)}
        ).paused

    result = watchdog.record_tool_failure("file_read", {"path": "9"})
    assert not result.paused
    assert sum(watchdog.tool_failures.values()) == 10


def test_segment_llm_failure_budget_pauses_at_three() -> None:
    watchdog = Watchdog(from_trigger("user", task_id="task-1"))

    assert not watchdog.record_llm_failure().paused
    assert not watchdog.record_llm_failure().paused
    result = watchdog.record_llm_failure()

    assert result.paused
    assert result.reason == "llm_failure_budget"
