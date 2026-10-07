from __future__ import annotations

import pytest

from llm.retry_utils import call_with_retries, retry_delays
from llm.types import ErrorCategory


def test_retry_delays_use_exponential_backoff() -> None:
    assert retry_delays(max_retries=3, base_delay=1.0) == [1.0, 2.0, 4.0]
    assert retry_delays(max_retries=2) == [2.0, 4.0]


def test_call_with_retries_retries_retryable_errors() -> None:
    attempts = 0
    sleeps: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise TimeoutError("timed out")
        return "ok"

    result = call_with_retries(operation, sleep=sleeps.append)

    assert result == "ok"
    assert attempts == 3
    assert sleeps == [2.0, 4.0]


def test_call_with_retries_does_not_retry_non_retryable_errors() -> None:
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("context length exceeded")

    with pytest.raises(ValueError):
        call_with_retries(
            operation,
            sleep=lambda _delay: None,
            classify_error=lambda _exc: ErrorCategory.CONTEXT_OVERFLOW,
        )

    assert attempts == 1
