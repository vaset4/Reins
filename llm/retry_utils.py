from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from llm.error_classifier import classify_provider_exception, is_retryable
from llm.types import ErrorCategory

T = TypeVar("T")

MAX_TOTAL_ATTEMPTS = 3
BASE_DELAY = 2.0
MAX_DELAY = 60.0
RETRY_AFTER_CAP = 120.0


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
    jitter_ratio: float = 0.5,
) -> float:
    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        delay = max_delay
    else:
        delay = min(base_delay * (2**exponent), max_delay)
    jitter = random.uniform(0, jitter_ratio * delay)
    return delay + jitter


def compute_wait(
    attempt: int,
    *,
    retry_after: float | None = None,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
) -> float:
    if retry_after is not None and retry_after > 0:
        return min(retry_after, RETRY_AFTER_CAP)
    return jittered_backoff(attempt, base_delay=base_delay, max_delay=max_delay)


def retry_delays(
    *,
    max_retries: int = MAX_TOTAL_ATTEMPTS - 1,
    base_delay: float = BASE_DELAY,
    factor: float = 2.0,
    max_delay: float = MAX_DELAY,
) -> list[float]:
    return [
        min(base_delay * (factor**attempt), max_delay) for attempt in range(max_retries)
    ]


def call_with_retries(
    operation: Callable[[], T],
    *,
    max_retries: int = MAX_TOTAL_ATTEMPTS - 1,
    sleep: Callable[[float], None] = time.sleep,
    classify_error: Callable[
        [BaseException], ErrorCategory
    ] = classify_provider_exception,
) -> T:
    delays = retry_delays(max_retries=max_retries)
    attempts = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            category = classify_error(exc)
            if attempts >= max_retries or not is_retryable(category):
                raise
            sleep(delays[attempts])
            attempts += 1


__all__ = [
    "MAX_TOTAL_ATTEMPTS",
    "call_with_retries",
    "compute_wait",
    "jittered_backoff",
    "retry_delays",
]
