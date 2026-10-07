from __future__ import annotations

from llm.error_classifier import classify_provider_exception, is_retryable
from llm.types import ErrorCategory


class RateLimitError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class APITimeoutError(Exception):
    pass


class BadRequestError(Exception):
    pass


class APIError(Exception):
    pass


def test_classify_provider_exception_covers_all_categories() -> None:
    cases = [
        (RuntimeError("api key missing"), ErrorCategory.MISSING_CONFIG),
        (APIConnectionError("connection reset"), ErrorCategory.TRANSPORT),
        (APITimeoutError("timeout"), ErrorCategory.TIMEOUT),
        (RateLimitError("rate limit"), ErrorCategory.RATE_LIMITED),
        (BadRequestError("maximum context length"), ErrorCategory.CONTEXT_OVERFLOW),
        (APIError("provider failed"), ErrorCategory.PROVIDER_ERROR),
        (
            RuntimeError("invalid provider response"),
            ErrorCategory.INVALID_PROVIDER_RESPONSE,
        ),
        (RuntimeError("invalid protocol"), ErrorCategory.INVALID_PROTOCOL),
        (RuntimeError("empty response"), ErrorCategory.EMPTY_RESPONSE),
        (RuntimeError("something else"), ErrorCategory.UNKNOWN),
    ]

    for exc, category in cases:
        assert classify_provider_exception(exc) is category


def test_retryable_categories_match_v2_1_contract() -> None:
    assert {category for category in ErrorCategory if is_retryable(category)} == {
        ErrorCategory.TRANSPORT,
        ErrorCategory.TIMEOUT,
        ErrorCategory.RATE_LIMITED,
        ErrorCategory.OVERLOADED,
        ErrorCategory.SERVER_ERROR,
    }
