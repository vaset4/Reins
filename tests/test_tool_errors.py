from __future__ import annotations

from tools.types import ToolError, ToolErrorCategory


def test_tool_error_categories_are_complete() -> None:
    assert {item.value for item in ToolErrorCategory} == {
        "timeout",
        "permission",
        "invalid_input",
        "transport",
        "unknown",
        "cancelled",
        "business",
    }


def test_tool_error_retryable_defaults_follow_category() -> None:
    assert ToolError(ToolErrorCategory.TIMEOUT, "timeout").retryable is True
    assert ToolError(ToolErrorCategory.TRANSPORT, "transport").retryable is True
    assert ToolError(ToolErrorCategory.PERMISSION, "permission").retryable is False
    assert ToolError(ToolErrorCategory.INVALID_INPUT, "invalid").retryable is False
    assert ToolError(ToolErrorCategory.UNKNOWN, "unknown").retryable is False
    assert ToolError(ToolErrorCategory.CANCELLED, "cancelled").retryable is False
    assert ToolError(ToolErrorCategory.BUSINESS, "business").retryable is False


def test_tool_error_accepts_explicit_retryable_and_partial_state() -> None:
    error = ToolError(
        ToolErrorCategory.TIMEOUT,
        "timed out",
        retryable=False,
        partial_state="killed after 30s, side effects unknown",
    )

    assert error.retryable is False
    assert error.partial_state == "killed after 30s, side effects unknown"
