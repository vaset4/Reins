from __future__ import annotations

from collections.abc import Mapping

from llm.types import ErrorCategory


MODEL_ERROR_DEFAULT_RETRYABLE: Mapping[ErrorCategory, bool] = {
    ErrorCategory.MISSING_CONFIG: False,
    ErrorCategory.AUTH: False,
    ErrorCategory.BILLING: False,
    ErrorCategory.TRANSPORT: True,
    ErrorCategory.TIMEOUT: True,
    ErrorCategory.RATE_LIMITED: True,
    ErrorCategory.OVERLOADED: True,
    ErrorCategory.SERVER_ERROR: True,
    ErrorCategory.CONTEXT_OVERFLOW: False,
    ErrorCategory.PAYLOAD_TOO_LARGE: False,
    ErrorCategory.MODEL_NOT_FOUND: False,
    ErrorCategory.FORMAT_ERROR: False,
    ErrorCategory.PROVIDER_ERROR: False,
    ErrorCategory.INVALID_PROVIDER_RESPONSE: False,
    ErrorCategory.INVALID_PROTOCOL: False,
    ErrorCategory.EMPTY_RESPONSE: False,
    ErrorCategory.UNKNOWN: False,
}


def classify_provider_exception(exc: BaseException) -> ErrorCategory:
    message = str(exc)
    lowered = message.lower()
    class_name = exc.__class__.__name__
    module_name = exc.__class__.__module__.lower()
    status_code = _extract_status_code(exc)

    if isinstance(exc, TimeoutError) or class_name in {"APITimeoutError", "Timeout"}:
        return ErrorCategory.TIMEOUT
    if isinstance(exc, (ConnectionError, OSError)) or class_name in {
        "APIConnectionError",
        "TransportError",
    }:
        return ErrorCategory.TRANSPORT

    if status_code == 401 or status_code == 403:
        return ErrorCategory.AUTH
    if status_code == 402:
        if _contains_any(lowered, ("try again", "retry", "resets at", "reset in")):
            return ErrorCategory.RATE_LIMITED
        return ErrorCategory.BILLING
    if status_code == 404 and _contains_any(
        lowered, ("model", "not found", "does not exist", "invalid model")
    ):
        return ErrorCategory.MODEL_NOT_FOUND
    if status_code == 413:
        return ErrorCategory.PAYLOAD_TOO_LARGE
    if status_code == 429:
        return ErrorCategory.RATE_LIMITED
    if status_code in (503, 529):
        return ErrorCategory.OVERLOADED
    if status_code in (500, 502):
        return ErrorCategory.SERVER_ERROR

    if class_name == "RateLimitError" or _contains_any(
        lowered, ("rate limit", "rate_limited", "too many requests")
    ):
        return ErrorCategory.RATE_LIMITED
    if _contains_any(
        lowered,
        (
            "unauthorized",
            "forbidden",
            "invalid api key",
            "invalid_api_key",
            "access denied",
            "authentication",
        ),
    ):
        return ErrorCategory.AUTH
    if _contains_any(
        lowered,
        (
            "insufficient credits",
            "insufficient_quota",
            "payment required",
            "billing",
            "credit balance",
            "credits have been exhausted",
        ),
    ):
        return ErrorCategory.BILLING
    if _contains_any(
        lowered,
        (
            "model not found",
            "model_not_found",
            "invalid model",
            "no such model",
            "unknown model",
            "does not exist",
        ),
    ):
        return ErrorCategory.MODEL_NOT_FOUND
    if _contains_any(lowered, ("api key", "missing config", "missing_config")):
        return ErrorCategory.MISSING_CONFIG
    if _looks_like_context_overflow(lowered):
        return ErrorCategory.CONTEXT_OVERFLOW
    if _contains_any(lowered, ("payload too large", "request entity too large", "413")):
        return ErrorCategory.PAYLOAD_TOO_LARGE
    if _contains_any(lowered, ("overloaded", "503", "529")):
        return ErrorCategory.OVERLOADED
    if _contains_any(lowered, ("server error", "internal server error", "500", "502")):
        return ErrorCategory.SERVER_ERROR
    if _contains_any(lowered, ("invalid protocol", "invalid_model_protocol")):
        return ErrorCategory.INVALID_PROTOCOL
    if _contains_any(
        lowered, ("invalid provider response", "invalid_provider_response")
    ):
        return ErrorCategory.INVALID_PROVIDER_RESPONSE
    if _contains_any(lowered, ("empty response", "empty_response", "empty content")):
        return ErrorCategory.EMPTY_RESPONSE
    if class_name in {"BadRequestError", "APIError"} or (
        module_name.startswith(("anthropic", "openai")) and class_name.endswith("Error")
    ):
        return ErrorCategory.PROVIDER_ERROR
    return ErrorCategory.UNKNOWN


def classify_http_status(
    status_code: int,
    message: str = "",
) -> ErrorCategory:
    lowered = message.lower()
    if status_code == 401 or status_code == 403:
        return ErrorCategory.AUTH
    if status_code == 402:
        if _contains_any(lowered, ("try again", "retry", "resets at", "reset in")):
            return ErrorCategory.RATE_LIMITED
        return ErrorCategory.BILLING
    if status_code == 404:
        if _contains_any(
            lowered, ("model", "not found", "does not exist", "invalid model")
        ):
            return ErrorCategory.MODEL_NOT_FOUND
        return ErrorCategory.PROVIDER_ERROR
    if status_code == 413:
        return ErrorCategory.PAYLOAD_TOO_LARGE
    if status_code == 429:
        return ErrorCategory.RATE_LIMITED
    if status_code in (503, 529):
        return ErrorCategory.OVERLOADED
    if status_code in (500, 502):
        return ErrorCategory.SERVER_ERROR
    if 400 <= status_code < 500:
        if _looks_like_context_overflow(lowered):
            return ErrorCategory.CONTEXT_OVERFLOW
        return ErrorCategory.FORMAT_ERROR
    if 500 <= status_code < 600:
        return ErrorCategory.SERVER_ERROR
    return ErrorCategory.UNKNOWN


def is_retryable(category: ErrorCategory) -> bool:
    return MODEL_ERROR_DEFAULT_RETRYABLE.get(category, False)


def _extract_status_code(exc: BaseException) -> int | None:
    current: BaseException | None = exc
    for _ in range(5):
        if current is None:
            break
        code = getattr(current, "status_code", None)
        if isinstance(code, int):
            return code
        code = getattr(current, "status", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
        current = current.__cause__ or current.__context__
    return None


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _looks_like_context_overflow(message: str) -> bool:
    return _contains_any(
        message,
        (
            "context length",
            "context window",
            "maximum context",
            "maximum tokens",
            "too many tokens",
            "token limit",
            "prompt is too long",
        ),
    )


__all__ = [
    "MODEL_ERROR_DEFAULT_RETRYABLE",
    "classify_http_status",
    "classify_provider_exception",
    "is_retryable",
]
