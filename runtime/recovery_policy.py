from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

DEFAULT_RECOVERABLE_ERROR_REPEATS = 5
DEFAULT_EMPTY_RESPONSE_REPEATS = 2
DEFAULT_PROTOCOL_ERROR_REPEATS = 1


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    recoverable_error_repeats: int = DEFAULT_RECOVERABLE_ERROR_REPEATS
    empty_response_repeats: int = DEFAULT_EMPTY_RESPONSE_REPEATS
    protocol_error_repeats: int = DEFAULT_PROTOCOL_ERROR_REPEATS


def recovery_policy_from_config(config: Mapping[str, object] | None) -> RecoveryPolicy:
    raw_policy = (config or {}).get("recovery_policy")
    if isinstance(raw_policy, RecoveryPolicy):
        return raw_policy
    if not isinstance(raw_policy, Mapping):
        return RecoveryPolicy()
    return RecoveryPolicy(
        recoverable_error_repeats=_positive_int(
            raw_policy.get("recoverable_error_repeats"),
            DEFAULT_RECOVERABLE_ERROR_REPEATS,
        ),
        empty_response_repeats=_positive_int(
            raw_policy.get("empty_response_repeats"),
            DEFAULT_EMPTY_RESPONSE_REPEATS,
        ),
        protocol_error_repeats=_positive_int(
            raw_policy.get("protocol_error_repeats"),
            DEFAULT_PROTOCOL_ERROR_REPEATS,
        ),
    )


def _positive_int(value: object, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, (str, bytes, bytearray, int)):
        raise ValueError("recovery policy values must be integers")
    parsed = int(value)
    if parsed < 0:
        raise ValueError("recovery policy values must be non-negative")
    return parsed


__all__ = ["RecoveryPolicy", "recovery_policy_from_config"]
