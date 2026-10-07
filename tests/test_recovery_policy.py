from __future__ import annotations

import pytest

from runtime.recovery_policy import RecoveryPolicy, recovery_policy_from_config


def test_recovery_policy_defaults() -> None:
    assert recovery_policy_from_config(None) == RecoveryPolicy(
        recoverable_error_repeats=5,
        empty_response_repeats=2,
        protocol_error_repeats=1,
    )


def test_recovery_policy_reads_runtime_config() -> None:
    policy = recovery_policy_from_config(
        {
            "recovery_policy": {
                "recoverable_error_repeats": 1,
                "empty_response_repeats": 0,
                "protocol_error_repeats": 3,
            }
        }
    )

    assert policy == RecoveryPolicy(
        recoverable_error_repeats=1,
        empty_response_repeats=0,
        protocol_error_repeats=3,
    )


def test_recovery_policy_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        recovery_policy_from_config(
            {"recovery_policy": {"recoverable_error_repeats": -1}}
        )
