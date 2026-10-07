from __future__ import annotations

import pytest

from llm.types import CacheTier, ErrorCategory, PromptSection, ProviderHint, TokenUsage


def test_token_usage_defaults_to_zero_counts() -> None:
    usage = TokenUsage()

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0


def test_token_usage_adds_all_count_fields() -> None:
    usage = TokenUsage(
        input_tokens=10,
        output_tokens=3,
        cache_read_input_tokens=7,
        cache_creation_input_tokens=2,
    ) + TokenUsage(
        input_tokens=5,
        output_tokens=4,
        cache_read_input_tokens=1,
        cache_creation_input_tokens=8,
    )

    assert usage == TokenUsage(
        input_tokens=15,
        output_tokens=7,
        cache_read_input_tokens=8,
        cache_creation_input_tokens=10,
    )


def test_token_usage_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TokenUsage(input_tokens=-1)


def test_error_category_values_match_v2_1_model_error_contract() -> None:
    assert {category.value for category in ErrorCategory} == {
        "missing_config",
        "auth",
        "billing",
        "transport_error",
        "timeout",
        "rate_limited",
        "overloaded",
        "server_error",
        "context_overflow",
        "payload_too_large",
        "model_not_found",
        "format_error",
        "provider_error",
        "invalid_provider_response",
        "invalid_model_protocol",
        "empty_response",
        "unknown",
    }


def test_error_category_retryable_set_is_narrow() -> None:
    retryable = {category.value for category in ErrorCategory if category.retryable}

    assert retryable == {
        "transport_error",
        "timeout",
        "rate_limited",
        "overloaded",
        "server_error",
    }


def test_provider_hint_accepts_cache_tier_enum_or_value() -> None:
    assert ProviderHint(CacheTier.STABLE).cache_tier is CacheTier.STABLE
    assert ProviderHint("semi_stable").cache_tier is CacheTier.SEMI_STABLE


def test_prompt_section_defaults_to_dynamic_cache_hint() -> None:
    section = PromptSection(name="conversation", content="hello")

    assert section.hint.cache_tier is CacheTier.DYNAMIC
