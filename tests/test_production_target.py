from __future__ import annotations

import dataclasses

import pytest

from llm.config import LLMProviderConfig
from llm.model_registry import ModelSelectionError, ModelSelector
from llm.model_request import Capability
from llm.production_target import (
    PRODUCTION_MODEL_KEY,
    _InlineCredential,
    production_allowed_model_keys,
    production_connection,
    production_model_registry,
)
from llm.provider_connection import ProviderConnectionError
from llm.resolved_target import ResolvedModelTarget
from tests.provider_test_support import text_request, tool_roundtrip_request


TEST_SECRET = "synthetic-production-token"


def make_target(**overrides: object) -> ResolvedModelTarget:
    """构造生产配置解析结果的测试替身。

    作者：LKX
    时间：2026-08-30 16:40:00
    传参：overrides 覆盖单个字段
    返回：ResolvedModelTarget
    """
    target = ResolvedModelTarget(
        provider="openai_compatible",
        model="glm-5.1",
        base_url="https://proxy.example.com/v1",
        api_mode="chat_completions",
        timeout_seconds=45.0,
        config_source="file_default",
        credential_source="secrets_vault",
        api_key_present=True,
        api_key=TEST_SECRET,
        context_window=128000,
        context_window_source="file_default",
        context_window_defaulted=False,
        profile_name="remote-proxy",
        credential_name="proxy_api_key",
    )
    return dataclasses.replace(target, **overrides)  # type: ignore[arg-type]


def make_config(target: ResolvedModelTarget) -> LLMProviderConfig:
    """按 app/cli.py:117-122 的同一口径从 target 造 config。"""
    return LLMProviderConfig(
        base_url=target.base_url,
        model=target.model,
        api_key=target.api_key,
        timeout_seconds=target.timeout_seconds,
    )


def test_descriptor_maps_identity_family_and_connection_reference() -> None:
    target = make_target()
    descriptor = production_model_registry(target, make_config(target)).require(
        PRODUCTION_MODEL_KEY,
    )
    assert descriptor.api_family == "openai_chat"
    assert descriptor.provider == "openai_compatible"
    assert descriptor.model_id == "glm-5.1"
    assert descriptor.connection_profile_key == "remote-proxy"
    assert production_allowed_model_keys() == (PRODUCTION_MODEL_KEY,)


def test_descriptor_declares_four_capabilities_with_configured_window() -> None:
    target = make_target(context_window=64000)
    descriptor = production_model_registry(target, make_config(target)).require(
        PRODUCTION_MODEL_KEY,
    )
    assert descriptor.capabilities[Capability.NATIVE_TOOLS] is True
    assert descriptor.capabilities[Capability.STREAMING] is True
    assert descriptor.capabilities[Capability.REASONING] is True
    assert descriptor.capabilities[Capability.CONTEXT_WINDOW_TOKENS] == 64000


def test_unsupported_api_mode_fails_instead_of_defaulting_to_a_family() -> None:
    target = make_target(api_mode="unknown")
    with pytest.raises(ModelSelectionError, match="unsupported_api_mode:unknown"):
        production_model_registry(target, make_config(target))


def test_connection_carries_base_url_timeout_and_bearer_header() -> None:
    target = make_target()
    connection = production_connection(target)
    assert connection.base_url == "https://proxy.example.com/v1"
    assert connection.timeout_seconds == 45.0
    assert connection.headers["Authorization"] == f"Bearer {TEST_SECRET}"


def test_missing_api_key_fails_closed_with_missing_config() -> None:
    target = make_target(api_key=None, api_key_present=False, credential_source="none")
    with pytest.raises(ProviderConnectionError, match="missing_config"):
        production_connection(target)


def test_connection_repr_does_not_leak_the_credential() -> None:
    target = make_target()
    connection = production_connection(target)
    assert TEST_SECRET not in repr(connection)
    assert connection.credential == TEST_SECRET


def test_blank_profile_name_still_yields_a_valid_connection_profile_key() -> None:
    target = make_target(profile_name="", credential_name="")
    descriptor = production_model_registry(target, make_config(target)).require(
        PRODUCTION_MODEL_KEY,
    )
    assert descriptor.connection_profile_key.strip()
    connection = production_connection(target)
    assert connection.headers["Authorization"] == f"Bearer {TEST_SECRET}"


def test_credential_resolver_redacts_repr_and_rejects_a_foreign_reference() -> None:
    credential = _InlineCredential("proxy_api_key", TEST_SECRET)
    assert credential.resolve("proxy_api_key") == TEST_SECRET
    assert TEST_SECRET not in repr(credential)
    with pytest.raises(ProviderConnectionError, match="unknown_credential_ref"):
        credential.resolve("other_api_key")


def test_selector_picks_production_model_for_real_tool_and_stream_requests() -> None:
    target = make_target()
    selector = ModelSelector(production_model_registry(target, make_config(target)))
    for request in (tool_roundtrip_request(), text_request(stream=True)):
        decision = selector.select(
            production_allowed_model_keys(),
            request.required_capabilities,
            request.optional_preferences,
        )
        assert decision.selected.model_key == PRODUCTION_MODEL_KEY
        assert decision.rejections == ()
