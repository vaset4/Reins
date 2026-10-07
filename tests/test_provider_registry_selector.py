from __future__ import annotations

import pytest

from llm.model_request import (
    Capability,
    CapabilityRequirement,
    ModelPreference,
    PreferenceKind,
)
from llm.model_registry import (
    ModelDescriptor,
    ModelRegistry,
    ModelSelectionError,
    ModelSelector,
)
from llm.provider_adapter import AdapterRegistry, ProviderAdapterError


class StubAdapter:
    api_family = "stub"

    def stream(
        self,
        request: object,
        *,
        model: object,
        connection: object,
        cancellation: object | None = None,
        prepared_body: object | None = None,
    ) -> object:
        """保留供应商流接口形状供注册测试；参数：请求、连接及发送快照；返回：空事件流。"""
        return iter(())


def descriptor(key: str, *, priority: int, reasoning: bool = False) -> ModelDescriptor:
    return ModelDescriptor(
        model_key=key,
        provider="fixture",
        model_id=key,
        api_family="stub",
        capabilities={
            Capability.STREAMING: True,
            Capability.REASONING: reasoning,
            Capability.CONTEXT_WINDOW_TOKENS: 32000,
            Capability.OUTPUT_TOKENS: 4096,
        },
        preference_values={PreferenceKind.LATENCY_PRIORITY: frozenset({"low_latency"})},
        connection_profile_key="fixture",
        priority=priority,
    )


def test_adapter_registry_rejects_duplicate_and_unknown_family() -> None:
    registry = AdapterRegistry([StubAdapter()])
    assert registry.require("stub").api_family == "stub"
    with pytest.raises(ProviderAdapterError, match="duplicate_api_family"):
        AdapterRegistry([StubAdapter(), StubAdapter()])
    with pytest.raises(ProviderAdapterError, match="unknown_api_family"):
        registry.require("missing")


def test_selector_records_required_rejection_and_optional_notice() -> None:
    registry = ModelRegistry(
        [
            descriptor("fast", priority=10),
            descriptor("deep", priority=1, reasoning=True),
        ]
    )
    decision = ModelSelector(registry).select(
        ("fast", "deep"),
        frozenset({CapabilityRequirement(Capability.REASONING)}),
        (ModelPreference(PreferenceKind.LATENCY_PRIORITY, "low_latency"),),
    )
    assert decision.selected.model_key == "deep"
    assert decision.rejections[0].model_key == "fast"
    assert decision.rejections[0].reasons == ("missing_required_capability:reasoning",)
    assert decision.reason == "highest_optional_score_then_priority_then_allowed_order"


def test_selector_fails_when_no_allowed_model_satisfies_requirements() -> None:
    selector = ModelSelector(ModelRegistry([descriptor("fast", priority=1)]))
    with pytest.raises(ModelSelectionError, match="required_capability_unsatisfied"):
        selector.select(
            ("fast",),
            frozenset({CapabilityRequirement(Capability.OUTPUT_TOKENS, 8192)}),
            (),
        )


def test_descriptor_rejects_string_preference_container() -> None:
    with pytest.raises(ModelSelectionError, match="preference_values"):
        ModelDescriptor(
            model_key="invalid",
            provider="fixture",
            model_id="invalid",
            api_family="stub",
            capabilities={Capability.STREAMING: True},
            preference_values={PreferenceKind.LATENCY_PRIORITY: "low_latency"},  # type: ignore[dict-item]
            connection_profile_key="fixture",
        )
