from __future__ import annotations

from dataclasses import fields

import pytest

from llm.messages import (
    AssistantMessage,
    DocumentRefPart,
    ImagePart,
    ProviderStateEnvelope,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
)
from llm.model_request import (
    Capability,
    CapabilityRequirement,
    ModelPreference,
    ModelRequest,
    ModelRequestContractError,
    ModelToolDefinition,
    PreferenceKind,
    RequestMetadata,
    model_request_from_mapping,
    model_request_to_mapping,
)
from llm.types import (
    MeasurementStatus,
    ModelUsage,
    TokenUsage,
    UsageContractError,
    UsageMeasurement,
    usage_from_mapping,
    usage_to_mapping,
)


def _requirement(
    capability: Capability, minimum: int | None = None
) -> CapabilityRequirement:
    return CapabilityRequirement(capability, minimum)


def _closed_tool_messages() -> tuple[object, ...]:
    return (
        UserMessage("user", (TextPart("read"), ImagePart("image", "image/png"))),
        AssistantMessage(
            "assistant-call",
            (
                ThinkingPart("inspect", "visible"),
                ToolCallPart("call-1", "read", {"path": "README.md"}),
            ),
            provider_state=ProviderStateEnvelope(
                "family", "provider", "model", 1, {"state": [1]}
            ),
        ),
        ToolResultMessage(
            "result",
            "call-1",
            "read",
            (DocumentRefPart("document", "contents"),),
            "success",
        ),
    )


def _rich_request() -> ModelRequest:
    return ModelRequest(
        instructions=(TextPart("system instruction"),),
        messages=_closed_tool_messages(),
        tools=(
            ModelToolDefinition(
                "read",
                "Read a file",
                {"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ),
        required_capabilities=frozenset(
            {
                _requirement(Capability.NATIVE_TOOLS),
                _requirement(Capability.IMAGE_INPUT),
                _requirement(Capability.DOCUMENT_INPUT),
                _requirement(Capability.REASONING),
                _requirement(Capability.PROVIDER_STATE_ROUND_TRIP),
                _requirement(Capability.STREAMING),
                _requirement(Capability.OUTPUT_TOKENS, 512),
            }
        ),
        optional_preferences=(
            ModelPreference(PreferenceKind.REASONING_LEVEL, "high"),
            ModelPreference(PreferenceKind.CACHE_RETENTION, "session"),
            ModelPreference(PreferenceKind.LATENCY_PRIORITY, "low_latency"),
        ),
        stream=True,
        max_output_tokens=512,
        request_metadata=RequestMetadata("session-1", "run-1", "turn-1"),
    )


def test_provider_neutral_request_round_trip_preserves_typed_contract() -> None:
    request = _rich_request()

    restored = model_request_from_mapping(model_request_to_mapping(request))

    assert restored == request
    assert [part.text for part in restored.instructions] == ["system instruction"]
    assert restored.request_metadata == RequestMetadata("session-1", "run-1", "turn-1")
    assert {field.name for field in fields(ModelRequest)} == {
        "instructions",
        "messages",
        "tools",
        "required_capabilities",
        "optional_preferences",
        "stream",
        "max_output_tokens",
        "request_metadata",
        "instruction_layers",
        "observations",
    }
    forbidden = {
        "role",
        "tool_calls",
        "tool_call_id",
        "response_format",
        "cache_control",
        "headers",
        "provider_params",
    }
    assert forbidden.isdisjoint(model_request_to_mapping(request))


def test_model_tool_definition_is_narrow_and_deeply_immutable() -> None:
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    definition = ModelToolDefinition("read", "Read", schema)
    schema["properties"]["path"]["type"] = "integer"

    assert {field.name for field in fields(ModelToolDefinition)} == {
        "name",
        "description",
        "input_schema",
    }
    assert model_request_to_mapping(
        ModelRequest(
            instructions=(),
            messages=(UserMessage("user", (TextPart("hello"),)),),
            tools=(definition,),
            required_capabilities=frozenset({_requirement(Capability.NATIVE_TOOLS)}),
        )
    )["tools"][0] == {
        "name": "read",
        "description": "Read",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    }
    with pytest.raises(TypeError):
        definition.input_schema["extra"] = True


@pytest.mark.parametrize(
    ("request_kwargs", "missing"),
    [
        (
            {"tools": (ModelToolDefinition("read", "Read", {"type": "object"}),)},
            Capability.NATIVE_TOOLS,
        ),
        (
            {"messages": (UserMessage("user", (ImagePart("image", "image/png"),)),)},
            Capability.IMAGE_INPUT,
        ),
        (
            {"messages": (UserMessage("user", (DocumentRefPart("doc", "summary"),)),)},
            Capability.DOCUMENT_INPUT,
        ),
        (
            {
                "messages": (
                    AssistantMessage("assistant", (ThinkingPart("x", "visible"),)),
                )
            },
            Capability.REASONING,
        ),
        (
            {
                "messages": (
                    AssistantMessage(
                        "assistant",
                        (TextPart("x"),),
                        provider_state=ProviderStateEnvelope("f", "p", "m", 1, {}),
                    ),
                )
            },
            Capability.PROVIDER_STATE_ROUND_TRIP,
        ),
        ({"stream": True}, Capability.STREAMING),
        ({"max_output_tokens": 128}, Capability.OUTPUT_TOKENS),
    ],
)
def test_request_rejects_missing_structural_capabilities(
    request_kwargs, missing: Capability
) -> None:
    defaults = {
        "instructions": (),
        "messages": (UserMessage("user", (TextPart("hello"),)),),
        "tools": (),
        "required_capabilities": frozenset(),
    }

    with pytest.raises(ModelRequestContractError) as exc_info:
        ModelRequest(**(defaults | request_kwargs))

    assert exc_info.value.code == "missing_required_capability"
    assert missing.value in exc_info.value.detail


def test_output_token_requirement_must_cover_requested_limit() -> None:
    with pytest.raises(ModelRequestContractError) as exc_info:
        ModelRequest(
            instructions=(),
            messages=(UserMessage("user", (TextPart("hello"),)),),
            tools=(),
            required_capabilities=frozenset(
                {_requirement(Capability.OUTPUT_TOKENS, 127)}
            ),
            max_output_tokens=128,
        )

    assert exc_info.value.code == "missing_required_capability"
    assert "minimum 128" in exc_info.value.detail


@pytest.mark.parametrize(
    "requirement",
    [
        lambda: CapabilityRequirement(Capability.NATIVE_TOOLS, 1),
        lambda: CapabilityRequirement(Capability.OUTPUT_TOKENS, None),
        lambda: CapabilityRequirement(Capability.CONTEXT_WINDOW_TOKENS, 0),
    ],
)
def test_capability_requirement_rejects_invalid_minimum_shape(requirement) -> None:
    with pytest.raises(ModelRequestContractError):
        requirement()


def test_request_mapping_rejects_duplicate_and_unknown_capabilities() -> None:
    payload = model_request_to_mapping(
        ModelRequest(
            instructions=(),
            messages=(UserMessage("user", (TextPart("hello"),)),),
            tools=(),
            required_capabilities=frozenset({_requirement(Capability.PROMPT_CACHE)}),
        )
    )
    duplicate = dict(payload)
    duplicate["required_capabilities"] = [
        {"capability": "prompt_cache", "minimum": None},
        {"capability": "prompt_cache", "minimum": None},
    ]
    unknown = dict(payload)
    unknown["required_capabilities"] = [{"capability": "audio_input", "minimum": None}]

    with pytest.raises(ModelRequestContractError) as duplicate_error:
        model_request_from_mapping(duplicate)
    with pytest.raises(ModelRequestContractError) as unknown_error:
        model_request_from_mapping(unknown)

    assert duplicate_error.value.code == "duplicate_capability"
    assert unknown_error.value.code == "unknown_capability"


def test_direct_request_rejects_duplicate_capabilities_before_freezing() -> None:
    requirement = _requirement(Capability.PROMPT_CACHE)

    with pytest.raises(ModelRequestContractError) as exc_info:
        ModelRequest(
            instructions=(),
            messages=(UserMessage("user", (TextPart("hello"),)),),
            tools=(),
            required_capabilities=[requirement, requirement],
        )

    assert exc_info.value.code == "duplicate_capability"
    assert exc_info.value.path == "required_capabilities"


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("messages", 1, "invalid_model_request"),
        ("required_capabilities", 1, "invalid_capability_requirement"),
        ("optional_preferences", 1, "invalid_model_request"),
    ],
)
def test_direct_request_rejects_non_container_fields_with_typed_error(
    field: str,
    value: object,
    code: str,
) -> None:
    kwargs = {
        "instructions": (),
        "messages": (UserMessage("user", (TextPart("hello"),)),),
        "tools": (),
        "required_capabilities": frozenset(),
    }
    kwargs[field] = value

    with pytest.raises(ModelRequestContractError) as exc_info:
        ModelRequest(**kwargs)

    assert exc_info.value.code == code
    assert exc_info.value.path == field


@pytest.mark.parametrize(
    ("kind", "value"),
    [
        (PreferenceKind.REASONING_LEVEL, "unbounded"),
        (PreferenceKind.CACHE_RETENTION, "forever"),
        (PreferenceKind.LATENCY_PRIORITY, "fastest"),
    ],
)
def test_optional_preferences_have_closed_values(
    kind: PreferenceKind, value: str
) -> None:
    with pytest.raises(ModelRequestContractError):
        ModelPreference(kind, value)


def test_optional_preference_rejects_unhashable_value_with_typed_error() -> None:
    with pytest.raises(ModelRequestContractError) as exc_info:
        ModelPreference(PreferenceKind.REASONING_LEVEL, [])

    assert exc_info.value.code == "invalid_preference"
    assert exc_info.value.path == "preferences.reasoning_level.value"


def test_request_rejects_duplicate_preferences_and_preference_cannot_replace_requirement() -> (
    None
):
    base = {
        "instructions": (),
        "messages": (
            AssistantMessage("assistant", (ThinkingPart("think", "visible"),)),
        ),
        "tools": (),
        "required_capabilities": frozenset(),
    }
    with pytest.raises(ModelRequestContractError) as missing_error:
        ModelRequest(
            **base,
            optional_preferences=(
                ModelPreference(PreferenceKind.REASONING_LEVEL, "high"),
            ),
        )
    with pytest.raises(ModelRequestContractError) as duplicate_error:
        ModelRequest(
            **(base | {"messages": (UserMessage("user", (TextPart("hello"),)),)}),
            optional_preferences=(
                ModelPreference(PreferenceKind.REASONING_LEVEL, "low"),
                ModelPreference(PreferenceKind.REASONING_LEVEL, "high"),
            ),
        )

    assert missing_error.value.code == "missing_required_capability"
    assert duplicate_error.value.code == "duplicate_preference"


def test_request_parser_rejects_legacy_messages_unknown_fields_and_metadata() -> None:
    payload = model_request_to_mapping(
        ModelRequest(
            instructions=(),
            messages=(UserMessage("user", (TextPart("hello"),)),),
            tools=(),
            required_capabilities=frozenset(),
        )
    )
    legacy = dict(payload)
    legacy["messages"] = [{"role": "user", "content": "hello"}]
    unknown = dict(payload)
    unknown["headers"] = {"Authorization": "secret"}
    metadata = dict(payload)
    metadata["request_metadata"] = {"session_id": "s", "credential": "secret"}

    with pytest.raises(ModelRequestContractError) as legacy_error:
        model_request_from_mapping(legacy)
    with pytest.raises(ModelRequestContractError) as unknown_error:
        model_request_from_mapping(unknown)
    with pytest.raises(ModelRequestContractError) as metadata_error:
        model_request_from_mapping(metadata)

    assert legacy_error.value.code == "unsupported_legacy_message"
    assert unknown_error.value.code == "unknown_field"
    assert metadata_error.value.code == "unknown_field"


def test_request_parser_rejects_non_string_field_names_with_typed_error() -> None:
    payload = model_request_to_mapping(
        ModelRequest(
            instructions=(),
            messages=(UserMessage("user", (TextPart("hello"),)),),
            tools=(),
            required_capabilities=frozenset(),
        )
    )
    payload[1] = "not a JSON field name"
    payload["extra"] = "also unknown"

    with pytest.raises(ModelRequestContractError) as exc_info:
        model_request_from_mapping(payload)

    assert exc_info.value.code == "unknown_field"
    assert exc_info.value.path == "request"


def test_usage_statuses_round_trip_without_turning_unknown_into_zero() -> None:
    usage = ModelUsage(
        input_tokens=UsageMeasurement(MeasurementStatus.REPORTED, 0),
        output_tokens=UsageMeasurement(MeasurementStatus.UNKNOWN, None),
        total_tokens=UsageMeasurement(
            MeasurementStatus.DERIVED,
            0,
            ("input_tokens", "output_tokens"),
        ),
        cache_read_input_tokens=UsageMeasurement(
            MeasurementStatus.NOT_APPLICABLE, None
        ),
    )

    mapping = usage_to_mapping(usage)

    assert usage_from_mapping(mapping) == usage
    assert mapping["input_tokens"] == {
        "status": "reported",
        "value": 0,
        "derived_from": [],
    }
    assert mapping["output_tokens"]["value"] is None
    assert mapping["cache_read_input_tokens"]["value"] is None
    assert ModelUsage().input_tokens.status is MeasurementStatus.UNKNOWN


@pytest.mark.parametrize(
    "measurement",
    [
        lambda: UsageMeasurement(MeasurementStatus.REPORTED, None),
        lambda: UsageMeasurement(MeasurementStatus.REPORTED, -1),
        lambda: UsageMeasurement(MeasurementStatus.REPORTED, 1, ("input",)),
        lambda: UsageMeasurement(MeasurementStatus.DERIVED, 1),
        lambda: UsageMeasurement(MeasurementStatus.UNKNOWN, 0),
        lambda: UsageMeasurement(MeasurementStatus.NOT_APPLICABLE, None, ("provider",)),
    ],
)
def test_usage_measurements_reject_invalid_status_value_combinations(
    measurement,
) -> None:
    with pytest.raises(UsageContractError) as exc_info:
        measurement()

    assert exc_info.value.code == "invalid_usage_measurement"


@pytest.mark.parametrize("sources", ["input_tokens", 1])
def test_usage_measurement_rejects_invalid_source_container_with_typed_error(
    sources: object,
) -> None:
    with pytest.raises(UsageContractError) as exc_info:
        UsageMeasurement(MeasurementStatus.DERIVED, 1, sources)

    assert exc_info.value.code == "invalid_usage_measurement"
    assert exc_info.value.path == "usage.derived_from"


def test_usage_parser_rejects_legacy_token_usage_and_unknown_keys() -> None:
    with pytest.raises(UsageContractError):
        usage_from_mapping(TokenUsage())
    with pytest.raises(UsageContractError) as exc_info:
        usage_from_mapping(
            {
                **usage_to_mapping(ModelUsage()),
                "provider_tokens": {
                    "status": "reported",
                    "value": 1,
                    "derived_from": [],
                },
            }
        )

    assert exc_info.value.code == "unknown_field"


def test_usage_parser_rejects_non_string_field_names_with_typed_error() -> None:
    payload = usage_to_mapping(ModelUsage())
    payload[1] = "not a JSON field name"
    payload["extra"] = "also unknown"

    with pytest.raises(UsageContractError) as exc_info:
        usage_from_mapping(payload)

    assert exc_info.value.code == "unknown_field"
    assert exc_info.value.path == "usage"
