from __future__ import annotations

from math import nan

import pytest

from llm.messages import (
    AssistantMessage,
    DocumentRefPart,
    ImagePart,
    MessageContractError,
    ProviderStateEnvelope,
    StopReason,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    agent_message_from_mapping,
    agent_message_to_mapping,
    require_compatible_state,
    validate_message_sequence,
)
from llm.types import MeasurementStatus, ModelUsage, UsageMeasurement, usage_to_mapping


def _reported(value: int) -> UsageMeasurement:
    return UsageMeasurement(MeasurementStatus.REPORTED, value)


def _usage() -> ModelUsage:
    return ModelUsage(
        input_tokens=_reported(12),
        output_tokens=_reported(4),
        total_tokens=UsageMeasurement(
            MeasurementStatus.DERIVED,
            16,
            ("input_tokens", "output_tokens"),
        ),
    )


def _tool_call_message() -> AssistantMessage:
    return AssistantMessage(
        message_id="assistant-call",
        content=(
            TextPart("I will inspect it."),
            ToolCallPart("call-1", "file_read", {"path": "README.md"}),
        ),
        stop_reason=StopReason.TOOL_CALL,
    )


def _tool_result_message(*, message_id: str = "tool-result") -> ToolResultMessage:
    return ToolResultMessage(
        message_id=message_id,
        call_id="call-1",
        tool_name="file_read",
        content=(TextPart("contents"),),
        status="success",
        artifact_refs=("artifact-1",),
    )


def test_all_message_variants_round_trip_without_losing_order_or_state() -> None:
    state_payload = {"encrypted": {"signature": "opaque"}, "blocks": [1, True, None]}
    state = ProviderStateEnvelope(
        api_family="anthropic_messages",
        provider="anthropic",
        model="claude-test",
        state_version=1,
        payload=state_payload,
    )
    messages = (
        UserMessage(
            message_id="user-1",
            content=(
                TextPart("inspect", ("artifact-input",)),
                ImagePart("artifact-image", "image/png", width=640, height=480),
                DocumentRefPart("artifact-doc", "design", "text/markdown"),
            ),
        ),
        AssistantMessage(
            message_id="assistant-1",
            content=(
                ThinkingPart("checking", "redacted"),
                TextPart("done"),
            ),
            stop_reason=StopReason.END_TURN,
            provider_state=state,
            usage=_usage(),
        ),
        ToolResultMessage(
            message_id="tool-1",
            call_id="call-1",
            tool_name="file_read",
            content=(
                TextPart("partial output"),
                ImagePart("artifact-result-image", "image/jpeg"),
                DocumentRefPart("artifact-result-doc", "full output"),
            ),
            status="partial",
            error="remaining page unavailable",
            artifact_refs=("artifact-full",),
        ),
    )

    restored = tuple(
        agent_message_from_mapping(agent_message_to_mapping(message))
        for message in messages
    )

    assert restored == messages
    assert [part.kind for part in restored[0].content] == [
        "text",
        "image",
        "document_ref",
    ]
    assert restored[1].provider_state == state
    assert restored[2].status == "partial"
    assert restored[2].artifact_refs == ("artifact-full",)


def test_json_payloads_are_deeply_frozen_and_serialization_returns_fresh_values() -> (
    None
):
    arguments = {"items": [{"name": "before"}]}
    part = ToolCallPart("call-1", "write", arguments)
    arguments["items"][0]["name"] = "after"

    serialized = agent_message_to_mapping(
        AssistantMessage("assistant-1", (part,), stop_reason=StopReason.TOOL_CALL)
    )
    serialized_arguments = serialized["content"][0]["arguments"]
    serialized_arguments["items"][0]["name"] = "serialized-change"
    serialized_again = agent_message_to_mapping(
        AssistantMessage("assistant-2", (part,), stop_reason=StopReason.TOOL_CALL)
    )

    assert serialized_again["content"][0]["arguments"] == {
        "items": [{"name": "before"}]
    }
    with pytest.raises(TypeError):
        part.arguments["new"] = "value"


@pytest.mark.parametrize(
    ("factory", "code"),
    [
        (
            lambda: UserMessage("user", (ThinkingPart("x", "visible"),)),
            "invalid_content_part",
        ),
        (
            lambda: AssistantMessage("assistant", (ImagePart("ref", "image/png"),)),
            "invalid_content_part",
        ),
        (
            lambda: ToolResultMessage(
                "result",
                "call",
                "tool",
                (ThinkingPart("x", "redacted"),),
                "success",
            ),
            "invalid_content_part",
        ),
    ],
)
def test_message_variants_reject_disallowed_content_parts(factory, code: str) -> None:
    with pytest.raises(MessageContractError) as exc_info:
        factory()

    assert exc_info.value.code == code
    assert ".content" in exc_info.value.path


@pytest.mark.parametrize(
    ("factory", "code", "path"),
    [
        (
            lambda: TextPart("x", "ab"),
            "invalid_content_part",
            "text_part.reference_ids",
        ),
        (lambda: TextPart("x", 1), "invalid_content_part", "text_part.reference_ids"),
        (lambda: UserMessage("user", "text"), "invalid_message", "user.content"),
        (lambda: UserMessage("user", 1), "invalid_message", "user.content"),
        (
            lambda: ToolResultMessage(
                "result",
                "call",
                "tool",
                (TextPart("ok"),),
                "success",
                artifact_refs="ab",
            ),
            "invalid_message",
            "tool_result.artifact_refs",
        ),
        (
            lambda: ToolResultMessage(
                "result",
                "call",
                "tool",
                (TextPart("ok"),),
                "success",
                artifact_refs=1,
            ),
            "invalid_message",
            "tool_result.artifact_refs",
        ),
        (
            lambda: ThinkingPart("x", []),
            "invalid_content_part",
            "thinking_part.visibility",
        ),
        (
            lambda: ToolResultMessage(
                "result",
                "call",
                "tool",
                (TextPart("ok"),),
                [],
            ),
            "invalid_message",
            "tool_result.status",
        ),
    ],
)
def test_direct_message_constructors_reject_invalid_runtime_shapes(
    factory,
    code: str,
    path: str,
) -> None:
    with pytest.raises(MessageContractError) as exc_info:
        factory()

    assert exc_info.value.code == code
    assert exc_info.value.path == path


def test_message_parser_rejects_legacy_openai_rows_and_unknown_fields() -> None:
    with pytest.raises(MessageContractError) as legacy_error:
        agent_message_from_mapping({"role": "user", "content": "hello"})
    with pytest.raises(MessageContractError) as field_error:
        agent_message_from_mapping(
            {
                "kind": "user",
                "message_id": "user-1",
                "content": [{"kind": "text", "text": "hello"}],
                "extra": "not allowed",
            }
        )

    assert legacy_error.value.code == "unsupported_legacy_message"
    assert field_error.value.code == "unknown_field"


def test_message_parser_rejects_non_string_field_names_with_typed_error() -> None:
    with pytest.raises(MessageContractError) as exc_info:
        agent_message_from_mapping(
            {
                "kind": "user",
                "message_id": "user-1",
                "content": [{"kind": "text", "text": "hello"}],
                1: "not a JSON field name",
                "extra": "also unknown",
            }
        )

    assert exc_info.value.code == "unknown_field"
    assert exc_info.value.path == "message"


def test_assistant_usage_error_preserves_exact_nested_path() -> None:
    usage = usage_to_mapping(ModelUsage())
    usage["input_tokens"] = {
        "status": "reported",
        "value": None,
        "derived_from": [],
    }

    with pytest.raises(MessageContractError) as exc_info:
        agent_message_from_mapping(
            {
                "kind": "assistant",
                "message_id": "assistant-1",
                "content": [{"kind": "text", "text": "hello"}],
                "usage": usage,
            }
        )

    assert exc_info.value.code == "invalid_usage_measurement"
    assert exc_info.value.path == "message.usage.input_tokens.value"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "api_family": "",
            "provider": "p",
            "model": "m",
            "state_version": 1,
            "payload": {},
        },
        {
            "api_family": "a",
            "provider": "",
            "model": "m",
            "state_version": 1,
            "payload": {},
        },
        {
            "api_family": "a",
            "provider": "p",
            "model": "",
            "state_version": 1,
            "payload": {},
        },
        {
            "api_family": "a",
            "provider": "p",
            "model": "m",
            "state_version": 0,
            "payload": {},
        },
        {
            "api_family": "a",
            "provider": "p",
            "model": "m",
            "state_version": 1,
            "payload": {"raw": b"x"},
        },
        {
            "api_family": "a",
            "provider": "p",
            "model": "m",
            "state_version": 1,
            "payload": {"number": nan},
        },
    ],
)
def test_provider_state_rejects_invalid_identity_version_and_json(payload) -> None:
    with pytest.raises(MessageContractError):
        ProviderStateEnvelope(**payload)


def test_provider_state_compatibility_checks_identity_but_not_opaque_keys() -> None:
    envelope = ProviderStateEnvelope(
        "openai_responses",
        "openai",
        "gpt-test",
        2,
        {"authorization": "opaque-to-core", "raw_response": {"id": "state-1"}},
    )

    require_compatible_state(
        envelope,
        api_family="openai_responses",
        provider="openai",
        model="gpt-test",
    )
    with pytest.raises(MessageContractError) as exc_info:
        require_compatible_state(
            envelope,
            api_family="openai_chat",
            provider="openai",
            model="gpt-test",
        )

    assert exc_info.value.code == "provider_state_mismatch"
    assert agent_message_to_mapping(
        AssistantMessage("assistant", (TextPart("ok"),), provider_state=envelope)
    )["provider_state"]["payload"] == {
        "authorization": "opaque-to-core",
        "raw_response": {"id": "state-1"},
    }


@pytest.mark.parametrize(
    ("messages", "code"),
    [
        (
            (
                UserMessage("same", (TextPart("a"),)),
                UserMessage("same", (TextPart("b"),)),
            ),
            "duplicate_message_id",
        ),
        (
            (
                _tool_call_message(),
                _tool_result_message(),
                AssistantMessage("a2", (ToolCallPart("call-1", "file_read", {}),)),
            ),
            "duplicate_call_id",
        ),
        (
            (
                _tool_call_message(),
                UserMessage("user", (TextPart("never mind"),)),
            ),
            "interleaved_tool_call",
        ),
        ((_tool_result_message(),), "orphan_tool_result"),
        (
            (
                _tool_call_message(),
                _tool_result_message(),
                _tool_result_message(message_id="tool-result-2"),
            ),
            "duplicate_tool_result",
        ),
        (
            (
                _tool_call_message(),
                ToolResultMessage(
                    "result", "call-1", "file_write", (TextPart("x"),), "success"
                ),
            ),
            "tool_name_mismatch",
        ),
        ((_tool_call_message(),), "dangling_tool_call"),
    ],
)
def test_sequence_validator_rejects_invalid_message_and_call_graphs(
    messages, code: str
) -> None:
    with pytest.raises(MessageContractError) as exc_info:
        validate_message_sequence(messages)

    assert exc_info.value.code == code
    assert exc_info.value.path.startswith("messages")


def test_sequence_validator_accepts_one_closed_tool_exchange() -> None:
    messages = (
        UserMessage("user", (TextPart("read"),)),
        _tool_call_message(),
        _tool_result_message(),
        AssistantMessage("assistant-final", (TextPart("done"),)),
    )

    validate_message_sequence(messages)


@pytest.mark.parametrize("messages", ["not messages", 1])
def test_sequence_validator_rejects_invalid_container_with_typed_error(
    messages,
) -> None:
    with pytest.raises(MessageContractError) as exc_info:
        validate_message_sequence(messages)

    assert exc_info.value.code == "invalid_message"
    assert exc_info.value.path == "messages"


@pytest.mark.parametrize(
    "message",
    [
        ToolResultMessage,
    ],
)
def test_tool_result_status_rules_expose_empty_or_inconsistent_results(message) -> None:
    with pytest.raises(MessageContractError):
        message("result", "call", "tool", (), "success")
    with pytest.raises(MessageContractError):
        message("result", "call", "tool", (TextPart("ok"),), "success", error="bad")
    with pytest.raises(MessageContractError):
        message("result", "call", "tool", (), "error")
    with pytest.raises(MessageContractError):
        message("result", "call", "tool", (), "partial", error="incomplete")
