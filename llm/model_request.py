from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Collection, Literal, Mapping, Sequence, cast

from llm.messages import (
    AgentMessage,
    AssistantMessage,
    DocumentRefPart,
    ImagePart,
    JsonValue,
    MessageContractError,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    agent_message_from_mapping,
    agent_message_to_mapping,
    content_part_from_mapping,
    content_part_to_mapping,
    freeze_json_object,
    thaw_json_value,
    validate_message_sequence,
)
from tools.tool_registry import ToolDefinition, ToolRegistry
from context.window import output_reserve, request_budget


PROMPT_POLICY_VERSION = "native-actions-v2"
MODEL_ACTIONS = frozenset({"final", "run_tools"})


class ModelRequestContractError(ValueError):
    """表示 Provider 无关请求合同构造或反序列化失败。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：code 为稳定错误码；path 为失败路径；detail 为具体原因
    返回：携带 code、path 和 detail 的 ValueError
    """

    def __init__(self, code: str, path: str, detail: str) -> None:
        """初始化可定位的模型请求合同错误。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：code 为稳定错误码；path 为失败路径；detail 为具体原因
        返回：无；构造异常对象
        """
        self.code = code
        self.path = path
        self.detail = detail
        super().__init__(f"{code} at {path}: {detail}")


def _request_error(code: str, path: str, detail: str) -> ModelRequestContractError:
    return ModelRequestContractError(code, path, detail)


def _require_request_text(value: object, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise _request_error(
            "invalid_model_request", path, "value must be a non-empty string"
        )


@dataclass(frozen=True, slots=True)
class ModelToolDefinition:
    """保存模型可见且不含执行状态的 Provider 无关工具定义。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：name/description 为模型可见文本；input_schema 为 JSON object schema
    返回：深冻结 schema 的窄工具 DTO
    """

    name: str
    description: str
    input_schema: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        """校验工具文本并深冻结输入 schema。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法定义抛 ModelRequestContractError
        """
        _require_request_text(self.name, "tool.name")
        _require_request_text(self.description, "tool.description")
        try:
            frozen = freeze_json_object(self.input_schema, path="tool.input_schema")
        except MessageContractError as exc:
            raise _request_error(
                "invalid_tool_definition", exc.path, exc.detail
            ) from exc
        object.__setattr__(self, "input_schema", frozen)


class Capability(str, Enum):
    """定义模型选择必须显式满足的闭合能力集合。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：枚举值来自调用方或严格反序列化入口
    返回：Provider 无关能力标识
    """

    NATIVE_TOOLS = "native_tools"
    IMAGE_INPUT = "image_input"
    DOCUMENT_INPUT = "document_input"
    REASONING = "reasoning"
    PROVIDER_STATE_ROUND_TRIP = "provider_state_round_trip"
    STREAMING = "streaming"
    PROMPT_CACHE = "prompt_cache"
    CONTEXT_WINDOW_TOKENS = "context_window_tokens"
    OUTPUT_TOKENS = "output_tokens"


_TOKEN_CAPABILITIES = frozenset(
    {Capability.CONTEXT_WINDOW_TOKENS, Capability.OUTPUT_TOKENS}
)

# 生产路径只走流式：Adapter 只消费 chunk 流，非流式响应会被判 invalid_provider_response
_PRODUCTION_STREAMING = True


@dataclass(frozen=True, slots=True)
class CapabilityRequirement:
    """保存一项必需能力及可选 token 下限。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：capability 为闭合能力；minimum 仅用于 token 能力且必须为正整数
    返回：不可变能力要求
    """

    capability: Capability
    minimum: int | None = None

    def __post_init__(self) -> None:
        """校验布尔能力与 token 能力的值形状。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法要求抛 ModelRequestContractError
        """
        if not isinstance(self.capability, Capability):
            raise _request_error(
                "unknown_capability",
                "required_capabilities.capability",
                "capability must be a Capability",
            )
        if self.capability in _TOKEN_CAPABILITIES:
            if not _positive_request_int(self.minimum):
                raise _request_error(
                    "invalid_capability_requirement",
                    f"required_capabilities.{self.capability.value}.minimum",
                    "token capability minimum must be a positive integer",
                )
            return
        if self.minimum is not None:
            raise _request_error(
                "invalid_capability_requirement",
                f"required_capabilities.{self.capability.value}.minimum",
                "boolean capability must not define minimum",
            )


class PreferenceKind(str, Enum):
    """定义可观察但不替代 required capability 的偏好种类。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：枚举值来自调用方或严格反序列化入口
    返回：Provider 无关偏好标识
    """

    REASONING_LEVEL = "reasoning_level"
    CACHE_RETENTION = "cache_retention"
    LATENCY_PRIORITY = "latency_priority"


_PREFERENCE_VALUES: Mapping[PreferenceKind, frozenset[str]] = {
    PreferenceKind.REASONING_LEVEL: frozenset(
        {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
    ),
    PreferenceKind.CACHE_RETENTION: frozenset({"request", "session"}),
    PreferenceKind.LATENCY_PRIORITY: frozenset({"balanced", "low_latency"}),
}


@dataclass(frozen=True, slots=True)
class ModelPreference:
    """保存一个闭合值域的 Provider 无关模型偏好。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：kind 为偏好种类；value 为该种类允许的闭合值
    返回：不可变模型偏好
    """

    kind: PreferenceKind
    value: str

    def __post_init__(self) -> None:
        """校验偏好种类和值域。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法偏好抛 ModelRequestContractError
        """
        if not isinstance(self.kind, PreferenceKind):
            raise _request_error(
                "unknown_preference", "preferences.kind", "unknown preference kind"
            )
        if (
            not isinstance(self.value, str)
            or self.value not in _PREFERENCE_VALUES[self.kind]
        ):
            raise _request_error(
                "invalid_preference",
                f"preferences.{self.kind.value}.value",
                f"unsupported value: {self.value!r}",
            )


@dataclass(frozen=True, slots=True)
class RequestMetadata:
    """保存不含任意扩展字段的请求关联标识。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：session_id/run_id/turn_id 为可选关联标识
    返回：不可变请求 metadata
    """

    session_id: str = ""
    run_id: str = ""
    turn_id: str = ""

    def __post_init__(self) -> None:
        """校验三个关联标识只接受字符串。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法标识抛 ModelRequestContractError
        """
        for name in ("session_id", "run_id", "turn_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or (value and not value.strip()):
                raise _request_error(
                    "invalid_request_metadata",
                    f"request_metadata.{name}",
                    "id must be a string and non-blank when present",
                )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """保存一次完整且能力声明充分的 Provider 无关模型请求。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：instructions/messages/tools 为语义输入；capabilities/preferences 为选择意图；其余为通用限制
    返回：经过消息图和结构能力校验的不可变请求
    """

    instructions: tuple[TextPart, ...]
    messages: tuple[AgentMessage, ...]
    tools: tuple[ModelToolDefinition, ...]
    required_capabilities: frozenset[CapabilityRequirement]
    optional_preferences: tuple[ModelPreference, ...] = ()
    stream: bool = False
    max_output_tokens: int | None = None
    request_metadata: RequestMetadata = RequestMetadata()
    instruction_layers: tuple[str, ...] = ()
    observations: tuple[TextPart, ...] = ()

    def __post_init__(self) -> None:
        """复制输入容器并校验类型、消息图和结构能力。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法请求抛 ModelRequestContractError
        """
        _copy_and_validate_request_fields(self)
        try:
            validate_message_sequence(self.messages)
        except MessageContractError as exc:
            raise _request_error(exc.code, exc.path, exc.detail) from exc
        _validate_structural_capabilities(self)


def _copy_and_validate_request_fields(request: ModelRequest) -> None:
    instructions = _copy_request_sequence(request.instructions, "instructions")
    observations = _copy_request_sequence(request.observations, "observations")
    if any(not isinstance(item, TextPart) for item in observations):
        raise _request_error(
            "invalid_observation", "observations", "only text observations are allowed"
        )
    object.__setattr__(request, "observations", observations)
    messages = _copy_request_sequence(request.messages, "messages")
    tools = _copy_request_sequence(request.tools, "tools")
    preferences = _copy_request_sequence(
        request.optional_preferences,
        "optional_preferences",
    )
    requirement_values = _copy_requirement_values(request.required_capabilities)
    _validate_request_field_types(
        instructions,
        messages,
        tools,
        requirements=requirement_values,
        preferences=preferences,
    )
    _validate_request_options(request)
    layers = _copy_request_sequence(request.instruction_layers, "instruction_layers")
    if layers and (
        len(layers) != len(instructions)
        or any(
            not isinstance(layer, str)
            or layer not in {"stable", "baseline", "delta", "dynamic", "ephemeral"}
            for layer in layers
        )
    ):
        raise _request_error(
            "invalid_instruction_layers",
            "instruction_layers",
            "layers must match ordered instructions",
        )
    object.__setattr__(request, "instruction_layers", layers)
    object.__setattr__(request, "instructions", instructions)
    object.__setattr__(request, "messages", messages)
    object.__setattr__(request, "tools", tools)
    object.__setattr__(
        request,
        "required_capabilities",
        frozenset(cast(tuple[CapabilityRequirement, ...], requirement_values)),
    )
    object.__setattr__(request, "optional_preferences", preferences)


def _copy_request_sequence(value: object, path: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _request_error(
            "invalid_model_request",
            path,
            "value must be an ordered sequence",
        )
    return tuple(value)


def _copy_requirement_values(value: object) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise _request_error(
            "invalid_capability_requirement",
            "required_capabilities",
            "value must be a collection",
        )
    return tuple(value)


def _validate_request_field_types(
    instructions: tuple[object, ...],
    messages: tuple[object, ...],
    tools: tuple[object, ...],
    *,
    requirements: tuple[object, ...],
    preferences: tuple[object, ...],
) -> None:
    if any(not isinstance(item, TextPart) for item in instructions):
        raise _request_error(
            "invalid_instruction", "instructions", "only TextPart is allowed"
        )
    if not messages or any(
        not isinstance(item, (UserMessage, AssistantMessage, ToolResultMessage))
        for item in messages
    ):
        raise _request_error(
            "invalid_message", "messages", "messages must contain AgentMessage"
        )
    if any(not isinstance(item, ModelToolDefinition) for item in tools):
        raise _request_error(
            "invalid_tool_definition", "tools", "invalid tool definition"
        )
    if any(not isinstance(item, CapabilityRequirement) for item in requirements):
        raise _request_error(
            "invalid_capability_requirement",
            "required_capabilities",
            "invalid requirement",
        )
    if any(not isinstance(item, ModelPreference) for item in preferences):
        raise _request_error(
            "invalid_preference", "optional_preferences", "invalid preference"
        )
    _reject_duplicate_capabilities(
        cast(tuple[CapabilityRequirement, ...], requirements)
    )
    _reject_duplicate_preferences(cast(tuple[ModelPreference, ...], preferences))


def _validate_request_options(request: ModelRequest) -> None:
    if not isinstance(request.stream, bool):
        raise _request_error("invalid_model_request", "stream", "stream must be bool")
    if request.max_output_tokens is not None and not _positive_request_int(
        request.max_output_tokens
    ):
        raise _request_error(
            "invalid_model_request",
            "max_output_tokens",
            "max_output_tokens must be a positive integer",
        )
    if not isinstance(request.request_metadata, RequestMetadata):
        raise _request_error(
            "invalid_request_metadata",
            "request_metadata",
            "value must be RequestMetadata",
        )


def _positive_request_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _reject_duplicate_capabilities(
    requirements: Sequence[CapabilityRequirement],
) -> None:
    capabilities = [item.capability for item in requirements]
    if len(set(capabilities)) != len(capabilities):
        raise _request_error(
            "duplicate_capability",
            "required_capabilities",
            "each capability may appear only once",
        )


def _reject_duplicate_preferences(preferences: tuple[ModelPreference, ...]) -> None:
    kinds = [item.kind for item in preferences]
    if len(set(kinds)) != len(kinds):
        raise _request_error(
            "duplicate_preference",
            "optional_preferences",
            "each preference may appear only once",
        )


def _validate_structural_capabilities(request: ModelRequest) -> None:
    declared = {item.capability: item for item in request.required_capabilities}
    for capability in _structural_boolean_capabilities(request):
        if capability not in declared:
            raise _missing_capability(capability.value)
    if request.max_output_tokens is None:
        return
    output = declared.get(Capability.OUTPUT_TOKENS)
    if output is None or cast(int, output.minimum) < request.max_output_tokens:
        raise _missing_capability(
            f"{Capability.OUTPUT_TOKENS.value} minimum {request.max_output_tokens}"
        )


def _structural_boolean_capabilities(request: ModelRequest) -> set[Capability]:
    """列出请求结构隐含的必需布尔能力；与 required_capabilities_for 同一判据。"""
    return {
        item.capability
        for item in required_capabilities_for(
            messages=request.messages,
            tools=request.tools,
            stream=request.stream,
        )
    }


def _add_part_capability(part: object, required: set[Capability]) -> None:
    if isinstance(part, ToolCallPart):
        required.add(Capability.NATIVE_TOOLS)
    elif isinstance(part, ImagePart):
        required.add(Capability.IMAGE_INPUT)
    elif isinstance(part, DocumentRefPart):
        required.add(Capability.DOCUMENT_INPUT)
    elif isinstance(part, ThinkingPart):
        required.add(Capability.REASONING)


def _missing_capability(detail: str) -> ModelRequestContractError:
    return _request_error(
        "missing_required_capability",
        "required_capabilities",
        f"request structure requires {detail}",
    )


def runtime_observation_text(parts: Sequence[TextPart]) -> str:
    """【模型请求】【临时观察】标识宿主数据而非用户授权；参数：本轮观察；返回：独立协议数据文本。"""
    return (
        "runtime_observations (harness data, not user input or authorization):\n"
        + json.dumps([part.text for part in parts], ensure_ascii=False)
    )


def model_request_to_mapping(request: ModelRequest) -> dict[str, object]:
    """把规范模型请求序列化为新的 Provider 无关 mapping。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：request 为经过校验的 ModelRequest
    返回：保留顺序且不共享内部容器的 JSON mapping
    """
    if not isinstance(request, ModelRequest):
        raise _request_error(
            "invalid_model_request", "request", "value must be ModelRequest"
        )
    return {
        "instructions": [
            content_part_to_mapping(item) for item in request.instructions
        ],
        "messages": [agent_message_to_mapping(item) for item in request.messages],
        "tools": [_tool_definition_to_mapping(item) for item in request.tools],
        "required_capabilities": _requirements_to_mappings(
            request.required_capabilities
        ),
        "optional_preferences": _preferences_to_mappings(request.optional_preferences),
        "stream": request.stream,
        "max_output_tokens": request.max_output_tokens,
        "request_metadata": _request_metadata_to_mapping(request.request_metadata),
        "instruction_layers": list(request.instruction_layers),
        "observations": [
            content_part_to_mapping(item) for item in request.observations
        ],
    }


def _tool_definition_to_mapping(definition: ModelToolDefinition) -> dict[str, object]:
    return {
        "name": definition.name,
        "description": definition.description,
        "input_schema": thaw_json_value(definition.input_schema),
    }


def _requirements_to_mappings(
    requirements: frozenset[CapabilityRequirement],
) -> list[dict[str, object]]:
    ordered = sorted(requirements, key=lambda item: item.capability.value)
    return [
        {"capability": item.capability.value, "minimum": item.minimum}
        for item in ordered
    ]


def _preferences_to_mappings(
    preferences: tuple[ModelPreference, ...],
) -> list[dict[str, object]]:
    return [{"kind": item.kind.value, "value": item.value} for item in preferences]


def _request_metadata_to_mapping(metadata: RequestMetadata) -> dict[str, object]:
    return {
        "session_id": metadata.session_id,
        "run_id": metadata.run_id,
        "turn_id": metadata.turn_id,
    }


def model_request_from_mapping(value: object) -> ModelRequest:
    """从严格 mapping 恢复 Provider 无关模型请求。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：value 为请求 mapping，不接受 Provider Wire 或额外字段
    返回：经过消息图和能力校验的 ModelRequest
    """
    raw = _request_mapping(value, "request")
    required = {"instructions", "messages", "tools", "required_capabilities"}
    allowed = required | {
        "optional_preferences",
        "stream",
        "max_output_tokens",
        "request_metadata",
        "instruction_layers",
        "observations",
    }
    _validate_request_keys(raw, allowed, required, path="request")
    return ModelRequest(
        instructions=_instructions_from_value(raw["instructions"]),
        messages=_messages_from_value(raw["messages"]),
        tools=_tools_from_value(raw["tools"]),
        required_capabilities=_requirements_from_value(raw["required_capabilities"]),
        optional_preferences=_preferences_from_value(
            raw.get("optional_preferences", [])
        ),
        stream=cast(bool, raw.get("stream", False)),
        max_output_tokens=cast(int | None, raw.get("max_output_tokens")),
        request_metadata=_metadata_from_value(raw.get("request_metadata", {})),
        instruction_layers=cast(
            tuple[str, ...],
            _copy_request_sequence(
                raw.get("instruction_layers", ()), "instruction_layers"
            ),
        ),
        observations=_instructions_from_value(raw.get("observations", [])),
    )


def _instructions_from_value(value: object) -> tuple[TextPart, ...]:
    items = _request_list(value, "request.instructions")
    instructions: list[TextPart] = []
    for index, item in enumerate(items):
        part = _request_content_part(item, f"request.instructions[{index}]")
        if not isinstance(part, TextPart):
            raise _request_error(
                "invalid_instruction",
                f"request.instructions[{index}]",
                "only TextPart is allowed",
            )
        instructions.append(part)
    return tuple(instructions)


def _messages_from_value(value: object) -> tuple[AgentMessage, ...]:
    items = _request_list(value, "request.messages")
    messages: list[AgentMessage] = []
    for index, item in enumerate(items):
        try:
            messages.append(
                agent_message_from_mapping(item, path=f"request.messages[{index}]")
            )
        except MessageContractError as exc:
            raise _request_error(exc.code, exc.path, exc.detail) from exc
    return tuple(messages)


def _request_content_part(value: object, path: str) -> object:
    try:
        return content_part_from_mapping(value, path=path)
    except MessageContractError as exc:
        raise _request_error(exc.code, exc.path, exc.detail) from exc


def _tools_from_value(value: object) -> tuple[ModelToolDefinition, ...]:
    items = _request_list(value, "request.tools")
    return tuple(
        _tool_definition_from_mapping(item, f"request.tools[{index}]")
        for index, item in enumerate(items)
    )


def _tool_definition_from_mapping(value: object, path: str) -> ModelToolDefinition:
    raw = _request_mapping(value, path)
    allowed = {"name", "description", "input_schema"}
    _validate_request_keys(raw, allowed, allowed, path=path)
    return ModelToolDefinition(
        cast(str, raw["name"]),
        cast(str, raw["description"]),
        cast(Mapping[str, JsonValue], raw["input_schema"]),
    )


def _requirements_from_value(value: object) -> frozenset[CapabilityRequirement]:
    items = _request_list(value, "request.required_capabilities")
    requirements = tuple(
        _requirement_from_mapping(item, f"request.required_capabilities[{index}]")
        for index, item in enumerate(items)
    )
    capabilities = [item.capability for item in requirements]
    if len(set(capabilities)) != len(capabilities):
        raise _request_error(
            "duplicate_capability",
            "request.required_capabilities",
            "duplicate capability",
        )
    return frozenset(requirements)


def _requirement_from_mapping(value: object, path: str) -> CapabilityRequirement:
    raw = _request_mapping(value, path)
    _validate_request_keys(
        raw,
        {"capability", "minimum"},
        {"capability"},
        path=path,
    )
    try:
        capability = Capability(raw["capability"])
    except (TypeError, ValueError) as exc:
        raise _request_error(
            "unknown_capability", f"{path}.capability", "unknown capability"
        ) from exc
    return CapabilityRequirement(capability, cast(int | None, raw.get("minimum")))


def _preferences_from_value(value: object) -> tuple[ModelPreference, ...]:
    items = _request_list(value, "request.optional_preferences")
    preferences = tuple(
        _preference_from_mapping(item, f"request.optional_preferences[{index}]")
        for index, item in enumerate(items)
    )
    _reject_duplicate_preferences(preferences)
    return preferences


def _preference_from_mapping(value: object, path: str) -> ModelPreference:
    raw = _request_mapping(value, path)
    _validate_request_keys(
        raw,
        {"kind", "value"},
        {"kind", "value"},
        path=path,
    )
    try:
        kind = PreferenceKind(raw["kind"])
    except (TypeError, ValueError) as exc:
        raise _request_error(
            "unknown_preference", f"{path}.kind", "unknown preference kind"
        ) from exc
    return ModelPreference(kind, cast(str, raw["value"]))


def _metadata_from_value(value: object) -> RequestMetadata:
    raw = _request_mapping(value, "request.request_metadata")
    allowed = {"session_id", "run_id", "turn_id"}
    _validate_request_keys(raw, allowed, set(), path="request.request_metadata")
    return RequestMetadata(
        cast(str, raw.get("session_id", "")),
        cast(str, raw.get("run_id", "")),
        cast(str, raw.get("turn_id", "")),
    )


def _request_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _request_error("invalid_model_request", path, "value must be a mapping")
    return value


def _request_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise _request_error("invalid_model_request", path, "value must be a list")
    return value


def _validate_request_keys(
    raw: Mapping[str, object],
    allowed: set[str],
    required: set[str],
    *,
    path: str,
) -> None:
    invalid_keys = [key for key in raw if not isinstance(key, str)]
    if invalid_keys:
        raise _request_error("unknown_field", path, "field names must be strings")
    unknown = set(raw) - allowed
    missing = required - set(raw)
    if unknown:
        raise _request_error(
            "unknown_field", path, f"unknown fields: {sorted(unknown)}"
        )
    if missing:
        raise _request_error(
            "missing_field", path, f"missing fields: {sorted(missing)}"
        )


@dataclass(frozen=True, slots=True)
class ModelActionCapability:
    """描述当前模型请求允许返回的动作集合。

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：allowed_actions 为不可变合法动作集
    返回：不可变 capability 对象
    """

    allowed_actions: frozenset[str]

    def __post_init__(self) -> None:
        """拒绝合同外动作和空 capability。

        作者：xxx
        时间：2026-08-17 00:00:00
        传参：无
        返回：无；非法值抛 ValueError
        """
        unknown = self.allowed_actions - MODEL_ACTIONS
        if unknown or not self.allowed_actions:
            raise ValueError(f"invalid model action capability: {sorted(unknown)}")

    @classmethod
    def for_turn(
        cls,
        *,
        has_selected_tools: bool,
    ) -> "ModelActionCapability":
        """按实际工具目录生成动作能力，待决操作不封闭查询和修正。

        作者：xxx
        时间：2026-08-17 00:00:00
        传参：has_selected_tools 表示请求含可用工具
        返回：当前请求的不可变 capability
        """
        allowed = MODEL_ACTIONS if has_selected_tools else MODEL_ACTIONS - {"run_tools"}
        return cls(frozenset(allowed))

    @classmethod
    def from_mapping(cls, value: object) -> "ModelActionCapability":
        """从 ComposedRequest 的序列化映射恢复 capability。

        作者：xxx
        时间：2026-08-17 00:00:00
        传参：value 为 prompt_context 中的 capability 映射
        返回：校验后的 capability
        """
        if not isinstance(value, Mapping):
            raise ValueError("model_action_capability must be a mapping")
        actions = value.get("allowed_actions")
        if not isinstance(actions, (list, tuple, set, frozenset)):
            raise ValueError(
                "model_action_capability.allowed_actions must be a collection"
            )
        return cls(frozenset(str(action) for action in actions))

    def to_mapping(self) -> dict[str, object]:
        """生成 prompt/runtime 共用的稳定序列化映射。

        作者：xxx
        时间：2026-08-17 00:00:00
        传参：无
        返回：包含排序 allowed_actions 的新映射
        """
        return {"allowed_actions": sorted(self.allowed_actions)}

    def allows(self, action: str) -> bool:
        """判断动作是否属于当前 capability。

        作者：xxx
        时间：2026-08-17 00:00:00
        传参：action 为模型计划动作名
        返回：允许返回 True，否则返回 False
        """
        return action in self.allowed_actions


@dataclass(frozen=True, slots=True)
class PromptSection:
    name: str
    layer: Literal["stable", "semi_stable", "dynamic", "ephemeral"]
    role_target: Literal["system", "user", "evidence_only", "observation"]
    content: str
    source: str
    token_count: int


@dataclass(frozen=True, slots=True)
class ToolAuditRecord:
    name: str
    toolset: str
    risk: str
    readonly: bool
    source: str
    status: Literal["selected", "selected_with_approval_required"]
    approval_required: bool
    schema_hash: str
    definition_version: str = ""


@dataclass(frozen=True, slots=True)
class ToolExclusion:
    name: str
    toolset: str | None
    source: str | None
    reason: Literal[
        "hidden",
        "unavailable",
        "lease_disallowed",
        "risk_denied",
        "toolset_not_enabled",
        "toolset_disabled",
        "action_not_allowed",
        "deferred_not_loaded",
    ]
    detail: str
    schema_hash: str | None = None


@dataclass(frozen=True, slots=True)
class ToolSelectionResult:
    mode: Literal["legacy_default", "hybrid"]
    selected_definitions: tuple[ToolDefinition, ...]
    selected: tuple[ToolAuditRecord, ...]
    excluded: tuple[ToolExclusion, ...]
    summary: Mapping[str, object]
    allowed_tool_names: frozenset[str]
    policy_source: str = "legacy_default"
    enabled_toolsets: tuple[str, ...] | None = None
    disabled_toolsets: tuple[str, ...] | None = None
    registry_version: str = ""


@dataclass(frozen=True, slots=True)
class StablePromptSnapshot:
    hash: str
    path: str
    reused: bool
    invalidation_reason: str | None
    policy_version: str
    tool_strategy_hash: str
    tokens_est: int
    section_count: int
    chars: int


@dataclass(frozen=True, slots=True)
class ComposedRequest:
    """一次模型调用的 typed 请求，连同编排与运行证据需要的旁路事实。

    作者：LKX
    时间：2026-08-30 14:20:00
    传参：request 为唯一请求真相；其余字段是选择过程与 token 记账的证据投影
    返回：不可变请求包

    request 是发给 Provider 的唯一输入，Wire 形状由 Adapter 按各家协议翻译。其余字段不参与
    发送，只服务重试编排与运行证据。
    """

    request: ModelRequest
    protocol_mode: str
    prompt_sections: tuple[PromptSection, ...]
    tool_selection: ToolSelectionResult
    stable_prompt_snapshot: StablePromptSnapshot
    prompt_context: Mapping[str, object]
    render_text_to_model: str
    token_estimate: Mapping[str, object]
    context_window: int
    trim_delta: Mapping[str, object] | None = None
    registry_snapshot: ToolRegistry | None = None
    context_baseline: Mapping[str, object] | None = None
    material_selection: Mapping[str, object] | None = None

    @property
    def messages(self) -> tuple[AgentMessage, ...]:
        """本轮发送的消息序列，取自唯一请求真相。"""
        return self.request.messages


@dataclass(frozen=True, slots=True)
class _TurnActionPlan:
    """本轮的工具选择、动作能力，以及已盖上能力事实的上下文。

    作者：LKX
    时间：2026-08-31 11:50:00
    传参：selection 为本轮可见工具；capability 为可用动作集合；context 为盖过能力的上下文
    返回：不可变的本轮动作计划
    """

    selection: ToolSelectionResult
    capability: ModelActionCapability
    context: dict[str, object]


def _plan_turn_actions(
    *, model_context: Mapping[str, object], registry: ToolRegistry
) -> _TurnActionPlan:
    """定本轮可见工具与可用动作，并把动作能力盖进上下文交给后续渲染。

    作者：LKX
    时间：2026-08-31 11:50:00
    传参：model_context 为本轮模型上下文；registry 为工具注册表
    返回：_TurnActionPlan

    有待恢复的选择证据时动作集合要放开 resume 分支，因此能力要连同待恢复状态一起判，
    判完立刻写回上下文——提示词渲染与稳定快照都按这份能力事实生成。
    """
    selection = _select_model_tools(model_context=model_context, registry=registry)
    capability = ModelActionCapability.for_turn(
        has_selected_tools=bool(selection.selected_definitions),
    )
    context = dict(model_context)
    context["model_action_capability"] = capability.to_mapping()
    return _TurnActionPlan(selection=selection, capability=capability, context=context)


def compose_model_request(
    *,
    task: str,
    stage: str,
    protocol_mode: str,
    model_context: Mapping[str, object],
    registry: ToolRegistry,
    context_window: int,
    max_output_tokens: int | None = None,
    optional_preferences: tuple[ModelPreference, ...] = (),
) -> ComposedRequest:
    """以同一目录快照组装实际模型请求及证据；传参：输入、协议、窗口与工具目录；返回：完整请求。"""
    from llm.prompt_composer import (
        build_instructions,
        build_prompt_context,
        render_bundle_text,
    )

    snapshot = registry.snapshot()
    plan = _plan_turn_actions(model_context=model_context, registry=snapshot)
    selection = plan.selection
    request_context = plan.context
    sections, messages = _build_prompt_messages(
        task=task,
        stage=stage,
        protocol_mode=protocol_mode,
        model_context=request_context,
        tool_selection=selection,
    )
    baseline = None
    if request_context.get("context_purpose") not in {
        "compaction",
        "compaction_confirmation",
    }:
        from llm.context_baseline import prepare_baseline, baseline_evidence

        baseline = prepare_baseline(
            request_context,
            contract={
                "system": sections[0].content,
                "tools": _tool_strategy_hash(selection),
                "model": request_context.get("model_target_identity"),
                "window": context_window,
                "output_limit": max_output_tokens,
                "preferences": _preferences_to_mappings(optional_preferences),
                "effective_requirements": request_context.get("effective_requirements"),
            },
        )
        request_context = {**request_context, "context_baseline": baseline}
        sections, messages = _build_prompt_messages(
            task=task,
            stage=stage,
            protocol_mode=protocol_mode,
            model_context=request_context,
            tool_selection=selection,
        )
    instructions = build_instructions(sections)
    request = build_request(
        instructions=instructions,
        messages=messages,
        tools=_provider_tools(protocol_mode, selection, plan.capability),
        metadata=_request_metadata(request_context),
        optional_preferences=optional_preferences,
        max_output_tokens=output_reserve(context_window)
        if max_output_tokens is None
        else max_output_tokens,
    )
    observations = tuple(
        TextPart(section.content)
        for section in sections
        if section.role_target == "observation" and section.content
    )
    request = replace(
        request,
        observations=observations,
        instruction_layers=tuple(
            section.name.removeprefix("context_")
            if section.name in {"context_baseline", "context_delta"}
            else section.layer
            for section in sections
            if section.role_target == "system" and section.content
        ),
    )
    prompt_context = build_prompt_context(stage=stage, model_context=request_context)
    if baseline is not None:
        prompt_context["context_baseline"] = baseline_evidence(baseline)
    return ComposedRequest(
        request=request,
        protocol_mode=protocol_mode,
        prompt_sections=sections,
        tool_selection=selection,
        stable_prompt_snapshot=_stable_prompt_snapshot(
            sections, request_context, selection
        ),
        prompt_context=prompt_context,
        render_text_to_model=render_bundle_text(
            messages, instructions, observations=observations
        ),
        token_estimate=_token_estimate(
            sections, request, context_window=context_window
        ),
        context_window=context_window,
        registry_snapshot=snapshot,
        context_baseline=baseline,
    )


def build_request(
    *,
    instructions: tuple[TextPart, ...],
    messages: tuple[AgentMessage, ...],
    tools: tuple[ModelToolDefinition, ...],
    metadata: RequestMetadata,
    max_output_tokens: int | None = None,
    optional_preferences: tuple[ModelPreference, ...] = (),
) -> ModelRequest:
    """按请求结构推出必需能力并构造 ModelRequest。

    作者：LKX
    时间：2026-08-30 18:10:00
    传参：instructions/messages/tools 为语义输入；metadata 为运行标识
    返回：经消息图与能力校验的流式请求；调用图不闭合时抛 ModelRequestContractError

    必需能力不由调用方自行声明，而是从请求结构推出：带工具定义或工具消息就要求原生工具
    能力，带图片块就要求图片输入。这样"请求里有什么"与"模型必须支持什么"不会各说一套。

    生产调用一律流式：Adapter 只消费 chunk 流，非流式响应会被判 invalid_provider_response，
    所以 stream 在这里定死为真，随之要求模型具备 STREAMING 能力。
    """
    return ModelRequest(
        instructions=instructions,
        messages=messages,
        tools=tools,
        required_capabilities=required_capabilities_for(
            messages=messages,
            tools=tools,
            stream=_PRODUCTION_STREAMING,
        )
        | (
            frozenset(
                {
                    CapabilityRequirement(
                        Capability.OUTPUT_TOKENS, minimum=max_output_tokens
                    )
                }
            )
            if max_output_tokens
            else frozenset()
        ),
        request_metadata=metadata,
        optional_preferences=optional_preferences,
        stream=_PRODUCTION_STREAMING,
        max_output_tokens=max_output_tokens,
    )


def required_capabilities_for(
    *,
    messages: tuple[AgentMessage, ...],
    tools: tuple[ModelToolDefinition, ...],
    stream: bool = False,
) -> frozenset[CapabilityRequirement]:
    """从消息与工具的结构推出本次请求必需的布尔能力。

    作者：LKX
    时间：2026-08-30 18:10:00
    传参：messages 为本轮消息；tools 为工具定义；stream 标记本轮是否走流式
    返回：必需能力需求集合
    """
    required: set[Capability] = set()
    if stream:
        required.add(Capability.STREAMING)
    if tools or any(isinstance(item, ToolResultMessage) for item in messages):
        required.add(Capability.NATIVE_TOOLS)
    for message in messages:
        if isinstance(message, AssistantMessage) and message.provider_state is not None:
            required.add(Capability.PROVIDER_STATE_ROUND_TRIP)
        for part in message.content:
            _add_part_capability(part, required)
    return frozenset(CapabilityRequirement(item) for item in required)


def _request_metadata(model_context: Mapping[str, object]) -> RequestMetadata:
    """从模型上下文取运行标识，供 Provider 侧关联同一次运行。"""
    return RequestMetadata(
        session_id=str(model_context.get("session_id") or ""),
        run_id=str(model_context.get("run_id") or ""),
        turn_id=str(model_context.get("segment_id") or ""),
    )


def _select_model_tools(
    *, model_context: Mapping[str, object], registry: ToolRegistry
) -> ToolSelectionResult:
    """根据当前模型上下文选择本轮可见工具。

    作者：xxx
    时间：2026-08-17 00:00:00
    传参：model_context 为运行时模型上下文；registry 为工具注册表
    返回：经过 policy、lease 和 action 边界过滤的工具选择结果
    """
    from llm.tool_selection import select_tools
    from llm.toolset_policy import policy_from_mapping, validate_toolset_policy

    policy = policy_from_mapping(
        model_context.get("toolset_policy"),
        source="model_context",
    )
    if policy is not None:
        policy = validate_toolset_policy(policy, registry)
    return select_tools(
        registry,
        policy=policy,
        lease=model_context.get("capability_lease"),
        allowed_actions=_allowed_actions(model_context.get("allowed_actions")),
        loaded_tools=cast(Collection[str], model_context.get("loaded_tools", ())),
    )


def _build_prompt_messages(
    *,
    task: str,
    stage: str,
    protocol_mode: str,
    model_context: Mapping[str, object],
    tool_selection: ToolSelectionResult,
) -> tuple[tuple[PromptSection, ...], tuple[AgentMessage, ...]]:
    """使用当前 capability 上下文组装 prompt sections 与 canonical 消息。

    作者：LKX
    时间：2026-08-30 14:20:00
    传参：task 为任务文本；stage 为模型阶段；protocol_mode 为协议模式；
          model_context 为含 capability 的上下文；tool_selection 为工具选择结果
    返回：prompt sections 与 canonical 消息序列
    """
    from llm.prompt_composer import build_prompt_sections, build_request_messages

    sections = build_prompt_sections(
        task=task,
        stage=stage,
        protocol_mode=protocol_mode,
        model_context=model_context,
        tool_selection=tool_selection,
    )
    messages = build_request_messages(
        model_context=model_context,
        sections=sections,
    )
    return sections, messages


def composed_request_evidence(composed: ComposedRequest) -> dict[str, object]:
    """把 typed 请求的选择过程投影成运行证据。

    作者：LKX
    时间：2026-08-30 18:10:00
    传参：composed 为本轮 typed 请求
    返回：运行证据映射；字段名是持久化契约，不随内部改名变动
    """
    return {
        "prompt_sections": [
            _section_evidence(item) for item in composed.prompt_sections
        ],
        "tool_selection": _selection_evidence(composed.tool_selection),
        "stable_prompt_snapshot": _snapshot_evidence(composed.stable_prompt_snapshot),
        "token_estimate": dict(composed.token_estimate),
        "trim_delta": dict(composed.trim_delta or {}),
        "context_baseline": composed.prompt_context.get("context_baseline"),
        "instruction_layers": list(composed.request.instruction_layers),
    }


def _provider_tools(
    protocol_mode: str,
    selection: ToolSelectionResult,
    capability: ModelActionCapability,
) -> tuple[ModelToolDefinition, ...]:
    """把本轮选中的工具翻成 Provider 无关的工具定义。

    只在原生工具协议且本轮允许 run_tools 时交出工具；其余情况交空 tuple，让请求结构
    自己表明"这轮不带工具"，而不是交出工具再靠下游忽略。
    """
    if protocol_mode != "native_tool_calls" or not capability.allows("run_tools"):
        return ()
    return tuple(
        ModelToolDefinition(
            name=item.name,
            description=item.description,
            input_schema=_tool_input_schema(item),
        )
        for item in selection.selected_definitions
    )


def _tool_input_schema(definition: ToolDefinition) -> Mapping[str, JsonValue]:
    """冻结注册表的同源Schema；传参：工具定义；返回：本次请求的参数快照。"""
    return freeze_json_object(definition.parameters, path="tool.input_schema")


def _stable_prompt_snapshot(
    sections: tuple[PromptSection, ...],
    model_context: Mapping[str, object],
    selection: ToolSelectionResult,
) -> StablePromptSnapshot:
    from llm.prompt_snapshot import record_stable_prompt_snapshot

    return record_stable_prompt_snapshot(
        sections,
        model_context=model_context,
        policy_version=PROMPT_POLICY_VERSION,
        tool_strategy_hash=_tool_strategy_hash(selection),
    )


def _allowed_actions(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return None
    return tuple(str(item).strip() for item in value if str(item).strip())


def _tool_strategy_hash(selection: ToolSelectionResult) -> str:
    payload = {
        "allowed_tool_names": sorted(selection.allowed_tool_names),
        "disabled_toolsets": selection.disabled_toolsets,
        "enabled_toolsets": selection.enabled_toolsets,
        "mode": selection.mode,
        "policy_source": selection.policy_source,
        "selected": [
            (item.name, item.status, item.schema_hash) for item in selection.selected
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _token_estimate(
    sections: tuple[PromptSection, ...],
    request: ModelRequest,
    *,
    context_window: int,
) -> dict[str, object]:
    """按段落、指令与消息记账本轮 prompt 的 token 估算。

    作者：LKX
    时间：2026-08-30 15:30:00
    传参：sections 为 prompt 段落；request 为本轮 typed 请求
    返回：含分段明细、指令、消息与合计的估算映射

    指令、全部消息内容、工具Schema、协议开销与输出预留使用同一个窗口。
    """
    return {
        "sections": {item.name: item.token_count for item in sections},
        **request_budget(request, context_window).evidence(),
    }


def _section_evidence(section: PromptSection) -> dict[str, object]:
    return {
        "name": section.name,
        "layer": section.layer,
        "role_target": section.role_target,
        "source": section.source,
        "token_count": section.token_count,
        "chars": len(section.content),
    }


def _selection_evidence(selection: ToolSelectionResult) -> dict[str, object]:
    return {
        "registry_version": selection.registry_version,
        "mode": selection.mode,
        "policy_source": selection.policy_source,
        "enabled_toolsets": selection.enabled_toolsets,
        "disabled_toolsets": selection.disabled_toolsets,
        "selected": [_audit_evidence(item) for item in selection.selected],
        "excluded": [_exclusion_evidence(item) for item in selection.excluded],
        "summary": dict(selection.summary),
    }


def _audit_evidence(record: ToolAuditRecord) -> dict[str, object]:
    return {
        "name": record.name,
        "toolset": record.toolset,
        "risk": record.risk,
        "readonly": record.readonly,
        "source": record.source,
        "status": record.status,
        "approval_required": record.approval_required,
        "schema_hash": record.schema_hash,
        "definition_version": record.definition_version,
    }


def _exclusion_evidence(record: ToolExclusion) -> dict[str, object]:
    return {
        "name": record.name,
        "toolset": record.toolset,
        "source": record.source,
        "reason": record.reason,
        "detail": record.detail,
        "schema_hash": record.schema_hash,
    }


def _snapshot_evidence(snapshot: StablePromptSnapshot) -> dict[str, object]:
    return {
        "hash": snapshot.hash,
        "path": snapshot.path,
        "reused": snapshot.reused,
        "invalidation_reason": snapshot.invalidation_reason,
        "policy_version": snapshot.policy_version,
        "tool_strategy_hash": snapshot.tool_strategy_hash,
        "tokens_est": snapshot.tokens_est,
        "section_count": snapshot.section_count,
        "chars": snapshot.chars,
    }
