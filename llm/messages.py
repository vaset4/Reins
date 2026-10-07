from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal, Mapping, Sequence, TypeAlias, cast

from llm.types import (
    ModelUsage,
    UsageContractError,
    usage_from_mapping,
    usage_to_mapping,
)


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


class MessageContractError(ValueError):
    """表示消息、内容块或 Provider 状态合同失败。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：code 为稳定错误码；path 为失败路径；detail 为具体原因
    返回：携带 code、path 和 detail 的 ValueError
    """

    def __init__(self, code: str, path: str, detail: str) -> None:
        """初始化可定位的消息合同错误。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：code 为稳定错误码；path 为失败路径；detail 为具体原因
        返回：无；构造异常对象
        """
        self.code = code
        self.path = path
        self.detail = detail
        super().__init__(f"{code} at {path}: {detail}")


class StopReason(str, Enum):
    """定义 Provider 无关的模型停止原因。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：枚举值来自 Adapter 或严格反序列化入口
    返回：闭合的停止原因
    """

    END_TURN = "end_turn"
    TOOL_CALL = "tool_call"
    MAX_OUTPUT_TOKENS = "max_output_tokens"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTER = "content_filter"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


def _message_error(code: str, path: str, detail: str) -> MessageContractError:
    return MessageContractError(code, path, detail)


def _require_non_empty_text(value: object, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise _message_error(
            "invalid_content_part", path, "value must be a non-empty string"
        )


def _copy_unique_ids(values: object, path: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise _message_error(
            "invalid_content_part", path, "ids must be an ordered sequence"
        )
    copied = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in copied):
        raise _message_error(
            "invalid_content_part", path, "ids must be non-empty strings"
        )
    if len(set(copied)) != len(copied):
        raise _message_error("invalid_content_part", path, "ids must be unique")
    return copied


def _copy_message_content(value: object, path: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise _message_error(
            "invalid_message", path, "content must be an ordered sequence"
        )
    return tuple(value)


def freeze_json_object(value: object, *, path: str) -> Mapping[str, JsonValue]:
    """深复制并冻结一个标准 JSON object。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：value 为 JSON object；path 为错误定位路径
    返回：不保留调用方可变引用的只读 Mapping
    """
    if not isinstance(value, Mapping):
        raise _message_error("invalid_json_value", path, "value must be a JSON object")
    frozen = _freeze_json_value(value, path)
    if not isinstance(frozen, Mapping):
        raise _message_error("invalid_json_value", path, "value must be a JSON object")
    return frozen


def thaw_json_value(value: JsonValue) -> JsonValue:
    """把内部冻结 JSON 值转换为新的普通 list/dict。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：value 为经过校验的冻结 JSON 值
    返回：可交给 JSON 编码器的新值，不共享内部容器
    """
    if isinstance(value, Mapping):
        return {key: thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json_value(item) for item in value]
    return value


def _freeze_json_value(value: object, path: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _message_error("invalid_json_value", path, "float must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value, path)
    raise _message_error(
        "invalid_json_value", path, f"unsupported JSON value: {type(value).__name__}"
    )


def _freeze_json_mapping(
    value: Mapping[object, object], path: str
) -> Mapping[str, JsonValue]:
    copied: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise _message_error(
                "invalid_json_value", path, "JSON object keys must be strings"
            )
        copied[key] = _freeze_json_value(item, f"{path}.{key}")
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class TextPart:
    """保存模型可见文本及其稳定引用。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：text 为非空文本；reference_ids 为去重后的稳定引用
    返回：不可变文本内容块
    """

    text: str
    reference_ids: tuple[str, ...] = ()
    kind: Literal["text"] = field(init=False, default="text")

    def __post_init__(self) -> None:
        """校验文本与引用标识。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法内容抛 MessageContractError
        """
        _require_non_empty_text(self.text, "text_part.text")
        object.__setattr__(
            self,
            "reference_ids",
            _copy_unique_ids(self.reference_ids, "text_part.reference_ids"),
        )


@dataclass(frozen=True, slots=True)
class ImagePart:
    """保存图片稳定引用与模型可见元数据。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：source_ref 为稳定引用；mime_type 为媒体类型；width/height 为可选正整数
    返回：不可变图片内容块
    """

    source_ref: str
    mime_type: str
    width: int | None = None
    height: int | None = None
    kind: Literal["image"] = field(init=False, default="image")

    def __post_init__(self) -> None:
        """校验图片引用、类型和尺寸。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法内容抛 MessageContractError
        """
        _require_non_empty_text(self.source_ref, "image_part.source_ref")
        _require_non_empty_text(self.mime_type, "image_part.mime_type")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and not _is_positive_int(value):
                raise _message_error(
                    "invalid_content_part",
                    f"image_part.{name}",
                    "dimension must be a positive integer",
                )


@dataclass(frozen=True, slots=True)
class DocumentRefPart:
    """保存文档 artifact 引用与模型可见摘要。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：artifact_ref 为稳定引用；summary 为摘要；mime_type 为可选媒体类型
    返回：不可变文档引用内容块
    """

    artifact_ref: str
    summary: str
    mime_type: str | None = None
    kind: Literal["document_ref"] = field(init=False, default="document_ref")

    def __post_init__(self) -> None:
        """校验 artifact 引用、摘要和可选媒体类型。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法内容抛 MessageContractError
        """
        _require_non_empty_text(self.artifact_ref, "document_ref_part.artifact_ref")
        _require_non_empty_text(self.summary, "document_ref_part.summary")
        if self.mime_type is not None:
            _require_non_empty_text(self.mime_type, "document_ref_part.mime_type")


@dataclass(frozen=True, slots=True)
class ThinkingPart:
    """保存 Provider 无关的模型思考文本可见性。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：text 为非空思考文本；visibility 为 visible 或 redacted
    返回：不可变思考内容块
    """

    text: str
    visibility: Literal["visible", "redacted"]
    kind: Literal["thinking"] = field(init=False, default="thinking")

    def __post_init__(self) -> None:
        """校验思考文本和值域。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法内容抛 MessageContractError
        """
        _require_non_empty_text(self.text, "thinking_part.text")
        if not isinstance(self.visibility, str) or self.visibility not in {
            "visible",
            "redacted",
        }:
            raise _message_error(
                "invalid_content_part",
                "thinking_part.visibility",
                "unknown thinking visibility",
            )


@dataclass(frozen=True, slots=True)
class ToolCallPart:
    """保存规范化工具调用标识、名称和 JSON object 参数。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：call_id 为稳定关联标识；tool_name 为工具名；arguments 为 JSON object
    返回：深冻结参数的不可变工具调用内容块
    """

    call_id: str
    tool_name: str
    arguments: Mapping[str, JsonValue]
    kind: Literal["tool_call"] = field(init=False, default="tool_call")

    def __post_init__(self) -> None:
        """校验调用标识、工具名并深冻结参数。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法内容抛 MessageContractError
        """
        _require_non_empty_text(self.call_id, "tool_call_part.call_id")
        _require_non_empty_text(self.tool_name, "tool_call_part.tool_name")
        object.__setattr__(
            self,
            "arguments",
            freeze_json_object(self.arguments, path="tool_call_part.arguments"),
        )


ContentPart: TypeAlias = (
    TextPart | ImagePart | DocumentRefPart | ThinkingPart | ToolCallPart
)
UserContentPart: TypeAlias = TextPart | ImagePart | DocumentRefPart
AssistantContentPart: TypeAlias = TextPart | ThinkingPart | ToolCallPart
ToolResultContentPart: TypeAlias = TextPart | ImagePart | DocumentRefPart


@dataclass(frozen=True, slots=True)
class ProviderStateEnvelope:
    """保存仅由原 API family Adapter 解释的连续性状态。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：api_family/provider/model 为目标身份；state_version 为正版本；payload 为 opaque JSON object
    返回：深冻结 payload 的 Provider 状态信封
    """

    api_family: str
    provider: str
    model: str
    state_version: int
    payload: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        """校验状态身份、版本并仅做 JSON 深冻结。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法状态抛 MessageContractError
        """
        for name in ("api_family", "provider", "model"):
            _require_non_empty_state_text(getattr(self, name), f"provider_state.{name}")
        if not _is_positive_int(self.state_version):
            raise _message_error(
                "invalid_provider_state",
                "provider_state.state_version",
                "state_version must be a positive integer",
            )
        try:
            frozen = freeze_json_object(self.payload, path="provider_state.payload")
        except MessageContractError as exc:
            raise _message_error(
                "invalid_provider_state", exc.path, exc.detail
            ) from exc
        object.__setattr__(self, "payload", frozen)


def _require_non_empty_state_text(value: object, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise _message_error(
            "invalid_provider_state", path, "value must be a non-empty string"
        )


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True, slots=True)
class UserMessage:
    """保存一条有序的用户输入消息。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：message_id 为稳定标识；content 为 user 允许的非空内容块 tuple
    返回：不可变用户消息
    """

    message_id: str
    content: tuple[UserContentPart, ...]
    kind: Literal["user"] = field(init=False, default="user")

    def __post_init__(self) -> None:
        """校验用户消息标识及允许的内容块。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法消息抛 MessageContractError
        """
        _validate_message_id(self.message_id, "user.message_id")
        content = _copy_message_content(self.content, "user.content")
        _validate_content(
            content,
            (TextPart, ImagePart, DocumentRefPart),
            "user.content",
            require_non_empty=True,
        )
        object.__setattr__(self, "content", content)


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    """保存模型文本、思考、工具调用、停止原因和连续性状态。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：message_id/content 为消息主体；stop_reason/state/usage 为可选归一化结果
    返回：不可变助手消息
    """

    message_id: str
    content: tuple[AssistantContentPart, ...]
    stop_reason: StopReason | None = None
    provider_state: ProviderStateEnvelope | None = None
    usage: ModelUsage | None = None
    kind: Literal["assistant"] = field(init=False, default="assistant")

    def __post_init__(self) -> None:
        """校验助手消息内容和可选归一化字段。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法消息抛 MessageContractError
        """
        _validate_message_id(self.message_id, "assistant.message_id")
        content = _copy_message_content(self.content, "assistant.content")
        _validate_content(
            content,
            (TextPart, ThinkingPart, ToolCallPart),
            "assistant.content",
            require_non_empty=True,
        )
        if self.stop_reason is not None and not isinstance(
            self.stop_reason, StopReason
        ):
            raise _message_error(
                "invalid_message", "assistant.stop_reason", "unknown stop reason"
            )
        if self.provider_state is not None and not isinstance(
            self.provider_state, ProviderStateEnvelope
        ):
            raise _message_error(
                "invalid_message", "assistant.provider_state", "invalid provider state"
            )
        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise _message_error(
                "invalid_message", "assistant.usage", "usage must be ModelUsage"
            )
        object.__setattr__(self, "content", content)


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    """保存与工具调用稳定关联的模型可见结果。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：message/call/tool 标识关联调用；content/status/error/artifact_refs 描述结果
    返回：不可变工具结果消息
    """

    message_id: str
    call_id: str
    tool_name: str
    content: tuple[ToolResultContentPart, ...]
    status: Literal["success", "error", "partial"]
    error: str | None = None
    artifact_refs: tuple[str, ...] = ()
    kind: Literal["tool_result"] = field(init=False, default="tool_result")

    def __post_init__(self) -> None:
        """校验工具结果关联、状态和模型可见结果不变量。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法消息抛 MessageContractError
        """
        _validate_message_id(self.message_id, "tool_result.message_id")
        _require_non_empty_message_text(self.call_id, "tool_result.call_id")
        _require_non_empty_message_text(self.tool_name, "tool_result.tool_name")
        content = _copy_message_content(self.content, "tool_result.content")
        _validate_content(
            content,
            (TextPart, ImagePart, DocumentRefPart),
            "tool_result.content",
            require_non_empty=False,
        )
        refs = _copy_result_refs(self.artifact_refs)
        _validate_tool_result_state(
            self.status,
            self.error,
            content=content,
            artifact_refs=refs,
        )
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "artifact_refs", refs)


AgentMessage: TypeAlias = UserMessage | AssistantMessage | ToolResultMessage


def _validate_message_id(value: object, path: str) -> None:
    _require_non_empty_message_text(value, path)


def _require_non_empty_message_text(value: object, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise _message_error(
            "invalid_message", path, "value must be a non-empty string"
        )


def _validate_content(
    content: tuple[object, ...],
    allowed: tuple[type[object], ...],
    path: str,
    *,
    require_non_empty: bool,
) -> None:
    if require_non_empty and not content:
        raise _message_error("invalid_message", path, "content must not be empty")
    for index, part in enumerate(content):
        if not isinstance(part, allowed):
            raise _message_error(
                "invalid_content_part",
                f"{path}[{index}]",
                f"part is not allowed for {path.split('.')[0]}",
            )


def _copy_result_refs(values: Sequence[str]) -> tuple[str, ...]:
    try:
        return _copy_unique_ids(values, "tool_result.artifact_refs")
    except MessageContractError as exc:
        raise _message_error("invalid_message", exc.path, exc.detail) from exc


def _validate_tool_result_state(
    status: object,
    error: object,
    *,
    content: tuple[object, ...],
    artifact_refs: tuple[str, ...],
) -> None:
    if not isinstance(status, str) or status not in {"success", "error", "partial"}:
        raise _message_error(
            "invalid_message", "tool_result.status", "unknown result status"
        )
    if error is not None and (not isinstance(error, str) or not error.strip()):
        raise _message_error(
            "invalid_message", "tool_result.error", "error must be non-empty"
        )
    if status == "success" and error is not None:
        raise _message_error(
            "invalid_message", "tool_result.error", "success cannot carry error"
        )
    if status == "error" and error is None:
        raise _message_error(
            "invalid_message", "tool_result.error", "error status requires summary"
        )
    if status == "partial" and not content and not artifact_refs:
        raise _message_error(
            "invalid_message", "tool_result.content", "partial requires visible result"
        )
    if not content and error is None and not artifact_refs:
        raise _message_error("invalid_message", "tool_result", "result cannot be empty")


def content_part_to_mapping(part: ContentPart) -> dict[str, object]:
    """把一个内容块序列化为带 discriminator 的新 mapping。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：part 为五种规范内容块之一
    返回：不共享内部可变状态的 JSON mapping
    """
    if isinstance(part, TextPart):
        return {
            "kind": part.kind,
            "text": part.text,
            "reference_ids": list(part.reference_ids),
        }
    if isinstance(part, ImagePart):
        return _image_to_mapping(part)
    if isinstance(part, DocumentRefPart):
        return _document_to_mapping(part)
    if isinstance(part, ThinkingPart):
        return {"kind": part.kind, "text": part.text, "visibility": part.visibility}
    if isinstance(part, ToolCallPart):
        return {
            "kind": part.kind,
            "call_id": part.call_id,
            "tool_name": part.tool_name,
            "arguments": thaw_json_value(part.arguments),
        }
    raise _message_error("invalid_content_part", "content", "unknown content part type")


def _image_to_mapping(part: ImagePart) -> dict[str, object]:
    return {
        "kind": part.kind,
        "source_ref": part.source_ref,
        "mime_type": part.mime_type,
        "width": part.width,
        "height": part.height,
    }


def _document_to_mapping(part: DocumentRefPart) -> dict[str, object]:
    return {
        "kind": part.kind,
        "artifact_ref": part.artifact_ref,
        "summary": part.summary,
        "mime_type": part.mime_type,
    }


def content_part_from_mapping(value: object, *, path: str = "content") -> ContentPart:
    """从严格 mapping 恢复一种规范内容块。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：value 为带 kind 的 mapping；path 为错误定位路径
    返回：校验后的内容块；未知 kind/字段会失败
    """
    raw = _require_mapping(value, path, "invalid_content_part")
    kind = raw.get("kind")
    parsers = {
        "text": _text_from_mapping,
        "image": _image_from_mapping,
        "document_ref": _document_from_mapping,
        "thinking": _thinking_from_mapping,
        "tool_call": _tool_call_from_mapping,
    }
    parser = parsers.get(kind) if isinstance(kind, str) else None
    if parser is None:
        raise _message_error(
            "unknown_kind", f"{path}.kind", f"unknown content kind: {kind!r}"
        )
    return cast(ContentPart, parser(raw, path))


def _text_from_mapping(raw: Mapping[str, object], path: str) -> TextPart:
    _validate_keys(
        raw,
        {"kind", "text", "reference_ids"},
        {"kind", "text"},
        path=path,
    )
    refs = _optional_string_list(raw, "reference_ids", path)
    return TextPart(cast(str, raw["text"]), refs)


def _image_from_mapping(raw: Mapping[str, object], path: str) -> ImagePart:
    allowed = {"kind", "source_ref", "mime_type", "width", "height"}
    _validate_keys(raw, allowed, {"kind", "source_ref", "mime_type"}, path=path)
    return ImagePart(
        cast(str, raw["source_ref"]),
        cast(str, raw["mime_type"]),
        width=cast(int | None, raw.get("width")),
        height=cast(int | None, raw.get("height")),
    )


def _document_from_mapping(raw: Mapping[str, object], path: str) -> DocumentRefPart:
    allowed = {"kind", "artifact_ref", "summary", "mime_type"}
    _validate_keys(raw, allowed, {"kind", "artifact_ref", "summary"}, path=path)
    return DocumentRefPart(
        cast(str, raw["artifact_ref"]),
        cast(str, raw["summary"]),
        cast(str | None, raw.get("mime_type")),
    )


def _thinking_from_mapping(raw: Mapping[str, object], path: str) -> ThinkingPart:
    allowed = {"kind", "text", "visibility"}
    _validate_keys(raw, allowed, allowed, path=path)
    return ThinkingPart(
        cast(str, raw["text"]),
        cast(Literal["visible", "redacted"], raw["visibility"]),
    )


def _tool_call_from_mapping(raw: Mapping[str, object], path: str) -> ToolCallPart:
    allowed = {"kind", "call_id", "tool_name", "arguments"}
    _validate_keys(raw, allowed, allowed, path=path)
    return ToolCallPart(
        cast(str, raw["call_id"]),
        cast(str, raw["tool_name"]),
        cast(Mapping[str, JsonValue], raw["arguments"]),
    )


def agent_message_to_mapping(message: AgentMessage) -> dict[str, object]:
    """把一条规范消息序列化为新的 JSON mapping。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：message 为 User/Assistant/ToolResultMessage
    返回：保留内容顺序和归一化状态的新 mapping
    """
    if isinstance(message, UserMessage):
        return _user_to_mapping(message)
    if isinstance(message, AssistantMessage):
        return _assistant_to_mapping(message)
    if isinstance(message, ToolResultMessage):
        return _tool_result_to_mapping(message)
    raise _message_error("invalid_message", "message", "unknown message type")


def _user_to_mapping(message: UserMessage) -> dict[str, object]:
    return {
        "kind": message.kind,
        "message_id": message.message_id,
        "content": [content_part_to_mapping(part) for part in message.content],
    }


def _assistant_to_mapping(message: AssistantMessage) -> dict[str, object]:
    return {
        "kind": message.kind,
        "message_id": message.message_id,
        "content": [content_part_to_mapping(part) for part in message.content],
        "stop_reason": message.stop_reason.value if message.stop_reason else None,
        "provider_state": _provider_state_to_mapping(message.provider_state),
        "usage": usage_to_mapping(message.usage) if message.usage is not None else None,
    }


def _tool_result_to_mapping(message: ToolResultMessage) -> dict[str, object]:
    return {
        "kind": message.kind,
        "message_id": message.message_id,
        "call_id": message.call_id,
        "tool_name": message.tool_name,
        "content": [content_part_to_mapping(part) for part in message.content],
        "status": message.status,
        "error": message.error,
        "artifact_refs": list(message.artifact_refs),
    }


def _provider_state_to_mapping(
    envelope: ProviderStateEnvelope | None,
) -> dict[str, object] | None:
    if envelope is None:
        return None
    return {
        "api_family": envelope.api_family,
        "provider": envelope.provider,
        "model": envelope.model,
        "state_version": envelope.state_version,
        "payload": thaw_json_value(envelope.payload),
    }


def agent_message_from_mapping(value: object, *, path: str = "message") -> AgentMessage:
    """从严格 mapping 恢复一条规范消息。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：value 为带 kind 的消息 mapping；path 为错误定位路径
    返回：校验后的 AgentMessage；OpenAI role dict 明确失败
    """
    raw = _require_mapping(value, path, "invalid_message")
    if "role" in raw:
        raise _message_error(
            "unsupported_legacy_message",
            path,
            "OpenAI role messages are not accepted",
        )
    kind = raw.get("kind")
    parsers = {
        "user": _user_from_mapping,
        "assistant": _assistant_from_mapping,
        "tool_result": _tool_result_from_mapping,
    }
    parser = parsers.get(kind) if isinstance(kind, str) else None
    if parser is None:
        raise _message_error(
            "unknown_kind", f"{path}.kind", f"unknown message kind: {kind!r}"
        )
    return cast(AgentMessage, parser(raw, path))


def _user_from_mapping(raw: Mapping[str, object], path: str) -> UserMessage:
    allowed = {"kind", "message_id", "content"}
    _validate_keys(raw, allowed, allowed, path=path)
    return UserMessage(
        cast(str, raw["message_id"]),
        _parse_user_content(raw["content"], path),
    )


def _assistant_from_mapping(raw: Mapping[str, object], path: str) -> AssistantMessage:
    allowed = {
        "kind",
        "message_id",
        "content",
        "stop_reason",
        "provider_state",
        "usage",
    }
    required = {"kind", "message_id", "content"}
    _validate_keys(raw, allowed, required, path=path)
    return AssistantMessage(
        cast(str, raw["message_id"]),
        _parse_assistant_content(raw["content"], path),
        stop_reason=_parse_stop_reason(raw.get("stop_reason"), path),
        provider_state=_provider_state_from_mapping(
            raw.get("provider_state"), f"{path}.provider_state"
        ),
        usage=_usage_from_message_mapping(raw.get("usage"), f"{path}.usage"),
    )


def _tool_result_from_mapping(
    raw: Mapping[str, object], path: str
) -> ToolResultMessage:
    allowed = {
        "kind",
        "message_id",
        "call_id",
        "tool_name",
        "content",
        "status",
        "error",
        "artifact_refs",
    }
    required = {"kind", "message_id", "call_id", "tool_name", "content", "status"}
    _validate_keys(raw, allowed, required, path=path)
    return ToolResultMessage(
        cast(str, raw["message_id"]),
        cast(str, raw["call_id"]),
        cast(str, raw["tool_name"]),
        _parse_result_content(raw["content"], path),
        cast(Literal["success", "error", "partial"], raw["status"]),
        error=cast(str | None, raw.get("error")),
        artifact_refs=_optional_string_list(raw, "artifact_refs", path),
    )


def _parse_user_content(value: object, path: str) -> tuple[UserContentPart, ...]:
    return cast(
        tuple[UserContentPart, ...],
        _parse_typed_content(value, path, (TextPart, ImagePart, DocumentRefPart)),
    )


def _parse_assistant_content(
    value: object, path: str
) -> tuple[AssistantContentPart, ...]:
    return cast(
        tuple[AssistantContentPart, ...],
        _parse_typed_content(value, path, (TextPart, ThinkingPart, ToolCallPart)),
    )


def _parse_result_content(
    value: object, path: str
) -> tuple[ToolResultContentPart, ...]:
    return cast(
        tuple[ToolResultContentPart, ...],
        _parse_typed_content(value, path, (TextPart, ImagePart, DocumentRefPart)),
    )


def _parse_typed_content(
    value: object,
    path: str,
    allowed: tuple[type[object], ...],
) -> tuple[ContentPart, ...]:
    items = _require_list(value, f"{path}.content", "invalid_content_part")
    parts = tuple(
        content_part_from_mapping(item, path=f"{path}.content[{index}]")
        for index, item in enumerate(items)
    )
    _validate_content(
        parts,
        allowed,
        f"{path}.content",
        require_non_empty=False,
    )
    return parts


def _parse_stop_reason(value: object, path: str) -> StopReason | None:
    if value is None:
        return None
    try:
        return StopReason(cast(str, value))
    except (TypeError, ValueError) as exc:
        raise _message_error(
            "invalid_message", f"{path}.stop_reason", "unknown stop reason"
        ) from exc


def _provider_state_from_mapping(
    value: object, path: str
) -> ProviderStateEnvelope | None:
    if value is None:
        return None
    raw = _require_mapping(value, path, "invalid_provider_state")
    allowed = {"api_family", "provider", "model", "state_version", "payload"}
    _validate_keys(raw, allowed, allowed, path=path)
    return ProviderStateEnvelope(
        cast(str, raw["api_family"]),
        cast(str, raw["provider"]),
        cast(str, raw["model"]),
        cast(int, raw["state_version"]),
        cast(Mapping[str, JsonValue], raw["payload"]),
    )


def _usage_from_message_mapping(value: object, path: str) -> ModelUsage | None:
    if value is None:
        return None
    try:
        return usage_from_mapping(value)
    except UsageContractError as exc:
        error_path = path
        if exc.path.startswith("usage."):
            error_path = f"{path}{exc.path[len('usage') :]}"
        raise _message_error(exc.code, error_path, exc.detail) from exc


def _require_mapping(value: object, path: str, code: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _message_error(code, path, "value must be a mapping")
    return value


def _require_list(value: object, path: str, code: str) -> list[object]:
    if not isinstance(value, list):
        raise _message_error(code, path, "value must be a list")
    return value


def _validate_keys(
    raw: Mapping[str, object],
    allowed: set[str],
    required: set[str],
    *,
    path: str,
) -> None:
    invalid_keys = [key for key in raw if not isinstance(key, str)]
    if invalid_keys:
        raise _message_error("unknown_field", path, "field names must be strings")
    unknown = set(raw) - allowed
    missing = required - set(raw)
    if unknown:
        raise _message_error(
            "unknown_field", path, f"unknown fields: {sorted(unknown)}"
        )
    if missing:
        raise _message_error(
            "missing_field", path, f"missing fields: {sorted(missing)}"
        )


def _optional_string_list(
    raw: Mapping[str, object], key: str, path: str
) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise _message_error(
            "invalid_content_part", f"{path}.{key}", "value must be a list"
        )
    if any(not isinstance(item, str) for item in value):
        raise _message_error(
            "invalid_content_part", f"{path}.{key}", "items must be strings"
        )
    return tuple(value)


def model_visible_text(message: AgentMessage) -> str:
    """取一条消息里模型能读到的正文文本。

    作者：LKX
    时间：2026-08-30 14:20:00
    传参：message 为任一 canonical 消息
    返回：拼接后的正文；工具调用公告与思考块不算正文，返回空串

    历史裁剪按 token 预算取舍、证据按 token 记账、prompt 渲染按正文拼接，三处都需要同一个
    "正文是什么"的定义。工具调用参数与 thinking 块由 Adapter 按各家协议单独承载，不计入
    正文，否则同一条消息在预算里被算两次。
    """
    return "".join(part.text for part in message.content if isinstance(part, TextPart))


def recent_user_text(history: Sequence[AgentMessage]) -> str:
    """取给定消息序列里最后一个 UserMessage 的正文。

    作者：LKX
    时间：2026-09-01 16:40:00
    传参：history 为按时间正序排列的 canonical 消息序列
    返回：最后一个 UserMessage 的正文；没有此类消息时返回空串

    消息对象不携带入站来源，后台输入也可能使用 UserMessage。
    生产召回在读取 SessionEntry 的来源后先排除 agent 消息，再调用本函数。
    """
    return next(
        (
            model_visible_text(message)
            for message in reversed(history)
            if isinstance(message, UserMessage)
        ),
        "",
    )


def group_tool_call_units(
    messages: Sequence[AgentMessage],
) -> tuple[tuple[AgentMessage, ...], ...]:
    """把消息序列按"不可分割的工具调用组"切块，供所有历史裁剪共用。

    作者：LKX
    时间：2026-08-30 14:20:00
    传参：messages 为按发送顺序排列的消息
    返回：分组后的消息块 tuple，拼回去与入参逐条等同；不复制也不修改消息对象

    一次工具调用横跨两条消息：AssistantMessage 用 ToolCallPart 公告 call_id，随后的
    ToolResultMessage 用同一个 call_id 回指。工具结果脱离发起它的公告就没有意义，所以
    这两类消息必须整组保留或整组丢弃，裁剪不允许从中间切开。

    公告缺失的孤立 ToolResultMessage 自成一组，交由 validate_message_sequence 在组装时
    显式报 orphan_tool_result，这里不掩盖也不替它补造公告。
    """
    groups: list[tuple[AgentMessage, ...]] = []
    current: list[AgentMessage] = []
    open_call_ids: set[str] = set()
    for message in messages:
        # 1. 工具结果回指当前组已公告的 call_id 时并入当前组，其余消息一律另起一组
        if current and _extends_call_unit(message, open_call_ids):
            current.append(message)
            continue
        if current:
            groups.append(tuple(current))
        current = [message]
        open_call_ids = _announced_call_ids(message)
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _extends_call_unit(message: AgentMessage, open_call_ids: set[str]) -> bool:
    """判断这条消息是否是当前组已公告调用的工具结果。"""
    return isinstance(message, ToolResultMessage) and message.call_id in open_call_ids


def _announced_call_ids(message: AgentMessage) -> set[str]:
    """取出 assistant 消息公告的全部工具调用 id；非公告消息返回空集合。"""
    if not isinstance(message, AssistantMessage):
        return set()
    return {part.call_id for part in message.content if isinstance(part, ToolCallPart)}


def validate_message_sequence(
    messages: Sequence[AgentMessage], *, allow_pending: bool = False
) -> None:
    """校验完整模型输入中的消息和工具调用关联图。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：messages 为按发送顺序排列的规范消息
    返回：无；重复、孤立、错名或悬空调用抛 MessageContractError
    """
    if isinstance(messages, (str, bytes, bytearray)) or not isinstance(
        messages,
        Sequence,
    ):
        raise _message_error(
            "invalid_message", "messages", "value must be an ordered sequence"
        )
    message_ids: set[str] = set()
    calls: dict[str, str] = {}
    pending: dict[str, str] = {}
    completed: set[str] = set()
    for index, message in enumerate(messages):
        _validate_sequence_message(message, index, message_ids)
        if pending and not isinstance(message, ToolResultMessage):
            raise _message_error(
                "interleaved_tool_call",
                f"messages[{index}]",
                "tool call group must finish before another message",
            )
        if isinstance(message, AssistantMessage):
            _record_call_parts(
                message,
                index=index,
                calls=calls,
                pending=pending,
            )
        elif isinstance(message, ToolResultMessage):
            _record_tool_result(
                message,
                index=index,
                pending=pending,
                completed=completed,
            )
    if pending and not allow_pending:
        call_id = next(iter(pending))
        raise _message_error(
            "dangling_tool_call",
            "messages",
            f"tool call has no result: {call_id}",
        )


def _validate_sequence_message(
    message: object,
    index: int,
    message_ids: set[str],
) -> None:
    if not isinstance(message, (UserMessage, AssistantMessage, ToolResultMessage)):
        raise _message_error(
            "invalid_message", f"messages[{index}]", "value must be AgentMessage"
        )
    if message.message_id in message_ids:
        raise _message_error(
            "duplicate_message_id",
            f"messages[{index}].message_id",
            f"duplicate message_id: {message.message_id}",
        )
    message_ids.add(message.message_id)


def _record_call_parts(
    message: AssistantMessage,
    *,
    index: int,
    calls: dict[str, str],
    pending: dict[str, str],
) -> None:
    for part_index, part in enumerate(message.content):
        if not isinstance(part, ToolCallPart):
            continue
        if part.call_id in calls:
            raise _message_error(
                "duplicate_call_id",
                f"messages[{index}].content[{part_index}].call_id",
                f"duplicate call_id: {part.call_id}",
            )
        calls[part.call_id] = part.tool_name
        pending[part.call_id] = part.tool_name


def _record_tool_result(
    message: ToolResultMessage,
    *,
    index: int,
    pending: dict[str, str],
    completed: set[str],
) -> None:
    if message.call_id in completed:
        raise _message_error(
            "duplicate_tool_result",
            f"messages[{index}].call_id",
            f"call already has a result: {message.call_id}",
        )
    if message.call_id not in pending:
        raise _message_error(
            "orphan_tool_result",
            f"messages[{index}].call_id",
            f"tool result has no earlier call: {message.call_id}",
        )
    if pending[message.call_id] != message.tool_name:
        raise _message_error(
            "tool_name_mismatch",
            f"messages[{index}].tool_name",
            f"expected {pending[message.call_id]!r}, got {message.tool_name!r}",
        )
    pending.pop(message.call_id)
    completed.add(message.call_id)


def require_compatible_state(
    envelope: ProviderStateEnvelope,
    *,
    api_family: str,
    provider: str,
    model: str,
) -> None:
    """确认 Provider 状态只能发回同一 API family、Provider 和模型。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：envelope 为状态；api_family/provider/model 为本次目标身份
    返回：无；身份不匹配抛 provider_state_mismatch
    """
    if not isinstance(envelope, ProviderStateEnvelope):
        raise _message_error(
            "invalid_provider_state", "provider_state", "invalid envelope"
        )
    actual = (envelope.api_family, envelope.provider, envelope.model)
    expected = (api_family, provider, model)
    if actual != expected:
        raise _message_error(
            "provider_state_mismatch",
            "provider_state",
            f"state target {actual!r} does not match {expected!r}",
        )


__all__ = [
    "AgentMessage",
    "AssistantContentPart",
    "AssistantMessage",
    "ContentPart",
    "DocumentRefPart",
    "ImagePart",
    "JsonValue",
    "MessageContractError",
    "ProviderStateEnvelope",
    "StopReason",
    "TextPart",
    "ThinkingPart",
    "ToolCallPart",
    "ToolResultContentPart",
    "ToolResultMessage",
    "UserContentPart",
    "UserMessage",
    "agent_message_from_mapping",
    "agent_message_to_mapping",
    "content_part_from_mapping",
    "content_part_to_mapping",
    "freeze_json_object",
    "group_tool_call_units",
    "model_visible_text",
    "recent_user_text",
    "require_compatible_state",
    "thaw_json_value",
    "validate_message_sequence",
]
