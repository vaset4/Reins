from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, cast
from urllib.parse import urlsplit

from llm.messages import (
    AssistantMessage,
    ImagePart,
    JsonValue,
    ProviderStateEnvelope,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    thaw_json_value,
)
from llm.providers.image_content import (
    validate_image_request_size,
    validated_image_data,
)
from llm.providers.reasoning import reasoning_parameters
from llm.model_registry import ModelDescriptor
from llm.model_request import Capability, ModelRequest, runtime_observation_text
from runtime.cancellation import CancellationToken
from llm.stream_control import mapped_sdk_stream
from llm.provider_adapter import ProviderAdapterError
from llm.provider_connection import ResolvedConnection
from llm.provider_result import (
    ProviderError,
    ProviderErrorCategory,
    normalize_provider_error,
    reported,
)
from llm.provider_stream import ModelStreamEvent, StreamInterruptedError
from llm.types import ModelUsage


# https://platform.claude.com/docs/en/build-with-claude/vision：直接API上限，伙伴更小上限保留远端错误
IMAGE_BASE64_MAX_BYTES = 10 * 1024 * 1024
IMAGE_MAX_DIMENSION = 8000
IMAGE_REQUEST_MAX_BYTES = 32 * 1024 * 1024
IMAGE_REQUEST_MAX_COUNT = 600
IMAGE_REQUEST_200K_MAX_COUNT = 100
IMAGE_CONTEXT_200K = 200000

RawStreamFactory = Callable[
    [Mapping[str, object], ResolvedConnection], Iterable[Mapping[str, object]]
]
ClientFactory = Callable[[ResolvedConnection], Any]


class AnthropicMessagesAdapter:
    """翻译 Anthropic Messages request 与六类 raw stream event。"""

    api_family = "anthropic_messages"

    def __init__(
        self,
        stream_factory: RawStreamFactory | None = None,
        client_factory: ClientFactory | None = None,
        http_client: Any | None = None,
    ) -> None:
        """注入负责真实 SDK I/O 的 stream factory。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：stream_factory 接收 Wire body 和已解析连接
        返回：无
        """
        self._stream_factory = stream_factory
        self._client_factory = client_factory or _default_client
        self._http_client = http_client

    def build_request(
        self, request: ModelRequest, *, model_id: str
    ) -> dict[str, object]:
        """将 canonical ModelRequest 翻译为 Anthropic Messages body。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：request 为 Child 1 合同；model_id 为 Wire 模型名
        返回：不含 credential 的 request body
        """
        body: dict[str, object] = {
            "model": model_id,
            "max_tokens": request.max_output_tokens or 4096,
            "system": [
                {"type": "text", "text": part.text} for part in request.instructions
            ],
            "messages": [_anthropic_message(message) for message in request.messages],
            "stream": request.stream,
        }
        if request.observations:
            cast(list[dict[str, object]], body["messages"]).append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": runtime_observation_text(request.observations),
                        }
                    ],
                }
            )
        if request.tools:
            # 工具 schema 是深冻结的，嵌套层只 dict() 会让 SDK 序列化请求体时 TypeError
            body["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": thaw_json_value(tool.input_schema),
                }
                for tool in request.tools
            ]
        if any(
            isinstance(part, ImagePart)
            for message in request.messages
            for part in message.content
        ):
            validate_image_request_size(body, max_bytes=IMAGE_REQUEST_MAX_BYTES)
        body.update(reasoning_parameters(request, model_id, "anthropic_messages"))
        return body

    def stream(
        self,
        request: ModelRequest,
        *,
        model: ModelDescriptor,
        connection: ResolvedConnection,
        cancellation: CancellationToken | None = None,
        prepared_body: Mapping[str, object] | None = None,
    ) -> Iterator[ModelStreamEvent]:
        """校验模型并把 SDK raw event 逐项翻译为统一事件。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：request/model/connection 为唯一 Provider port 输入
        返回：ModelStreamEvent 迭代器
        """
        _validate_target(request, model)
        emitted = -1
        try:
            # 【模型调用】【发送证据】1. 已准备的发送体直接交给发送层，不再次读取附件或组装历史
            body = (
                prepared_body
                if prepared_body is not None
                else self.apply_cache_hints(
                    self.build_request(request, model_id=model.model_id),
                    request=request,
                    model=model,
                    connection=connection,
                )
            )
            raw = self._raw_stream(body, connection, cancellation)
            for event in self.translate_stream(
                raw, provider=model.provider, model=model.model_id
            ):
                emitted = event.sequence
                yield event
        except _known_errors() as exc:
            yield from _error_events(exc, emitted, model.provider, model.model_id)

    def _raw_stream(
        self,
        body: Mapping[str, object],
        connection: ResolvedConnection,
        cancellation: CancellationToken | None = None,
    ) -> Iterable[Mapping[str, object]]:
        if self._stream_factory is not None:
            return self._stream_factory(body, connection)
        client = (
            self._client_factory(connection)
            if self._client_factory is not _default_client
            else _default_client(
                connection,
                http_client=self._http_client,
            )
        )
        if cancellation is not None and callable(getattr(client, "close", None)):
            cancellation.register_closer(client.close)
        stream = client.messages.create(
            **cast(dict[str, Any], thaw_json_value(cast(JsonValue, body)))
        )
        return mapped_sdk_stream(stream, _raw_mapping, cancellation)

    def apply_cache_hints(
        self,
        body: Mapping[str, object],
        *,
        request: ModelRequest,
        model: ModelDescriptor,
        connection: ResolvedConnection,
    ) -> dict[str, object]:
        """只给明确支持的官方目标标记稳定前缀；参数：已构造正文、模型及实际连接；返回：发送正文。"""
        endpoint = urlsplit(connection.base_url)
        if (
            endpoint.scheme != "https"
            or endpoint.hostname != "api.anthropic.com"
            or model.capabilities.get(Capability.PROMPT_CACHE) is not True
        ):
            return dict(body)
        boundary = -1
        for index, layer in enumerate(request.instruction_layers):
            if layer not in {"stable", "baseline", "delta"}:
                break
            boundary = index
        if boundary < 0:
            return dict(body)
        blocks = [
            dict(block) for block in cast(list[dict[str, object]], body["system"])
        ]
        # 1. 【模型调用】【缓存提示】一个稳定前缀只设一个边界，不指定TTL，也不把提示当命中
        blocks[boundary]["cache_control"] = {"type": "ephemeral"}
        return {**body, "system": blocks}

    def translate_stream(
        self,
        raw_events: Iterable[Mapping[str, object]],
        *,
        provider: str,
        model: str,
    ) -> Iterator[ModelStreamEvent]:
        """将冻结 SDK 校验的六类 raw event 翻译为统一流。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：raw_events 为官方类型导出的 dict；provider/model 为身份
        返回：保持 raw 顺序的统一事件
        """
        sequence = 0
        stop_reason = "unknown"
        terminal = False
        started = False
        thinking_state: dict[str, object] = {}
        for raw in raw_events:
            translated, next_stop = _translate_anthropic_event(
                raw,
                sequence,
                provider,
                model,
                stop_reason,
                thinking_state,
            )
            if next_stop is not None:
                stop_reason = next_stop
            for event in translated:
                yield event
                started = started or event.kind == "response_start"
            sequence += len(translated)
            terminal = terminal or any(
                item.kind in {"response_done", "response_error"} for item in translated
            )
        if not terminal:
            if started:
                raise StreamInterruptedError("missing_message_stop")
            raise ProviderAdapterError("invalid_provider_response:missing_message_stop")


def _anthropic_message(message: object) -> dict[str, object]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": _user_content(message)}
    if isinstance(message, ToolResultMessage):
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.call_id,
                    "content": _text_parts(message.content),
                    "is_error": message.status != "success",
                }
            ],
        }
    if isinstance(message, AssistantMessage):
        return {"role": "assistant", "content": _assistant_blocks(message)}
    raise ProviderAdapterError("invalid_model_protocol:unsupported_anthropic_message")


def _user_content(message: UserMessage) -> list[dict[str, object]]:
    """把图片的真实字节送入Messages base64 source；参数：消息；返回：有序内容块。"""
    if all(isinstance(part, TextPart) for part in message.content):
        return [{"type": "text", "text": _text_parts(message.content)}]
    content: list[dict[str, object]] = []
    for part in message.content:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            data = validated_image_data(part, max_dimension=IMAGE_MAX_DIMENSION)
            if len(data) > IMAGE_BASE64_MAX_BYTES:
                raise ProviderAdapterError(
                    f"image_too_large:{len(data)}>{IMAGE_BASE64_MAX_BYTES}"
                )
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.mime_type,
                        "data": data,
                    },
                }
            )
        else:
            raise ProviderAdapterError("unsupported_content:document_ref")
    return content


def _assistant_blocks(message: AssistantMessage) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    for part in message.content:
        if isinstance(part, TextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ToolCallPart):
            # 历史回灌的工具参数是深冻结的，嵌套对象要一起解冻才能编码
            blocks.append(
                {
                    "type": "tool_use",
                    "id": part.call_id,
                    "name": part.tool_name,
                    "input": thaw_json_value(part.arguments),
                }
            )
        elif isinstance(part, ThinkingPart):
            blocks.append(_thinking_block(part, message.provider_state))
    return blocks


def _thinking_block(
    part: ThinkingPart, state: ProviderStateEnvelope | None
) -> dict[str, object]:
    if state is None or state.api_family != "anthropic_messages":
        raise ProviderAdapterError("provider_state_mismatch:anthropic_messages")
    signatures = state.payload.get("signatures")
    signature = signatures.get(part.text) if isinstance(signatures, Mapping) else None
    if part.visibility == "redacted":
        return {"type": "redacted_thinking", "data": part.text}
    if not isinstance(signature, str) or not signature:
        raise ProviderAdapterError("invalid_model_protocol:thinking_signature_missing")
    return {"type": "thinking", "thinking": part.text, "signature": signature}


def _translate_anthropic_event(
    raw: Mapping[str, object],
    sequence: int,
    provider: str,
    model: str,
    stop_reason: str,
    thinking_state: dict[str, object],
) -> tuple[list[ModelStreamEvent], str | None]:
    event_type = raw.get("type")
    if event_type == "message_start":
        return _message_start(raw, sequence, provider, model), None
    if event_type == "content_block_start":
        return _content_start(raw, sequence, provider, model, thinking_state), None
    if event_type == "content_block_delta":
        return _content_delta(raw, sequence, provider, model, thinking_state), None
    if event_type == "content_block_stop":
        return [
            _event("content_end", sequence, provider, model, block_id=_block_id(raw))
        ], None
    if event_type == "message_delta":
        delta = _mapping(raw.get("delta"))
        next_stop = _anthropic_stop_reason(str(delta.get("stop_reason") or "unknown"))
        usage = _anthropic_usage(raw.get("usage"), include_input=False)
        return (
            [_event("usage_update", sequence, provider, model, usage=usage)]
            if usage is not None
            else []
        ), next_stop
    if event_type == "message_stop":
        payload = _anthropic_state_payload(thinking_state)
        state = (
            ProviderStateEnvelope("anthropic_messages", provider, model, 1, payload)
            if payload
            else None
        )
        return [
            _event(
                "response_done",
                sequence,
                provider,
                model,
                stop_reason=stop_reason,
                provider_state=state,
            )
        ], None
    if event_type == "error":
        return [_anthropic_error(raw, sequence, provider, model)], None
    raise ProviderAdapterError(
        f"invalid_provider_response:unsupported_anthropic_event:{event_type}"
    )


def _message_start(
    raw: Mapping[str, object],
    sequence: int,
    provider: str,
    model: str,
) -> list[ModelStreamEvent]:
    message = _mapping(raw.get("message"))
    events = [
        _event(
            "response_start",
            sequence,
            provider,
            model,
            message_id=str(message.get("id") or "anthropic-message"),
        )
    ]
    usage = _anthropic_usage(message.get("usage"), include_input=True)
    if usage is not None:
        events.append(
            _event("usage_update", sequence + 1, provider, model, usage=usage)
        )
    return events


def _content_start(
    raw: Mapping[str, object],
    sequence: int,
    provider: str,
    model: str,
    thinking_state: dict[str, object],
) -> list[ModelStreamEvent]:
    block = _mapping(raw.get("content_block"))
    block_type = block.get("type")
    values: dict[str, object] = {"block_id": _block_id(raw)}
    if block_type == "tool_use":
        values.update(
            {
                "content_kind": "tool_call",
                "call_id": str(block.get("id") or ""),
                "tool_name": str(block.get("name") or ""),
            }
        )
    elif block_type in {"thinking", "redacted_thinking"}:
        values["content_kind"] = "thinking"
        if block_type == "redacted_thinking":
            data = str(block.get("data") or "")
            values["thinking_visibility"] = "redacted"
            thinking_state[f"redacted:{raw.get('index', 0)}"] = data
            return [
                _event("content_start", sequence, provider, model, **values),
                _event(
                    "content_delta",
                    sequence + 1,
                    provider,
                    model,
                    block_id=_block_id(raw),
                    delta=data,
                ),
            ]
        thinking_state[f"text:{raw.get('index', 0)}"] = ""
    elif block_type == "text":
        values["content_kind"] = "text"
    else:
        raise ProviderAdapterError(
            f"invalid_provider_response:unsupported_anthropic_block:{block_type}"
        )
    return [_event("content_start", sequence, provider, model, **values)]


def _content_delta(
    raw: Mapping[str, object],
    sequence: int,
    provider: str,
    model: str,
    thinking_state: dict[str, object],
) -> list[ModelStreamEvent]:
    delta = _mapping(raw.get("delta"))
    delta_type = delta.get("type")
    text = delta.get("text") if delta_type == "text_delta" else delta.get("thinking")
    if delta_type == "input_json_delta":
        text = delta.get("partial_json")
    if delta_type == "signature_delta":
        thinking_state[f"signature:{raw.get('index', 0)}"] = str(
            delta.get("signature") or ""
        )
        return []
    if delta_type not in {"text_delta", "thinking_delta", "input_json_delta"}:
        raise ProviderAdapterError(
            f"invalid_provider_response:unsupported_anthropic_delta:{delta_type}"
        )
    if delta_type == "thinking_delta":
        key = f"text:{raw.get('index', 0)}"
        thinking_state[key] = str(thinking_state.get(key) or "") + str(text or "")
    return [
        _event(
            "content_delta",
            sequence,
            provider,
            model,
            block_id=_block_id(raw),
            delta=str(text or ""),
        )
    ]


def _anthropic_state_payload(state: Mapping[str, object]) -> Mapping[str, JsonValue]:
    signatures: dict[str, JsonValue] = {}
    redacted: list[JsonValue] = []
    for key, value in state.items():
        if key.startswith("text:") and isinstance(value, str) and value:
            signature = state.get(f"signature:{key.removeprefix('text:')}")
            if isinstance(signature, str) and signature:
                signatures[value] = signature
        if key.startswith("redacted:") and isinstance(value, str) and value:
            redacted.append(value)
    payload: dict[str, JsonValue] = {}
    if signatures:
        payload["signatures"] = signatures
    if redacted:
        payload["redacted_blocks"] = redacted
    return payload


def _anthropic_usage(raw: object, *, include_input: bool) -> ModelUsage | None:
    if not isinstance(raw, Mapping):
        return None
    unknown = ModelUsage()
    return ModelUsage(
        input_tokens=reported(int(raw["input_tokens"]))
        if include_input and raw.get("input_tokens") is not None
        else unknown.input_tokens,
        output_tokens=reported(int(raw["output_tokens"]))
        if raw.get("output_tokens") is not None
        else unknown.output_tokens,
        cache_read_input_tokens=reported(int(raw["cache_read_input_tokens"]))
        if raw.get("cache_read_input_tokens") is not None
        else unknown.cache_read_input_tokens,
        cache_write_input_tokens=reported(int(raw["cache_creation_input_tokens"]))
        if raw.get("cache_creation_input_tokens") is not None
        else unknown.cache_write_input_tokens,
    )


def _anthropic_error(
    raw: Mapping[str, object], sequence: int, provider: str, model: str
) -> ModelStreamEvent:
    detail = _mapping(raw.get("error"))
    category = (
        "rate_limited" if detail.get("type") == "rate_limit_error" else "provider_error"
    )
    error = ProviderError(
        cast(ProviderErrorCategory, category),
        "stream_decode",
        category == "rate_limited",
        str(detail.get("message") or "provider error"),
        provider,
        model,
        "anthropic_messages",
    )
    return _event("response_error", sequence, provider, model, error=error)


def _validate_target(request: ModelRequest, model: ModelDescriptor) -> None:
    if model.api_family != "anthropic_messages":
        raise ProviderAdapterError(
            f"adapter_family_mismatch:anthropic_messages:{model.api_family}"
        )
    image_count = sum(
        isinstance(part, ImagePart)
        for message in request.messages
        for part in message.content
    )
    context_window = model.capabilities.get(Capability.CONTEXT_WINDOW_TOKENS)
    max_images = (
        IMAGE_REQUEST_200K_MAX_COUNT
        if context_window == IMAGE_CONTEXT_200K
        else IMAGE_REQUEST_MAX_COUNT
    )
    if image_count > max_images:
        raise ProviderAdapterError(f"too_many_images:{image_count}>{max_images}")
    for requirement in request.required_capabilities:
        value = model.capabilities.get(requirement.capability, False)
        satisfied = (
            value >= requirement.minimum
            if requirement.minimum is not None and isinstance(value, int)
            else value is True
        )
        if not satisfied:
            raise ProviderAdapterError(
                f"unsupported_capability:{requirement.capability.value}"
            )
    for message in request.messages:
        if not isinstance(message, AssistantMessage) or message.provider_state is None:
            continue
        state = message.provider_state
        if (state.api_family, state.provider, state.model) != (
            "anthropic_messages",
            model.provider,
            model.model_id,
        ):
            raise ProviderAdapterError("provider_state_mismatch:anthropic_messages")


def _text_parts(parts: object) -> str:
    if not isinstance(parts, tuple):
        return ""
    return "\n".join(
        part.text for part in parts if isinstance(part, (TextPart, ThinkingPart))
    )


def _block_id(raw: Mapping[str, object]) -> str:
    return f"block:{raw.get('index', 0)}"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _anthropic_stop_reason(value: str) -> str:
    return {
        "end_turn": "end_turn",
        "tool_use": "tool_call",
        "max_tokens": "max_output_tokens",
        "stop_sequence": "stop_sequence",
        "refusal": "content_filter",
    }.get(value, "unknown")


def _event(
    kind: str, sequence: int, provider: str, model: str, **values: Any
) -> ModelStreamEvent:
    return ModelStreamEvent(
        kind, sequence, "anthropic_messages", provider, model, **values
    )


def _default_client(
    connection: ResolvedConnection, *, http_client: Any | None = None
) -> Any:
    import anthropic

    owned = {"x-api-key", "anthropic-version"}
    headers = {
        key: value
        for key, value in connection.headers.items()
        if key.lower() not in owned
    }
    return anthropic.Anthropic(
        api_key=connection.credential,
        base_url=connection.base_url,
        timeout=connection.timeout_seconds,
        max_retries=0,
        default_headers=headers,
        http_client=http_client,
    )


def _raw_mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    dump = getattr(value, "model_dump", None)
    raw = dump(mode="json", exclude_none=True) if callable(dump) else None
    if isinstance(raw, Mapping):
        return raw
    raise ProviderAdapterError("invalid_provider_response:anthropic_event_type")


def _known_errors() -> tuple[type[BaseException], ...]:
    import anthropic
    import httpx

    return (
        anthropic.APIError,
        httpx.TransportError,
        asyncio.CancelledError,
        TimeoutError,
        ConnectionError,
    )


def _error_events(
    error: BaseException, emitted: int, provider: str, model: str
) -> Iterator[ModelStreamEvent]:
    sequence = emitted + 1
    if emitted < 0:
        yield _event("response_start", 0, provider, model, message_id="provider-error")
        sequence = 1
    normalized = normalize_provider_error(
        error,
        provider=provider,
        model=model,
        api_family="anthropic_messages",
        stage="transport",
    )
    yield _event("response_error", sequence, provider, model, error=normalized)


__all__ = ["AnthropicMessagesAdapter"]
