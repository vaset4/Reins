from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, cast

from llm.messages import (
    AssistantMessage,
    ImagePart,
    JsonValue,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    thaw_json_value,
)
from llm.providers.image_content import (
    OPENAI_IMAGE_REQUEST_BYTES,
    OPENAI_IMAGE_REQUEST_COUNT,
    validate_image_request_size,
    validated_image_data,
)
from llm.providers.reasoning import reasoning_parameters
from llm.model_registry import ModelDescriptor
from llm.model_request import (
    Capability,
    ModelRequest,
    PreferenceKind,
    runtime_observation_text,
)
from runtime.cancellation import CancellationToken
from llm.stream_control import mapped_sdk_stream
from llm.provider_adapter import ProviderAdapterError
from llm.provider_connection import ResolvedConnection
from llm.provider_result import normalize_provider_error, not_applicable, reported
from llm.provider_stream import ModelStreamEvent, StreamInterruptedError
from llm.types import ModelUsage


RawStreamFactory = Callable[
    [Mapping[str, object], ResolvedConnection], Iterable[Mapping[str, object]]
]
ClientFactory = Callable[[ResolvedConnection], Any]


class OpenAIChatAdapter:
    """翻译 OpenAI Chat Completions request 与 chunk stream。"""

    api_family = "openai_chat"

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
        """将 canonical ModelRequest 翻译为 Chat Completions body。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：request 为 Child 1 合同；model_id 为 Wire 模型名
        返回：不含 credential 的 request body
        """
        messages: list[dict[str, object]] = []
        if request.instructions:
            messages.append(
                {"role": "system", "content": _text_parts(request.instructions)}
            )
        reasoning_history = (
            model_id.lower().rsplit("/", 1)[-1].startswith("deepseek-")
            and bool(request.tools)
            and any(
                item.kind == PreferenceKind.REASONING_LEVEL and item.value != "none"
                for item in request.optional_preferences
            )
        )
        previous = None
        for message in request.messages:
            # 1. Chat 历史不回传思考块；只有思考的保存记录不生成空 assistant 行
            if (
                not reasoning_history
                and isinstance(message, AssistantMessage)
                and all(isinstance(part, ThinkingPart) for part in message.content)
            ):
                continue
            row = _chat_message(message)
            if reasoning_history and isinstance(message, AssistantMessage):
                thought = "\n\n".join(
                    part.text
                    for part in message.content
                    if isinstance(part, ThinkingPart)
                )
                if thought:
                    row["reasoning_content"] = thought
                # 【模型调用】【思考续接】只合并持久化来源ID明确一致的同一响应，不借用其他轮次思考
                if (
                    isinstance(previous, AssistantMessage)
                    and row.get("tool_calls")
                    and message.message_id == f"{previous.message_id}:tool-calls"
                    and messages
                ):
                    preamble = messages.pop()
                    row = {**preamble, **row, "content": preamble.get("content", "")}
            messages.append(row)
            previous = message
        if request.observations:
            messages.append(
                {
                    "role": "user",
                    "content": runtime_observation_text(request.observations),
                }
            )
        body: dict[str, object] = {
            "model": model_id,
            "messages": messages,
            "stream": request.stream,
        }
        if request.tools:
            body["tools"] = [
                _chat_tool(tool.name, tool.description, tool.input_schema)
                for tool in request.tools
            ]
        if request.stream:
            body["stream_options"] = {"include_usage": True}
        if request.max_output_tokens is not None:
            body["max_completion_tokens"] = request.max_output_tokens
        image_count = sum(
            isinstance(part, ImagePart)
            for message in request.messages
            for part in message.content
        )
        if image_count > OPENAI_IMAGE_REQUEST_COUNT:
            raise ProviderAdapterError(
                f"too_many_images:{image_count}>{OPENAI_IMAGE_REQUEST_COUNT}"
            )
        if image_count:
            validate_image_request_size(body, max_bytes=OPENAI_IMAGE_REQUEST_BYTES)
        body.update(reasoning_parameters(request, model_id, "chat_completions"))
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
        """校验选中模型并把 SDK chunk 流逐项翻译为统一事件。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：request/model/connection 为唯一 Provider port 输入
        返回：ModelStreamEvent 迭代器
        """
        _validate_target(request, model, self.api_family, allow_state=False)
        emitted = -1
        try:
            # 【模型调用】【发送证据】1. 已准备的发送体直接交给发送层，不再次读取附件或组装历史
            body = (
                prepared_body
                if prepared_body is not None
                else self.build_request(request, model_id=model.model_id)
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
        stream = client.chat.completions.create(
            **cast(dict[str, Any], thaw_json_value(cast(JsonValue, body)))
        )
        return mapped_sdk_stream(stream, _raw_mapping, cancellation)

    def translate_stream(
        self,
        chunks: Iterable[Mapping[str, object]],
        *,
        provider: str,
        model: str,
    ) -> Iterator[ModelStreamEvent]:
        """将已由冻结 SDK 校验的 Chat chunks 翻译为统一流。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：chunks 为官方类型导出的 dict；provider/model 为身份
        返回：保持 chunk 顺序的统一事件
        """
        sequence = 0
        started = False
        open_blocks: dict[str, tuple[str, str]] = {}
        stop_reason = "unknown"
        saw_finish_reason = False
        for chunk in chunks:
            if not started:
                yield _event(
                    "response_start",
                    sequence,
                    provider,
                    model,
                    message_id=str(chunk.get("id") or "chat-response"),
                )
                sequence += 1
                started = True
            events, stop = _chat_chunk_events(
                chunk, sequence, provider, model, open_blocks
            )
            for item in events:
                # 【模型调用】【尝试用量】已报告的用量即时进入组装器，后续断流也保留这次计费证据
                yield item
                sequence += 1
            if stop is not None:
                stop_reason = stop
                saw_finish_reason = True
        if not started:
            raise StreamInterruptedError("empty_chat_stream")
        if not saw_finish_reason:
            raise StreamInterruptedError("incomplete_chat_stream")
        for block_id in tuple(open_blocks):
            yield _event("content_end", sequence, provider, model, block_id=block_id)
            sequence += 1
        yield _event(
            "response_done", sequence, provider, model, stop_reason=stop_reason
        )


def _chat_message(message: object) -> dict[str, object]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": _user_content(message)}
    if isinstance(message, ToolResultMessage):
        return {
            "role": "tool",
            "tool_call_id": message.call_id,
            "content": _text_parts(message.content),
        }
    if isinstance(message, AssistantMessage):
        text = "\n".join(
            part.text for part in message.content if isinstance(part, TextPart)
        )
        calls = [
            _chat_call(part)
            for part in message.content
            if isinstance(part, ToolCallPart)
        ]
        result: dict[str, object] = {"role": "assistant", "content": text}
        if calls:
            result["tool_calls"] = calls
        return result
    raise ProviderAdapterError("invalid_model_protocol:unsupported_chat_message")


def _chat_call(part: ToolCallPart) -> dict[str, object]:
    return {
        "id": part.call_id,
        "type": "function",
        "function": {"name": part.tool_name, "arguments": _json(part.arguments)},
    }


def _user_content(message: UserMessage) -> str | list[dict[str, object]]:
    """保留文字与图片原始顺序；参数：用户消息；返回：Chat内容，未知文档引用明确拒绝。"""
    if all(isinstance(part, TextPart) for part in message.content):
        return _text_parts(message.content)
    content: list[dict[str, object]] = []
    for part in message.content:
        if isinstance(part, TextPart):
            content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            validated_image_data(part)
            content.append({"type": "image_url", "image_url": {"url": part.source_ref}})
        else:
            raise ProviderAdapterError("unsupported_content:document_ref")
    return content


def _chat_tool(
    name: str, description: str, schema: Mapping[str, JsonValue]
) -> dict[str, object]:
    # 工具 schema 在 ModelToolDefinition 里被逐层冻结，properties/items 这些嵌套层是
    # mappingproxy；只做 dict() 只解冻最外层，SDK 序列化请求体时会 TypeError，
    # 于是整轮对话在"带工具"这一步就发不出去。这里必须深解冻。
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": thaw_json_value(schema),
        },
    }


def _chat_chunk_events(
    chunk: Mapping[str, object],
    sequence: int,
    provider: str,
    model: str,
    open_blocks: dict[str, tuple[str, str]],
) -> tuple[list[ModelStreamEvent], str | None]:
    events: list[ModelStreamEvent] = []
    stop_reason: str | None = None
    choices = chunk.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, Mapping):
            events.extend(
                _chat_delta_events(
                    choice.get("delta"), sequence, provider, model, open_blocks
                )
            )
            finish = choice.get("finish_reason")
            if isinstance(finish, str):
                stop_reason = _chat_stop_reason(finish)
    usage = _chat_usage(chunk.get("usage"))
    if usage is not None:
        events.append(
            _event("usage_update", sequence + len(events), provider, model, usage=usage)
        )
    return events, stop_reason


def _chat_delta_events(
    raw_delta: object,
    sequence: int,
    provider: str,
    model: str,
    open_blocks: dict[str, tuple[str, str]],
) -> list[ModelStreamEvent]:
    if not isinstance(raw_delta, Mapping):
        return []
    events: list[ModelStreamEvent] = []
    # 1. 推理正文先于回答正文到达，按到达顺序译成独立 thinking 块，缺字段或空串时不开块
    reasoning = _delta_reasoning(raw_delta)
    if reasoning:
        events.extend(
            _block_delta(
                "reasoning:0",
                "thinking",
                reasoning,
                sequence,
                provider,
                model,
                open_blocks,
            )
        )
    # 2. 同一 chunk 里推理与回答可同时出现，回答块的 sequence 要接在已产出事件之后
    content = raw_delta.get("content")
    if isinstance(content, str) and content:
        events.extend(
            _block_delta(
                "text:0",
                "text",
                content,
                sequence + len(events),
                provider,
                model,
                open_blocks,
            )
        )
    calls = raw_delta.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            events.extend(
                _chat_tool_delta(
                    call, sequence + len(events), provider, model, open_blocks
                )
            )
    return events


def _delta_reasoning(raw_delta: Mapping[str, object]) -> str:
    """取一个 delta 分片里的推理正文。

    作者：LKX
    时间：2026-08-30 20:10:00
    传参：raw_delta 为 chunk 的 delta 映射
    返回：推理正文；两个字段都缺或为空时返回空串

    同一端点会在 reasoning_content 与 reasoning 两个字段名之间切换：本仓 179 份运行证据里
    前者独占 18 份、后者独占 14 份、两者同时非空 0 份。只读其中一个会在另一半响应上丢掉推理
    正文，且不报任何错，所以按固定优先级依次取，取到即止。
    """
    for key in ("reasoning_content", "reasoning"):
        value = raw_delta.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _chat_tool_delta(
    raw: object,
    sequence: int,
    provider: str,
    model: str,
    open_blocks: dict[str, tuple[str, str]],
) -> list[ModelStreamEvent]:
    if not isinstance(raw, Mapping):
        raise ProviderAdapterError("invalid_provider_response:chat_tool_delta")
    block_id = f"tool:{raw.get('index', 0)}"
    function = raw.get("function")
    function_map = function if isinstance(function, Mapping) else {}
    call_id = str(raw.get("id") or open_blocks.get(block_id, ("", ""))[0])
    tool_name = str(function_map.get("name") or open_blocks.get(block_id, ("", ""))[1])
    arguments = function_map.get("arguments")
    delta = arguments if isinstance(arguments, str) else ""
    return _block_delta(
        block_id,
        "tool_call",
        delta,
        sequence,
        provider,
        model,
        open_blocks,
        call_id=call_id,
        tool_name=tool_name,
    )


def _block_delta(
    block_id: str,
    kind: str,
    delta: str,
    sequence: int,
    provider: str,
    model: str,
    open_blocks: dict[str, tuple[str, str]],
    *,
    call_id: str = "",
    tool_name: str = "",
) -> list[ModelStreamEvent]:
    events: list[ModelStreamEvent] = []
    if block_id not in open_blocks:
        open_blocks[block_id] = (call_id, tool_name)
        events.append(
            _event(
                "content_start",
                sequence,
                provider,
                model,
                block_id=block_id,
                content_kind=kind,
                call_id=call_id,
                tool_name=tool_name,
            )
        )
    if delta:
        events.append(
            _event(
                "content_delta",
                sequence + len(events),
                provider,
                model,
                block_id=block_id,
                delta=delta,
            )
        )
    return events


def _chat_usage(raw: object) -> ModelUsage | None:
    if not isinstance(raw, Mapping):
        return None
    details = raw.get("prompt_tokens_details")
    detail_map = details if isinstance(details, Mapping) else {}
    return ModelUsage(
        input_tokens=reported(_usage_int(raw["prompt_tokens"]))
        if raw.get("prompt_tokens") is not None
        else ModelUsage().input_tokens,
        output_tokens=reported(_usage_int(raw["completion_tokens"]))
        if raw.get("completion_tokens") is not None
        else ModelUsage().output_tokens,
        total_tokens=reported(_usage_int(raw["total_tokens"]))
        if raw.get("total_tokens") is not None
        else ModelUsage().total_tokens,
        cache_read_input_tokens=reported(_usage_int(detail_map["cached_tokens"]))
        if detail_map.get("cached_tokens") is not None
        else ModelUsage().cache_read_input_tokens,
        cache_write_input_tokens=not_applicable(),
    )


def _usage_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderAdapterError("invalid_provider_response:usage_integer")
    return value


def _validate_target(
    request: ModelRequest, model: ModelDescriptor, family: str, *, allow_state: bool
) -> None:
    if model.api_family != family:
        raise ProviderAdapterError(
            f"adapter_family_mismatch:{family}:{model.api_family}"
        )
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
    if not allow_state and Capability.PROVIDER_STATE_ROUND_TRIP in {
        item.capability for item in request.required_capabilities
    }:
        raise ProviderAdapterError("unsupported_capability:provider_state_round_trip")


def _text_parts(parts: object) -> str:
    if not isinstance(parts, tuple):
        return ""
    return "\n".join(
        part.text for part in parts if isinstance(part, (TextPart, ThinkingPart))
    )


def _json(value: Mapping[str, JsonValue]) -> str:
    # 历史回灌的工具参数同样是深冻结的，嵌套对象要一起解冻才能编码
    return json.dumps(
        thaw_json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _chat_stop_reason(value: str) -> str:
    return {
        "stop": "end_turn",
        "tool_calls": "tool_call",
        "length": "max_output_tokens",
        "content_filter": "content_filter",
    }.get(value, "unknown")


def _event(
    kind: str, sequence: int, provider: str, model: str, **values: Any
) -> ModelStreamEvent:
    return ModelStreamEvent(kind, sequence, "openai_chat", provider, model, **values)


def _default_client(
    connection: ResolvedConnection, *, http_client: Any | None = None
) -> Any:
    import openai

    headers = {
        key: value
        for key, value in connection.headers.items()
        if key.lower() != "authorization"
    }
    return openai.OpenAI(
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
    raise ProviderAdapterError("invalid_provider_response:chat_chunk_type")


def _known_errors() -> tuple[type[BaseException], ...]:
    import httpx2
    import openai

    return (
        openai.APIError,
        httpx2.TransportError,
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
        api_family="openai_chat",
        stage="transport",
    )
    yield _event("response_error", sequence, provider, model, error=normalized)


__all__ = ["OpenAIChatAdapter"]
