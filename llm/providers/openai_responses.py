from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, cast

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
    OPENAI_IMAGE_REQUEST_BYTES,
    OPENAI_IMAGE_REQUEST_COUNT,
    validate_image_request_size,
    validated_image_data,
)
from llm.providers.reasoning import reasoning_parameters
from llm.model_registry import ModelDescriptor
from llm.model_request import ModelRequest, runtime_observation_text
from runtime.cancellation import CancellationToken
from llm.stream_control import mapped_sdk_stream
from llm.provider_adapter import ProviderAdapterError
from llm.provider_connection import ResolvedConnection
from llm.provider_result import (
    ProviderError,
    normalize_provider_error,
    not_applicable,
    reported,
)
from llm.provider_stream import ModelStreamEvent, StreamInterruptedError
from llm.types import ModelUsage


RawStreamFactory = Callable[
    [Mapping[str, object], ResolvedConnection], Iterable[Mapping[str, object]]
]
ClientFactory = Callable[[ResolvedConnection], Any]
_TEXT_DELTA_EVENTS = frozenset(
    {
        "response.output_text.delta",
        "response.reasoning_text.delta",
        "response.reasoning_summary_text.delta",
    }
)
_TEXT_DONE_EVENTS = frozenset(
    {
        "response.output_text.done",
        "response.reasoning_text.done",
        "response.reasoning_summary_text.done",
    }
)
_TEXT_EVENTS = (
    _TEXT_DELTA_EVENTS
    | _TEXT_DONE_EVENTS
    | {"response.reasoning_summary_part.added", "response.reasoning_summary_part.done"}
)


class OpenAIResponsesAdapter:
    """翻译 OpenAI Responses request 与判别事件 union。"""

    api_family = "openai_responses"

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
        """将 canonical ModelRequest 翻译为 Responses body。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：request 为 Child 1 合同；model_id 为 Wire 模型名
        返回：不含 credential 的 request body
        """
        body: dict[str, object] = {
            "model": model_id,
            "instructions": _text_parts(request.instructions),
            "input": [
                item
                for message in request.messages
                for item in _response_items(message)
            ],
            "stream": request.stream,
        }
        if request.observations:
            cast(list[dict[str, object]], body["input"]).append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": runtime_observation_text(request.observations),
                        }
                    ],
                }
            )
        if request.tools:
            # 1. 深冻结Schema转为JSON；保留可选字段语义，关闭供应商strict，实际参数仍由Harness按原Schema校验
            body["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": thaw_json_value(tool.input_schema),
                    "strict": False,
                }
                for tool in request.tools
            ]
        if request.max_output_tokens is not None:
            body["max_output_tokens"] = request.max_output_tokens
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
        body.update(reasoning_parameters(request, model_id, "responses"))
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
        """校验模型并把 SDK Responses 事件逐项翻译为统一事件。

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
        stream = client.responses.create(
            **cast(dict[str, Any], thaw_json_value(cast(JsonValue, body)))
        )
        return mapped_sdk_stream(stream, _raw_mapping, cancellation)

    def translate_stream(
        self,
        raw_events: Iterable[Mapping[str, object]],
        *,
        provider: str,
        model: str,
    ) -> Iterator[ModelStreamEvent]:
        """将冻结 SDK 校验的 Responses event union 翻译为统一流。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：raw_events 为官方类型导出的判别 dict；provider/model 为身份
        返回：保持 sequence 的统一事件
        """
        sequence = 0
        terminal = False
        started = False
        open_text_blocks: set[str] = set()
        provider_state_payload: dict[str, JsonValue] = {"input_items": []}
        for raw in raw_events:
            translated = _translate_response_event(
                raw,
                sequence,
                provider=provider,
                model=model,
                open_text_blocks=open_text_blocks,
                provider_state_payload=provider_state_payload,
            )
            for event in translated:
                yield event
                started = started or event.kind == "response_start"
            sequence += len(translated)
            terminal = terminal or any(
                item.kind in {"response_done", "response_error"} for item in translated
            )
        if not terminal:
            if started:
                raise StreamInterruptedError("incomplete_responses_stream")
            raise ProviderAdapterError(
                "invalid_provider_response:incomplete_responses_stream"
            )


def _response_items(message: object) -> list[dict[str, object]]:
    if isinstance(message, UserMessage):
        return [{"role": "user", "content": _user_content(message)}]
    if isinstance(message, ToolResultMessage):
        return [
            {
                "type": "function_call_output",
                "call_id": message.call_id,
                "output": _text_parts(message.content),
            }
        ]
    if isinstance(message, AssistantMessage):
        return _assistant_items(message)
    raise ProviderAdapterError("invalid_model_protocol:unsupported_responses_message")


def _user_content(message: UserMessage) -> list[dict[str, object]]:
    """把原始用户内容逐块交给Responses；参数：消息；返回：文字和冻结图片，拒绝未实现文档引用。"""
    if all(isinstance(part, TextPart) for part in message.content):
        return [{"type": "input_text", "text": _text_parts(message.content)}]
    content: list[dict[str, object]] = []
    for part in message.content:
        if isinstance(part, TextPart):
            content.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImagePart):
            validated_image_data(part)
            content.append(
                {"type": "input_image", "image_url": part.source_ref, "detail": "auto"}
            )
        else:
            raise ProviderAdapterError("unsupported_content:document_ref")
    return content


def _assistant_items(message: AssistantMessage) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    text = "\n".join(
        part.text for part in message.content if isinstance(part, TextPart)
    )
    if text:
        items.append(
            {"role": "assistant", "content": [{"type": "output_text", "text": text}]}
        )
    for part in message.content:
        if isinstance(part, ToolCallPart):
            items.append(
                {
                    "type": "function_call",
                    "call_id": part.call_id,
                    "name": part.tool_name,
                    "arguments": _json(part.arguments),
                }
            )
    if message.provider_state is not None:
        _require_state_target(message.provider_state)
        state_items = message.provider_state.payload.get("input_items", ())
        if isinstance(state_items, tuple):
            # 回灌的 reasoning item 里 summary 等嵌套层同样是冻结的，要整棵解冻；
            # 入参是 Mapping，thaw_json_value 必然回 dict，收窄成请求体条目类型
            items.extend(
                cast(dict[str, object], thaw_json_value(item))
                for item in state_items
                if isinstance(item, Mapping)
            )
    return items


def _translate_response_event(
    raw: Mapping[str, object],
    sequence: int,
    *,
    provider: str,
    model: str,
    open_text_blocks: set[str],
    provider_state_payload: dict[str, JsonValue],
) -> list[ModelStreamEvent]:
    """按真实事件类别转交内容、工具和终态；传参：原事件、序号与响应状态；返回：规范事件。"""
    event_type = raw.get("type")
    if event_type == "response.created":
        response = _mapping(raw.get("response"))
        return [
            _event(
                "response_start",
                sequence,
                provider,
                model,
                message_id=str(response.get("id") or "responses-message"),
            )
        ]
    if event_type == "response.output_item.added":
        return _output_item_added(raw, sequence, provider, model)
    if event_type in _TEXT_EVENTS:
        return _translate_text_event(
            raw,
            sequence,
            provider=provider,
            model=model,
            open_text_blocks=open_text_blocks,
        )
    if event_type in {
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
    }:
        return [_function_argument_event(raw, sequence, provider, model)]
    if event_type == "response.output_item.done":
        _capture_response_state(raw, provider_state_payload)
        return []
    if event_type == "response.completed":
        return _completed_events(raw, sequence, provider, model, provider_state_payload)
    if event_type in {"response.failed", "response.incomplete", "error"}:
        return [_failed_event(raw, sequence, provider, model)]
    if event_type in {
        "response.in_progress",
        "response.content_part.added",
        "response.content_part.done",
    }:
        return []
    raise ProviderAdapterError(
        f"invalid_provider_response:unsupported_responses_event:{event_type}"
    )


def _translate_text_event(
    raw: Mapping[str, object],
    sequence: int,
    *,
    provider: str,
    model: str,
    open_text_blocks: set[str],
) -> list[ModelStreamEvent]:
    """保留正文与摘要的独立生命周期，完成快照不重复追加；传参：内容事件及流状态；返回：规范内容事件。"""
    event_type = raw.get("type")
    if event_type in _TEXT_DELTA_EVENTS:
        return _text_delta(
            raw,
            sequence,
            provider=provider,
            model=model,
            open_text_blocks=open_text_blocks,
        )
    if event_type == "response.reasoning_summary_part.done":
        # 1. 【模型协议】【推理摘要】容器完成携带全文快照，文字与结束已由text事件提交
        return []
    block_id = _text_block_id(raw)
    if event_type == "response.reasoning_summary_part.added":
        part = _mapping(raw.get("part"))
        text = part.get("text")
        if part.get("type") != "summary_text" or not isinstance(text, str):
            raise ProviderAdapterError(
                "invalid_provider_response:reasoning_summary_part"
            )
        if block_id in open_text_blocks:
            raise ProviderAdapterError(
                "invalid_provider_response:duplicate_reasoning_summary_part"
            )
        open_text_blocks.add(block_id)
        events = [
            _event(
                "content_start",
                sequence,
                provider,
                model,
                block_id=block_id,
                content_kind="thinking",
            )
        ]
        if text:
            events.append(
                _event(
                    "content_delta",
                    sequence + 1,
                    provider,
                    model,
                    block_id=block_id,
                    delta=text,
                )
            )
        return events
    open_text_blocks.discard(block_id)
    return [_event("content_end", sequence, provider, model, block_id=block_id)]


def _function_argument_event(
    raw: Mapping[str, object],
    sequence: int,
    provider: str,
    model: str,
) -> ModelStreamEvent:
    values: dict[str, object] = {"block_id": _function_block_id(raw)}
    if raw.get("type") == "response.function_call_arguments.delta":
        values["delta"] = str(raw.get("delta") or "")
        return _event("content_delta", sequence, provider, model, **values)
    return _event("content_end", sequence, provider, model, **values)


def _output_item_added(
    raw: Mapping[str, object],
    sequence: int,
    provider: str,
    model: str,
) -> list[ModelStreamEvent]:
    item = _mapping(raw.get("item"))
    if item.get("type") == "function_call":
        return [
            _event(
                "content_start",
                sequence,
                provider,
                model,
                block_id=_function_block_id(raw),
                content_kind="tool_call",
                call_id=str(item.get("call_id") or ""),
                tool_name=str(item.get("name") or ""),
            )
        ]
    if item.get("type") in {"message", "reasoning"}:
        return []
    raise ProviderAdapterError(
        f"invalid_provider_response:unsupported_output_item:{item.get('type')}"
    )


def _text_delta(
    raw: Mapping[str, object],
    sequence: int,
    *,
    provider: str,
    model: str,
    open_text_blocks: set[str],
) -> list[ModelStreamEvent]:
    """将真实文本增量追加到对应内容块；传参：原事件、序号与流状态；返回：开始及增量事件。"""
    block_id = _text_block_id(raw)
    kind = "thinking" if "reasoning" in str(raw.get("type")) else "text"
    events: list[ModelStreamEvent] = []
    if block_id not in open_text_blocks:
        open_text_blocks.add(block_id)
        events.append(
            _event(
                "content_start",
                sequence,
                provider,
                model,
                block_id=block_id,
                content_kind=kind,
            )
        )
    events.append(
        _event(
            "content_delta",
            sequence + len(events),
            provider,
            model,
            block_id=block_id,
            delta=str(raw.get("delta") or ""),
        )
    )
    return events


def _completed_events(
    raw: Mapping[str, object],
    sequence: int,
    provider: str,
    model: str,
    provider_state_payload: dict[str, JsonValue],
) -> list[ModelStreamEvent]:
    response = _mapping(raw.get("response"))
    events: list[ModelStreamEvent] = []
    usage = _responses_usage(response.get("usage"))
    if usage is not None:
        events.append(_event("usage_update", sequence, provider, model, usage=usage))
    provider_state_payload["response_id"] = str(response.get("id") or "")
    state = ProviderStateEnvelope(
        "openai_responses", provider, model, 1, provider_state_payload
    )
    events.append(
        _event(
            "response_done",
            sequence + len(events),
            provider,
            model,
            stop_reason="end_turn",
            provider_state=state,
        )
    )
    return events


def _capture_response_state(
    raw: Mapping[str, object],
    payload: dict[str, JsonValue],
) -> None:
    item = _mapping(raw.get("item"))
    if item.get("type") != "reasoning":
        return
    encrypted = item.get("encrypted_content")
    if not isinstance(encrypted, str) or not encrypted:
        return
    values = payload.get("input_items")
    if not isinstance(values, list):
        raise ProviderAdapterError("invalid_model_protocol:responses_state_items")
    values.append(
        {
            "type": "reasoning",
            "id": str(item.get("id") or ""),
            "encrypted_content": encrypted,
            "summary": [],
        }
    )


def _failed_event(
    raw: Mapping[str, object], sequence: int, provider: str, model: str
) -> ModelStreamEvent:
    response = _mapping(raw.get("response"))
    error_data = _mapping(response.get("error") or raw.get("error"))
    summary = str(
        error_data.get("message")
        or response.get("incomplete_details")
        or "provider response failed"
    )
    error = ProviderError(
        "provider_error",
        "stream_decode",
        False,
        summary,
        provider,
        model,
        "openai_responses",
    )
    return _event("response_error", sequence, provider, model, error=error)


def _responses_usage(raw: object) -> ModelUsage | None:
    if not isinstance(raw, Mapping):
        return None
    input_details = _mapping(raw.get("input_tokens_details"))
    return ModelUsage(
        input_tokens=reported(_as_int(raw.get("input_tokens")))
        if raw.get("input_tokens") is not None
        else ModelUsage().input_tokens,
        output_tokens=reported(_as_int(raw.get("output_tokens")))
        if raw.get("output_tokens") is not None
        else ModelUsage().output_tokens,
        total_tokens=reported(_as_int(raw.get("total_tokens")))
        if raw.get("total_tokens") is not None
        else ModelUsage().total_tokens,
        cache_read_input_tokens=reported(_as_int(input_details.get("cached_tokens")))
        if input_details.get("cached_tokens") is not None
        else ModelUsage().cache_read_input_tokens,
        cache_write_input_tokens=not_applicable(),
    )


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise ProviderAdapterError("invalid_provider_response:usage_integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    raise ProviderAdapterError("invalid_provider_response:usage_integer")


def _validate_target(request: ModelRequest, model: ModelDescriptor) -> None:
    if model.api_family != "openai_responses":
        raise ProviderAdapterError(
            f"adapter_family_mismatch:openai_responses:{model.api_family}"
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
    for message in request.messages:
        if not isinstance(message, AssistantMessage) or message.provider_state is None:
            continue
        state = message.provider_state
        if (state.api_family, state.provider, state.model) != (
            "openai_responses",
            model.provider,
            model.model_id,
        ):
            raise ProviderAdapterError("provider_state_mismatch:openai_responses")


def _require_state_target(state: ProviderStateEnvelope) -> None:
    if state.api_family != "openai_responses":
        raise ProviderAdapterError("provider_state_mismatch:openai_responses")


def _text_parts(parts: object) -> str:
    if not isinstance(parts, tuple):
        return ""
    return "\n".join(
        part.text for part in parts if isinstance(part, (TextPart, ThinkingPart))
    )


def _function_block_id(raw: Mapping[str, object]) -> str:
    item = _mapping(raw.get("item"))
    return (
        f"function:{raw.get('item_id') or item.get('id') or raw.get('output_index', 0)}"
    )


def _text_block_id(raw: Mapping[str, object]) -> str:
    """区分同一输出项的正文序号和摘要序号；传参：原内容事件；返回：稳定内容块身份。"""
    if str(raw.get("type")).startswith("response.reasoning_summary_"):
        index = raw.get("summary_index")
        if type(index) is not int or index < 0:
            raise ProviderAdapterError(
                "invalid_provider_response:reasoning_summary_index"
            )
        return f"summary:{raw.get('item_id') or raw.get('output_index', 0)}:{index}"
    return f"text:{raw.get('item_id') or raw.get('output_index', 0)}:{raw.get('content_index', 0)}"


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _json(value: Mapping[str, JsonValue]) -> str:
    # 历史回灌的工具参数同样是深冻结的，嵌套对象要一起解冻才能编码
    return json.dumps(
        thaw_json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _event(
    kind: str, sequence: int, provider: str, model: str, **values: Any
) -> ModelStreamEvent:
    return ModelStreamEvent(
        kind, sequence, "openai_responses", provider, model, **values
    )


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
    raise ProviderAdapterError("invalid_provider_response:responses_event_type")


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
        api_family="openai_responses",
        stage="transport",
    )
    yield _event("response_error", sequence, provider, model, error=normalized)


__all__ = ["OpenAIResponsesAdapter"]
