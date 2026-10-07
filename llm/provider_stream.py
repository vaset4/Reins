from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Mapping, cast

from llm.messages import (
    AssistantContentPart,
    AssistantMessage,
    ProviderStateEnvelope,
    StopReason,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from llm.provider_result import ProviderCallResult, ProviderError, UsageAccumulator
from llm.types import ModelUsage


EventKind = Literal[
    "response_start",
    "content_start",
    "content_delta",
    "content_end",
    "usage_update",
    "response_done",
    "response_error",
]
ContentKind = Literal["text", "thinking", "tool_call"]
ThinkingVisibility = Literal["visible", "redacted"]


class StreamProtocolError(ValueError):
    """表示统一事件违反 block、sequence 或 terminal 状态机。"""


class StreamInterruptedError(StreamProtocolError):
    """【模型调用】【流中断】响应已开始但未收到完成标记，允许同一请求重试。

    作者：xxx
    时间：2026-09-29 10:45:00
    """


@dataclass(frozen=True, slots=True)
class ModelStreamEvent:
    """保存 Adapter 输出的一个 Provider 无关流事件。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：kind/sequence/identity 为公共元数据；其余字段由事件类型约束
    返回：不可变统一事件
    """

    kind: EventKind | str
    sequence: int
    api_family: str
    provider: str
    model: str
    message_id: str = ""
    block_id: str = ""
    content_kind: ContentKind | str | None = None
    thinking_visibility: ThinkingVisibility | str = "visible"
    delta: str = ""
    call_id: str = ""
    tool_name: str = ""
    usage: ModelUsage | None = None
    stop_reason: StopReason | str | None = None
    provider_state: ProviderStateEnvelope | None = None
    error: ProviderError | None = None

    def __post_init__(self) -> None:
        """校验公共元数据和闭合事件种类。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无；非法事件抛 StreamProtocolError
        """
        allowed = {
            "response_start",
            "content_start",
            "content_delta",
            "content_end",
            "usage_update",
            "response_done",
            "response_error",
        }
        if self.kind not in allowed:
            raise StreamProtocolError(f"unknown_stream_event:{self.kind}")
        if self.sequence < 0:
            raise StreamProtocolError("invalid_event_sequence")
        if any(
            not value.strip() for value in (self.api_family, self.provider, self.model)
        ):
            raise StreamProtocolError("stream_event_identity_required")


@dataclass(slots=True)
class _OpenBlock:
    kind: ContentKind
    thinking_visibility: ThinkingVisibility = "visible"
    call_id: str = ""
    tool_name: str = ""
    fragments: list[str] | None = None

    def __post_init__(self) -> None:
        if self.fragments is None:
            self.fragments = []


class StreamAssembler:
    """组装 AssistantMessage 并独占流顺序、tool JSON 和 terminal 不变量。"""

    def __init__(self) -> None:
        """初始化单条流的状态。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：无
        """
        self._started = False
        self._terminal = False
        self._last_sequence = -1
        self._message_id = "provider-message"
        self._identity: tuple[str, str, str] | None = None
        self._open: dict[str, _OpenBlock] = {}
        self._ended: set[str] = set()
        self._call_ids: set[str] = set()
        self._content: list[AssistantContentPart] = []
        self._usage = UsageAccumulator()
        self._result: ProviderCallResult | None = None

    def assemble(
        self, events: list[ModelStreamEvent] | tuple[ModelStreamEvent, ...]
    ) -> ProviderCallResult:
        """消费完整事件序列并要求恰好一个 terminal。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：events 为一个 Adapter 调用的有序统一事件
        返回：成功消息或保留 partial state 的 ProviderCallResult
        """
        for event in events:
            self.accept(event)
        return self.finish()

    def finish(self) -> ProviderCallResult:
        """在事件已通过 accept 消费后检查并返回 terminal 结果。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：无
        返回：ProviderCallResult；已开始却缺失 terminal 时抛断流错误，其他非法状态抛协议错误
        """
        if not self._terminal:
            if self._started:
                raise StreamInterruptedError("missing_terminal_event")
            raise StreamProtocolError("missing_terminal_event")
        if self._result is None:
            raise StreamProtocolError("terminal_result_missing")
        return self._result

    def accept(self, event: ModelStreamEvent) -> None:
        """校验并应用一个统一事件。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：event 为下一个单调事件
        返回：无；协议失败抛 StreamProtocolError
        """
        if self._terminal:
            raise StreamProtocolError("event_after_terminal")
        self._validate_common(event)
        if event.kind == "response_start":
            self._start(event)
            return
        if not self._started:
            raise StreamProtocolError("response_not_started")
        handlers = {
            "content_start": self._content_start,
            "content_delta": self._content_delta,
            "content_end": self._content_end,
            "usage_update": self._usage_update,
            "response_done": self._response_done,
            "response_error": self._response_error,
        }
        handlers[event.kind](event)

    def _validate_common(self, event: ModelStreamEvent) -> None:
        if event.sequence <= self._last_sequence:
            raise StreamProtocolError("non_monotonic_event_sequence")
        self._last_sequence = event.sequence
        identity = (event.api_family, event.provider, event.model)
        if self._identity is not None and self._identity != identity:
            raise StreamProtocolError("stream_identity_changed")

    def _start(self, event: ModelStreamEvent) -> None:
        if self._started:
            raise StreamProtocolError("duplicate_response_start")
        self._started = True
        self._identity = (event.api_family, event.provider, event.model)
        if event.message_id:
            self._message_id = event.message_id

    def _content_start(self, event: ModelStreamEvent) -> None:
        if (
            not event.block_id
            or event.block_id in self._open
            or event.block_id in self._ended
        ):
            raise StreamProtocolError("duplicate_or_blank_block_id")
        if event.content_kind not in {"text", "thinking", "tool_call"}:
            raise StreamProtocolError("invalid_content_kind")
        if event.thinking_visibility not in {"visible", "redacted"}:
            raise StreamProtocolError("invalid_thinking_visibility")
        if event.content_kind != "thinking" and event.thinking_visibility != "visible":
            raise StreamProtocolError("thinking_visibility_on_non_thinking_block")
        if event.content_kind == "tool_call":
            if (
                not event.call_id
                or not event.tool_name
                or event.call_id in self._call_ids
            ):
                raise StreamProtocolError("duplicate_or_invalid_call_id")
            self._call_ids.add(event.call_id)
        self._open[event.block_id] = _OpenBlock(
            cast(ContentKind, event.content_kind),
            cast(ThinkingVisibility, event.thinking_visibility),
            event.call_id,
            event.tool_name,
        )

    def _content_delta(self, event: ModelStreamEvent) -> None:
        block = self._open.get(event.block_id)
        if block is None:
            raise StreamProtocolError("delta_before_content_start")
        if not isinstance(event.delta, str):
            raise StreamProtocolError("content_delta_must_be_text")
        assert block.fragments is not None
        block.fragments.append(event.delta)

    def _content_end(self, event: ModelStreamEvent) -> None:
        block = self._open.pop(event.block_id, None)
        if block is None:
            raise StreamProtocolError("end_before_content_start")
        self._ended.add(event.block_id)
        # 推理模型常吐出只含空格/换行的正文或思考分片，拼完 strip 为空的块没有语义；
        # 放进消息就会被 TextPart/ThinkingPart 的"正文非空白"合同校验拒收，整轮 run 失败，
        # 所以在组装层直接丢弃（无损规范化，不影响后续 empty_assistant_message 判定）
        if block.kind in {"text", "thinking"} and _has_blank_body(block):
            return
        self._content.append(_finish_block(block))

    def _usage_update(self, event: ModelStreamEvent) -> None:
        if event.usage is None:
            raise StreamProtocolError("usage_update_missing_usage")
        self._usage.merge(event.usage)

    def _response_done(self, event: ModelStreamEvent) -> None:
        if self._open:
            raise StreamProtocolError("terminal_with_open_content")
        if not self._content:
            raise StreamProtocolError("empty_assistant_message")
        stop_reason = _normalize_stop_reason(event.stop_reason)
        message = AssistantMessage(
            self._message_id,
            tuple(self._content),
            stop_reason,
            event.provider_state,
            self._usage.finalize(),
        )
        self._terminal = True
        self._result = ProviderCallResult(message=message, usage=self._usage.finalize())

    def _response_error(self, event: ModelStreamEvent) -> None:
        if event.error is None:
            raise StreamProtocolError("response_error_missing_error")
        self._terminal = True
        self._result = self.failure_result(
            event.error, provider_state=event.provider_state
        )

    def failure_result(
        self,
        error: ProviderError,
        *,
        provider_state: ProviderStateEnvelope | None = None,
    ) -> ProviderCallResult:
        """保留流失败前已确认的内容与用量，不把半截工具参数当成可执行调用。

        传参：error 为真实失败；provider_state 为可选续接信息；返回：失败结果快照
        """
        partial_content = list(self._content)
        for block in self._open.values():
            # 未闭合块只保留有实际内容的正文/思考；tool_call 的 JSON 半截无法解析故整块跳过
            if block.kind not in {"text", "thinking"} or _has_blank_body(block):
                continue
            partial_content.append(_finish_block(block))
        partial = _partial_message(
            self._message_id,
            partial_content,
            usage=self._usage.finalize(),
            provider_state=provider_state,
        )
        return ProviderCallResult(
            error=error, partial_message=partial, usage=self._usage.finalize()
        )


def _has_blank_body(block: _OpenBlock) -> bool:
    """判断一个块收集到的分片拼起来是否只剩空白。

    作者：xxx
    时间：2026-09-05 00:00:00
    传参：block 为待收尾的开放块
    返回：拼接结果 strip 后为空返回 True
    """
    return not "".join(block.fragments or []).strip()


def _finish_block(block: _OpenBlock) -> AssistantContentPart:
    text = "".join(block.fragments or [])
    if block.kind == "text":
        return TextPart(text)
    if block.kind == "thinking":
        return ThinkingPart(text, block.thinking_visibility)
    try:
        arguments = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StreamProtocolError("invalid_tool_arguments_json") from exc
    if not isinstance(arguments, Mapping):
        raise StreamProtocolError("tool_arguments_must_be_object")
    return ToolCallPart(block.call_id, block.tool_name, arguments)


def _normalize_stop_reason(value: StopReason | str | None) -> StopReason:
    if value is None:
        return StopReason.UNKNOWN
    if isinstance(value, StopReason):
        return value
    try:
        return StopReason(value)
    except ValueError as exc:
        raise StreamProtocolError(f"unknown_stop_reason:{value}") from exc


def _partial_message(
    message_id: str,
    content: list[AssistantContentPart],
    *,
    usage: ModelUsage,
    provider_state: ProviderStateEnvelope | None,
) -> AssistantMessage | None:
    if not content:
        return None
    return AssistantMessage(
        message_id,
        tuple(content),
        StopReason.UNKNOWN,
        provider_state,
        usage=usage,
    )


__all__ = [
    "ContentKind",
    "EventKind",
    "ModelStreamEvent",
    "StreamAssembler",
    "StreamInterruptedError",
    "StreamProtocolError",
    "ThinkingVisibility",
]
