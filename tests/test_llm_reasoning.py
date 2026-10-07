from __future__ import annotations
from scripts.testing.llm import from_test_text_json_stub

from collections.abc import Iterator

from llm.client import RealLLMClient
from scripts.testing.llm import (
    _ScriptedAdapter,
    _test_config,
    _test_connection,
    _test_model_registry,
)
from llm.messages import StopReason
from llm.model_registry import ModelDescriptor
from llm.model_request import ModelRequest
from llm.provider_adapter import AdapterRegistry
from llm.provider_connection import ResolvedConnection
from llm.provider_stream import ModelStreamEvent
from llm.reasoning import split_think_blocks

_PROVIDER_REASONING = "Provider chain of thought."
_INLINE_REASONING = "Inline note."
_FINAL_JSON = '{"type":"final","content":"done"}'
_VISIBLE_TEXT = f"<think>{_INLINE_REASONING}</think>{_FINAL_JSON}"
_STUB_MESSAGE_ID = "reasoning-stub-message"


def test_split_think_blocks_returns_visible_text_and_reasoning() -> None:
    visible, reasoning = split_think_blocks(
        "before <think>first</think> middle <think>second</think> after"
    )

    assert visible == "before  middle  after"
    assert reasoning == "first\n\nsecond"


def test_client_moves_inline_think_block_to_plan_reasoning() -> None:
    client = from_test_text_json_stub(
        '<think>Inspect intent.</think>{"type":"final","content":"done"}'
    )

    plan = client.plan("12345")

    assert plan.final_output == "done"
    assert plan.reasoning_content == "Inspect intent."


def _event(
    model: ModelDescriptor,
    kind: str,
    sequence: int,
    *,
    block_id: str = "",
    content_kind: str | None = None,
    delta: str = "",
    stop_reason: StopReason | None = None,
) -> ModelStreamEvent:
    """造一个带完整身份的流事件，身份取本轮选中模型。

    作者：LKX
    时间：2026-08-31 15:40:00
    传参：model 为选中模型，事件的 provider 与 model 取自它；kind 为事件种类；
          sequence 为事件序号；其余关键字为该种类用到的事件字段
    返回：ModelStreamEvent

    字段逐个显式列出而不是收成 **kwargs：事件构造写错字段是这类测试桩最容易犯的错，
    收成 kwargs 会让错字段一路滑到跑流的时候才炸。
    """
    return ModelStreamEvent(
        kind=kind,
        sequence=sequence,
        api_family=_ScriptedAdapter.api_family,
        provider=model.provider,
        model=model.model_id,
        message_id=_STUB_MESSAGE_ID,
        block_id=block_id,
        content_kind=content_kind,
        delta=delta,
        stop_reason=stop_reason,
    )


def _two_source_reasoning_events(model: ModelDescriptor) -> list[ModelStreamEvent]:
    """发一轮同时带两处推理来源的事件流：独立 thinking 块 + 正文内联 think 块。

    作者：LKX
    时间：2026-08-31 15:40:00
    传参：model 为本轮选中模型
    返回：从 response_start 到 response_done 的完整事件列表

    真实端点把 delta.reasoning_content 交成独立 thinking 块，同一条消息的正文里还可能
    另有内联 <think>。两处推理会同时出现，所以这里两处都发，用来守客户端把它们合起来而
    不是只取一处。
    """
    events: list[ModelStreamEvent] = [_event(model, "response_start", 0)]
    blocks = (
        ("thinking", _PROVIDER_REASONING),
        ("text", _VISIBLE_TEXT),
    )
    for content_kind, text in blocks:
        block_id = f"block-{content_kind}"
        events.append(
            _event(
                model,
                "content_start",
                len(events),
                block_id=block_id,
                content_kind=content_kind,
            )
        )
        events.append(
            _event(model, "content_delta", len(events), block_id=block_id, delta=text)
        )
        events.append(_event(model, "content_end", len(events), block_id=block_id))
    events.append(
        _event(model, "response_done", len(events), stop_reason=StopReason.END_TURN)
    )
    return events


class _TwoSourceReasoningAdapter:
    """按"推理块 + 内联 think 正文"发一轮事件流的测试 Adapter。

    作者：LKX
    时间：2026-08-31 15:40:00
    传参：无
    返回：满足 ProviderAdapter 协议的测试适配器

    六个 from_test_* 工厂都表达不了独立 thinking 块——_ScriptedTurn 只有文本、工具调用
    与错误三种形状。这里只替换最外层"事件从哪来"，StreamAssembler 与推理合并都仍是生产
    实现，所以断言覆盖的是真实装配路径。
    """

    api_family = _ScriptedAdapter.api_family

    def __init__(self) -> None:
        self._scripted = _ScriptedAdapter(())

    def build_request(
        self, request: ModelRequest, *, model_id: str
    ) -> dict[str, object]:
        """把请求体证据的构造原样交回脚本化 Adapter，保持与生产同一份 body。"""
        return self._scripted.build_request(request, model_id=model_id)

    def stream(
        self,
        request: ModelRequest,
        *,
        model: ModelDescriptor,
        connection: ResolvedConnection,
        cancellation=None,
        prepared_body=None,
    ) -> Iterator[ModelStreamEvent]:
        """发一轮带两处推理来源的事件流。"""
        del request, connection, cancellation
        return iter(_two_source_reasoning_events(model))


def test_client_keeps_provider_reasoning_content() -> None:
    client = RealLLMClient(
        _test_config(),
        adapter_registry=AdapterRegistry([_TwoSourceReasoningAdapter()]),
        model_registry=_test_model_registry(),
        connection=_test_connection(),
        protocol_mode="text_json",
    )

    plan = client.plan("explain")

    # Provider 交来的推理块不能被正文解析吃掉：终稿只留 final 内容，推理另走 reasoning_content
    assert plan.final_output == "done"
    # 两处来源都要保住，且按"独立 thinking 块在前、内联 think 在后"合并
    assert plan.reasoning_content == f"{_PROVIDER_REASONING}\n\n{_INLINE_REASONING}"
