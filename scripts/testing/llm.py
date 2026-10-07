"""脚本化 Provider 测试支持；经真实客户端组合与解析，不发网络请求。

作者：xxx；时间：2026-09-28 18:00:00
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final

from llm.client import RealLLMClient
from llm.config import LLMProviderConfig
from llm.messages import (
    StopReason,
    ToolCallPart,
    thaw_json_value,
)
from llm.model_registry import (
    ModelDescriptor,
    ModelRegistry,
)
from llm.model_request import (
    Capability,
    ModelRequest,
)
from llm.production_target import (
    PRODUCTION_MODEL_KEY,
)
from llm.provider_adapter import (
    AdapterRegistry,
)
from llm.provider_connection import ResolvedConnection
from llm.provider_result import (
    ProviderError,
    ProviderErrorCategory,
)
from llm.provider_stream import ModelStreamEvent
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.resolved_target import ResolvedModelTarget
from runtime.cancellation import CancellationToken

_DEFAULT_CONTEXT_WINDOW = 128000


@dataclass(frozen=True, slots=True)
class _ScriptedTurn:
    """一轮脚本化响应：可见文本、原生工具调用块，或一个 Provider 错误。"""

    text: str = ""
    calls: tuple[ToolCallPart, ...] = ()
    error: ProviderError | None = None
    # 分片形态的思考链与答案正文：多于一片才能验出「边生成边显示」而不是轮末一次性。
    # answer 非空时取代 text，两者不同时使用
    thinking: tuple[str, ...] = ()
    answer: tuple[str, ...] = ()
    stop_reason: StopReason | None = None


@dataclass(frozen=True, slots=True)
class _ScriptedOptions:
    """脚本化客户端的编排口径。"""

    protocol_mode: str = "native_tool_calls"
    allowed_actions: list[str] | None = None
    repeat: bool = False
    resolved_target: ResolvedModelTarget | None = None


_DEFAULT_SCRIPTED_OPTIONS = _ScriptedOptions()


@dataclass(frozen=True, slots=True)
class ScriptedTurnOptions:
    """from_test_turns 的编排口径，供测试按需覆盖。

    作者：LKX
    时间：2026-08-31 16:40:00
    传参：repeat 让末轮无限重复以模拟持续故障；context_window 决定完整请求窗口；
          protocol_mode/allowed_actions 为本轮协议与动作口径
    返回：不可变口径对象
    """

    repeat: bool = False
    context_window: int = _DEFAULT_CONTEXT_WINDOW
    protocol_mode: str = "native_tool_calls"
    allowed_actions: list[str] | None = None


_DEFAULT_TURN_OPTIONS = ScriptedTurnOptions()


def from_test_stub(provider_text: str) -> RealLLMClient:
    """构造只回一段固定文本的测试客户端。"""
    return _from_scripted([_ScriptedTurn(text=provider_text)])


def from_test_text_json_stub(provider_text: str) -> RealLLMClient:
    """构造 text_json 协议模式下只回一段固定文本的测试客户端。"""
    return _from_scripted(
        [_ScriptedTurn(text=provider_text)],
        options=_ScriptedOptions(protocol_mode="text_json"),
    )


def from_test_streaming_turn(
    thinking: tuple[str, ...] = (),
    *,
    answer: tuple[str, ...] = (),
) -> RealLLMClient:
    """构造一轮按分片交付思考链与答案正文的测试客户端。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：thinking 为 provider 原生思考块的分片；answer 为答案正文的分片，
          两者都按到达顺序排列
    返回：测试客户端

    用来验流式：分片必须在模型还没答完时就逐条到界面，而不是轮末一次性一整块。
    """
    return _from_scripted([_ScriptedTurn(thinking=thinking, answer=answer)])


def from_test_native_tool_calls(
    tool_calls: list[ToolCallPart], *, content: str = ""
) -> RealLLMClient:
    """构造一轮就返回原生工具调用的测试客户端。

    作者：LKX
    时间：2026-08-30 18:40:00
    传参：tool_calls 为 typed 工具调用块；content 为同轮可见文本
    返回：测试客户端
    """
    return _from_scripted([_ScriptedTurn(text=content, calls=tuple(tool_calls))])


def from_test_native_tool_then_final(
    tool_calls: list[ToolCallPart], final_text: str
) -> RealLLMClient:
    """构造先回工具调用、再回终稿文本的两轮测试客户端。

    作者：LKX
    时间：2026-08-30 18:40:00
    传参：tool_calls 为首轮 typed 工具调用块；final_text 为次轮终稿文本
    返回：测试客户端
    """
    return _from_scripted(
        [
            _ScriptedTurn(calls=tuple(tool_calls)),
            _ScriptedTurn(text=final_text),
        ]
    )


def from_test_error(
    message: str,
    *,
    category: ProviderErrorCategory = "provider_error",
    retryable: bool = False,
) -> RealLLMClient:
    """构造总是返回指定 Provider 错误的测试客户端。

    作者：LKX
    时间：2026-08-30 18:40:00
    传参：message 为错误摘要；category 必须是 ProviderErrorCategory 值域内的分类；
          retryable 标记是否可重试
    返回：测试客户端；越界分类由类型检查拦下，运行期再由 ProviderError 抛 ValueError

    错误在这里就地构造而不是延迟到流里，越界分类会在工厂调用处暴露，
    而不是等发请求时从生成器深处冒出来。
    """
    error = ProviderError(
        category=category,
        stage="transport",
        retryable=retryable,
        summary=message,
        provider=_TEST_PROVIDER,
        model=_TEST_MODEL_ID,
        api_family=_TEST_API_FAMILY,
    )
    # 持续故障：每次调用都撞同一个错，可重试分类才能真正走完重试预算
    return _from_scripted(
        [_ScriptedTurn(error=error)], options=_ScriptedOptions(repeat=True)
    )


def from_test_sequence(
    provider_texts: Iterable[str],
    *,
    protocol_mode: str = "native_tool_calls",
    allowed_actions: list[str] | None = None,
) -> RealLLMClient:
    """构造按序返回多段文本的测试客户端。

    作者：LKX
    时间：2026-08-30 18:40:00
    传参：provider_texts 为逐轮文本；protocol_mode 为协议模式；
          allowed_actions 为本轮允许动作
    返回：测试客户端；脚本用尽后按 provider_error 报错
    """
    return _from_scripted(
        [_ScriptedTurn(text=text) for text in provider_texts],
        options=_ScriptedOptions(
            protocol_mode=protocol_mode, allowed_actions=allowed_actions
        ),
    )


def from_test_turns(
    turns: Iterable[str | ProviderError],
    *,
    options: ScriptedTurnOptions = _DEFAULT_TURN_OPTIONS,
) -> RealLLMClient:
    """构造按序混排"成功文本"与"Provider 错误"两类轮次的测试客户端。

    作者：LKX
    时间：2026-08-31 16:40:00
    传参：turns 里 str 表示一轮成功文本、ProviderError 表示一轮失败；
          options 为重复末轮、上下文窗口与协议口径
    返回：测试客户端

    另外六个工厂各自只表达单一形态（全成功、全失败、工具调用），而重试编排、
    预裁剪这类行为要看"先错后成"的转折，所以这里放开逐轮混排。上下文窗口可调是
    因为预裁剪阈值由它推出，用生产默认值根本触发不到那条路径。
    """
    return _from_scripted(
        [_scripted_turn_from(turn) for turn in turns],
        options=_ScriptedOptions(
            protocol_mode=options.protocol_mode,
            allowed_actions=options.allowed_actions,
            repeat=options.repeat,
            resolved_target=_test_target(options.context_window),
        ),
    )


def _from_scripted(
    turns: list[_ScriptedTurn],
    *,
    options: _ScriptedOptions = _DEFAULT_SCRIPTED_OPTIONS,
) -> RealLLMClient:
    """用脚本化 Adapter 组装测试客户端，注入面与生产完全同一条。

    作者：LKX
    时间：2026-08-30 19:20:00
    传参：turns 为逐轮脚本；options 为协议模式、允许动作与末轮重复开关
    返回：测试客户端
    """
    adapter = _ScriptedAdapter(tuple(turns), repeat=options.repeat)
    return RealLLMClient(
        _test_config(),
        adapter_registry=AdapterRegistry([adapter]),
        model_registry=_test_model_registry(),
        connection=_test_connection(),
        allowed_actions=options.allowed_actions,
        protocol_mode=options.protocol_mode,
        resolved_target=options.resolved_target,
    )


_TEST_PROVIDER = "stub"
_TEST_MODEL_ID = "stub-model"
_TEST_API_FAMILY = "scripted_test"
_TEST_BASE_URL = "http://stub.local/v1"
_TEST_TIMEOUT_SECONDS = 10
_TEST_CONTEXT_WINDOW = 128000
# 脚本用尽属于测试脚本没写够轮次，按 Provider 失败暴露而不是静默回空
_SEQUENCE_EXHAUSTED_SUMMARY = "test stub sequence exhausted"
# 测试脚本按 chat completions 语义写，请求体证据也就借它的 body 构造器，只不发网络
_TEST_BODY_BUILDER: Final = OpenAIChatAdapter()


class _ScriptedAdapter:
    """按预置脚本发 ModelStreamEvent 的测试 Adapter，走与生产同一条装配链。

    作者：LKX
    时间：2026-08-30 18:50:00
    传参：turns 为逐轮脚本
    返回：满足 ProviderAdapter 协议的测试适配器

    只替换最外层的"事件从哪来"，StreamAssembler、错误桥、证据投影都仍是生产实现，
    所以测试覆盖的是真实装配路径而不是一条影子路径。
    """

    api_family = _TEST_API_FAMILY

    def __init__(
        self, turns: tuple[_ScriptedTurn, ...], *, repeat: bool = False
    ) -> None:
        self._turns = turns
        self._repeat = repeat
        self._calls = 0

    def build_request(
        self, request: ModelRequest, *, model_id: str
    ) -> dict[str, object]:
        """交出本轮请求体证据，形状与真实 chat completions 调用一致。

        作者：LKX
        时间：2026-08-30 21:40:00
        传参：request 为本轮 canonical 请求；model_id 为 Wire 模型名
        返回：不含 credential 的 request body

        直接复用生产 body 构造器而不是自造一份摘要：raw_model_request 是要落盘的
        请求证据，如果测试路径记的形状和生产不同，靠它做断言的用例就在核对一条
        永远不会真正发出去的请求。
        """
        return _TEST_BODY_BUILDER.build_request(request, model_id=model_id)

    def stream(
        self,
        request: ModelRequest,
        *,
        model: ModelDescriptor,
        connection: ResolvedConnection,
        cancellation: CancellationToken | None = None,
        prepared_body: Mapping[str, object] | None = None,
    ) -> Iterator[ModelStreamEvent]:
        """按脚本发一轮事件流。

        作者：LKX
        时间：2026-08-30 18:50:00
        传参：request 为本轮请求；model 为选中模型；connection 为已解析连接
        返回：ModelStreamEvent 迭代器
        """
        del request, connection
        turn = self._next_turn()
        return iter(_scripted_events(turn, model))

    def _next_turn(self) -> _ScriptedTurn:
        """取下一轮脚本；用尽后按 Provider 错误暴露。

        作者：LKX
        时间：2026-08-30 19:20:00
        传参：无
        返回：本轮脚本

        repeat 模式下末轮无限重复，用于模拟"每次调用都撞同一个错"的持续故障；
        非 repeat 模式用尽即报错，测试少写了一轮不会被静默当成正常结束。
        """
        index = self._calls
        self._calls += 1
        if self._repeat and self._turns:
            return self._turns[min(index, len(self._turns) - 1)]
        if index < len(self._turns):
            return self._turns[index]
        return _ScriptedTurn(
            error=ProviderError(
                category="provider_error",
                stage="transport",
                retryable=False,
                summary=_SEQUENCE_EXHAUSTED_SUMMARY,
                provider=_TEST_PROVIDER,
                model=_TEST_MODEL_ID,
                api_family=_TEST_API_FAMILY,
            )
        )


def _scripted_events(
    turn: _ScriptedTurn, model: ModelDescriptor
) -> list[ModelStreamEvent]:
    """把一轮脚本翻成合规的事件序列。

    作者：LKX
    时间：2026-08-30 18:50:00
    传参：turn 为本轮脚本；model 为选中模型
    返回：从 response_start 到终结事件的完整事件列表

    文本与工具调用都为空时不发内容块，StreamAssembler 会按 empty_assistant_message
    抛错，再由错误桥转成 empty_response，与旧路径"空响应"行为一致。
    """
    builder = _ScriptedEventBuilder(model)
    if turn.error is not None:
        return builder.error_events(turn.error)
    if turn.thinking:
        builder.thinking_block(turn.thinking)
    answer = turn.answer or ((turn.text,) if turn.text else ())
    if answer:
        builder.text_block(answer)
    for part in turn.calls:
        builder.tool_call_block(part)
    return builder.done_events(stop_reason=turn.stop_reason)


class _ScriptedEventBuilder:
    """按事件序与 block 状态机的要求逐个拼装测试事件。"""

    def __init__(self, model: ModelDescriptor) -> None:
        self._model = model
        self._events: list[ModelStreamEvent] = []
        self._blocks = 0
        self._append("response_start")

    def text_block(self, fragments: tuple[str, ...]) -> None:
        """发一个文本内容块，正文按分片逐条交付。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：fragments 为按到达顺序排列的正文分片
        返回：无
        """
        block_id = self._open_block("text")
        for fragment in fragments:
            self._append("content_delta", block_id=block_id, delta=fragment)
        self._append("content_end", block_id=block_id)

    def thinking_block(self, fragments: tuple[str, ...]) -> None:
        """发一个 provider 原生思考块，正文按分片逐条交付。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：fragments 为按到达顺序排列的思考正文分片
        返回：无

        分片而不是整段：流式路径下每个 content_delta 对应界面上一条增量，
        只发一片就验不出「边生成边显示」和「轮末一次性」的区别。
        """
        block_id = self._open_block("thinking")
        for fragment in fragments:
            self._append("content_delta", block_id=block_id, delta=fragment)
        self._append("content_end", block_id=block_id)

    def tool_call_block(self, part: ToolCallPart) -> None:
        """发一个原生工具调用块，参数以 JSON 增量交付。"""
        block_id = self._open_block(
            "tool_call", call_id=part.call_id, tool_name=part.tool_name
        )
        self._append(
            "content_delta",
            block_id=block_id,
            delta=json.dumps(thaw_json_value(part.arguments), ensure_ascii=False),
            call_id=part.call_id,
            tool_name=part.tool_name,
        )
        self._append("content_end", block_id=block_id)

    def done_events(
        self, *, stop_reason: StopReason | None = None
    ) -> list[ModelStreamEvent]:
        """收尾响应事件；传参：显式供应商停止原因；返回：完整事件，省略原因时按内容正常结束。"""
        if stop_reason is None:
            stop_reason = (
                StopReason.TOOL_CALL
                if any(event.content_kind == "tool_call" for event in self._events)
                else StopReason.END_TURN
            )
        self._append("response_done", stop_reason=stop_reason)
        return self._events

    def error_events(self, error: ProviderError) -> list[ModelStreamEvent]:
        """以 response_error 收尾，交出 Provider 失败事实。"""
        self._append("response_error", error=error)
        return self._events

    def _open_block(
        self, content_kind: str, *, call_id: str = "", tool_name: str = ""
    ) -> str:
        """开一个内容块并返回块 id。

        作者：LKX
        时间：2026-08-31 10:20:00
        传参：content_kind 为块类型；call_id/tool_name 仅工具调用块需要
        返回：本块 id
        """
        self._blocks += 1
        block_id = f"block-{self._blocks}"
        self._append(
            "content_start",
            block_id=block_id,
            content_kind=content_kind,
            call_id=call_id,
            tool_name=tool_name,
        )
        return block_id

    def _append(
        self,
        kind: str,
        *,
        block_id: str = "",
        content_kind: str | None = None,
        delta: str = "",
        call_id: str = "",
        tool_name: str = "",
        stop_reason: StopReason | None = None,
        error: ProviderError | None = None,
    ) -> None:
        """按共同身份追加一个事件，序号自增。

        作者：LKX
        时间：2026-08-31 10:20:00
        传参：kind 为事件种类；其余为该种类用到的事件字段
        返回：无

        字段逐个显式列出而不是收成 **kwargs：事件构造错字段是这个桩最容易犯的错，
        写成 kwargs 会让类型检查放过它，测试要到跑流的时候才炸。
        """
        self._events.append(
            ModelStreamEvent(
                kind=kind,
                sequence=len(self._events),
                api_family=_TEST_API_FAMILY,
                provider=self._model.provider,
                model=self._model.model_id,
                message_id=_TEST_MESSAGE_ID,
                block_id=block_id,
                content_kind=content_kind,
                delta=delta,
                call_id=call_id,
                tool_name=tool_name,
                stop_reason=stop_reason,
                error=error,
            )
        )


_TEST_MESSAGE_ID = "scripted-message"


def _test_model_registry() -> ModelRegistry:
    """构造只含脚本化候选的注册表，key 与生产同名以复用允许候选清单。"""
    return ModelRegistry(
        [
            ModelDescriptor(
                model_key=PRODUCTION_MODEL_KEY,
                provider=_TEST_PROVIDER,
                model_id=_TEST_MODEL_ID,
                api_family=_TEST_API_FAMILY,
                capabilities={
                    Capability.NATIVE_TOOLS: True,
                    Capability.STREAMING: True,
                    Capability.REASONING: True,
                    Capability.OUTPUT_TOKENS: _TEST_CONTEXT_WINDOW,
                    Capability.CONTEXT_WINDOW_TOKENS: _TEST_CONTEXT_WINDOW,
                },
                preference_values={},
                connection_profile_key="stub_profile",
            )
        ]
    )


def _test_connection() -> ResolvedConnection:
    """构造不含真实凭据的测试连接；脚本化 Adapter 不会用它发请求。"""
    return ResolvedConnection(
        base_url=_TEST_BASE_URL,
        timeout_seconds=float(_TEST_TIMEOUT_SECONDS),
        credential="stub-credential",
        headers={},
    )


def _test_config() -> LLMProviderConfig:
    return LLMProviderConfig(
        base_url=_TEST_BASE_URL,
        model=_TEST_MODEL_ID,
        api_key=None,
        timeout_seconds=_TEST_TIMEOUT_SECONDS,
    )


def scripted_provider_error(
    category: ProviderErrorCategory,
    summary: str,
    *,
    retryable: bool = False,
) -> ProviderError:
    """按脚本化桩的身份构造一个 Provider 错误，供测试排错误轮。

    作者：LKX
    时间：2026-08-31 16:40:00
    传参：category 为错误分类；summary 为错误摘要；retryable 标记是否可重试
    返回：带测试身份的 ProviderError；category 越界时当场抛 ValueError

    provider/model/api_family 三件身份由这里统一盖，测试就不必为了拼一个错误去
    引用桩的内部常量。
    """
    return ProviderError(
        category=category,
        stage="transport",
        retryable=retryable,
        summary=summary,
        provider=_TEST_PROVIDER,
        model=_TEST_MODEL_ID,
        api_family=_TEST_API_FAMILY,
    )


def _scripted_turn_from(turn: str | ProviderError) -> _ScriptedTurn:
    """把 from_test_turns 的轮次声明翻成内部脚本轮次。

    作者：LKX
    时间：2026-08-31 16:40:00
    传参：turn 为一轮声明，str 表示成功文本、ProviderError 表示失败
    返回：内部 _ScriptedTurn
    """
    if isinstance(turn, ProviderError):
        return _ScriptedTurn(error=turn)
    return _ScriptedTurn(text=turn)


def _test_target(context_window: int) -> ResolvedModelTarget:
    """构造只为定上下文窗口而存在的测试模型目标。

    作者：LKX
    时间：2026-08-31 16:40:00
    传参：context_window 为本轮上下文窗口 token 数
    返回：ResolvedModelTarget

    预裁剪阈值由上下文窗口推出，测试要触发裁剪就得把窗口压小；其余字段与
    _test_config 保持同一套测试身份。
    """
    return ResolvedModelTarget(
        provider=_TEST_PROVIDER,
        model=_TEST_MODEL_ID,
        base_url=_TEST_BASE_URL,
        api_mode="chat_completions",
        timeout_seconds=float(_TEST_TIMEOUT_SECONDS),
        config_source="test",
        credential_source="test",
        api_key_present=False,
        context_window=context_window,
    )
