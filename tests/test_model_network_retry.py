"""验证真实 SDK 断流的等待、重试隔离与取消边界。

作者：xxx
时间：2026-09-29 10:45:00
"""

from __future__ import annotations

import json
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from threading import Timer
from time import perf_counter
from types import ModuleType
from typing import Any

import httpx
import httpx2
import openai
import pytest

from llm.client import RealLLMClient, _interruptible_backoff
from llm.config import LLMProviderConfig
from llm.model_registry import ModelDescriptor, ModelRegistry
from llm.model_request import Capability
from llm.provider_adapter import AdapterRegistry
from llm.provider_connection import ResolvedConnection
from llm.provider_result import normalize_provider_error
from llm.providers.anthropic_messages import AnthropicMessagesAdapter
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.providers.openai_responses import OpenAIResponsesAdapter
from llm.types import LLMPlan, ModelRetryNotice, ModelStreamOutput, ModelAttemptStarted
from runtime.cancellation import CancellationToken, ExecutionCancelled
from tools.tool_registry import ToolRegistry


_MODEL = "retry-test-model"
_CONTEXT_WINDOW = 128000
_UPSTREAM_ERROR = 'data: {"error":{"message":"Upstream stream disconnected"}}\n\n'
_OUTPUT_TOKENS = 4096


def _chunk(
    delta: dict[str, object],
    *,
    finish: str | None = None,
    usage: dict[str, int] | None = None,
) -> str:
    """构造真实 SDK 可读取的 SSE；传参：增量、完成原因与用量；返回：一条 SSE。"""
    body: dict[str, object] = {
        "id": "retry-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": _MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        body["usage"] = usage
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def _success() -> str:
    """构造完整模型响应；传参：无；返回：带完成标记和已知用量的 SSE。"""
    return (
        _chunk({"content": "完整回答"})
        + _chunk(
            {},
            finish="stop",
            usage={"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
        )
        + "data: [DONE]\n\n"
    )


@dataclass
class _NetworkScript:
    """只模拟网络字节与传输故障，保留真实 SDK 和模型客户端；作者：xxx。"""

    replies: tuple[
        str | tuple[int, dict[str, str], str] | BaseException | httpx2.SyncByteStream,
        ...,
    ]
    http: ModuleType = httpx2
    requests: list[dict[str, object]] = field(default_factory=list)

    def __call__(self, request: Any) -> Any:
        """按发送顺序返回网络结果；传参：真实 SDK 请求；返回：HTTP 响应或抛传输异常。"""
        index = len(self.requests)
        self.requests.append(json.loads(request.content))
        reply = self.replies[index]
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, httpx2.SyncByteStream):
            return self.http.Response(
                200, headers={"content-type": "text/event-stream"}, stream=reply
            )
        if isinstance(reply, tuple):
            status, headers, message = reply
            return self.http.Response(
                status, headers=headers, json={"error": {"message": message}}
            )
        return self.http.Response(
            200, headers={"content-type": "text/event-stream"}, text=reply
        )


def _client(
    script: _NetworkScript, *, api_family: str = "openai_chat"
) -> RealLLMClient:
    """把网络脚本注入生产模型客户端；传参：脚本与协议族；返回：经过真实适配器与 SDK 的客户端。"""
    adapter_type = {
        "openai_chat": OpenAIChatAdapter,
        "openai_responses": OpenAIResponsesAdapter,
        "anthropic_messages": AnthropicMessagesAdapter,
    }[api_family]
    adapter = adapter_type(
        http_client=script.http.Client(transport=script.http.MockTransport(script))
    )
    descriptor = ModelDescriptor(
        "production",
        "fixture",
        _MODEL,
        api_family,
        {
            Capability.STREAMING: True,
            Capability.NATIVE_TOOLS: True,
            Capability.CONTEXT_WINDOW_TOKENS: _CONTEXT_WINDOW,
            Capability.OUTPUT_TOKENS: _OUTPUT_TOKENS,
        },
        {},
        "fixture",
    )
    return RealLLMClient(
        LLMProviderConfig(
            base_url="https://fixture.invalid/v1",
            model=_MODEL,
            api_key="synthetic",
            timeout_seconds=10,
        ),
        adapter_registry=AdapterRegistry([adapter]),
        model_registry=ModelRegistry([descriptor]),
        connection=ResolvedConnection(
            "https://fixture.invalid/v1", 10, "synthetic", {}
        ),
    )


def _drain(
    stream: Generator[ModelStreamOutput, None, LLMPlan],
) -> tuple[list[ModelStreamOutput], LLMPlan]:
    """同时收集流式提示和最终结果；传参：模型流；返回：事件与计划。"""
    events = []
    while True:
        try:
            events.append(next(stream))
        except StopIteration as done:
            return events, done.value


def _context(**values: object) -> dict[str, object]:
    """提供不含机器状态的调用上下文；传参：本例上下文；返回：隔离的工具注册表与上下文。"""
    return {"tool_registry": ToolRegistry(), **values}


@pytest.mark.parametrize("body", [None, {"message": "Upstream stream disconnected"}])
def test_sdk_upstream_disconnect_is_retryable(body: object) -> None:
    """历史上游断流必须可重试；传参：SDK 错误体；返回：无。"""
    error = openai.APIError(
        "Upstream stream disconnected",
        httpx2.Request("POST", "https://fixture.invalid"),
        body=body,
    )
    normalized = normalize_provider_error(
        error,
        provider="fixture",
        model=_MODEL,
        api_family="openai_chat",
        stage="transport",
    )
    assert normalized.category == "transport_error"
    assert normalized.retryable


@pytest.mark.parametrize(
    "first",
    [
        "",
        "data: [DONE]\n\n",
        _chunk({"content": "首轮半截"}) + _UPSTREAM_ERROR,
        _chunk({"content": "首轮半截"}),
        httpx2.ReadTimeout("network read timed out"),
        httpx2.RemoteProtocolError("peer closed connection"),
        httpx2.ConnectError("network temporarily offline"),
    ],
)
def test_transient_failures_wait_then_succeed_without_merging_attempts(
    monkeypatch: pytest.MonkeyPatch, first: object
) -> None:
    """暂时断流等待后成功，正文及用量不跨尝试拼接；传参：替换器、首轮故障；返回：无。"""
    waits: list[float] = []
    monkeypatch.setattr("llm.client.sleep", waits.append)
    script = _NetworkScript((first, _success()))
    events, plan = _drain(_client(script).plan_stream("继续回答", _context()))
    assert plan.model_error is None
    assert plan.final_output == "完整回答"
    assert len(script.requests) == 2
    assert script.requests[0] == script.requests[1]
    assert len(waits) == 1 and 2 <= waits[0] <= 3
    notices = [event for event in events if isinstance(event, ModelRetryNotice)]
    assert len(notices) == 1
    assert (
        notices[0].attempt_index,
        notices[0].max_attempts,
        notices[0].wait_seconds,
    ) == (2, 3, waits[0])
    assert [attempt.attempt_index for attempt in plan.model_attempts] == [1, 2]
    assert plan.model_attempts[0].error is not None
    assert plan.model_attempts[0].usage.total_tokens.value is None
    assert plan.model_attempts[1].usage.total_tokens.value == 16
    assert plan.observation is not None
    assert plan.observation.total_tokens is None


@pytest.mark.parametrize(
    ("response", "summary"),
    [(_UPSTREAM_ERROR, "Upstream stream disconnected"), ("", "empty_chat_stream")],
)
def test_disconnect_exhaustion_preserves_failure_and_all_attempts(
    monkeypatch: pytest.MonkeyPatch, response: str, summary: str
) -> None:
    """断流耗尽仅三次发送，保留真实错误；传参：替换器；返回：无。"""
    waits: list[float] = []
    monkeypatch.setattr("llm.client.sleep", waits.append)
    script = _NetworkScript((response,) * 3)
    events, plan = _drain(_client(script).plan_stream("回答", _context()))
    assert plan.model_error is not None
    assert plan.model_error.category == "transport_error"
    assert len(script.requests) == len(plan.model_attempts) == 3
    assert len(waits) == 2 and 2 <= waits[0] <= 3 and 4 <= waits[1] <= 6
    assert [
        event.attempt_index for event in events if isinstance(event, ModelRetryNotice)
    ] == [2, 3]
    assert all(
        attempt.error is not None and attempt.error.summary == summary
        for attempt in plan.model_attempts
    )
    assert plan.run_tools_request is None


def test_retry_after_wait_is_announced_before_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """服务端等待秒数必须先上屏再等待；传参：替换器；返回：无。"""
    waits: list[float] = []
    monkeypatch.setattr("llm.client.sleep", waits.append)
    script = _NetworkScript(((429, {"retry-after": "7"}, "rate limited"), _success()))
    stream = _client(script).plan_stream("回答", _context())
    started_notice = next(stream)
    assert isinstance(started_notice, ModelAttemptStarted)
    assert len(script.requests) == 0
    notice = next(stream)
    assert isinstance(notice, ModelRetryNotice)
    assert notice.wait_seconds == 7
    assert notice.error_category == "rate_limited"
    assert waits == [] and len(script.requests) == 1
    _, plan = _drain(stream)
    assert waits == [7] and plan.model_error is None


@pytest.mark.parametrize("response", [_UPSTREAM_ERROR, ""])
def test_user_cancel_during_backoff_interrupts_without_second_dispatch(
    response: str,
) -> None:
    """真正停止可以打断空流或断流后的等待且不重发；传参：失败响应；返回：无。"""
    token = CancellationToken()
    script = _NetworkScript((response, _success()))
    stream = _client(script).plan_stream("回答", _context(cancellation=token))
    started_notice = next(stream)
    assert isinstance(started_notice, ModelAttemptStarted)
    assert len(script.requests) == 0
    notice = next(stream)
    assert isinstance(notice, ModelRetryNotice)
    assert notice.attempt_index == 2
    token.cancel("user_stop")
    started = perf_counter()
    _, plan = _drain(stream)
    assert perf_counter() - started < 1
    assert len(script.requests) == 1
    assert plan.model_error is not None
    assert plan.model_error.category == "cancelled"
    assert plan.model_error.stage == "backoff"


def test_backoff_observes_cancel_while_waiting() -> None:
    """实际等待期间停止必须抛出取消；传参：无；返回：无。"""
    token = CancellationToken()
    timer = Timer(0.05, token.cancel)
    timer.start()
    try:
        with pytest.raises(ExecutionCancelled):
            _interruptible_backoff(5, token)
    finally:
        timer.cancel()


@pytest.mark.parametrize(
    ("reply", "category"),
    [
        ((401, {}, "Invalid API key"), "auth"),
        ((402, {}, "Your credit balance is too low"), "billing"),
        ((400, {}, "Unrecognized request argument"), "format_error"),
        ('data: {"error":{"message":"provider failed"}}\n\n', "provider_error"),
        (
            _chunk(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "invalid",
                            "type": "function",
                            "function": {
                                "name": "terminal_run",
                                "arguments": "not-json",
                            },
                        }
                    ]
                },
                finish="tool_calls",
            ),
            "invalid_model_protocol",
        ),
    ],
)
def test_permanent_or_protocol_failures_do_not_retry(
    monkeypatch: pytest.MonkeyPatch, reply: object, category: str
) -> None:
    """认证、欠费、请求及真实协议错误不可盲重试；传参：替换器、响应及分类；返回：无。"""
    waits: list[float] = []
    monkeypatch.setattr("llm.client.sleep", waits.append)
    script = _NetworkScript((reply,))
    events, plan = _drain(_client(script).plan_stream("回答", _context()))
    assert plan.model_error is not None
    assert plan.model_error.category == category
    assert len(script.requests) == 1 and waits == []
    assert not any(isinstance(event, ModelRetryNotice) for event in events)


def test_partial_tool_call_never_reaches_successful_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首轮半截工具调用不会带到重试结果；传参：替换器；返回：无。"""
    monkeypatch.setattr("llm.client.sleep", lambda _seconds: None)
    partial = _chunk(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "partial",
                    "type": "function",
                    "function": {"name": "terminal_run", "arguments": '{"command":'},
                }
            ]
        }
    )
    script = _NetworkScript((partial + _UPSTREAM_ERROR, _success()))
    _, plan = _drain(_client(script).plan_stream("回答", _context()))
    assert plan.model_error is None and plan.run_tools_request is None
    assert plan.final_output == "完整回答"
    assert plan.model_attempts[0].error is not None
    assert plan.model_attempts[0].error.category == "transport_error"


@dataclass
class _InterruptedBytes(httpx2.SyncByteStream):
    """在真实 SDK 消费正文期间断开网络；作者：xxx。"""

    first_chunk: str
    failure: Exception

    def __iter__(self) -> Iterator[bytes]:
        """先交付部分 SSE 再暴露网络异常；传参：无；返回：字节流。"""
        yield self.first_chunk.encode("utf-8")
        raise self.failure


@pytest.mark.parametrize(
    "failure",
    [
        httpx2.ReadTimeout("mid-stream timeout"),
        httpx2.RemoteProtocolError("mid-stream disconnect"),
    ],
)
def test_sdk_transport_failure_after_partial_bytes_retries(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    """SDK 读取半截响应后失败也必须重试；传参：替换器与网络故障；返回：无。"""
    monkeypatch.setattr("llm.client.sleep", lambda _seconds: None)
    script = _NetworkScript(
        (_InterruptedBytes(_chunk({"content": "首轮半截"}), failure), _success())
    )
    events, plan = _drain(_client(script).plan_stream("回答", _context()))
    assert plan.model_error is None and plan.final_output == "完整回答"
    assert plan.model_attempts[0].response["text"] == "首轮半截"
    assert any(isinstance(event, ModelRetryNotice) for event in events)


@pytest.mark.parametrize("ending", ["", _UPSTREAM_ERROR])
def test_reported_usage_survives_following_disconnect(
    monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    """已报告用量不能因随后断流丢失；传参：替换器与断流形式；返回：无。"""
    monkeypatch.setattr("llm.client.sleep", lambda _seconds: None)
    partial = _chunk(
        {"content": "首轮半截"},
        usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    )
    script = _NetworkScript((partial + ending, _success()))
    _, plan = _drain(_client(script).plan_stream("回答", _context()))
    assert plan.model_error is None
    assert plan.model_attempts[0].usage.total_tokens.value == 9
    assert plan.observation is not None
    assert plan.observation.total_tokens == 25


@pytest.mark.parametrize(
    ("family", "http"), [("openai_responses", httpx2), ("anthropic_messages", httpx)]
)
def test_other_sdk_families_retry_only_started_incomplete_streams(
    monkeypatch: pytest.MonkeyPatch, family: str, http: ModuleType
) -> None:
    """其他协议族已开始的流缺终止标记也可恢复；传参：替换器、协议族与 HTTP 实现；返回：无。"""
    monkeypatch.setattr("llm.client.sleep", lambda _seconds: None)
    path = (
        Path(__file__).parent
        / "fixtures"
        / "llm"
        / "providers"
        / family
        / "stream_text.jsonl"
    )
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    chunks = [f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in raw]
    script = _NetworkScript(("".join(chunks[:-1]), "".join(chunks)), http=http)
    events, plan = _drain(
        _client(script, api_family=family).plan_stream("回答", _context())
    )
    assert plan.model_error is None and plan.final_output == "hello"
    assert len(script.requests) == 2
    assert plan.model_attempts[0].error is not None
    assert plan.model_attempts[0].error.category == "transport_error"
    assert len([event for event in events if isinstance(event, ModelRetryNotice)]) == 1


def test_already_cancelled_request_never_dispatches() -> None:
    """取消后的请求不能发送；传参：无；返回：无。"""
    token = CancellationToken()
    token.cancel("user_stop")
    script = _NetworkScript((_success(),))
    events, plan = _drain(
        _client(script).plan_stream("回答", _context(cancellation=token))
    )
    assert plan.model_error is not None
    assert plan.model_error.category == "cancelled"
    assert script.requests == [] and events == []
