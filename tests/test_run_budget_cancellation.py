"""真实模型尝试的预算结算和流取消边界。

作者：xxx
时间：2026-09-14 15:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_stub

from threading import Event

import pytest

from llm.provider_result import ProviderError, reported
from llm.types import ModelAttemptStarted, ModelUsage
from runtime.cancellation import (
    CancellationToken,
    ExecutionCancelled,
    RunBudgetExceeded,
)
from runtime.lease import from_trigger
from runtime.watchdog import Watchdog
from tests.test_model_attempts import _event


@pytest.mark.parametrize(
    "failure, category",
    [(ExecutionCancelled, "cancelled"), (RunBudgetExceeded, "budget_exhausted")],
)
def test_dispatch_exception_keeps_its_explicit_model_category(
    monkeypatch, failure, category
):
    """仍可由上游产生的边界异常保持分类且不派发供应商请求；传参：替换器及异常；返回：无。"""
    client = from_test_stub("must not dispatch")
    dispatched = []

    def reserve(_reservation):
        """在真实派发前暴露上游边界；传参：本次预留；返回：不返回。"""
        raise failure("upstream boundary")

    def stream(_request, **_kwargs):
        """记录不应到达的供应商调用；传参：请求；返回：空事件流。"""
        dispatched.append(True)
        return iter(())

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    plan = client.plan("核对运行边界", {"reserve_model_attempt": reserve})
    assert plan.model_error.category == category and not dispatched


def test_cancelled_stream_keeps_partial_content_usage_and_closes_backend(monkeypatch):
    """取消已生成内容的流会关闭后端并结算已有消耗；传参：替换器；返回：无。"""
    closed = Event()
    token = CancellationToken()
    client = from_test_stub("unused")
    adapter = client._adapter_registry.require("scripted_test")
    watchdog = Watchdog(from_trigger("user", task_id="budget"), cancellation=token)
    records = []

    def record(attempt):
        """按相同尝试重复结算以检查去重；传参：尝试；返回：无。"""
        records.append(attempt)
        watchdog.settle_model_attempt(attempt)
        watchdog.settle_model_attempt(attempt)

    def stream(_request, **_kwargs):
        """模拟后端一直生成直到连接被关闭；传参：请求；返回：原始Provider事件。"""
        token.register_closer(closed.set)
        yield _event("response_start", 0, message_id="cancelled-response")
        yield _event("usage_update", 1, usage=ModelUsage(input_tokens=reported(17)))
        yield _event("content_start", 2, block_id="text", content_kind="text")
        yield _event("content_delta", 3, block_id="text", delta="已收到正文")
        assert closed.wait(5)

    monkeypatch.setattr(adapter, "stream", stream)
    output = client.plan_stream(
        "处理取消",
        {
            "cancellation": token,
            "model_attempt_recorder": record,
            "reserve_model_attempt": watchdog.reserve_model_attempt,
        },
    )
    started = next(output)
    assert isinstance(started, ModelAttemptStarted)
    delta = next(output)
    assert delta.text == "已收到正文"
    token.cancel()
    with pytest.raises(StopIteration) as end:
        next(output)
    plan = end.value.value
    assert closed.wait(1)
    assert plan.model_error.category == "cancelled"
    assert plan.raw_model_response["text"] == "已收到正文"
    assert len(records) == 2
    assert records[-1].usage.input_tokens.value == 17
    assert records[-1].usage.output_tokens.value is None
    assert watchdog.tokens_used == 17
    assert watchdog.unknown_usage_attempts == 1
    assert watchdog.steps_taken == 1


def test_retry_continues_past_step_limit_and_records_each_attempt(monkeypatch):
    """步数上限取消后重试不再被拦，每次失败尝试的用量仍然入账；传参：替换器；返回：无。"""
    client = from_test_stub("unused")
    adapter = client._adapter_registry.require("scripted_test")
    watchdog = Watchdog(from_trigger("user", task_id="budget", max_steps=1))
    dispatched, records = [], []

    def record(attempt):
        """将实际尝试交给唯一计量者；传参：尝试；返回：无。"""
        records.append(attempt)
        watchdog.settle_model_attempt(attempt)

    def stream(_request, **_kwargs):
        """返回可重试的真实失败事件；传参：请求；返回：用量及错误。"""
        dispatched.append(True)
        yield _event("response_start", 0, message_id="retry-response")
        yield _event(
            "usage_update",
            1,
            usage=ModelUsage(
                input_tokens=reported(9),
                output_tokens=reported(1),
                total_tokens=reported(10),
            ),
        )
        yield _event(
            "response_error",
            2,
            error=ProviderError(
                category="rate_limited",
                stage="transport",
                retryable=True,
                summary="temporarily unavailable",
                provider="fixture",
                model="fixture-model",
                api_family="fixture",
            ),
        )

    monkeypatch.setattr(adapter, "stream", stream)
    monkeypatch.setattr("llm.client.compute_wait", lambda *_args, **_kwargs: 0)
    plan = client.plan(
        "继续",
        {
            "model_attempt_recorder": record,
            "reserve_model_attempt": watchdog.reserve_model_attempt,
        },
    )
    # 步数超出上限不再拦派发，重试一直走到次数用尽
    assert len(dispatched) == 3
    # 每个尝试留 started 与 finished 两条记录
    assert len(records) == 6
    assert plan.model_error.category == "rate_limited"
    assert watchdog.steps_taken == 3
    assert watchdog.tokens_used == 30
    assert plan.observation.total_tokens == 30


@pytest.mark.parametrize(
    "usage, expected",
    [
        ({}, "unknown/unknown"),
        ({"prompt_tokens": 17}, "17/unknown"),
        ({"input_tokens": 0, "output_tokens": 0}, "0/0"),
    ],
)
def test_rendered_usage_distinguishes_unknown_from_zero(usage, expected):
    """通过用户事件渲染核对未知与零；传参：已知用量及预期；返回：无。"""
    from app.repl.console import capture_console
    from app.repl.render import EventRenderer
    from runtime.stream_events import AssistantTurnComplete

    with capture_console() as console:
        EventRenderer(trace_on=True).render(
            AssistantTurnComplete(content="本次结束", usage=usage)
        )
    assert f"tokens={expected}" in console.export_text()
