"""真实调用边界的尝试记录与失败用量验证（脚本 Provider 机制测试）。

作者：xxx
时间：2026-09-13 20:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_stub

from collections.abc import Iterator

import pytest

from llm.provider_adapter import ProviderAdapterError
from llm.provider_result import ProviderError, reported
from llm.provider_stream import ModelStreamEvent
from llm.types import ModelAttemptEvent, ModelUsage


def _event(kind: str, sequence: int, **values: object) -> ModelStreamEvent:
    """构造同一 Provider 的事件；传参：类别、序号与内容；返回：流事件。"""
    return ModelStreamEvent(
        kind=kind,
        sequence=sequence,
        provider="fixture",
        model="fixture-model",
        api_family="fixture",
        **values,
    )


@pytest.mark.parametrize("failure", ["protocol", "error_without_text"])
def test_failed_attempt_retains_reported_usage_and_partial_content(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """流协议失败或无正文失败都保留已报用量；传参：替换器和故障；返回：无。"""
    client = from_test_stub("unused")
    adapter = client._adapter_registry.require("scripted_test")

    def stream(_request: object, **_kwargs: object) -> Iterator[ModelStreamEvent]:
        """交付已计费事件后失败；传参：原请求；返回：有真实先后顺序的流。"""
        yield _event("response_start", 0, message_id="partial-message")
        yield _event(
            "usage_update",
            1,
            usage=ModelUsage(input_tokens=reported(17), output_tokens=reported(2)),
        )
        if failure == "protocol":
            yield _event("content_start", 2, block_id="text", content_kind="text")
            yield _event("content_delta", 3, block_id="text", delta="已收到的部分正文")
            yield _event("content_end", 4, block_id="missing-block")
        else:
            yield _event(
                "response_error",
                2,
                error=ProviderError(
                    category="billing",
                    stage="transport",
                    retryable=False,
                    summary="quota exhausted",
                    provider="fixture",
                    model="fixture-model",
                    api_family="fixture",
                ),
            )

    monkeypatch.setattr(adapter, "stream", stream)
    records: list[ModelAttemptEvent] = []
    plan = client.plan(
        "验证失败证据",
        {"model_request_id": "request-usage", "model_attempt_recorder": records.append},
    )
    assert plan.model_error is not None
    assert [row.phase for row in records] == ["started", "finished"]
    assert records[0].attempt_id == records[1].attempt_id
    assert records[1].error is not None
    assert records[1].usage.input_tokens.value == 17
    assert plan.observation.total_tokens == 19
    assert plan.raw_model_response["prompt_tokens"] == 17
    if failure == "protocol":
        assert plan.raw_model_response["text"] == "已收到的部分正文"
    else:
        assert plan.raw_model_response["text"] is None


def test_required_attempt_record_failure_prevents_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """发送前必要记录失败时不派发 Provider；传参：替换器；返回：无。"""
    client = from_test_stub("unused")
    adapter = client._adapter_registry.require("scripted_test")
    dispatched: list[bool] = []

    def stream(_request: object, **_kwargs: object) -> Iterator[ModelStreamEvent]:
        """观察是否触及发送边界；传参：原请求；返回：空事件迭代器。"""
        dispatched.append(True)
        return iter(())

    def fail_record(_attempt: ModelAttemptEvent) -> None:
        """模拟必要持久化失败；传参：尝试；返回：无。"""
        raise OSError("evidence disk failure")

    monkeypatch.setattr(adapter, "stream", stream)
    with pytest.raises(OSError, match="evidence disk failure"):
        client.plan("不能派发", {"model_attempt_recorder": fail_record})
    assert dispatched == []


def test_adapter_failure_before_iterator_finishes_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """适配器尚未返回响应流就失败时仍闭合尝试证据；传参：替换器；返回：无。"""
    client = from_test_stub("unused")
    adapter = client._adapter_registry.require("scripted_test")

    def stream(_request: object, **_kwargs: object) -> Iterator[ModelStreamEvent]:
        """在建立响应流时暴露适配失败；传参：原请求；返回：无，抛出实际错误。"""
        raise ProviderAdapterError("cannot construct provider stream")

    monkeypatch.setattr(adapter, "stream", stream)
    records: list[ModelAttemptEvent] = []
    plan = client.plan(
        "验证派发失败",
        {
            "model_request_id": "request-dispatch",
            "model_attempt_recorder": records.append,
        },
    )
    assert plan.model_error is not None
    assert "cannot construct provider stream" in plan.model_error.summary
    assert [row.phase for row in records] == ["started", "finished"]
    assert records[0].attempt_id == records[1].attempt_id
    assert records[1].error is not None
    assert records[1].usage.input_tokens.value is None
    assert plan.model_attempts == (records[1],)
