from __future__ import annotations

import json

from llm.messages import (
    AssistantMessage,
    ThinkingPart,
    ToolCallPart,
    ToolResultMessage,
    agent_message_to_mapping,
    model_visible_text,
    thaw_json_value,
)
from llm.model_request import ComposedRequest, composed_request_evidence
from llm.provider_result import ProviderCallResult
from llm.types import MeasurementStatus, ModelError, ModelUsage, UsageMeasurement

# 未上报与不适用两种状态都表示"没有这项数字"，不能折成 0
_ABSENT_MEASUREMENT_STATUSES = frozenset(
    {MeasurementStatus.UNKNOWN, MeasurementStatus.NOT_APPLICABLE}
)

# Run Evidence 的响应字段名沿用 Wire 词汇（tool_calls / id / function.arguments）。
# 这些名字是已持久化的证据契约，跨进程消费方按字面取值：
# frontends/observe/panels/response_anatomy.py:29 与
# frontends/tui/data/run_evidence_detail.py:131 都对 response["tool_calls"] 取 len()。
# 改名会让历史证据与前端一起坏，故在退休扫描里按 allowlist 保留。
_NO_RESPONSE_MESSAGE = "provider call did not return a response"


def request_source_evidence(composed: ComposedRequest) -> dict[str, object]:
    """冻结发送材料的真实来源，正文相同部分由存储复用；参数：本次组合；返回：来源映射。"""
    messages: list[dict[str, object]] = []
    for position, message in enumerate(composed.request.messages):
        source: dict[str, object] = {
            "position": position,
            "message_id": message.message_id,
            "kind": message.kind,
        }
        if isinstance(message, ToolResultMessage):
            source.update(
                {
                    "call_id": message.call_id,
                    "tool_name": message.tool_name,
                    "fed_back": agent_message_to_mapping(message),
                }
            )
        messages.append(source)
    # 【模型调用】【材料来源】1. 来源由组合时身份与版本给出，不对今天的文件做文本相似匹配
    return {
        "messages": messages,
        "composition": composed_request_evidence(composed),
        "mapping": "canonical message order; provider adaptation may combine or omit blocks",
    }


def provider_call_evidence(
    result: ProviderCallResult | None,
    error: ModelError | None = None,
) -> dict[str, object]:
    """把一次 Provider 调用结果投影成 raw_model_response 证据。

    作者：LKX
    时间：2026-08-30 18:30:00
    传参：result 为 Provider 调用结果，未拿到终态时为 None；error 为已桥接的 ModelError
    返回：运行证据映射；键名沿用既有 Wire 词汇以保住跨进程消费方

    error 由调用方传入而不是就地从 ProviderError 推：证据里的 category 必须与
    ModelError 同值域，否则前端与 agent_loop 的字面值分支会读到新值域而静默失效。
    """
    # 1. 连终态都没拿到（如流协议异常从 assemble 逃出）时保留旧形状，便于既有消费方识别
    if result is None:
        return _no_result_evidence(error)
    message = result.message or result.partial_message
    usage = (
        result.usage
        if result.usage is not None
        else message.usage
        if message is not None
        else None
    )
    payload: dict[str, object] = {
        "ok": result.message is not None,
        "message_id": message.message_id if message is not None else None,
        "text": _visible_text(message),
        "error_message": result.error.summary if result.error is not None else None,
        "tool_calls": _tool_call_evidence(message),
        "reasoning_content": _reasoning_text(message),
        "stop_reason": message.stop_reason.value
        if message is not None and message.stop_reason is not None
        else None,
        "prompt_tokens": _token_value(usage, "input_tokens"),
        "completion_tokens": _token_value(usage, "output_tokens"),
        "total_tokens": _token_value(usage, "total_tokens"),
    }
    _attach_error_facts(payload, result, error)
    return payload


def _no_result_evidence(error: ModelError | None) -> dict[str, object]:
    """无终态结果时的证据形状：沿用旧的 ok/error_message 两键。"""
    payload: dict[str, object] = {
        "ok": False,
        "error_message": error.summary if error is not None else _NO_RESPONSE_MESSAGE,
    }
    if error is not None:
        payload["error"] = _error_evidence(error, None)
    return payload


def _attach_error_facts(
    payload: dict[str, object],
    result: ProviderCallResult,
    error: ModelError | None,
) -> None:
    """把退避秒数与错误明细挂到证据上；成功调用不写这两项。"""
    provider_error = result.error
    if provider_error is not None and provider_error.retry_after_seconds is not None:
        payload["retry_after"] = provider_error.retry_after_seconds
    if error is not None:
        payload["error"] = _error_evidence(error, result)


def _error_evidence(
    error: ModelError,
    result: ProviderCallResult | None,
) -> dict[str, object]:
    """错误明细：ModelError 五项沿用旧契约，Provider 侧事实按有则补。"""
    payload: dict[str, object] = {
        "category": error.category,
        "summary": error.summary,
        "stage": error.stage,
        "retryable": error.retryable,
        "retry_after": error.retry_after,
    }
    provider_error = result.error if result is not None else None
    if provider_error is None:
        return payload
    payload["provider_stage"] = provider_error.stage
    payload["provider_category"] = provider_error.category
    if provider_error.http_status is not None:
        payload["http_status"] = provider_error.http_status
    if provider_error.request_id:
        payload["request_id"] = provider_error.request_id
    return payload


def _visible_text(message: AssistantMessage | None) -> str | None:
    """模型可见文本；无消息时为 None，与旧 ProviderResponse.text 的空态一致。"""
    if message is None:
        return None
    return model_visible_text(message)


def _reasoning_text(message: AssistantMessage | None) -> str | None:
    """思考文本：拼接全部 ThinkingPart，没有则为 None。"""
    if message is None:
        return None
    blocks = [part.text for part in message.content if isinstance(part, ThinkingPart)]
    return "\n\n".join(blocks) if blocks else None


def _tool_call_evidence(message: AssistantMessage | None) -> list[dict[str, object]]:
    """把 ToolCallPart 投影成证据里的工具调用列表。

    作者：LKX
    时间：2026-08-30 18:30:00
    传参：message 为本轮助手消息，可能为 None
    返回：工具调用证据列表；无调用时为空列表

    形状沿用 Wire 词汇（id/type/function），因为 tests/test_real_llm_runtime.py:101 与两个
    前端面板都按这套键名读历史证据。
    """
    if message is None:
        return []
    return [
        {
            "id": part.call_id,
            "type": "function",
            "function": {
                "name": part.tool_name,
                "arguments": json.dumps(
                    thaw_json_value(part.arguments), ensure_ascii=False
                ),
            },
        }
        for part in message.content
        if isinstance(part, ToolCallPart)
    ]


def _token_value(usage: ModelUsage | None, field_name: str) -> int | None:
    """取一项 token 计量；未知计量返回 None 而不是 0。

    作者：LKX
    时间：2026-08-30 18:30:00
    传参：usage 为本轮用量；field_name 为 ModelUsage 上的计量字段名
    返回：已上报或已推导的数值；UNKNOWN/NOT_APPLICABLE 返回 None

    把未知折成 0 会让"模型没报用量"和"真的用了 0 个 token"变成同一件事，
    下游记账与成本核算都会读到假数。
    """
    if usage is None:
        return None
    measurement = getattr(usage, field_name)
    return _measurement_value(measurement)


def _measurement_value(measurement: UsageMeasurement) -> int | None:
    """未知与不适用的计量投影成 None，其余返回整数值。"""
    if measurement.status in _ABSENT_MEASUREMENT_STATUSES:
        return None
    return measurement.value
