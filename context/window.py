"""完整模型请求共用的窗口预算，不删除原始会话内容。

作者：xxx
时间：2026-09-14 20:00:00
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from context.token_estimate import (
    MESSAGE_OVERHEAD_TOKENS,
    estimate_agent_messages_tokens,
    estimate_tokens,
)
from llm.messages import thaw_json_value

if TYPE_CHECKING:
    from llm.model_request import ModelRequest

DEFAULT_OUTPUT_TOKENS = 4096
OUTPUT_WINDOW_DIVISOR = 4
REQUEST_OVERHEAD_TOKENS = 12


@dataclass(frozen=True, slots=True)
class RequestBudget:
    """记录同一窗口的估算与输出硬额度；字段分别为窗口、指令、消息、工具、协议及输出token数。"""

    context_window: int
    instructions: int
    messages: int
    tools: int
    protocol: int
    output_reserved: int
    observations: int = 0

    @property
    def total(self) -> int:
        """求本次完整输入的估算；传参：无；返回：输入token数。"""
        return (
            self.instructions
            + self.messages
            + self.tools
            + self.protocol
            + self.observations
        )

    @property
    def required_total(self) -> int:
        """求输入与输出共用的窗口需求；传参：无；返回：token数。"""
        return self.total + self.output_reserved

    def evidence(self) -> dict[str, int]:
        """导出请求证据；传参：无；返回：各组成部分及合计，均为估算。"""
        return {
            "context_window": self.context_window,
            "instructions": self.instructions,
            "messages": self.messages,
            "tools": self.tools,
            "protocol": self.protocol,
            "observations": self.observations,
            "output_reserved": self.output_reserved,
            "total": self.total,
            "required_total": self.required_total,
        }


class ContextWindowExceeded(ValueError):
    """请求尚未发送时暴露实际窗口不足，供上下文管理生成摘要或返回明确错误。"""

    def __init__(self, budget: RequestBudget) -> None:
        """保存不足的窗口证据；传参：完整预算；返回：无。"""
        self.budget = budget
        super().__init__(
            f"context window exceeded: estimated input {budget.total} + output {budget.output_reserved} > window {budget.context_window}"
        )


def output_reserve(context_window: int) -> int:
    """为生成保留且实际限制输出额度；传参：模型窗口；返回：输出token上限。"""
    if type(context_window) is not int or context_window <= 0:
        raise ValueError("context window must be a positive integer")
    return min(DEFAULT_OUTPUT_TOKENS, max(1, context_window // OUTPUT_WINDOW_DIVISOR))


def request_budget(request: ModelRequest, context_window: int) -> RequestBudget:
    """按完整请求估算窗口占用；传参：请求及模型窗口；返回：同源预算。"""
    from llm.model_request import runtime_observation_text

    instruction_text = "\n".join(part.text for part in request.instructions)
    tools = [
        {
            "name": item.name,
            "description": item.description,
            "input_schema": thaw_json_value(item.input_schema),
        }
        for item in request.tools
    ]
    return RequestBudget(
        context_window=context_window,
        instructions=estimate_tokens(instruction_text)
        + (MESSAGE_OVERHEAD_TOKENS if instruction_text else 0),
        messages=estimate_agent_messages_tokens(request.messages),
        tools=estimate_tokens(json.dumps(tools, ensure_ascii=False)) if tools else 0,
        protocol=REQUEST_OVERHEAD_TOKENS,
        output_reserved=request.max_output_tokens or 0,
        observations=(
            estimate_tokens(runtime_observation_text(request.observations))
            + MESSAGE_OVERHEAD_TOKENS
            if request.observations
            else 0
        ),
    )


def require_request_fits(request: ModelRequest, context_window: int) -> None:
    """在网络派发前检查完整窗口；传参：请求和窗口；返回：无，超出时抛出带证据的错误。"""
    budget = request_budget(request, context_window)
    if budget.required_total > context_window:
        raise ContextWindowExceeded(budget)
