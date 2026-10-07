from __future__ import annotations

from llm.messages import (
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    AssistantMessage,
)
from llm.model_request import (
    Capability,
    CapabilityRequirement,
    ModelRequest,
    ModelToolDefinition,
)


def text_request(*, stream: bool = True) -> ModelRequest:
    """构造 Provider Adapter 测试共用的纯文本请求。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：stream 表示是否要求流式能力
    返回：只包含规范消息合同的 ModelRequest
    """
    required = {CapabilityRequirement(Capability.STREAMING)} if stream else set()
    return ModelRequest(
        instructions=(TextPart("You are Reins."),),
        messages=(UserMessage("m1", (TextPart("hello"),)),),
        tools=(),
        required_capabilities=frozenset(required),
        stream=stream,
    )


def tool_roundtrip_request() -> ModelRequest:
    """构造含完整工具调用图的 Provider Adapter 测试请求。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：无
    返回：含工具定义、调用和结果的 ModelRequest
    """
    tool = ModelToolDefinition("read", "Read one path", {"type": "object"})
    return ModelRequest(
        instructions=(TextPart("You are Reins."),),
        messages=(
            UserMessage("m1", (TextPart("read a.txt"),)),
            AssistantMessage("m2", (ToolCallPart("c1", "read", {"path": "a.txt"}),)),
            ToolResultMessage("m3", "c1", "read", (TextPart("content"),), "success"),
        ),
        tools=(tool,),
        required_capabilities=frozenset(
            {CapabilityRequirement(Capability.NATIVE_TOOLS)}
        ),
    )
