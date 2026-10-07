"""守住"请求体必须能被 JSON 编码"这条线。

作者：LKX
时间：2026-09-03 00:00:00

三个 Adapter 都从深冻结的合同对象（ModelToolDefinition.input_schema、
ToolCallPart.arguments、ProviderStateEnvelope.payload）取值。只解冻最外层时，
properties/items/summary 这些嵌套层还是 mappingproxy，SDK 发请求前 json.dumps
整个 body 就会 TypeError，表现是任何带工具的对话在第一轮直接失败。
现有 golden fixture 里的 schema 和参数都是单层的，抓不到这个漏洞。
"""

from __future__ import annotations

import json

import pytest

from llm.messages import (
    AssistantMessage,
    ProviderStateEnvelope,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
)
from llm.model_request import (
    Capability,
    CapabilityRequirement,
    ModelRequest,
    ModelToolDefinition,
)
from llm.providers.anthropic_messages import AnthropicMessagesAdapter
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.providers.openai_responses import OpenAIResponsesAdapter


def _nested_tool_request() -> ModelRequest:
    """构造嵌套 schema 加嵌套工具参数的请求，贴近注册表里真实工具的形状。"""
    tool = ModelToolDefinition(
        "write",
        "Write one path",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "edits": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["path"],
        },
    )
    return ModelRequest(
        instructions=(TextPart("You are Reins."),),
        messages=(
            UserMessage("m1", (TextPart("write a.txt"),)),
            AssistantMessage(
                "m2",
                (
                    ToolCallPart(
                        "c1",
                        "write",
                        {"path": "a.txt", "edits": [{"old": "x", "new": "y"}]},
                    ),
                ),
            ),
            ToolResultMessage("m3", "c1", "write", (TextPart("done"),), "success"),
        ),
        tools=(tool,),
        required_capabilities=frozenset(
            {CapabilityRequirement(Capability.NATIVE_TOOLS)}
        ),
    )


@pytest.mark.parametrize(
    "adapter,model_id,schema_path",
    [
        (OpenAIChatAdapter(), "gpt-fixture", ("function", "parameters")),
        (OpenAIResponsesAdapter(), "gpt-fixture", ("parameters",)),
        (AnthropicMessagesAdapter(), "claude-fixture", ("input_schema",)),
    ],
)
def test_nested_tool_schema_and_arguments_encode_to_json(
    adapter: object, model_id: str, schema_path: tuple[str, ...]
) -> None:
    """带嵌套工具 schema 的请求体必须能整体编码，且嵌套内容不被丢掉。"""
    # 1. build_request 内部就会编码工具参数，参数没深解冻时这一步先炸
    body = adapter.build_request(_nested_tool_request(), model_id=model_id)  # type: ignore[attr-defined]
    # 2. SDK 真正发请求前对整个 body 做 json.dumps，冻结容器漏到这里就是 TypeError
    encoded = json.loads(json.dumps(body))
    schema: object = encoded["tools"][0]
    for key in schema_path:
        assert isinstance(schema, dict)
        schema = schema[key]
    assert isinstance(schema, dict)
    assert schema["properties"]["edits"]["items"]["type"] == "object"
    assert schema["required"] == ["path"]


def test_replayed_reasoning_state_items_encode_to_json() -> None:
    """回灌 reasoning state：summary 里的嵌套对象也要能编码。"""
    state = ProviderStateEnvelope(
        api_family="openai_responses",
        provider="openai",
        model="gpt-fixture",
        state_version=1,
        payload={
            "input_items": [
                {
                    "id": "rs-1",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "considering"}],
                    "encrypted_content": "synthetic-encrypted-state",
                }
            ]
        },
    )
    request = ModelRequest(
        instructions=(TextPart("You are Reins."),),
        messages=(
            UserMessage("m1", (TextPart("hello"),)),
            AssistantMessage("m2", (TextPart("hi"),), provider_state=state),
        ),
        tools=(),
        required_capabilities=frozenset(
            {CapabilityRequirement(Capability.PROVIDER_STATE_ROUND_TRIP)}
        ),
    )
    body = OpenAIResponsesAdapter().build_request(request, model_id="gpt-fixture")
    encoded = json.loads(json.dumps(body))
    replayed = [item for item in encoded["input"] if item.get("type") == "reasoning"]
    assert replayed[0]["summary"] == [{"type": "summary_text", "text": "considering"}]
