"""验证无正文的思考与工具调用在真实消息边界正常保存。

作者：xxx；时间：2026-09-29 22:00:00
"""

from dataclasses import replace

import pytest

from llm.messages import (
    AssistantMessage,
    MessageContractError,
    TextPart,
    ThinkingPart,
    ToolResultMessage,
    model_visible_text,
    validate_message_sequence,
)
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.model_request import ModelPreference, PreferenceKind
from llm.resolved_target import resolve_model_target
from llm.client import RealLLMClient
from llm.config import LLMProviderConfig
from runtime.agent_loop import State
from runtime.session_messages import append_assistant_message, materialize_messages
from tests.test_tool_batch_execution import make_run
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk


@pytest.mark.parametrize("body", ["", " \n", "先读取文件"])
@pytest.mark.parametrize("effort", [None, "high"])
def test_reasoning_tool_response_executes_and_survives_reload(
    tmp_path, monkeypatch, body, effort
):
    """思考加工具响应经真实适配器、循环和存储后继续回答；参数：测试目录/替换器/正文；返回：无。"""

    def read_document(args):
        """实际读取临时项目说明；参数：工具参数；返回：文件原文。"""
        return (tmp_path / args["path"]).read_text(encoding="utf-8")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "read_document",
            "读项目说明",
            {"path": {"type": "string", "required": True}},
            "agent",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=read_document,
        )
    )
    loop, context, _ = make_run(tmp_path, registry, [])
    (tmp_path / "README.md").write_text("Reins 项目说明", encoding="utf-8")
    target = resolve_model_target(
        cli_overrides={
            "model": "deepseek-v4.1-flash",
            "base_url": "http://localhost/v1",
            "reasoning_effort": effort,
            "api_key": "test-only",
        }
    )
    client = RealLLMClient(
        config=LLMProviderConfig(target.base_url, target.model, target.api_key, 30),
        resolved_target=target,
    )
    adapter = client._adapter_registry.require("openai_chat")
    requests = []

    def stream(request, *, model, **kwargs):
        """通过生产 Chat 适配器翻译推理与工具 SSE；参数：真实请求/模型/连接；返回：统一事件。"""
        requests.append(request)
        delta = {"content": "我是 Reins"}
        finish = "stop"
        if len(requests) == 1:
            delta = {
                "content": body,
                "reasoning_content": "先读取项目说明",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "read-doc",
                        "type": "function",
                        "function": {
                            "name": "read_document",
                            "arguments": '{"path":"README.md"}',
                        },
                    }
                ],
            }
            finish = "tool_calls"
        chunks = [
            {
                "id": f"chat-{len(requests)}",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": f"chat-{len(requests)}",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            },
        ]
        for event in OpenAIChatAdapter().translate_stream(
            chunks, provider=model.provider, model=model.model_id
        ):
            yield replace(event, api_family=adapter.api_family)

    monkeypatch.setattr(adapter, "stream", stream)
    loop.llm_client = client
    list(loop.run_stream(context))
    messages = materialize_messages(tmp_path, context.session_id)
    validate_message_sequence(messages)
    assert loop.state == State.DONE
    assert len(requests) == 2
    thought = next(
        message
        for message in messages
        if isinstance(message, AssistantMessage)
        and any(isinstance(part, ThinkingPart) for part in message.content)
    )
    assert thought.content == (
        (ThinkingPart("先读取项目说明", "visible"), TextPart(body))
        if body.strip()
        else (ThinkingPart("先读取项目说明", "visible"),)
    )
    assert sum(isinstance(message, ToolResultMessage) for message in messages) == 1
    result = next(
        message
        for message in requests[1].messages
        if isinstance(message, ToolResultMessage)
    )
    assert "Reins 项目说明" in model_visible_text(result)
    wire = OpenAIChatAdapter().build_request(
        replace(requests[1], optional_preferences=()), model_id="fixture"
    )
    assert all(row.get("content") or row.get("tool_calls") for row in wire["messages"])
    if effort:
        assert requests[1].optional_preferences == (
            ModelPreference(PreferenceKind.REASONING_LEVEL, effort),
        )
        wire = OpenAIChatAdapter().build_request(
            requests[1], model_id="deepseek-v4.1-flash"
        )
        tool_row = next(row for row in wire["messages"] if row.get("tool_calls"))
        assert tool_row["reasoning_content"] == "先读取项目说明"
        assert tool_row["content"] == (body if body.strip() else "")
        assert sum("reasoning_content" in row for row in wire["messages"]) == 1
    assert model_visible_text(messages[-1]) == "我是 Reins"


@pytest.mark.parametrize("body,reasoning", [("", ""), (" \n", "\t")])
def test_empty_assistant_message_still_rejected(tmp_path, body, reasoning):
    """完全无内容的回答继续拒绝保存；参数：目录/空白正文/空白思考；返回：无。"""
    with pytest.raises(MessageContractError):
        append_assistant_message(tmp_path, "empty-session", body, reasoning=reasoning)
    assert materialize_messages(tmp_path, "empty-session") == ()
