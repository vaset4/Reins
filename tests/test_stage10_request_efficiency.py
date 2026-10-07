"""【模型请求】【稳定前缀】通过三个真实适配器核对临时观察与消息/指令边界。

作者：xxx
时间：2026-10-02 11:42:00
"""

import json

import pytest

from llm.model_request import model_request_from_mapping, model_request_to_mapping
from llm.providers.anthropic_messages import AnthropicMessagesAdapter
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.providers.openai_responses import OpenAIResponsesAdapter
from tests.test_stage9_requests import setup_request


@pytest.mark.parametrize(
    "adapter",
    [OpenAIChatAdapter(), OpenAIResponsesAdapter(), AnthropicMessagesAdapter()],
)
def test_step_cost_and_error_changes_follow_the_same_actual_history(tmp_path, adapter):
    """仅油表和临时错误变化不再打断稳定历史，观察不成为用户身份；参数：隔离根/协议；返回：无。"""
    _client, views, context, _actions = setup_request(tmp_path)
    first = views.prepare(
        "继续", {**context, "budget_evidence": {"steps_used": 1, "steps_limit": 20}}
    )
    second = views.prepare(
        "继续",
        {
            **context,
            "budget_evidence": {"steps_used": 2, "steps_limit": 20},
            "recoverable_error_notice": "刚才的网络连接失败",
        },
    )
    assert first.request.instructions == second.request.instructions
    assert (
        first.request.messages
        == second.request.messages
        == context["conversation_history"]
    )
    assert first.request.observations != second.request.observations
    left = adapter.build_request(first.request, model_id="fixture")
    right = adapter.build_request(second.request, model_id="fixture")
    key = "input" if adapter.api_family == "openai_responses" else "messages"
    assert left[key][:-1] == right[key][:-1]
    if "instructions" in left:
        assert left["instructions"] == right["instructions"]
    if "system" in left:
        assert left["system"] == right["system"]
    assert "not user input or authorization" in json.dumps(
        right[key][-1], ensure_ascii=False
    )
    assert "runtime_budget" not in "\n".join(
        part.text for part in first.request.instructions
    )
    assert "runtime_budget" in first.render_text_to_model
    assert first.token_estimate["observations"] > 0
    assert (
        model_request_from_mapping(model_request_to_mapping(second.request))
        == second.request
    )
