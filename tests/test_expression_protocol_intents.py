"""专用意图协议退休后，原生动作共用工具入口。

作者：xxx
时间：2026-09-14 15:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final

import json

import pytest

from llm.messages import ToolCallPart, ToolResultMessage
from llm.parser import parse_llm_response, parse_tool_call_parts
from runtime.session_messages import materialize_messages
from runtime.types import Trigger
from tests.test_native_actions import native_registry
from tests.test_session_runtime import capture_requests
from tests.test_tool_batch_execution import make_run


@pytest.mark.parametrize(
    "kind", ["clarify", "goal_op", "request_context", "resume_choice", "unknown"]
)
def test_retired_special_intents_are_explicit_protocol_errors(kind):
    """不保留同一动作的第二个隐藏协议入口；传参：旧类型；返回：无。"""
    registry = native_registry()
    plan = parse_llm_response(
        json.dumps({"type": kind}),
        allowed_tool_names=set(registry.list_tool_names()),
        registry=registry,
    )
    assert plan.model_error is not None
    assert plan.run_tools_request is None


@pytest.mark.parametrize(
    "name,args",
    [
        ("ask_user", {"question": "哪一天？"}),
        ("goal", {"action": "new", "goal_body": "核对材料"}),
        ("read_history", {"limit": 2}),
        ("operation_status", {}),
        ("resume_operation", {"action": "skip", "operation_id": "op-known"}),
    ],
)
def test_native_actions_use_shared_schema_in_both_protocols(name, args):
    """文本协议与原生调用产生同样的工具请求；传参：动作名/参数；返回：无。"""
    registry = native_registry()
    allowed = set(registry.list_tool_names())
    text = parse_llm_response(
        json.dumps({"type": "run_tools", "tool": name, "arguments": args}),
        allowed_tool_names=allowed,
        registry=registry,
    )
    native = parse_tool_call_parts(
        (ToolCallPart("native-call", name, args),),
        allowed_tool_names=allowed,
        registry=registry,
    )
    for plan in (text, native):
        assert plan.model_error is None
        assert plan.run_tools_request.tool_name == name
        assert plan.run_tools_request.arguments == args


def test_legacy_pending_operation_allows_query_without_replaying(tmp_path, monkeypatch):
    """旧恢复证据不会封锁查询，也不会按旧幂等标签自动执行；传参：目录/替换器；返回：无。"""
    registry = native_registry()
    loop, context, _client = make_run(tmp_path, registry, [])
    context.trigger = Trigger.RESUME
    context.payload["pending_tool_call"] = {
        "tool_name": "file_write",
        "args": {"path": "old.txt", "content": "must not replay"},
        "call_id": "legacy",
    }
    context.payload["resume_action"] = "replay"
    client = from_test_native_tool_then_final(
        [ToolCallPart("inspect-pending", "operation_status", {})], "先核实原操作"
    )
    loop.llm_client = client
    requests = capture_requests(client, monkeypatch)
    list(loop.run_stream(context))
    results = [
        message
        for message in materialize_messages(tmp_path, context.session_id)
        if isinstance(message, ToolResultMessage)
    ]
    assert len(results) == 1
    assert results[0].tool_name == "operation_status"
    assert results[0].status == "success"
    assert "legacy_pending" in str(requests[-1])
    assert not (tmp_path / "old.txt").exists()
