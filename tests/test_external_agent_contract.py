"""验证真实外部协议的回填、身份和用量语义；网络互通证据另由真实案例提供。

作者：xxx
时间：2026-09-14 17:15:00
"""

from __future__ import annotations

import pytest

from llm.messages import AssistantMessage, TextPart, ToolCallPart, ToolResultMessage
from runtime.cancellation import CancellationToken
from runtime.external_agent import ClaudeAgentClient, _claude_usage
from runtime.session_message_store import SessionMessageStore
from tools.builtin_tools import build_tool_registry
from tests.test_collaboration_boundaries import make_host, complete


def recorded_client(root, *, failed=False):
    """创建持久配对结果及其外部会话视图；传参：目录/原结果状态；返回：客户端和来源文件。"""
    host, context = make_host(root, lambda execution: complete(root, execution))
    messages = SessionMessageStore(root)
    call = ToolCallPart("external-call", "file_read", {"path": "artifact.txt"})
    messages.append_message(
        context.session_id, AssistantMessage("request", (call,)), run_id=context.run_id
    )
    messages.append_message(
        context.session_id,
        ToolResultMessage(
            "result",
            call.call_id,
            call.tool_name,
            (TextPart("original persisted artifact result"),),
            "error" if failed else "success",
            error="original source unavailable" if failed else None,
        ),
        run_id=context.run_id,
    )
    client = ClaudeAgentClient(
        context,
        data_root=root,
        member={"agent_id": "external", "task": "inspect"},
        collaboration=host,
        cancellation=CancellationToken(),
    )
    client._registry = build_tool_registry(repo_root=root, data_root=root)
    client._tools = {"file_read": {}}
    return client, messages


def rpc_call(**overrides):
    """构造CLI的实际MCP身份关联格式；传参：要替换的字段；返回：协议请求。"""
    return {
        "method": "tools/call",
        "id": 7,
        "params": {
            "name": "file_read",
            "arguments": {"path": "artifact.txt"},
            "_meta": {"claudecode/toolUseId": "external-call"},
            **overrides,
        },
    }


@pytest.mark.parametrize("failed", [False, True])
def test_external_redelivery_returns_original_result_without_reexecution(
    tmp_path, failed
):
    """断线重投读取原持久结果，不回到文件执行也不把业务失败改成功；传参：目录/结果；返回：无。"""
    client, messages = recorded_client(tmp_path, failed=failed)
    (tmp_path / "artifact.txt").write_text(
        "content changed after original execution", encoding="utf-8"
    )
    before = messages.read_entries(client.context.session_id)
    first, duplicate = client._mcp(rpc_call()), client._mcp(rpc_call())
    assert first == duplicate
    assert first["content"][0]["text"] == "original persisted artifact result"
    assert first["isError"] is failed
    assert messages.read_entries(client.context.session_id) == before
    assert (
        tmp_path / "artifact.txt"
    ).read_text() == "content changed after original execution"


@pytest.mark.parametrize(
    "overrides",
    [
        {"name": "file_write"},
        {"arguments": {"path": "other.txt"}},
        {"_meta": {"claudecode/toolUseId": "unknown"}},
    ],
)
def test_external_call_cannot_change_its_persisted_identity(tmp_path, overrides):
    """工具名、参数或身份改变都不能冒领旧结果；传参：目录/变更；返回：无。"""
    client, messages = recorded_client(tmp_path)
    before = messages.read_entries(client.context.session_id)
    with pytest.raises(ValueError, match="identity or arguments"):
        client._mcp(rpc_call(**overrides))
    assert messages.read_entries(client.context.session_id) == before


def test_replayed_external_decision_does_not_create_another_execution(tmp_path):
    """恢复时重复的已闭合调用不再次进入执行器，新调用保留原ID；传参：目录；返回：无。"""
    client, _ = recorded_client(tmp_path)
    old = {
        "type": "tool_use",
        "id": "external-call",
        "name": "mcp__reins__file_read",
        "input": {"path": "artifact.txt"},
    }
    assert client._parts_plan([old]).run_tools_request is None
    new = {**old, "id": "external-new", "input": {"path": "next.txt"}}
    plan = client._parts_plan([old, new])
    assert plan.run_tools_request.call_id == "external-new"
    with pytest.raises(ValueError, match="different arguments"):
        client._parts_plan([{**old, "input": {"path": "forged.txt"}}])


def test_sdk_placeholder_zeros_do_not_erase_observed_usage(tmp_path):
    """流包装层的零不是消费更新，缓存输入与已知输出保持可追溯；传参：目录；返回：无。"""
    client, _ = recorded_client(tmp_path)
    client._stream_event(
        {
            "type": "message_start",
            "message": {
                "model": "reported-model",
                "usage": {
                    "input_tokens": 3,
                    "cache_read_input_tokens": 2,
                    "cache_creation_input_tokens": 1,
                    "output_tokens": 0,
                },
            },
        }
    )
    assert _claude_usage(client._usage).output_tokens.value is None
    client._stream_event({"type": "message_delta", "usage": {"output_tokens": 5}})
    client._stream_event(
        {"type": "message_stop", "usage": {"input_tokens": 0, "output_tokens": 0}}
    )
    usage = _claude_usage(client._usage)
    assert (
        usage.input_tokens.value == 6
        and usage.output_tokens.value == 5
        and usage.total_tokens.value == 11
    )
    assert client.model == "reported-model"
    assert _claude_usage({}).total_tokens.value is None
