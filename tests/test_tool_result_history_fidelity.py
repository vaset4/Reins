from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.messages import (
    AssistantMessage,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    freeze_json_object,
    model_visible_text,
    validate_message_sequence,
)
from llm.prompt_composer import build_prompt_sections, build_request_messages
from llm.tool_selection import select_tools
from runtime.tool_results import _render_tool_conversation
from runtime.agent_loop import AgentLoop
from runtime.tool_results import MAX_TOOL_OUTPUT_CHARS
from runtime.lease import Lease
from runtime.run_facts import RunFactStore
from runtime.session_messages import (
    ToolExchange,
    append_tool_exchange,
    append_user_message,
)
from runtime.types import RunContext, RunToolsRequest, RunToolsResult, Trigger
from tools.readonly_inspection import DEFAULT_READ_MAX_CHARS
from tools.tool_registry import ToolRegistry

SESSION_ID = "session-tool-history"


def _minimal_sections() -> tuple[object, ...]:
    """造一组最小 prompt 段落，只提供组装消息必需的当前任务文本。"""
    return build_prompt_sections(
        task="next step",
        stage="continue",
        protocol_mode="native_tool_calls",
        model_context={},
        tool_selection=select_tools(ToolRegistry()),
    )


def _paged_meta() -> dict[str, object]:
    return {
        "truncated": True,
        "total_count": 5000,
        "returned_count": 4000,
        "offset": 0,
        "next_offset": 4000,
    }


def test_tool_conversation_envelope_preserves_truncation_meta(
    tmp_path: Path,
) -> None:
    result = RunToolsResult.ok(
        action="file_read",
        tool_name="file_read",
        content="A" * (MAX_TOOL_OUTPUT_CHARS + 500),
        summary="read file",
        meta=_paged_meta(),
    )

    payload = json.loads(_render_tool_conversation(result))

    assert payload["output_complete"] is False
    assert payload["source_truncated"] is True
    assert payload["prompt_truncated"] is True
    assert payload["truncation_layers"] == ["source", "prompt"]
    assert payload["next_offset"] == 4000
    assert payload["total_count"] == 5000
    assert payload["returned_count"] == 4000
    assert "output" in payload and "output_preview" not in payload


def test_full_read_span_reaches_the_model_without_a_hole(tmp_path: Path) -> None:
    """读满一整个窗口的正文要整段进 prompt，中间不许被掐掉一截。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：tmp_path 为 pytest 提供的独占数据根
    返回：无

    两个数各管一层：DEFAULT_READ_MAX_CHARS 决定 file_read 一次读多少，
    MAX_TOOL_OUTPUT_CHARS 决定多少能进 prompt。闸一旦小于窗口，这里就会把正文掐成
    头 + 尾、中间静默丢掉，而 offset / next_offset 契约仍宣称这一段已完整送达——模型
    于是拿着一个中间有洞的文档继续做，自己还以为读全了。这条把两层锁在一起：谁把闸
    调回窗口以下，它就红。
    """
    span = " ".join(f"w{index}" for index in range(DEFAULT_READ_MAX_CHARS))
    content = span[:DEFAULT_READ_MAX_CHARS]
    result = RunToolsResult.ok(
        action="file_read",
        tool_name="file_read",
        content=content,
        summary="read",
        meta=_paged_meta(),
    )

    payload = json.loads(_render_tool_conversation(result))

    assert payload["prompt_truncated"] is False
    assert payload["output"] == content


def test_tool_response_fact_preserves_required_meta(tmp_path: Path) -> None:
    store = RunFactStore(tmp_path)

    store.append_from_trajectory(
        {
            "type": "event",
            "event": "tool:response",
            "session_id": "session-a",
            "run_id": "run-a",
            "tool_call_id": "call-a",
            "tool_name": "file_read",
            "status": "ok",
            "summary": "read file",
            "output": "A" * 8500,
            "meta": _paged_meta(),
        }
    )

    fact = store.read_run("run-a")[0]
    tool = fact["tool"]

    assert tool["meta"]["next_offset"] == 4000
    assert tool["meta"]["total_count"] == 5000
    assert tool["output_complete"] is False
    assert tool["source_truncated"] is True


def test_agent_loop_tool_response_fact_preserves_prompt_truncation(
    tmp_path: Path,
) -> None:
    loop = AgentLoop(tmp_path)
    context = RunContext(
        trigger=Trigger.USER,
        payload={},
        capability_lease=Lease(),
        session_id="session-a",
        run_id="run-a",
        segment_id="segment-a",
    )
    request = RunToolsRequest(
        action="file_read",
        tool_name="file_read",
        arguments={"path": "large.txt"},
        call_id="call-a",
    )
    from runtime.tool_executor import ToolBatchExecutor

    executor = ToolBatchExecutor(
        tmp_path,
        registry=loop.tool_registry,
        authorizer=loop.authorizer,
        cancellation=loop.cancellation,
        extension_execution=loop.extension_execution,
        policy=loop.tool_policy,
        operations=loop.operations,
        facts=loop.run_facts,
        evidence=loop.run_evidence,
        states=loop.session_states,
        progress=loop.progress,
        history=loop.tool_history,
        client=None,
        collaboration=None,
    )
    call = executor.build_call(request, context=context, request_id="request-a")
    result = RunToolsResult.ok(
        action="file_read",
        tool_name="file_read",
        content="A" * (MAX_TOOL_OUTPUT_CHARS + 500),
        summary="read file",
        meta=_paged_meta(),
    )

    executor._append_tool_response_fact(context, call, result)

    fact = RunFactStore(tmp_path).read_run("run-a")[0]
    tool = fact["tool"]
    assert tool["prompt_truncated"] is True
    assert tool["source_truncated"] is True
    assert tool["output_complete"] is False
    assert tool["truncation_layers"] == ["source", "prompt"]


def test_conversation_tail_truncation_is_visible_to_model_history(
    tmp_path: Path,
) -> None:
    for index in range(25):
        append_user_message(tmp_path, SESSION_ID, f"message {index}")

    selection = AgentLoop(tmp_path)._read_conversation_history(SESSION_ID, limit=20)

    # 截断事实随 HistorySelection 单独携带，由 prompt 组装渲染进 instructions
    assert selection.truncated is True
    assert len(selection.messages) == 20
    assert selection.retained_count == 20
    assert model_visible_text(selection.messages[0]) == "message 5"

    sections = build_prompt_sections(
        task="next",
        stage="continue",
        protocol_mode="native_tool_calls",
        model_context={
            "conversation_history": selection.messages,
            "history_truncated_retained": selection.retained_count,
        },
        tool_selection=select_tools(ToolRegistry()),
    )
    notice = next(
        item.content for item in sections if item.name == "history_truncation_notice"
    )
    assert "conversation_tail_truncated" in notice
    assert "earlier_messages_omitted=true" in notice
    assert "retained_count=20" in notice


# --- Stage 2/3: native tool-dialog history ---


def test_native_history_replays_tool_role_with_tool_call_id(
    tmp_path: Path,
) -> None:
    """工具结果必须保留 call_id 关联，不被降级成 assistant 文本（F4 修复）。"""
    append_tool_exchange(
        tmp_path,
        SESSION_ID,
        ToolExchange(
            call_id="call-1",
            tool_name="list",
            args={"path": "."},
            rendered="[tool_result] ...",
            status="ok",
        ),
    )

    messages = AgentLoop(tmp_path)._read_conversation_history(SESSION_ID).messages

    assert [item.kind for item in messages] == ["assistant", "tool_result"]
    calls = [part for part in messages[0].content if isinstance(part, ToolCallPart)]
    assert [part.call_id for part in calls] == ["call-1"]
    assert messages[1].call_id == "call-1"


def test_request_messages_preserve_tool_structured_fields() -> None:
    """组装进请求的历史必须保住工具调用的结构化关联，不能塌成纯文本。"""
    history = (
        AssistantMessage(
            "a1",
            (ToolCallPart("c1", "grep", freeze_json_object({}, path="args")),),
        ),
        ToolResultMessage("t1", "c1", "grep", (TextPart("matched lines"),), "success"),
    )

    messages = build_request_messages(
        model_context={"conversation_history": history},
        sections=_minimal_sections(),
    )

    assert [item.kind for item in messages[1:]] == ["assistant", "tool_result"]
    calls = [part for part in messages[1].content if isinstance(part, ToolCallPart)]
    assert [part.call_id for part in calls] == ["c1"]
    assert messages[2].call_id == "c1"


def test_orphan_tool_message_is_rejected_not_fabricated() -> None:
    """孤立工具结果必须显式报错：补造一条假公告等于把没发生过的调用喂给模型。"""
    from llm.messages import MessageContractError

    orphan = (ToolResultMessage("t1", "orphan-1", "grep", (TextPart("r"),), "success"),)
    messages = build_request_messages(
        model_context={"conversation_history": orphan},
        sections=_minimal_sections(),
    )

    # 组装只把消息排好序，不就地补造公告
    assert [item.kind for item in messages] == ["user", "tool_result"]

    # 断开的调用图在构造 ModelRequest 时被拒，而不是被静默补上
    with pytest.raises(MessageContractError) as caught:
        validate_message_sequence(messages)
    assert caught.value.code == "orphan_tool_result"


def test_native_history_end_to_end_flows_to_request_messages(
    tmp_path: Path,
) -> None:
    """工具调用配对要能走完 SessionMessageStore -> 历史读取 -> 请求组装整条链。"""
    append_tool_exchange(
        tmp_path,
        SESSION_ID,
        ToolExchange(
            call_id="call-x",
            tool_name="list",
            args={"path": "."},
            rendered="[tool_result] ok",
            status="ok",
        ),
    )

    selection = AgentLoop(tmp_path)._read_conversation_history(SESSION_ID)
    messages = build_request_messages(
        model_context={"conversation_history": selection.messages},
        sections=_minimal_sections(),
    )

    assert [item.kind for item in messages[1:]] == ["assistant", "tool_result"]
    calls = [part for part in messages[1].content if isinstance(part, ToolCallPart)]
    assert [part.call_id for part in calls] == ["call-x"]
    assert messages[2].call_id == "call-x"
    # 组装出的序列本身是合法调用图
    validate_message_sequence(messages)


# --- Stage 4: truncated output shows a readable continuation hint ---


def test_truncated_output_carries_continue_hint(tmp_path: Path) -> None:
    """Stage 4: a truncated tool output embeds a human-readable continuation
    marker so the model can resume via offset without parsing the envelope."""

    result = RunToolsResult.ok(
        action="file_read",
        tool_name="file_read",
        content="A" * (MAX_TOOL_OUTPUT_CHARS + 500),
        summary="read file",
        meta={"truncated": True, "next_offset": 4000, "total_count": 5000},
    )

    payload = json.loads(_render_tool_conversation(result))

    assert "use offset=4000 to continue" in payload["output"]
    assert "of 5000" in payload["output"]
