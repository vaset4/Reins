from __future__ import annotations

import pytest

from runtime.persistence import RuntimeStore
from llm.context_baseline import adopt_request_context

from llm.messages import (
    AssistantMessage,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    model_visible_text,
)
from llm.model_request import (
    ComposedRequest,
    compose_model_request,
    composed_request_evidence,
)
from tools.tool_registry import (
    IDEMPOTENT_YES,
    TARGET_SCOPE_PATH,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_FILE,
    ToolDefinition,
    ToolRegistry,
)


def _instruction_text(bundle: ComposedRequest) -> str:
    """把本轮 instructions 拼成一段文本，供断言检查系统指令内容。"""
    return "\n".join(part.text for part in bundle.request.instructions)


def test_native_request_uses_tools_schema_without_text_parameter_catalog() -> None:
    registry = ToolRegistry()
    registry.register(_file_read_tool("file_read"))

    bundle = compose_model_request(
        task="inspect tools",
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context={"artifact_output_dir": ".reins/workspace/task/outputs"},
        registry=registry,
        context_window=30000,
    )

    system_prompt = _instruction_text(bundle)
    assert "native tool schemas" in system_prompt
    assert "ask_user tool call" not in system_prompt
    assert "Allowed model actions this turn:" in system_prompt
    assert "Read efficiently" in system_prompt
    assert "write it directly with file_write" in system_prompt
    assert "Tool catalog:" not in system_prompt
    assert "parameters=" not in system_prompt
    assert bundle.request.tools[0].name == "file_read"
    assert bundle.request.tools[0].input_schema["properties"]["path"]
    evidence = composed_request_evidence(bundle)
    assert "parameters" not in evidence["tool_selection"]["selected"][0]


def test_text_json_request_catalog_comes_only_from_selected_tools() -> None:
    registry = ToolRegistry()
    registry.register(_file_read_tool("file_read"))
    registry.register(_file_read_tool("hidden_read", model_visible=False))

    bundle = compose_model_request(
        task="inspect tools",
        stage="plan",
        protocol_mode="text_json",
        model_context={},
        registry=registry,
        context_window=30000,
    )

    system_prompt = _instruction_text(bundle)
    assert "Tool catalog:" in system_prompt
    assert "file_read:" in system_prompt
    assert "parameters=path<string required>" in system_prompt
    assert "hidden_read:" not in system_prompt
    assert bundle.request.tools == ()


def test_text_json_without_tools_keeps_non_tool_actions() -> None:
    bundle = compose_model_request(
        task="answer directly",
        stage="plan",
        protocol_mode="text_json",
        model_context={},
        registry=ToolRegistry(),
        context_window=30000,
    )

    system_prompt = _instruction_text(bundle)
    assert "No tools are available as a model action this turn" in system_prompt
    assert "Return strict JSON only" in system_prompt
    assert '{"type":"final","content":"..."}' in system_prompt
    assert '{"type":"clarify"' not in system_prompt
    assert '{"type":"run_tools"' not in system_prompt
    assert bundle.request.tools == ()


def test_allowed_actions_limits_composed_request_tools() -> None:
    registry = ToolRegistry()
    registry.register(_file_read_tool("file_read"))
    registry.register(_file_read_tool("file_write"))

    bundle = compose_model_request(
        task="inspect one file",
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context={"allowed_actions": ["file_read"]},
        registry=registry,
        context_window=30000,
    )

    assert [item.name for item in bundle.request.tools] == ["file_read"]
    assert bundle.tool_selection.allowed_tool_names == frozenset({"file_read"})
    assert {item.name: item.reason for item in bundle.tool_selection.excluded} == {
        "file_write": "action_not_allowed",
    }


def test_composed_request_preserves_history_and_runtime_directive_order() -> None:
    registry = ToolRegistry()
    registry.register(_file_read_tool("file_read"))

    bundle = compose_model_request(
        task="next step",
        stage="continue",
        protocol_mode="native_tool_calls",
        model_context={
            "system_reminder": "[system_reminder]decide[/system_reminder]",
            "conversation_history": (
                UserMessage("u1", (TextPart("first"),)),
                AssistantMessage("a1", (TextPart("answer"),)),
            ),
        },
        registry=registry,
        context_window=30000,
    )

    # 独立调用的任务先于历史，历史本身保持原有顺序
    assert [message.kind for message in bundle.messages] == [
        "user",
        "user",
        "assistant",
    ]
    assert model_visible_text(bundle.messages[0]).startswith("Task: next step")
    assert model_visible_text(bundle.messages[1]) == "first"
    assert model_visible_text(bundle.messages[2]) == "answer"
    # 运行时提示排在系统指令之后，仍在 instructions 里保持顺序
    instructions = [part.text for part in bundle.request.instructions]
    assert len(instructions) == 2
    assert instructions[1].startswith("[system_reminder]")


def test_runtime_feedback_reaches_request_in_ephemeral_observations() -> None:
    """协议错误和无进展观察进入实际请求，但不伪造会话消息。"""
    bundle = compose_model_request(
        task="continue after tool",
        stage="continue",
        protocol_mode="native_tool_calls",
        model_context={
            "recoverable_error_notice": "invalid_model_protocol: arguments",
            "no_progress_observation": "NO_PROGRESS_OBSERVED: change strategy",
            "conversation_history": (
                UserMessage("u1", (TextPart("inspect"),)),
                AssistantMessage("a1", (TextPart("done"),)),
            ),
        },
        registry=ToolRegistry(),
        context_window=30000,
    )

    observations = "\n".join(part.text for part in bundle.request.observations)
    assert "[recoverable_error_notice]" in observations
    assert "invalid_model_protocol: arguments" in observations
    assert "[no_progress_observation]" in observations
    assert "NO_PROGRESS_OBSERVED: change strategy" in observations
    assert all(
        "[recoverable_error_notice]" not in part.text
        for part in bundle.request.instructions
    )
    assert [message.kind for message in bundle.messages] == [
        "user",
        "user",
        "assistant",
    ]
    assert {section.name for section in bundle.prompt_sections} >= {
        "recoverable_error_notice",
        "no_progress_observation",
    }
    for section in bundle.prompt_sections:
        if section.name in {"recoverable_error_notice", "no_progress_observation"}:
            assert section.layer == "ephemeral"


def test_request_budget_preserves_unpublished_tool_content() -> None:
    """请求准备只记录窗口需求，不丢弃未总结的原文；传参：无；返回：无。"""
    bulky = " ".join(f"word{index}" for index in range(1000))

    bundle = compose_model_request(
        task="continue after tool",
        stage="continue",
        protocol_mode="native_tool_calls",
        model_context={
            "conversation_history": (
                AssistantMessage(
                    "a1",
                    (ToolCallPart("call-1", "file_read", {"path": "big.txt"}),),
                ),
                ToolResultMessage(
                    "t1", "call-1", "file_read", (TextPart(bulky),), "success"
                ),
            ),
        },
        registry=ToolRegistry(),
        context_window=200,
    )

    assert bundle.trim_delta is None
    assert bundle.token_estimate["required_total"] > bundle.context_window
    original = bundle.messages[2]
    assert model_visible_text(original) == bulky
    assert original.call_id == "call-1"


def test_corrupt_prompt_snapshot_fails_without_rebuilding_facts(tmp_path) -> None:
    """提示元数据损坏时明确失败，不能抹掉原记录；参数：临时根；返回：无。"""
    options = dict(
        task="inspect tools",
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context={"data_root": str(tmp_path), "session_id": "session-1"},
        registry=ToolRegistry(),
        context_window=30000,
    )
    adopt_request_context(compose_model_request(**options), request_id="request-test")
    database = RuntimeStore(tmp_path)
    path = database.source_path("prompt_snapshot", "session-1")
    path.write_bytes(b"{not-json")
    with pytest.raises(ValueError):
        compose_model_request(**options)
    assert path.read_bytes() == b"{not-json"


def test_prompt_snapshot_rebuilds_when_tool_strategy_changes(tmp_path) -> None:
    registry = ToolRegistry()
    registry.register(_file_read_tool("file_read"))
    first = compose_model_request(
        task="inspect tools",
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context={"data_root": str(tmp_path), "session_id": "session-1"},
        registry=registry,
        context_window=30000,
    )
    adopt_request_context(first, request_id="request-first")
    registry.register(_file_read_tool("file_write"))

    second = compose_model_request(
        task="inspect tools",
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context={"data_root": str(tmp_path), "session_id": "session-1"},
        registry=registry,
        context_window=30000,
    )
    adopt_request_context(second, request_id="request-second")
    database = RuntimeStore(tmp_path)
    with database.snapshot() as source:
        snapshot = source.get("prompt_snapshot", "session-1")
    assert snapshot is not None

    assert first.stable_prompt_snapshot.hash == second.stable_prompt_snapshot.hash
    assert second.stable_prompt_snapshot.reused is False
    assert second.stable_prompt_snapshot.invalidation_reason == "tool_strategy_changed"
    assert (
        snapshot["tool_strategy_hash"]
        == second.stable_prompt_snapshot.tool_strategy_hash
    )
    assert "stable_prompt" not in snapshot
    with database.snapshot() as source:
        assert source.get("session", "session-1") is None


def _file_read_tool(name: str, *, model_visible: bool = True) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="Read a file.",
        parameters={
            "path": {
                "type": "string",
                "description": "Path to read",
                "required": True,
            }
        },
        toolset=TOOLSET_FILE,
        risk_level=TOOL_RISK_SAFE,
        readonly=True,
        target_scope_rule=TARGET_SCOPE_PATH,
        source=TOOL_SOURCE_BUILTIN,
        model_visible=model_visible,
        idempotent=IDEMPOTENT_YES,
        executor=lambda _args: {"content": "ok"},
    )
