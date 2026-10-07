"""完整请求窗口、输出预留和结构化内容的预算验证。

作者：xxx
时间：2026-09-14 20:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final, from_test_turns

from context.token_estimate import estimate_agent_messages_tokens
from scripts.testing.llm import ScriptedTurnOptions
from llm.messages import AssistantMessage, ToolCallPart
from llm.model_request import compose_model_request
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk
from context.production_builder import ProductionContextBuilder
from runtime.session_messages import append_user_message


def test_complete_request_budget_includes_schema_and_output() -> None:
    """长工具说明与输出额度必须进入同一窗口；传参：无；返回：无。"""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "inspect",
            "字段说明 " * 700,
            {"query": {"type": "string"}},
            "agent",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
        )
    )
    composed = compose_model_request(
        task="检查材料",
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context={},
        registry=registry,
        context_window=4096,
    )
    budget = composed.token_estimate
    assert budget["tools"] > 700
    assert composed.request.max_output_tokens == budget["output_reserved"]
    assert budget["output_reserved"] > 0
    assert budget["total"] == sum(
        budget[key] for key in ("instructions", "messages", "tools", "protocol")
    )
    assert budget["required_total"] == budget["total"] + budget["output_reserved"]
    assert budget["context_window"] == 4096


def test_tool_arguments_are_counted_even_without_visible_text() -> None:
    """没有正文的工具请求仍然占用模型窗口；传参：无；返回：无。"""
    short = AssistantMessage(
        "short", (ToolCallPart("short-call", "inspect", {"query": "x"}),)
    )
    long = AssistantMessage(
        "long", (ToolCallPart("long-call", "inspect", {"query": "x" * 12000}),)
    )
    assert (
        estimate_agent_messages_tokens((long,))
        > estimate_agent_messages_tokens((short,)) + 2000
    )


def test_unpublished_history_is_not_dropped_to_meet_window(tmp_path) -> None:
    """语义摘要发布前不能静默删掉早先约束；传参：临时存储；返回：无。"""
    first = append_user_message(
        tmp_path, "context-window", "费用不能超过500，原文不能上传。"
    )
    for index in range(5):
        append_user_message(
            tmp_path, "context-window", f"材料{index}：" + "研究内容 " * 300
        )
    builder = ProductionContextBuilder(tmp_path, system_prompt_provider=lambda: "")
    selected = builder.read_conversation_history("context-window")
    assert first in {message.message_id for message in selected.messages}
    assert not selected.truncated


def test_oversized_current_input_is_rejected_before_network_dispatch(
    monkeypatch,
) -> None:
    """不可容纳的当前输入保留明确失败且不发送越窗请求；传参：替换器；返回：无。"""
    client = from_test_turns(
        ["不能冒充成功"], options=ScriptedTurnOptions(context_window=1200)
    )
    adapter = client._adapter_registry.require("scripted_test")
    calls = []
    original = adapter.stream

    def capture(request, **kwargs):
        """记录真实派发；传参：请求和连接；返回：模型事件流。"""
        calls.append(request)
        return original(request, **kwargs)

    monkeypatch.setattr(adapter, "stream", capture)
    plan = client.plan(
        "必须完整保留的新约束。" * 2000, {"tool_registry": ToolRegistry()}
    )
    assert plan.model_error is not None
    assert plan.model_error.category == "context_overflow"
    assert calls == []
    assert plan.observation.attempt_count == 0


def test_actual_tool_working_directory_reaches_model_in_nested_project(
    tmp_path, monkeypatch
):
    """进程工作目录与实际工具项目不同，模型仍看到准确目录且相对写入落在该目录；传参：根和替换器；返回：无。"""
    from app.run_task import run_task
    from approval import ApprovalDecision
    from tests.test_session_runtime import capture_requests

    project = tmp_path / "nested-project"
    project.mkdir()
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "write-local",
                "file_write",
                {"path": "result.json", "content": '{"ready":true}'},
            )
        ],
        "已保存",
    )
    requests = capture_requests(client, monkeypatch)
    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval",
        lambda _request: ApprovalDecision.TASK,
    )
    response = run_task(
        "把结果保存到result.json",
        project,
        data_root=project / "data",
        llm_client=client,
    )
    assert response.status == "done"
    instructions = "\n".join(part.text for part in requests[0].instructions)
    assert "tool_working_directory=" in instructions and str(project) in instructions
    assert (project / "result.json").read_text(encoding="utf-8") == '{"ready":true}'
    assert not (tmp_path / "result.json").exists()
