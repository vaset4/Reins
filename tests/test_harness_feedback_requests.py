"""核对主循环生成的反馈是否进入适配器实际请求。

作者：xxx
时间：2026-09-22 10:30:00
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import pytest

from approval import ApprovalDecision, ApprovalUnavailable

from llm.client import RealLLMClient
from scripts.testing.llm import (
    _ScriptedAdapter,
    _ScriptedTurn,
    _test_config,
    _test_connection,
    _test_model_registry,
)
from llm.messages import ToolCallPart, ToolResultMessage, model_visible_text
from llm.provider_adapter import AdapterRegistry
from runtime.agent_loop import AgentLoop, State
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.run_evidence import RunEvidenceStore
from runtime.session_messages import append_user_message, materialize_messages
from runtime.types import RunContext, Trigger
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk
from tasks.store import TaskStore


class CapturingAdapter(_ScriptedAdapter):
    """保留真实请求，只有供应商回复使用确定性事件。"""

    def __init__(self, turns: tuple[_ScriptedTurn, ...]) -> None:
        """保存回复序列；传参：供应商响应；返回：无。"""
        super().__init__(turns)
        self.requests = []

    def stream(
        self, request, *, model, connection, cancellation=None, prepared_body=None
    ):
        """捕获将发给供应商的请求；传参：请求及连接；返回：响应事件流。"""
        self.requests.append(request)
        return super().stream(
            request,
            model=model,
            connection=connection,
            cancellation=cancellation,
            prepared_body=prepared_body,
        )


def _run(
    root: Path,
    turns: tuple[_ScriptedTurn, ...],
    *,
    definition: ToolDefinition | None = None,
):
    """执行生产主循环与请求链；传参：临时根、响应序列及可选工具；返回：循环、上下文、捕获适配器。"""
    with closing(TaskStore(root)) as store:
        task = store.create_task("读取证据并回答")
    registry = ToolRegistry()
    registry.register(
        definition
        or ToolDefinition(
            "probe",
            "只读证据",
            {"value": {"type": "string", "required": True}},
            "agent",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=lambda _arguments: "unchanged-evidence",
        )
    )
    adapter = CapturingAdapter(turns)
    client = RealLLMClient(
        _test_config(),
        adapter_registry=AdapterRegistry([adapter]),
        model_registry=_test_model_registry(),
        connection=_test_connection(),
        protocol_mode="native_tool_calls",
    )
    loop = AgentLoop(root, llm_client=client, tool_registry=registry)
    context = RunContext(
        task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": task.goal},
        capability_lease=from_trigger("user", task_id=task.task_id),
    )
    append_user_message(root, context.session_id, task.goal)
    list(loop.run_stream(context))
    return loop, context, adapter


def test_protocol_repair_reaches_adapter_request_without_polluting_history(
    tmp_path: Path,
) -> None:
    """模型协议错误经主循环回喂下一次真实请求；传参：临时目录；返回：无。"""
    loop, context, adapter = _run(
        tmp_path,
        (
            _ScriptedTurn(calls=(ToolCallPart("invalid", "probe", {"value": 123}),)),
            _ScriptedTurn(text="已改正"),
        ),
    )
    assert loop.state == State.DONE
    assert len(adapter.requests) == 2
    instructions = "\n".join(part.text for part in adapter.requests[1].observations)
    assert "[recoverable_error_notice]" in instructions
    assert "value" in instructions
    assert all(
        "[recoverable_error_notice]" not in model_visible_text(message)
        for message in materialize_messages(tmp_path, context.session_id)
    )


def test_repeated_read_evidence_reaches_adapter_once_without_tiered_duplicate(
    tmp_path: Path,
) -> None:
    """重复结果仅作为本轮观察，保留原工具结果且不叠加分级催促；传参：临时目录；返回：无。"""
    turns = tuple(
        _ScriptedTurn(calls=(ToolCallPart(f"read-{index}", "probe", {"value": "A"}),))
        for index in range(5)
    )
    loop, context, adapter = _run(
        tmp_path, (*turns, _ScriptedTurn(text="完成证据分析"))
    )
    assert loop.state == State.DONE
    instructions = "\n".join(part.text for part in adapter.requests[-1].observations)
    assert instructions.count("[no_progress_observation]") == 1
    assert "NO_PROGRESS" in instructions
    assert "[system_reminder]" not in instructions
    messages = materialize_messages(tmp_path, context.session_id)
    results = [
        message for message in messages if isinstance(message, ToolResultMessage)
    ]
    assert len(results) == 5
    assert all(
        "unchanged-evidence" in model_visible_text(result)
        and "NO_PROGRESS" not in model_visible_text(result)
        for result in results
    )
    assert any(
        fact.get("event") == "progress:no_progress"
        for fact in RunFactStore(tmp_path).read_run(context.run_id)
    )


@pytest.mark.parametrize(
    ("decision", "category", "state"),
    [
        (ApprovalDecision.DENY, "permission", "denied"),
        (ApprovalDecision.CANCELLED, "cancelled", "cancelled"),
        (
            ApprovalUnavailable("private-approval-backend-detail"),
            "transport",
            "unavailable",
        ),
    ],
)
def test_approval_outcomes_reach_model_without_private_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: ApprovalDecision | ApprovalUnavailable,
    category: str,
    state: str,
) -> None:
    """审批三类失败进入真实下一次请求且诊断独立保存；传参：临时根、替换器、出口及期望；返回：无。"""
    # 1. 固定真实审批边界的出口，文件工具的副作用始终不能发生
    decide = (
        Mock(side_effect=decision)
        if isinstance(decision, Exception)
        else Mock(return_value=decision)
    )
    monkeypatch.setattr("approval._backend", decide)
    execute = Mock(return_value="unauthorized-executor-output")
    definition = ToolDefinition(
        "probe",
        "读取需要授权的材料",
        {"value": {"type": "string"}},
        "agent",
        ToolRisk.CONFIRM,
        True,
        "logical_scope",
        "builtin",
        idempotent=Idempotent.YES,
        executor=execute,
    )
    _, context, adapter = _run(
        tmp_path,
        (
            _ScriptedTurn(
                calls=(ToolCallPart("approval-case", "probe", {"value": "A"}),)
            ),
            _ScriptedTurn(text="已经收到审批结果"),
        ),
        definition=definition,
    )
    execute.assert_not_called()
    assert len(adapter.requests) == 2
    # 2. 检查生产适配器实际收到的结果，不能只验证已落盘会话
    request = adapter.requests[1]
    results = [
        message
        for message in request.messages
        if isinstance(message, ToolResultMessage)
    ]
    assert len(results) == 1
    result = json.loads(model_visible_text(results[0]))
    assert result["meta"]["approval_state"] == state
    assert result["meta"]["tool_error_category"] == category
    assert result["meta"]["execution_state"] == "not_started"
    assert result["meta"]["retryable"] is False
    visible = "\n".join(part.text for part in request.instructions)
    visible += "\n".join(model_visible_text(message) for message in request.messages)
    assert "private-approval-backend-detail" not in visible
    assert "unauthorized-executor-output" not in visible
    # 3. 原始设施故障仍可排查，但不能重新进入会话或工具操作正文
    history = materialize_messages(tmp_path, context.session_id)
    assert all(
        "private-approval-backend-detail" not in model_visible_text(message)
        for message in history
    )
    diagnostics = [
        row["payload"]
        for row in RunEvidenceStore(tmp_path).list_records(
            session_id=context.session_id, run_id=context.run_id, kind="tool_diagnostic"
        )
    ]
    if state == "unavailable":
        assert any(
            item["diagnostics"].get("approval_cause") == str(decision)
            for item in diagnostics
        )
    else:
        assert not diagnostics
