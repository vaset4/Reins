"""协作产物的操作引用和消息来源不能改变目标、提问的授权语义。

作者：xxx
时间：2026-09-14 17:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final

import pytest
from dataclasses import asdict, replace

from llm.messages import ToolCallPart
from runtime.goal_manager import GoalManager
from runtime.native_actions import link_question_answers
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore, SessionMessageStoreError
from runtime.tool_operations import ToolOperationStore
from tests.test_goal_completion import _complete, _environment, _result
from tests.test_native_actions import native_registry
from tests.test_session_runtime import make_runtime


def test_parent_can_complete_with_actual_operation_id(tmp_path):
    """模型复制工具回执中的操作ID即可引用原成果；传参：目录；返回：无。"""
    tasks, messages, _ = _environment(tmp_path)
    try:
        _result(messages)
        operations = ToolOperationStore(tmp_path)
        operations.write(
            {
                "session_id": "session",
                "run_id": "run-source",
                "operation_id": "op-verified",
            },
            {
                "state": "completed",
                "call": {"call_id": "call", "task_id": "goal"},
                "result": {"status": "ok"},
            },
        )
        manager = GoalManager(tasks, messages=messages, operations=operations)
        _complete(manager, reference="op-verified")
        evidence = tasks.require_task("goal").completion["evidence"][0]
        assert evidence["operation_id"] == "op-verified"
        assert evidence["message_id"] == "result" and evidence["call_id"] == "call"
        assert evidence["run_id"] == "run-source"
    finally:
        tasks.close()


@pytest.mark.parametrize("invalid", ["failed", "other_goal", "other_session"])
def test_operation_reference_cannot_bypass_actual_result_or_ownership(
    tmp_path, invalid
):
    """失败或别的会话/目标的操作不能关闭父目标；传参：目录/场景；返回：无。"""
    tasks, messages, _ = _environment(tmp_path)
    try:
        _result(messages, failed=invalid == "failed")
        operations = ToolOperationStore(tmp_path)
        operations.write(
            {
                "session_id": "other" if invalid == "other_session" else "session",
                "run_id": "run-source",
                "operation_id": "op-result",
            },
            {
                "state": "completed",
                "call": {
                    "call_id": "call",
                    "task_id": "other" if invalid == "other_goal" else "goal",
                },
                "result": {"status": "ok"},
            },
        )
        with pytest.raises(ValueError, match="evidence"):
            _complete(
                GoalManager(tasks, messages=messages, operations=operations),
                reference="op-result",
            )
        assert tasks.require_task("goal").status == "active"
    finally:
        tasks.close()


def test_agent_message_cannot_act_as_user_confirmation(tmp_path):
    """同伴的自述不能冒充用户完成确认，即使文本相同；传参：目录；返回：无。"""
    tasks, messages, manager = _environment(tmp_path)
    try:
        messages.accept_input(
            "session",
            "I confirm this is complete",
            input_id="peer",
            task_id="goal",
            input_source="agent",
        )
        messages.deliver_inputs("session", run_id="run-peer", task_id="goal")
        with pytest.raises(ValueError, match="recorded user action"):
            _complete(manager, kind="user_confirmation", reference="peer")
        with pytest.raises(SessionMessageStoreError, match="input_identity_conflict"):
            messages.accept_input(
                "session", "I confirm this is complete", input_id="peer", task_id="goal"
            )
        assert tasks.require_task("goal").status == "active"
    finally:
        tasks.close()


def test_peer_update_does_not_answer_an_open_user_question(tmp_path):
    """等待真人答复时，同伴进展只能作为材料；传参：目录；返回：无。"""
    client = from_test_native_tool_then_final(
        [ToolCallPart("question", "ask_user", {"question": "Which date?"})],
        "waiting for user",
    )
    runtime, runs = make_runtime(tmp_path, client, native_registry())
    runtime.submit("Schedule a meeting", input_id="start")
    assert runtime.wait_idle(5)
    messages, operations = SessionMessageStore(tmp_path), ToolOperationStore(tmp_path)
    messages.accept_input(
        "session-live",
        "The research agent suggests Tuesday",
        input_id="peer",
        input_source="agent",
    )
    messages.deliver_inputs("session-live", run_id="run-delivery", task_id=None)
    link_question_answers(
        messages, operations, run=runs[-1], facts=RunFactStore(tmp_path)
    )
    assert operations.for_session("session-live")[0]["state"] == "waiting_user"
    messages.accept_input("session-live", "Tuesday", input_id="user-answer")
    messages.deliver_inputs("session-live", run_id="run-delivery", task_id=None)
    link_question_answers(
        messages, operations, run=runs[-1], facts=RunFactStore(tmp_path)
    )
    question = operations.for_session("session-live")[0]
    assert (
        question["state"] == "answered" and question["answer_input_id"] == "user-answer"
    )


def test_resume_preserves_artifact_access_scope_with_fresh_budget(
    tmp_path, monkeypatch
):
    """恢复后仍可核实原授权目录的产物，同时不复用已经耗尽的运行额度；传参：目录/替换器；返回：无。"""
    from app.run_task import execute_resume
    from runtime.checkpoint import checkpoint_to_ledger_state
    from runtime.ledger import LedgerStore
    from runtime.ledger_writer import LedgerWriter
    from runtime.lease import from_trigger
    from tests.test_cli_resume_contract import _save_checkpoint
    from tools.builtin_tools import build_tool_registry
    from triggers.resume import make_run_context

    workspace, data = tmp_path / "workspace", tmp_path / "data"
    workspace.mkdir()
    (workspace / "artifact.txt").write_text("verified child artifact", encoding="utf-8")
    old = from_trigger(
        "user",
        max_steps=1,
        max_tokens=100,
        capabilities={
            "fs": {
                "project_root": str(workspace),
                "read": [str(workspace)],
                "write": [],
            },
            "network": {"enabled": False},
        },
    )
    checkpoint = replace(
        _save_checkpoint(data),
        lease_snapshot=asdict(old),
        checkpoint_id="checkpoint-with-scope",
    )
    LedgerWriter(LedgerStore(data), source="tests.scope").record_checkpoint_saved(
        checkpoint.checkpoint_id,
        checkpoint_to_ledger_state(checkpoint),
        checkpoint.reason,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        run_id=checkpoint.run_id,
    )
    renewed = make_run_context(data_root=data, checkpoint=checkpoint)
    assert renewed.capability_lease.capabilities == old.capabilities
    assert renewed.capability_lease.max_steps > old.max_steps

    def no_new_approval(_request):
        """原授权目录的只读核验不应丢失范围而再次询问；传参：审批；返回：无。"""
        pytest.fail("restored authorized file unexpectedly requires approval")

    monkeypatch.setattr(
        "tools.tool_registry.approval.request_approval", no_new_approval
    )
    client = from_test_native_tool_then_final(
        [ToolCallPart("read-artifact", "file_read", {"path": "artifact.txt"})],
        "verified",
    )
    response = execute_resume(
        checkpoint.checkpoint_id,
        workspace,
        data_root=data,
        llm_client=client,
        tool_registry=build_tool_registry(repo_root=workspace, data_root=data),
    )
    assert response.status == "done"
    reads = [
        row
        for row in ToolOperationStore(data).for_session(renewed.session_id)
        if row["call"]["call_id"] == "read-artifact"
    ]
    assert reads[0]["result"]["status"] == "ok"
    assert "verified child artifact" in reads[0]["result"]["output"]
