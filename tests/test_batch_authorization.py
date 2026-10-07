"""验证整批授权先于任何副作用，决定只覆盖已展示操作。

作者：xxx
时间：2026-09-24 12:00:00
"""

from scripts.testing.llm import from_test_native_tool_then_final
from contextlib import closing
from functools import partial

import pytest

from approval import ApprovalDecision
from approval.batch_types import ApprovalChoice, BatchDecision
from llm.messages import ToolCallPart
from runtime.agent_loop import AgentLoop
from runtime.ledger import LedgerStore
from runtime.lease import from_trigger
from runtime.session_messages import append_user_message
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk


def _effect(effects, name, _args):
    """记录实际业务执行而非准备；传参：列表、工具及参数；返回：真实结果。"""
    effects.append(name)
    return f"{name} executed"


def _batch_run(root, monkeypatch, *, cancel=False, safe_first=False):
    """运行生产主循环与真实审批边界；传参：隔离根、替换器及场景；返回：执行和展示证据。"""
    with closing(TaskStore(root)) as tasks:
        task = tasks.create_task("执行经我逐项批准的操作")
    effects, prompts = [], []
    registry = ToolRegistry()
    for name in ("a", "b", "c"):
        risk = ToolRisk.SAFE if safe_first and name == "a" else ToolRisk.CONFIRM
        registry.register(
            ToolDefinition(
                name,
                f"操作{name}",
                {"type": "object", "properties": {}},
                "agent",
                risk,
                safe_first and name == "a",
                "logical_scope",
                "builtin",
                idempotent=Idempotent.NO,
                executor=partial(_effect, effects, name),
            )
        )

    def decide(batch):
        """一次展示全部项目，只同意A/C；传参：真实批次；返回：明确逐项决定。"""
        prompts.append(
            (tuple(request.tool for request in batch.requests), tuple(effects))
        )
        if cancel:
            return BatchDecision(cancelled=True)
        return BatchDecision(
            tuple(
                ApprovalChoice(
                    request.operation_id,
                    ApprovalDecision.DENY
                    if request.tool == "b"
                    else ApprovalDecision.ONCE,
                )
                for request in batch.requests
            )
        )

    monkeypatch.setattr("approval.batch._batch_backend", decide)
    monkeypatch.setattr("approval._backend", lambda _: ApprovalDecision.ONCE)
    monkeypatch.setattr("approval._config_path", lambda: root / "config.yaml")
    context = RunContext(
        task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": task.goal},
        capability_lease=from_trigger(
            "user",
            task_id=task.task_id,
            capabilities={
                "fs": {
                    "project_root": str(root),
                    "read": [str(root)],
                    "write": [str(root)],
                }
            },
        ),
    )
    WorkspaceStore(root).bind_session(context.session_id, root)
    append_user_message(root, context.session_id, task.goal)
    client = from_test_native_tool_then_final(
        [ToolCallPart(name, name, {}) for name in ("a", "b", "c")], "执行结果已返回"
    )
    list(AgentLoop(root, llm_client=client, tool_registry=registry).run_stream(context))
    return effects, prompts, context


def test_batch_choice_only_dispatches_approved_operations(tmp_path, monkeypatch):
    """批准A/C不放行B，全部副作用晚于整批决定；传参：目录和替换器；返回：无。"""
    effects, prompts, context = _batch_run(tmp_path, monkeypatch)
    assert prompts == [(("a", "b", "c"), ())]
    assert effects == ["a", "c"]
    decisions = [
        row
        for row in LedgerStore(tmp_path).read_session_events(context.session_id)
        if row.event == "approval.decided"
    ]
    assert [row.payload["decision"] for row in decisions] == ["once", "deny", "once"]
    assert all(
        row.source == "user_action" and row.payload["source_input_id"]
        for row in decisions
    )


@pytest.mark.parametrize("safe_first", [False, True])
def test_cancelling_batch_prevents_even_previously_covered_actions(
    tmp_path, monkeypatch, safe_first
):
    """整批取消时，自动策略已覆盖项也不能提前执行；传参：目录、替换器和首项风险；返回：无。"""
    effects, prompts, _ = _batch_run(
        tmp_path, monkeypatch, cancel=True, safe_first=safe_first
    )
    assert len(prompts) == 1 and prompts[0][1] == ()
    assert effects == []
