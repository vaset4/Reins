"""子运行装配失败时释放自有资源；作者：xxx；时间：2026-09-28 18:40:00。"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from approval.session import ApprovalSession
from runtime.cancellation import CancellationToken
from runtime.child_execution import execute_child
from runtime.collaboration import ChildExecution, CollaborationRuntime
from runtime.extensions import RuntimeExtensions
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.shared_budget import BudgetOwner, SharedRunBudget
from runtime.types import RunContext, Trigger
from tools.tool_registry import ToolRegistry


def test_child_assembly_failure_closes_owned_client_and_keeps_shared_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """客户端创建后子循环装配失败仍关闭它，父Registry仅借用；传参：目录与故障注入；返回：无。"""
    context = RunContext(
        trigger=Trigger.DELEGATE, payload={}, capability_lease=from_trigger("delegate")
    )
    token, registry = CancellationToken(), ToolRegistry()
    budget = SharedRunBudget(
        BudgetOwner(context.session_id, context.run_id, context.capability_lease),
        RunFactStore(tmp_path),
    )
    execution = ChildExecution(
        context, token, budget, {"backend": "claude"}, Mock(spec=CollaborationRuntime)
    )
    owned = Mock()
    close_shared = Mock()
    monkeypatch.setattr(
        "runtime.external_agent.ClaudeAgentClient", lambda *_args, **_kwargs: owned
    )
    monkeypatch.setattr(registry, "close", close_shared)

    def fail_assembly(*_args: object, **_kwargs: object) -> None:
        """在获得自有客户端后模拟装配故障；传参：子运行依赖；返回：不返回。"""
        raise RuntimeError("child assembly failed")

    monkeypatch.setattr("runtime.agent_loop.AgentLoop", fail_assembly)
    with pytest.raises(RuntimeError, match="child assembly failed"):
        execute_child(
            execution,
            data_root=tmp_path,
            client=None,
            registry=registry,
            runtime_config={},
            extensions=RuntimeExtensions(),
            approval_session=ApprovalSession(),
        )
    owned.close.assert_called_once_with()
    close_shared.assert_not_called()
