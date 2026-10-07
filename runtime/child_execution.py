"""协作子运行装配，仅借用父共享依赖；作者：xxx；时间：2026-09-28 18:00:00。"""

from __future__ import annotations
from pathlib import Path
from approval.session import ApprovalSession
from llm.base import LLMClient
from runtime.collaboration import ChildExecution, ChildOutcome
from runtime.extensions import RuntimeExtensions
from tools.tool_registry import ToolRegistry


def execute_child(
    execution: ChildExecution,
    *,
    data_root: Path,
    client: LLMClient | None,
    registry: ToolRegistry,
    runtime_config: dict[str, object],
    extensions: RuntimeExtensions,
    approval_session: ApprovalSession,
) -> ChildOutcome:
    """以明确依赖创建独立子运行，关闭自有外部客户端；传参：执行归属及借用依赖；返回：真实边界和答复。"""
    from runtime.agent_loop import AgentLoop

    external = None
    if execution.member["backend"] == "claude":
        from runtime.external_agent import ClaudeAgentClient

        external = ClaudeAgentClient(
            execution.context,
            data_root=data_root,
            member=execution.member,
            collaboration=execution.collaboration,
            cancellation=execution.cancellation,
        )
        client = external
    elif execution.member["backend"] != "internal":
        raise ValueError(f"unsupported agent backend: {execution.member['backend']}")
    # 1. 【协作】【资源归属】从客户端创建成功起覆盖后续装配异常，只关闭本次拥有的客户端
    try:
        loop = AgentLoop(
            data_root,
            llm_client=client,
            tool_registry=registry,
            runtime_config=runtime_config,
            cancellation=execution.cancellation,
            extensions=extensions,
            shared_budget=execution.budget,
            collaboration=execution.collaboration,
            approval_session=approval_session,
        )
        state = loop.run(execution.context)
        return ChildOutcome(state.value.lower(), loop.last_output)
    finally:
        if external is not None:
            external.close()
