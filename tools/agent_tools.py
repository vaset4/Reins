"""协作工具共享普通工具的Schema、审批、执行身份与回填边界。

作者：xxx
时间：2026-09-15 01:00:00
"""

from __future__ import annotations

from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk

AGENT_ACTIONS = frozenset(
    {
        "delegate",
        "agent_send",
        "agent_status",
        "agent_wait",
        "agent_cancel",
        "agent_decision",
    }
)


def register_agent_tools(registry: ToolRegistry) -> None:
    """注册可执行的协作动作，依赖由当前运行提供；传参：工具目录；返回：无。"""
    text = {"type": "string", "minLength": 1}
    schemas: dict[str, tuple[str, dict[str, object], bool]] = {
        "delegate": (
            "Start an independent agent on a self-contained assignment. It inherits this run's permissions and shares "
            "the TOTAL budget. Returns an accepted identity, not completion. Read its status, exchange discoveries while it works, "
            "and verify artifacts before integrating. backend=claude uses the installed authenticated Claude Code through controlled Reins tools.",
            {
                "name": {**text, "required": True},
                "task": {**text, "required": True},
                "backend": {"type": "string", "enum": ["internal", "claude"]},
                "model": {
                    **text,
                    "description": "Optional model for the external Claude Code session; internal agents use the parent's model.",
                },
            },
            False,
        ),
        "agent_send": (
            "Send a sourced discovery or follow-up to a teammate by name/id, or to parent. "
            "Wakes an idle agent in its own session. Delivery is not proof of understanding. Set resume=true to explicitly restart cancelled work.",
            {
                "target": {**text, "required": True},
                "message": {**text, "required": True},
                "resume": {"type": "boolean"},
            },
            False,
        ),
        "agent_status": (
            "Read the actual status, output, external session and waiting relationships of collaborators. "
            "A done run does not prove its assignment or artifacts are correct.",
            {"target": text},
            True,
        ),
        "agent_wait": (
            "Wait for collaborators, a new input, completion, failure or cancellation. Mutual waits return evidence "
            "so you can change assignments. Defaults to waiting for other members; timeout is in seconds.",
            {
                "targets": {"type": "array", "items": text},
                "timeout_seconds": {"type": "number", "minimum": 0},
            },
            True,
        ),
        "agent_cancel": (
            "Request cancellation of a child and its descendants. The result distinguishes the request from actual stopping.",
            {"target": {**text, "required": True}},
            False,
        ),
        "agent_decision": (
            "The integrating parent can publish a sourced shared decision and notify collaborators. "
            "Use the current version from agent_status (0 for a new key). Other agents should send proposals to parent.",
            {
                "key": {**text, "required": True},
                "text": {**text, "required": True},
                "expected_version": {"type": "integer", "minimum": 0, "required": True},
            },
            False,
        ),
    }
    for name, (description, parameters, readonly) in schemas.items():
        if registry.get(name) is None:
            registry.register(
                ToolDefinition(
                    name,
                    description,
                    parameters,
                    "agent",
                    ToolRisk.CONFIRM if name == "delegate" else ToolRisk.SAFE,
                    readonly,
                    "logical_scope",
                    "builtin",
                    idempotent=Idempotent.YES if readonly else Idempotent.NO,
                    runtime_action=True,
                    semantics=("polling",)
                    if name in {"agent_status", "agent_wait"}
                    else (),
                )
            )
