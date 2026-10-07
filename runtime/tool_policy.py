"""请求选择与派发共用的工具策略；作者：xxx；时间：2026-09-28 18:00:00。"""

from __future__ import annotations
from dataclasses import replace
from approval.session import ApprovalMode, ApprovalSession
from llm.types import LLMPlan, ModelError
from llm.toolset_policy import (
    ToolsetPolicy,
    ToolsetPolicyError,
    resolve_toolset_policy,
    load_toolset_runtime_config,
    allowed_tool_actions,
)
from llm.tool_selection import select_tools
from runtime.session_state import SessionStateStore
from runtime.types import RunContext, RunToolsRequest
from tools.tool_registry import ToolRegistry


class RuntimeToolPolicy:
    """共同读取当前工具策略，派发再次校验现行允许集合。"""

    def __init__(
        self,
        registry: ToolRegistry,
        states: SessionStateStore,
        config: dict[str, object],
        *,
        approval_session: ApprovalSession | None = None,
    ) -> None:
        """接收目录、会话状态与配置；返回：无。"""
        self.tool_registry = registry
        self.session_states = states
        self.runtime_config = config
        self.approval_session = approval_session

    def resolve(self, context: RunContext) -> ToolsetPolicy | LLMPlan:
        """结合当前会话和配置解析工具策略；传参：运行；返回：策略或显式模型错误。"""
        payload = dict(context.payload)
        payload.setdefault("trigger", context.trigger.value)
        payload.setdefault("task", payload.get("message", ""))
        try:
            policy = resolve_toolset_policy(
                payload=payload,
                session_state=self.session_states.load(context.session_id),
                config=self.config(),
                registry=self.tool_registry,
            )
            # 1. 【工具策略】【只读模式】使用宿主实际权限模式，空写入根仍保留原路径审批语义
            read_only = (
                context.trigger.value != "cron"
                and self.approval_session is not None
                and self.approval_session.mode is ApprovalMode.READ_ONLY
            )
            return replace(policy, read_only=read_only)
        except ToolsetPolicyError as exc:
            error = ModelError.create(
                category="invalid_model_protocol",
                summary=f"invalid toolset policy: {exc}",
                raw_summary=str(exc),
                stage="policy",
            )
            return LLMPlan(
                final_output=error.render_output(),
                model_error=error,
                protocol_mode="",
                request_bundle_evidence={"toolset_policy_error": str(exc)},
            )

    def validate(
        self,
        context: RunContext,
        request: RunToolsRequest,
    ) -> ModelError | None:
        """派发前按当前策略重核实际动作；传参：运行、候选；返回：拒绝错误或None。"""
        policy = self.resolve(context)
        if isinstance(policy, LLMPlan):
            return policy.model_error or ModelError.create(
                category="invalid_model_protocol",
                summary="invalid toolset policy",
                raw_summary=str(policy.final_output or ""),
                stage="policy",
            )
        selection = select_tools(
            self.tool_registry,
            policy=policy,
            lease=context.capability_lease,
            include_deferred=True,
        )
        tool_name = request.tool_name or request.action
        allowed = allowed_tool_actions(policy, self.tool_registry.get(tool_name))
        if _tool_name_allowed(tool_name, selection.allowed_tool_names) and (
            allowed is None or request.arguments.get("action") in allowed
        ):
            return None
        return ModelError.create(
            category="invalid_tool_arguments",
            summary=f"tool {tool_name} is not allowed in current request",
            raw_summary=f"tool={tool_name}; execution policy check failed",
            stage="execute",
        )

    def config(self) -> dict[str, object]:
        """读取公开配置并应用本次运行覆盖；传参：无；返回：完整策略配置。"""
        config = load_toolset_runtime_config()
        config.update(self.runtime_config)
        return config


def _tool_name_allowed(tool_name: str, allowed_names: frozenset[str]) -> bool:
    """核对实际工具或别名是否在允许集合；传参：工具名、允许集合；返回：是否允许。"""
    if tool_name in allowed_names:
        return True
    return any(
        allowed.endswith("*") and tool_name.startswith(allowed[:-1])
        for allowed in allowed_names
    )
