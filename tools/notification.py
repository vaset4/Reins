"""持久通知动作声明，交付由后台 outbox 负责。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk


def register_tools(registry: ToolRegistry) -> None:
    """注册持久接纳和状态查询，创建通知不能伪称已送达；传参：工具目录；返回：无。"""
    descriptions: dict[str, tuple[str, dict[str, object], bool]] = {
        "notification_send": (
            "Queue a durable local notification. Returns an accepted notification_id, not delivery or reading. "
            "Use notification_status for actual channel receipts. The desktop preview is delivered by the local background host.",
            {
                "title": {"type": "string", "minLength": 1, "required": True},
                "message": {"type": "string", "minLength": 1, "required": True},
            },
            False,
        ),
        "notification_status": (
            "Read durable notification content and actual submitted/failed/unknown, visible and read receipts. "
            "Only the user's explicit UI acknowledgement marks a notification read.",
            {"notification_id": {"type": "string", "minLength": 1}},
            True,
        ),
    }
    for name, (description, parameters, readonly) in descriptions.items():
        if registry.get(name) is None:
            registry.register(
                ToolDefinition(
                    name,
                    description,
                    parameters,
                    "agent",
                    ToolRisk.SAFE,
                    readonly,
                    "logical_scope",
                    "builtin",
                    idempotent=Idempotent.YES if readonly else Idempotent.NO,
                    runtime_action=True,
                )
            )
