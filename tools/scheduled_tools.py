"""时间意图的模型动作声明。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk


def register_scheduled_tools(registry: ToolRegistry) -> None:
    """提供创建、查询、变更和接续计划的统一动作；传参：工具目录；返回：无。"""
    if registry.get("schedule") is not None:
        return
    registry.register(
        ToolDefinition(
            "schedule",
            "Manage authorized local reminders and scheduled work. Computer must remain on; closing the chat keeps the background host running. "
            "time is at:<ISO time>, interval:<positive seconds>, or a five-field cron expression; timezone is an IANA name such as Asia/Shanghai. "
            "For a relative one-time reminder, call list to read the exact current clock before computing its at time. "
            "kind=reminder only notifies and does not invoke a model; kind=work runs the self-contained prompt with the captured permissions/model. "
            "Optional profile_name selects a saved model profile. update applies to future occurrences; accepted occurrences keep their original content. "
            "pause prevents future starts but lets an active run finish; cancel also requests stopping active work. "
            "resume enables future starts; resume_occurrence explicitly continues a paused/failed occurrence using message and original permissions. "
            "list includes occurrence IDs and actual results. A finished run or queued notification is not proof the user's goal is complete.",
            _schedule_schema(),
            "agent",
            ToolRisk.SAFE,
            False,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.NO,
            runtime_action=True,
            readonly_actions=("list",),
        )
    )


def _schedule_schema() -> dict[str, object]:
    """约束明确时间与持久工作动作；传参：无；返回：同一模型/执行 Schema。"""
    text = {"type": "string", "minLength": 1}
    properties = {
        "action": {
            "enum": [
                "list",
                "create",
                "update",
                "pause",
                "resume",
                "cancel",
                "resume_occurrence",
            ]
        },
        "schedule_id": text,
        "occurrence_id": text,
        "name": text,
        "prompt": text,
        "time": text,
        "timezone": text,
        "kind": {"enum": ["reminder", "work"]},
        "profile_name": text,
        "message": text,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": ["action"],
        "allOf": [
            {
                "if": {"properties": {"action": {"const": "create"}}},
                "then": {"required": ["name", "prompt", "time", "timezone", "kind"]},
            },
            {
                "if": {
                    "properties": {
                        "action": {"enum": ["update", "pause", "resume", "cancel"]}
                    }
                },
                "then": {"required": ["schedule_id"]},
            },
            {
                "if": {"properties": {"action": {"const": "resume_occurrence"}}},
                "then": {"required": ["occurrence_id", "message"]},
            },
        ],
    }
