"""知识查询与修订工具声明，来源由原生运行边界提供。

作者：xxx
时间：2026-09-15 05:40:00
"""

from __future__ import annotations

from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk


def register_knowledge_tools(registry: ToolRegistry) -> None:
    """声明精确资料查询和有版本的知识修改；传参：工具目录；返回：无。"""
    definitions = (
        ToolDefinition(
            "memory_query",
            "Read exact canonical memories and original references, list by subject/field/scope/exact_fields, or search by relevance. "
            "read supports an old version; sources reads the exact saved original input or tool receipt, including a previous session. list includes archived/withdrawn records with explicit state. "
            "Search only uses current index content; index_status explains unavailable or stale indexes. "
            "memory_scope is global, session:<id>, goal:<id>, or project:<workspace_id>; session, goal and project select this run's identities. Different scopes are different facts. "
            "Use list without filters to discover existing subjects and scope labels. To save or correct a fact, load memory_manage through capabilities.",
            _query_schema(),
            "memory",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            runtime_action=True,
        ),
        ToolDefinition(
            "memory_manage",
            "Save or write long-term memory facts; update, revise, verify, archive, restore or withdraw persistent knowledge. rebuild_index repairs only the derived search index. "
            "Use kind=fact for enduring conclusions/preferences, note for scoped working notes, archive for an original-reading tool reference. "
            "Creating or using knowledge is not verification. Corrections require the exact expected_version, reason and actual sources; old versions remain readable. "
            "memory_scope is applicability, NOT storage lifetime: project facts also persist across sessions. "
            "Use memory_scope=project for this workspace; the harness binds its stable identity, never a folder name or task tag. global means applicable across projects. subject and fact_key identify one claim. "
            "Keep one subject-field claim per content: do not copy other projects' values or superseded amounts into it. "
            "Revisions automatically retain old versions and sources; cite only the new correction source, not input IDs from older sessions. "
            "Prefer revise for the same memory. For explicit cross-record replacement, supersedes lists the read memory_id/version and reason in the same commit. "
            "Replacement is permanent history: withdrawing the replacement does not revive the old record. "
            "Correct objective facts using applicable evidence; user requirements/preferences require user input evidence to change. "
            "tool_results can cite failed or unknown receipts for lessons, never as proof of successful completion; archive requires a successful readable original. "
            "Choose whether and when to save based on usefulness; current working evidence does not require a memory save before continuing. "
            "source_mode=current_input uses the latest real user input, user_inputs requires source_input_ids, tool_results requires source_operation_ids; "
            "In accepted background work, use origin_inputs and omit source_input_ids for the latest original user input; "
            "other original user IDs and operation IDs come from knowledge_read, not branch anchors. "
            "inference labels a model conclusion and cannot verify a fact. observed_at/expires_at are optional explicit ISO instants with timezone.",
            _manage_schema(),
            "memory",
            ToolRisk.SAFE,
            False,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.NO,
            runtime_action=True,
        ),
    )
    for definition in definitions:
        if registry.get(definition.name) is None:
            registry.register(definition)
    _register_skill_actions(registry)
    _register_reflection_actions(registry)


def _query_schema() -> dict[str, object]:
    """精确筛选和相关性搜索各有明确动作；传参：无；返回：JSON Schema。"""
    text = {"type": "string", "minLength": 1}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action"],
        "properties": {
            "action": {"enum": ["read", "sources", "list", "search", "index_status"]},
            "memory_id": text,
            "version": text,
            "query": text,
            "subject": text,
            "fact_key": text,
            "memory_scope": text,
            "state": {
                "enum": ["active", "archived", "withdrawn", "draft", "superseded"]
            },
            "exact_fields": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
        "allOf": [
            {
                "if": {"properties": {"action": {"enum": ["read", "sources"]}}},
                "then": {"required": ["memory_id"]},
            },
            {
                "if": {"properties": {"action": {"const": "search"}}},
                "then": {"required": ["query"]},
            },
        ],
    }


def _manage_schema() -> dict[str, object]:
    """所有内容修改基于明确版本与出处；传参：无；返回：JSON Schema。"""
    text = {"type": "string", "minLength": 1}
    texts = {"type": "array", "items": text, "minItems": 1}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action"],
        "properties": {
            "action": {
                "enum": [
                    "create",
                    "revise",
                    "archive",
                    "restore",
                    "withdraw",
                    "verify",
                    "rebuild_index",
                ]
            },
            "memory_id": text,
            "expected_version": text,
            "reason": text,
            "content": text,
            "type": {"enum": ["fact", "preference", "rule", "lesson", "experience"]},
            "kind": {"enum": ["fact", "note", "archive"]},
            "subject": text,
            "fact_key": text,
            "memory_scope": text,
            "tags": {"type": "array", "items": text},
            "observed_at": text,
            "expires_at": text,
            "exact_fields": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "supersedes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["memory_id", "version", "reason"],
                    "properties": {"memory_id": text, "version": text, "reason": text},
                },
            },
            "source_mode": {
                "enum": [
                    "current_input",
                    "user_inputs",
                    "tool_results",
                    "inference",
                    "origin_inputs",
                    "origin_results",
                ]
            },
            "source_input_ids": texts,
            "source_operation_ids": texts,
        },
        "allOf": [
            {
                "if": {"properties": {"action": {"const": "create"}}},
                "then": {
                    "required": [
                        "content",
                        "kind",
                        "memory_scope",
                        "subject",
                        "fact_key",
                    ]
                },
            },
            {
                "if": {
                    "properties": {
                        "action": {
                            "enum": [
                                "revise",
                                "archive",
                                "restore",
                                "withdraw",
                                "verify",
                            ]
                        }
                    }
                },
                "then": {"required": ["memory_id", "expected_version", "reason"]},
            },
            {
                "if": {"properties": {"action": {"const": "revise"}}},
                "then": {"required": ["content"]},
            },
        ],
    }


def _register_skill_actions(registry: ToolRegistry) -> None:
    """将版本修订和效果反馈作为模型可选动作；传参：目录；返回：无。"""
    if registry.get("skill_manage") is not None:
        return
    registry.register(
        ToolDefinition(
            "skill_manage",
            "Inspect versions or original sources, create or revise a reusable method, publish/withdraw an exact version, or record task feedback. "
            "Changes take effect in the current run; this tool does not queue a separate background job. "
            "For requested background method refinement, discover and load knowledge_reflect through capabilities. "
            "Create/revise default to an unpublished draft; publish=true explicitly makes that version available without claiming it is evaluated. "
            "Revise requires expected_version and reason. Existing IDs are revised, never overwritten. "
            "body is the actual guide; optional script defines main(args) and is versioned with the guide. Saving never runs the script. "
            "Publishing compiles Python resources and rejects syntax errors without replacing the current version; compilation does not prove task correctness. "
            "validation is only an author claim. feedback separately assesses an actual case as achieved/failed/unknown using real source inputs or operations. "
            "source_mode=current_input/user_inputs/tool_results/inference labels the source. In accepted background work, including ordinary scheduled work, origin_inputs/origin_results refer to its frozen source branch. "
            "Use origin_inputs and omit source_input_ids to cite its latest original user input. For other inputs use actual user message IDs from knowledge_read; branch anchors are not input IDs.",
            _skill_schema(),
            "memory",
            ToolRisk.SAFE,
            False,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.NO,
            runtime_action=True,
        )
    )


def _skill_schema() -> dict[str, object]:
    """固定内容版本、来源及任务效果字段；传参：无；返回：JSON Schema。"""
    text = {"type": "string", "minLength": 1}
    texts = {"type": "array", "items": text}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "skill_id"],
        "properties": {
            "action": {
                "enum": [
                    "versions",
                    "sources",
                    "create",
                    "revise",
                    "publish",
                    "withdraw",
                    "feedback",
                ]
            },
            "skill_id": text,
            "version": text,
            "expected_version": text,
            "reason": text,
            "name": text,
            "body": text,
            "script": text,
            "validation": text,
            "publish": {"type": "boolean"},
            "tags": texts,
            "trigger_keywords": texts,
            "required_capabilities": texts,
            "outcome": {"enum": ["achieved", "failed", "unknown"]},
            "source_mode": {
                "enum": [
                    "current_input",
                    "user_inputs",
                    "tool_results",
                    "inference",
                    "origin_inputs",
                    "origin_results",
                ]
            },
            "source_input_ids": texts,
            "source_operation_ids": texts,
        },
        "allOf": [
            {
                "if": {"properties": {"action": {"enum": ["create", "revise"]}}},
                "then": {"required": ["body", "reason"]},
            },
            {
                "if": {"properties": {"action": {"const": "revise"}}},
                "then": {"required": ["expected_version"]},
            },
            {
                "if": {
                    "properties": {
                        "action": {"enum": ["publish", "withdraw", "feedback"]}
                    }
                },
                "then": {"required": ["version", "reason"]},
            },
            {
                "if": {"properties": {"action": {"const": "feedback"}}},
                "then": {"required": ["outcome"]},
            },
        ],
    }


def _register_reflection_actions(registry: ToolRegistry) -> None:
    """暴露主动提炼和限定来源补读，默认轮末不触发；传参：目录；返回：无。"""
    text = {"type": "string", "minLength": 1}
    definitions = (
        ToolDefinition(
            "knowledge_reflect",
            "Request one durable background reflection of this session's current evidence. This queues work and does not run an extra model call inside this turn. "
            "Requires the existing background_run permission. objective explains what reusable method or lesson to extract. "
            "Optional skill_id/expected_version targets an existing skill version; publish=false keeps the revision as a draft, publish=true permits publication. "
            "The returned schedule/occurrence IDs prove acceptance, not completion; use them to inspect results or resume/cancel work. "
            "Set knowledge_only=true to select old or current sources for memory-only maintenance with durable coverage. "
            "Optional source_message_ids selects exact messages from this session; omitted means the current complete source range. "
            "Do not call automatically for every reply; request only when the evidence warrants useful reusable knowledge.",
            {
                "objective": {**text, "required": True},
                "skill_id": text,
                "expected_version": text,
                "publish": {"type": "boolean"},
                "knowledge_only": {"type": "boolean"},
                "source_message_ids": {"type": "array", "items": text, "minItems": 1},
            },
            "memory",
            ToolRisk.SAFE,
            False,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.NO,
            runtime_action=True,
        ),
        ToolDefinition(
            "knowledge_read",
            "Read the original evidence of accepted background work, including ordinary schedules and reflection requests. Default action=user_inputs reads only actual original user inputs, without tool catalogs and intermediate agent messages. "
            "action=messages provides the full frozen message set; operations returns a compact paged directory of operation IDs/status/previews for origin_results. "
            "action=operation with operation_id reads an exact original in character pages using offset/max_chars and next_offset; neither operation view advances message coverage. "
            "Automatic maintenance can finish only after all frozen message_ids have been read, including non-user messages; user_inputs is a subset. "
            "The source session and branch boundary are fixed; later inputs and other branches are excluded. "
            "User message IDs can be cited with origin_inputs/source_input_ids; branch_entry_id identifies the frozen branch, not a cursor or a user input. "
            "For paged messages or user_inputs, pass only the same action's returned non-null next_cursor unchanged. next_cursor=null ends that view; do not substitute branch_entry_id.",
            {
                "action": {
                    "enum": ["user_inputs", "messages", "operations", "operation"]
                },
                "cursor": {
                    **text,
                    "description": "For messages/user_inputs, use the non-null next_cursor returned by the same action, unchanged. Omit for the first page; null next_cursor means that view is complete. Never use branch_entry_id as a cursor.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "operation_id": text,
                "offset": {"type": "integer", "minimum": 0},
                "max_chars": {"type": "integer", "minimum": 1, "maximum": 32768},
            },
            "memory",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            runtime_action=True,
        ),
    )
    for definition in definitions:
        if registry.get(definition.name) is None:
            registry.register(definition)
