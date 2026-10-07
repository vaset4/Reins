"""运行时动作与其他工具共享 Schema、选择和调用配对。

作者：xxx
时间：2026-09-14 11:00:00
"""

from __future__ import annotations

from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk


def register_native_actions(registry: ToolRegistry) -> None:
    """注册提问、目标、历史及恢复工具；传参：注册表；返回：无，执行由当前运行提供依赖。"""
    schemas: dict[str, tuple[str, dict[str, object], bool]] = {
        **_context_schemas(),
        "capabilities": (
            "Discover tools by name, English or Chinese business descriptions. List/search by query or domain; use next_cursor for more. "
            "Load one tool by name, or a whole domain such as web, browser, memory, skills, collaboration or schedules. "
            "Only currently available authorized members are loaded; unavailable members are explained separately. "
            "Loading does not execute tools or grant permissions. Use refresh to reconnect configured sources or discover updates.",
            _capabilities_schema(),
            True,
        ),
        "ask_user": (
            "Ask a focused question and wait for the user's reply. The result includes a persistent question_id. "
            "For goals requiring explicit user acceptance, include completion with the goal revision and delivered evidence. "
            "After the user chooses, load the operations domain through capabilities and inspect operation_status(question_id) for confirmation_event_id. Ordinary replies are not completion confirmation.",
            _ask_user_schema(),
            False,
        ),
        "goal": (
            "Create, switch or complete a long-term goal record and update the current focus. "
            "This does not schedule work or start a separate executor; active is the goal state, not a running job. "
            "For later or separate background execution, discover scheduling or delegation capabilities. "
            "A reply does not complete a goal. Completion requires "
            "a revision, a summary assessing the full goal, and real tool_result, answer or user_confirmation references. "
            "For an answer in this same reply, provide its actual text alongside the tool call and use reference=current_answer. "
            "For user_confirmation use the recorded confirmation_event_id, never a plain user message_id.",
            _goal_schema(),
            False,
        ),
        "read_history": (
            "Read earlier messages from the current session branch. Pass next_cursor as cursor to continue "
            "the same snapshot; new messages do not shift its pages. Use before for a specific message, without cursor. "
            "Use summary_id to read its covered originals, view=topics with optional query to locate old topics, "
            "or view=messages with query to scan original interactions directly. Pass a topic source_ref to read its evidence. "
            "Use view=segments for the goal-segment directory; select summary_id/segment_id and level=P1/P2/P3/P4 "
            "to read a saved representation without another model call. Each segment source_ref retrieves its exact originals. "
            "source_ref cannot combine with summary_id/query/topics; before is only for plain paging. "
            "A page can be empty while scan_complete=false; continue with its cursor. Missing directory is not missing history.",
            {
                "before": {"type": "string"},
                "cursor": {"type": "string", "minLength": 1},
                "view": {"enum": ["topics", "messages", "segments"]},
                "query": {"type": "string", "minLength": 1},
                "segment_id": {"type": "string", "minLength": 1},
                "level": {"enum": ["P1", "P2", "P3", "P4"]},
                "summary_id": {"type": "string", "minLength": 1},
                "source_ref": {"type": "string", "minLength": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            True,
        ),
        "operation_status": (
            "Inspect actual execution state, late result and question reply, or list current branch operations. "
            "Includes tracked file changes with versions and uncertainty. Use run_id=current for files changed in this run; "
            "records describe past execution, not current disk state or a backup.",
            {"operation_id": {"type": "string"}, "run_id": {"type": "string"}},
            True,
        ),
        "resume_operation": (
            "Skip or retry a known operation. Retry is allowed only when execution was not started, "
            "or the backend declares read-only idempotency. Existing retries are returned without repeating effects.",
            {
                "operation_id": {"type": "string", "required": True},
                "action": {
                    "type": "string",
                    "enum": ["retry", "skip"],
                    "required": True,
                },
            },
            False,
        ),
    }
    for name, (description, schema, readonly) in schemas.items():
        if registry.get(name) is None:
            registry.register(
                ToolDefinition(
                    name,
                    description,
                    schema,
                    "agent",
                    ToolRisk.SAFE,
                    readonly,
                    "logical_scope",
                    "builtin",
                    idempotent=Idempotent.YES if readonly else Idempotent.NO,
                    runtime_action=True,
                    semantics=("polling",) if name == "operation_status" else (),
                )
            )


def _context_schemas() -> dict[str, tuple[str, dict[str, object], bool]]:
    """声明已采用材料的查看与释放动作；参数：无；返回：沿用同一原生注册合同的声明。"""
    return {
        "context_inspect": (
            "Inspect the materials actually adopted by the current session branch: identity, version, scope, "
            "representation, protection and exact read reference. Inspecting does not release content or call a model.",
            {"type": "object", "additionalProperties": False, "properties": {}},
            True,
        ),
        "context_release": (
            "Release a currently adopted material body to its recoverable reference by exact identity and version. "
            "Original messages, files, tool arguments and execution states remain intact. Current requirements, "
            "pending interactions and still-needed recovery reads cannot be released. Check the applied or rejected result.",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "identity": {"type": "string", "minLength": 1},
                    "version": {"type": "string", "minLength": 1},
                },
                "required": ["identity", "version"],
            },
            False,
        ),
    }


def _capabilities_schema() -> dict[str, object]:
    """用统一Schema描述发现、搜索与加载；传参：无；返回：工具声明。"""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"enum": ["list", "search", "load", "refresh"]},
            "query": {"type": "string"},
            "name": {"type": "string", "minLength": 1},
            "domain": {"type": "string", "minLength": 1},
            "registry_version": {"type": "string"},
            "cursor": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        "required": ["action"],
        "allOf": [
            {
                "if": {"properties": {"action": {"const": "load"}}},
                "then": {
                    "oneOf": [
                        {"required": ["name"], "not": {"required": ["domain"]}},
                        {"required": ["domain"], "not": {"required": ["name"]}},
                    ]
                },
            },
            {
                "if": {"properties": {"action": {"const": "search"}}},
                "then": {"required": ["query"]},
            },
        ],
    }


def _goal_schema() -> dict[str, object]:
    """用同一声明约束三种目标操作；传参：无；返回：完整对象Schema。"""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {"type": "string", "enum": ["new", "switch", "complete"]},
            "goal_ref": {"type": "string", "minLength": 1},
            "goal_body": {"type": "string", "minLength": 1},
            "expected_revision": {"type": "integer", "minimum": 0},
            "evidence": _evidence_schema(),
        },
        "required": ["action"],
        "allOf": [
            {
                "if": {"properties": {"action": {"const": "new"}}},
                "then": {"required": ["goal_body"]},
            },
            {
                "if": {"properties": {"action": {"const": "switch"}}},
                "then": {"required": ["goal_ref"]},
            },
            {
                "if": {"properties": {"action": {"const": "complete"}}},
                "then": {"required": ["goal_body", "expected_revision", "evidence"]},
            },
        ],
    }


def _evidence_schema() -> dict[str, object]:
    """统一成果与确认的引用形状；传参：无；返回：非空证据列表的 Schema。"""
    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"enum": ["tool_result", "answer", "user_confirmation"]},
                "reference": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Exact result/answer message_id, operation_id, or recorded confirmation_event_id. "
                    "current_answer selects this run's latest published answer. Ordinary user text is never confirmation.",
                },
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["kind", "reference", "reason"],
        },
    }


def _ask_user_schema() -> dict[str, object]:
    """为普通提问与明确完成确认声明不同参数；传参：无；返回：提问 Schema。"""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "question": {"type": "string", "minLength": 1},
            "completion": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "goal_id": {"type": "string", "minLength": 1},
                    "expected_revision": {"type": "integer", "minimum": 0},
                    "evidence": _evidence_schema(),
                },
                "required": ["goal_id", "expected_revision", "evidence"],
            },
        },
        "required": ["question"],
    }
