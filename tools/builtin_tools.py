from __future__ import annotations

from functools import partial

from pathlib import Path
from typing import Literal, Protocol, cast

from runtime.types import RunToolsRequest, RunToolsResult
from skills import executor as skill_executor
from tools import code_execution_tool, skill_tool, terminal_tool
from tools import browser as browser_tools
from tools import clipboard as clipboard_tools
from tools import notification as notification_tools
from tools import vision as vision_tools
from tools.agent_tools import register_agent_tools
from tools.native_actions import register_native_actions
from tools.scheduled_tools import register_scheduled_tools
from tools.knowledge_tools import register_knowledge_tools
from tools.memory_tools import MemoryToolExecutor
from tools.read_artifact import DEFAULT_ARTIFACT_PAGE_CHARS, read_artifact_page
from runtime.lease import Lease
from tools.readonly_file_tools import ReadOnlyFileToolExecutor
from tools.readonly_inspection import DEFAULT_READ_MAX_CHARS, ReadOnlyInspectionExecutor
from tools.readonly_web_tools import ReadOnlyWebToolExecutor
from tools.todo_tool import add_todo, list_todos, update_todo
from tools.tool_registry import (
    IDEMPOTENT_CONDITIONAL,
    IDEMPOTENT_NO,
    IDEMPOTENT_YES,
    TARGET_SCOPE_DOMAIN,
    TARGET_SCOPE_LOGICAL,
    TARGET_SCOPE_PATH,
    TOOL_RISK_CONFIRM,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_AGENT,
    TOOLSET_FILE,
    TOOLSET_MEMORY,
    TOOLSET_WEB,
    ToolDefinition,
    ToolRegistry,
)
from tools.types import ToolError, ToolErrorCategory
from tools.write_file_tools import WriteFileToolExecutor
from tools.redacted_files import RedactedFiles

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = Path.home() / ".reins" / "data"
_INSPECTION = ReadOnlyInspectionExecutor(_REPO_ROOT, 50, DEFAULT_READ_MAX_CHARS, 50)
_FILE_TOOLS = ReadOnlyFileToolExecutor(_INSPECTION)
_WRITE_FILE_TOOLS = WriteFileToolExecutor(_REPO_ROOT)
_WEB_TOOLS = ReadOnlyWebToolExecutor()
_MEMORY_TOOLS = MemoryToolExecutor(_DATA_ROOT)
_ArtifactReadMode = Literal["summary", "head", "full"]
_WORKSPACE_PATH_RULES = (
    "Use an absolute path inside tool_working_directory or a path relative to that directory. "
    "Copy the complete user-specified path verbatim, retaining every folder name, filename and internal space. "
    "Resolve relative paths from tool_working_directory; preserve absolute paths as supplied."
)


class _RunToolsExecutor(Protocol):
    def execute(self, request: RunToolsRequest) -> RunToolsResult: ...


def build_tool_registry(
    *,
    repo_root: Path | str,
    data_root: Path | str | None = None,
    redacted_files: RedactedFiles | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(redacted_files=redacted_files, data_root=data_root)
    register_tools(registry, repo_root=repo_root, data_root=data_root)
    return registry


def register_tools(
    registry: ToolRegistry,
    *,
    repo_root: Path | str | None = None,
    data_root: Path | str | None = None,
) -> None:
    """装配同源内置目录；参数：注册表、项目及数据目录；返回：无。"""
    register_native_actions(registry)
    register_scheduled_tools(registry)
    register_knowledge_tools(registry)
    register_agent_tools(registry)
    if repo_root is None and data_root is None:
        file_tools = _FILE_TOOLS
        write_file_tools = _WRITE_FILE_TOOLS
        web_tools = _WEB_TOOLS
        memory_tools = _MEMORY_TOOLS
    else:
        resolved_repo_root = (
            Path(repo_root).resolve() if repo_root is not None else _REPO_ROOT
        )
        resolved_data_root = Path(data_root) if data_root is not None else _DATA_ROOT
        inspection = ReadOnlyInspectionExecutor(
            resolved_repo_root, 50, DEFAULT_READ_MAX_CHARS, 50
        )
        file_tools = ReadOnlyFileToolExecutor(inspection)
        write_file_tools = WriteFileToolExecutor(resolved_repo_root)
        web_tools = ReadOnlyWebToolExecutor()
        memory_tools = MemoryToolExecutor(resolved_data_root)

    _register_readonly_queries(registry, file_tools)

    _register_file_write(registry, write_file_tools)

    _register_file_patch(registry, write_file_tools)

    _register_web_tools(registry, web_tools)

    _register_memory_note(registry, memory_tools)

    _register_execution_tools(registry)

    _register_skill_tools(registry)

    _register_task_material_tools(registry)

    _register_internal_tools(registry, file_tools)

    from tools import secret_tool

    secret_tool.register_tools(registry)
    browser_tools.register_tools(registry)
    vision_tools.register_tools(registry)
    clipboard_tools.register_tools(registry)
    notification_tools.register_tools(registry)
    _defer_optional_tools(registry)


def _register_file_write(
    registry: ToolRegistry, write_file_tools: WriteFileToolExecutor
) -> None:
    """注册完整文件写入工具；参数：注册表及所需执行器；返回：无。"""
    if registry.get("file_write") is None:
        registry.register(
            ToolDefinition(
                name="file_write",
                description="Create a workspace file, or replace an existing file using expected_sha256 from file_read. "
                "The returned resolved_path identifies the actual destination.",
                parameters={
                    "path": {
                        "type": "string",
                        "required": True,
                        "description": "File path. " + _WORKSPACE_PATH_RULES,
                    },
                    "content": {
                        "type": "string",
                        "required": True,
                        "description": "Exact full file content, including whitespace; empty text creates or empties a file.",
                    },
                    "expected_sha256": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "Required when replacing an existing file; use content_sha256 from file_read or the last successful write.",
                    },
                    "view_id": {
                        "type": "string",
                        "description": "Current redacted view identity from file_read; keep all protected tokens exactly once.",
                    },
                    "secret_replacements": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Explicit map from original protected token to new value. Requires user confirmation for this operation.",
                    },
                },
                toolset=TOOLSET_FILE,
                risk_level=TOOL_RISK_CONFIRM,
                readonly=False,
                target_scope_rule=TARGET_SCOPE_PATH,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_CONDITIONAL,
                executor=lambda args: _run_result("file_write", args, write_file_tools),
                parallel_safe=True,
            )
        )


def _register_file_patch(
    registry: ToolRegistry, write_file_tools: WriteFileToolExecutor
) -> None:
    """注册版本补丁工具；参数：注册表及所需执行器；返回：无。"""
    if registry.get("file_patch") is None:
        registry.register(
            ToolDefinition(
                name="file_patch",
                description="Replace exact text in the file version read earlier. Multiple matches require replace_all=true.",
                parameters={
                    "path": {
                        "type": "string",
                        "required": True,
                        "description": "File path. " + _WORKSPACE_PATH_RULES,
                    },
                    "old_text": {
                        "type": "string",
                        "required": True,
                        "minLength": 1,
                        "description": "Exact existing text, including indentation and line endings; must match uniquely unless replace_all=true.",
                    },
                    "new_text": {
                        "type": "string",
                        "required": True,
                        "description": "Exact replacement text; an empty string deletes the matched text.",
                    },
                    "expected_sha256": {
                        "type": "string",
                        "required": True,
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "content_sha256 returned by file_read or the last successful write; stale versions are rejected.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Explicitly replace all matching fragments; defaults to false and requires a unique match.",
                    },
                    "view_id": {
                        "type": "string",
                        "description": "Current redacted view identity from file_read. Patch the redacted text, keeping protected tokens.",
                    },
                    "secret_replacements": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Explicit map from original protected token to new value. Requires user confirmation for this operation.",
                    },
                },
                toolset=TOOLSET_FILE,
                risk_level=TOOL_RISK_CONFIRM,
                readonly=False,
                target_scope_rule=TARGET_SCOPE_PATH,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_CONDITIONAL,
                executor=lambda args: _run_result("file_patch", args, write_file_tools),
                parallel_safe=True,
            )
        )


def _register_web_tools(
    registry: ToolRegistry, web_tools: ReadOnlyWebToolExecutor
) -> None:
    """注册网络查询工具；参数：注册表及所需执行器；返回：无。"""
    if registry.get("web_search") is None:
        registry.register(
            ToolDefinition(
                name="web_search",
                description="Run a lightweight web search and return top links.",
                parameters={
                    "query": {
                        "type": "string",
                        "required": True,
                        "description": "Search query text.",
                    },
                    "domain": {
                        "type": "string",
                        "required": False,
                        "description": "Optional domain filter such as example.com.",
                    },
                },
                toolset=TOOLSET_WEB,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_DOMAIN,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_YES,
                executor=lambda args: _run_result("web_search", args, web_tools),
            )
        )

    if registry.get("web_fetch") is None:
        registry.register(
            ToolDefinition(
                name="web_fetch",
                description="Fetch one web page and return its readable text.",
                parameters={
                    "url": {
                        "type": "string",
                        "required": True,
                        "description": "Absolute HTTP or HTTPS URL.",
                    }
                },
                toolset=TOOLSET_WEB,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_DOMAIN,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_YES,
                executor=lambda args: _run_result("web_fetch", args, web_tools),
            )
        )

    if registry.get("web_scan") is None:
        registry.register(
            ToolDefinition(
                name="web_scan",
                description="Fetch one web page and return a light structural scan.",
                parameters={
                    "url": {
                        "type": "string",
                        "required": True,
                        "description": "Absolute HTTP or HTTPS URL.",
                    }
                },
                toolset=TOOLSET_WEB,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_DOMAIN,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_YES,
                executor=lambda args: _run_result("web_scan", args, web_tools),
            )
        )


def _register_memory_note(
    registry: ToolRegistry, memory_tools: MemoryToolExecutor
) -> None:
    """注册会话便签工具；参数：注册表及所需执行器；返回：无。"""
    if registry.get("memory_note") is None:
        registry.register(
            ToolDefinition(
                name="memory_note",
                description="Persist a working note for this session only. It survives reconnection but is not a cross-session fact or verification. Use memory_manage for long-term knowledge.",
                parameters={
                    "note": {
                        "type": "string",
                        "required": True,
                        "description": "Note written into working memory.",
                    },
                    "scope": {
                        "type": "string",
                        "required": True,
                        "description": "Logical scope such as working_memory.",
                    },
                    "related_sop": {
                        "type": "array",
                        "required": False,
                        "description": "Preferred related SOP or skill keys kept with the note.",
                    },
                },
                toolset=TOOLSET_MEMORY,
                risk_level=TOOL_RISK_CONFIRM,
                readonly=False,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_CONDITIONAL,
                executor=lambda args: _run_result("memory_note", args, memory_tools),
                runtime_action=True,
            )
        )


def _register_execution_tools(registry: ToolRegistry) -> None:
    """注册受控执行工具；参数：注册表及所需执行器；返回：无。"""
    if registry.get("terminal_tool") is None:
        registry.register(
            ToolDefinition(
                name="terminal_tool",
                description=(
                    "Run one Windows cmd.exe command under the active lease. "
                    "Use cwd='.' to run in the project; omitted cwd uses the task output directory. "
                    "Check exit_code and stderr to determine command success."
                ),
                parameters={
                    "command": {"type": "string", "required": True},
                    "cwd": {
                        "type": "string",
                        "required": False,
                        "description": "Existing working directory, absolute or relative to the project root; subject to approval.",
                    },
                    "scope": {"type": "string", "required": False},
                },
                toolset=TOOLSET_AGENT,
                risk_level=TOOL_RISK_CONFIRM,
                readonly=False,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                exec_boundary=True,
                idempotent=IDEMPOTENT_NO,
                executor=terminal_tool.executor,
            )
        )

    if registry.get("code_execution_tool") is None:
        registry.register(
            ToolDefinition(
                name="code_execution_tool",
                description=(
                    "Run one short Python script in a controlled subprocess "
                    "(not an OS sandbox) under the active lease workspace."
                ),
                parameters={
                    "code": {"type": "string", "required": True},
                    "timeout_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "required": False,
                        "description": "Maximum subprocess seconds, capped by the host execution limit. File capture is timed separately.",
                    },
                    "scope": {"type": "string", "required": False},
                },
                toolset=TOOLSET_AGENT,
                risk_level=TOOL_RISK_CONFIRM,
                readonly=False,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                exec_boundary=True,
                idempotent=IDEMPOTENT_CONDITIONAL,
                executor=code_execution_tool.executor,
            )
        )


def _register_skill_tools(registry: ToolRegistry) -> None:
    """注册技能工具；参数：注册表及所需执行器；返回：无。"""
    if registry.get("skill_run") is None:
        registry.register(
            ToolDefinition(
                name="skill_run",
                description=(
                    "Run one stored skill's script in a controlled subprocess "
                    "under the active lease. Gated by the skill's own state, "
                    "required capabilities, code_execution capability, and path "
                    "security."
                ),
                parameters={
                    "skill_id": {
                        "type": "string",
                        "required": True,
                        "description": "Stored skill id to execute.",
                    },
                    "version": {
                        "type": "string",
                        "description": "Exact content version selected by skill_read; the result records the version actually run.",
                    },
                    "args": {
                        "type": "object",
                        "required": False,
                        "description": "Input object passed to main(args). Construct its fields from the selected skill guide and the task's actual input data.",
                    },
                    "script_name": {
                        "type": "string",
                        "required": False,
                        "description": "Optional script filename under the skill's scripts dir; defaults to the skill's default script.",
                    },
                },
                toolset=TOOLSET_AGENT,
                risk_level=TOOL_RISK_CONFIRM,
                readonly=False,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_NO,
                executor=skill_executor.executor,
            )
        )

    if registry.get("skill_search") is None:
        registry.register(
            ToolDefinition(
                name="skill_search",
                description="Search stored skills, or omit query to list all active methods. Continue with next_cursor, then use skill_read with the returned skill_id and version.",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "Text to match against stored skills.",
                    },
                    "cursor": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                toolset=TOOLSET_AGENT,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_YES,
                executor=skill_tool.executor,
            )
        )

    if registry.get("skill_read") is None:
        registry.register(
            ToolDefinition(
                "skill_read",
                "Read the complete body of a discovered active skill by skill_id and optional version. This reads instructions; it does not run scripts.",
                {
                    "skill_id": {"type": "string", "required": True, "minLength": 1},
                    "version": {"type": "string"},
                    "trial": {
                        "type": "boolean",
                        "description": "Explicitly read the named unpublished draft version; withdrawn versions remain unavailable.",
                    },
                },
                TOOLSET_AGENT,
                TOOL_RISK_SAFE,
                True,
                TARGET_SCOPE_LOGICAL,
                TOOL_SOURCE_BUILTIN,
                idempotent=IDEMPOTENT_YES,
                executor=skill_tool.read_executor,
            )
        )


def _register_task_material_tools(registry: ToolRegistry) -> None:
    """注册产物和待办工具；参数：注册表及所需执行器；返回：无。"""
    if registry.get("read_artifact") is None:
        registry.register(
            ToolDefinition(
                name="read_artifact",
                description=(
                    "Read a stored Reins artifact by artifact_id only. Do not use "
                    "for ordinary workspace filenames or paths. Defaults to full-text paging. "
                    "Pass next_cursor as cursor to continue the same version."
                ),
                parameters={
                    "artifact_id": {
                        "type": "string",
                        "required": True,
                        "description": "Stored artifact id, not a filename or path.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["summary", "head", "full"],
                        "required": False,
                    },
                    "cursor": {
                        "type": "string",
                        "minLength": 1,
                        "description": "next_cursor from the preceding page; carries mode, position and version.",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Character offset; use next_offset from the preceding page with mode=full.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": DEFAULT_ARTIFACT_PAGE_CHARS,
                        "description": "Maximum characters in this page.",
                    },
                    "expected_sha256": {
                        "type": "string",
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "Optional content_sha256 from the previous page; rejects changed content.",
                    },
                    "scope": {"type": "string", "required": False},
                },
                toolset=TOOLSET_AGENT,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=True,
                idempotent=IDEMPOTENT_YES,
                executor=_read_artifact_executor,
                parallel_safe=True,
            )
        )

    if registry.get("todo") is None:
        registry.register(
            ToolDefinition(
                "todo",
                "Manage task-scoped todo items. list only reads; add/update change the task list and retain approval. "
                "Completing todos does not complete a goal. idx is a zero-based integer from list.",
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action"],
                    "properties": {
                        "action": {"enum": ["list", "add", "update"]},
                        "content": {"type": "string", "minLength": 1},
                        "idx": {"type": "integer", "minimum": 0},
                        "status": {
                            "enum": ["pending", "in_progress", "done", "blocked"]
                        },
                        "filter": {
                            "enum": ["pending", "in_progress", "done", "blocked"]
                        },
                        "task_id": {"type": "string", "minLength": 1},
                    },
                    "allOf": [
                        {
                            "if": {"properties": {"action": {"const": "add"}}},
                            "then": {"required": ["content"]},
                        },
                        {
                            "if": {"properties": {"action": {"const": "update"}}},
                            "then": {"required": ["idx", "status"]},
                        },
                    ],
                },
                TOOLSET_AGENT,
                TOOL_RISK_CONFIRM,
                False,
                TARGET_SCOPE_LOGICAL,
                TOOL_SOURCE_BUILTIN,
                idempotent=IDEMPOTENT_CONDITIONAL,
                executor=_todo_executor,
                readonly_actions=("list",),
            )
        )


def _register_internal_tools(
    registry: ToolRegistry, file_tools: ReadOnlyFileToolExecutor
) -> None:
    """注册内部检查工具；参数：注册表及所需执行器；返回：无。"""
    if registry.get("inspect") is None:
        registry.register(
            ToolDefinition(
                name="inspect",
                description=(
                    "Current compatibility bridge for repository inspection. "
                    "This keeps the existing read-only inspect path runnable while "
                    "the first-class file tools are introduced in the next task."
                ),
                parameters={
                    "payload": {
                        "type": "string",
                        "description": (
                            'Use "workspace", "dir <path>", "file <path>", '
                            'or "search <path> for <text>".'
                        ),
                    }
                },
                toolset=TOOLSET_FILE,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_PATH,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=False,
                idempotent=IDEMPOTENT_YES,
                executor=lambda args: _run_result("inspect", args, file_tools),
            )
        )

    if registry.get("echo") is None:
        registry.register(
            ToolDefinition(
                name="echo",
                description=(
                    "Legacy local echo bridge kept for compatibility tests. "
                    "It is not part of the v1 model-visible tool catalog."
                ),
                parameters={
                    "payload": {
                        "type": "string",
                        "description": "Plain text echoed back by the local adapter.",
                    }
                },
                toolset=TOOLSET_AGENT,
                risk_level=TOOL_RISK_SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                model_visible=False,
                idempotent=IDEMPOTENT_YES,
                executor=lambda args: args.get("payload", ""),
            )
        )


def _defer_optional_tools(registry: ToolRegistry) -> None:
    """把常用集合和领域说明一起发布到现有目录；参数：注册表；返回：无。"""
    from tools.tool_domains import configure_builtin_catalog

    configure_builtin_catalog(registry)


def _run_result(
    tool_name: str, args: dict[str, object], executor: _RunToolsExecutor
) -> object:
    visible_args = _visible_tool_args(args)
    request = RunToolsRequest(
        action=tool_name,
        tool_name=tool_name,
        arguments=_executor_request_args(tool_name, args, visible_args),
        payload=str(visible_args.get("payload", "")).strip(),
        target_scope=str(args.get("path") or args.get("scope") or "") or None,
    )
    result = executor.execute(request)
    return _to_executor_result(result)


def _visible_tool_args(args: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in args.items() if not key.startswith("__")}


def _executor_request_args(
    tool_name: str,
    args: dict[str, object],
    visible_args: dict[str, object],
) -> dict[str, object]:
    request_args = dict(visible_args)
    if tool_name in {
        "file_read",
        "list",
        "grep",
        "find_path",
        "inspect",
        "file_write",
        "file_patch",
    }:
        lease = args.get("__lease__")
        if lease is not None:
            request_args["__lease__"] = lease
        for name in (
            "__redacted_files__",
            "__session_id__",
            "__protected_edit__",
            "__validate_file__",
            "__cancellation__",
            "__file_capture__",
        ):
            if name in args:
                request_args[name] = args[name]
    return request_args


def _to_executor_result(result: RunToolsResult) -> object:
    if result.status == "ok":
        return {
            "content": result.content or result.output,
            "summary": result.summary,
            "meta": dict(result.meta),
        }
    error = result.error or result.output
    # 【文件工具】【搜索停止】仅接纳执行端确认的停止证据，不将请求取消当作已经停止
    if (
        result.tool_name in {"find_path", "inspect"}
        and result.meta.get("stopped") is True
    ):
        category = ToolErrorCategory(str(result.meta["error_category"]))
        return ToolError(category, error, retryable=False, details=dict(result.meta))
    if result.status == "denied":
        category = ToolErrorCategory.PERMISSION
    elif error.startswith("INVALID"):
        category = ToolErrorCategory.INVALID_INPUT
    else:
        category = ToolErrorCategory.UNKNOWN
    return ToolError(category, error, retryable=False)


def _read_artifact_executor(args: dict[str, object]) -> object:
    """通过当前租约读取产物页面；传参：模型参数和内部授权；返回：页面或明确错误。"""
    try:
        lease = args.get("__lease__")
        return read_artifact_page(
            _data_root_arg(args),
            str(args.get("artifact_id", "")).strip(),
            lease=lease if isinstance(lease, Lease) else None,
            mode=_artifact_mode_arg(args.get("mode")),
            offset=cast(int | None, args.get("offset")),
            limit=cast(int, args.get("limit", DEFAULT_ARTIFACT_PAGE_CHARS)),
            expected_sha256=cast(str | None, args.get("expected_sha256")),
            cursor=cast(str | None, args.get("cursor")),
        )
    except PermissionError as exc:
        return ToolError(ToolErrorCategory.PERMISSION, str(exc), retryable=False)
    except (FileNotFoundError, ValueError) as exc:
        return ToolError(ToolErrorCategory.INVALID_INPUT, str(exc), retryable=False)


def _todo_executor(args: dict[str, object]) -> object:
    """在原任务存储执行明确待办动作，完成不影响goal；参数：已校验动作和宿主身份；返回：实际待办记录。"""
    try:
        task, root = _task_id_arg(args), _data_root_arg(args)
        action = args["action"]
        if action == "list":
            return [
                {"idx": item.idx, "content": item.content, "status": item.status}
                for item in list_todos(
                    task, filter=cast(str | None, args.get("filter")), data_root=root
                )
            ]
        if action == "add":
            item = add_todo(task, str(args["content"]), data_root=root)
        else:
            item = update_todo(
                task, cast(int, args["idx"]), str(args["status"]), data_root=root
            )
        return {"idx": item.idx, "content": item.content, "status": item.status}
    except (IndexError, ValueError) as exc:
        return ToolError(ToolErrorCategory.INVALID_INPUT, str(exc), retryable=False)


def _task_id_arg(args: dict[str, object]) -> str:
    task_id = str(args.get("task_id") or args.get("__task_id__") or "").strip()
    if not task_id:
        raise ValueError("missing task_id")
    return task_id


def _data_root_arg(args: dict[str, object]) -> Path:
    value = args.get("__data_root__")
    return Path(value) if isinstance(value, str | Path) else _DATA_ROOT


def _artifact_mode_arg(value: object) -> _ArtifactReadMode | None:
    """保留未指定模式以便续读沿用游标；传参：模式值；返回：有效模式或None。"""
    if value is None:
        return None
    mode = str(value).strip()
    if mode not in {"summary", "head", "full"}:
        raise ValueError(f"invalid artifact read mode: {mode}")
    return cast(_ArtifactReadMode, mode)


def _register_readonly_queries(
    registry: ToolRegistry, file_tools: _RunToolsExecutor
) -> None:
    """【文件工具】【接口声明】从同一读取合同发布目录、路径、内容与搜索；参数：注册表和执行器；返回：无。"""
    text = {"type": "string", "minLength": 1}
    path = {**text, "description": _WORKSPACE_PATH_RULES}
    cursor = {
        **text,
        "description": "Continue only this query/resource/version using its returned next_cursor.",
    }
    entries = {
        "path": path,
        "cursor": cursor,
        "limit": {"type": "integer", "minimum": 1, "maximum": 200},
    }
    definitions = {
        "file_read": (
            "Read a UTF-8 file or extracted PDF text, not a directory. start_line is 1-based; line_count counts lines. "
            "Omit both to read from the beginning. Long lines/pages have a next_cursor: continue with path and cursor, "
            "without start_line/line_count. A changed source invalidates the cursor. Legacy character offset is not accepted.",
            {
                "path": path,
                "start_line": {"type": "integer", "minimum": 1},
                "line_count": {"type": "integer", "minimum": 1},
                "cursor": cursor,
            },
            ["path"],
        ),
        "list": (
            "List one directory. Entries have usable workspace paths and file/directory kind. "
            "Continue with the same path and next_cursor; listing a file does not read its content.",
            entries,
            ["path"],
        ),
        "grep": (
            "Search a literal substring in UTF-8 files under a path. Results identify paths and 1-based lines. "
            "Previews are not complete file reads. Continue with the same path/query and next_cursor. Binary files are reported as skipped.",
            {**entries, "query": text},
            ["path", "query"],
        ),
        "find_path": (
            "Find workspace files or directories by name fragment or glob. kind is file, directory or any (default). "
            "Results carry actual path kinds; no matches means this query completed without matches. "
            "Continue with the same query/path/kind and next_cursor.",
            {**entries, "query": text, "kind": {"enum": ["file", "directory", "any"]}},
            ["query"],
        ),
    }
    for name, (description, properties, required) in definitions.items():
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        }
        if name == "file_read":
            schema["allOf"] = [
                {
                    "if": {"required": ["cursor"]},
                    "then": {
                        "not": {
                            "anyOf": [
                                {"required": ["start_line"]},
                                {"required": ["line_count"]},
                            ]
                        }
                    },
                }
            ]
        if registry.get(name) is None:
            registry.register(
                ToolDefinition(
                    name,
                    description,
                    schema,
                    TOOLSET_FILE,
                    TOOL_RISK_SAFE,
                    True,
                    TARGET_SCOPE_PATH,
                    TOOL_SOURCE_BUILTIN,
                    idempotent=IDEMPOTENT_YES,
                    parallel_safe=True,
                    semantics=("verification",),
                    executor=partial(
                        _run_readonly_query, name=name, executor=file_tools
                    ),
                )
            )


def _run_readonly_query(
    arguments: dict[str, object], *, name: str, executor: _RunToolsExecutor
) -> object:
    """派发已声明的只读操作；参数：参数、工具名和执行器；返回：原有标准工具回执。"""
    return _run_result(name, arguments, executor)
