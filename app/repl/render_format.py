"""Formatting helpers for REPL stream-event rendering."""

from __future__ import annotations

import json
from collections.abc import Mapping
from time import perf_counter

from app.repl.status import ReplStatusSummary


TOOL_OUTPUT_LIMIT = 1024
ARG_PREVIEW_LIMIT = 80
JSON_ARG_LIMIT = 200
ERROR_OUTPUT_LIMIT = 240


def format_normal_status_hint(summary: ReplStatusSummary) -> str:
    continue_text = "是" if summary.can_continue else "否"
    return (
        f"状态提示: {summary.label} ({summary.category}); "
        f"可继续: {continue_text}; 下一步: {summary.next_step}"
    )


def format_process_line(
    *,
    step: int,
    tool_name: str,
    args: Mapping[str, object],
    is_error: bool,
    error_category: str | None,
) -> str:
    target = tool_target_text(tool_name, args)
    target_part = f" {target}" if target and target != "(none)" else ""
    result = _process_result(is_error, error_category)
    return f"过程 {step}. {tool_name}{target_part} -> {result}"


def compact_error_output(output: str) -> str:
    line = _first_non_empty_line(output)
    if not line:
        return ""
    return _shorten(line, ERROR_OUTPUT_LIMIT)


def format_tool_args(args: Mapping[str, object]) -> str:
    """Render tool args as compact JSON for trace panels."""
    try:
        rendered = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except Exception:
        rendered = repr(args)
    if len(rendered) > JSON_ARG_LIMIT:
        return rendered[:JSON_ARG_LIMIT] + " …"
    return rendered


def truncate_tool_output(output: str) -> str:
    """Truncate verbose tool output for trace panels."""
    if len(output) <= TOOL_OUTPUT_LIMIT:
        return output
    head = output[:TOOL_OUTPUT_LIMIT]
    omitted = len(output) - TOOL_OUTPUT_LIMIT
    return (
        f"{head}\n... [{omitted} more chars; inline output truncated; "
        "full output is in run evidence/task artifacts/]"
    )


def tool_panel_target_summary(
    tool_name: str,
    args: Mapping[str, object],
) -> str:
    return f"target={tool_target_text(tool_name, args)}"


def tool_target_text(tool_name: str, args: Mapping[str, object]) -> str:
    if tool_name == "grep":
        return _grep_target(args)
    if tool_name in {"file_read", "list"}:
        return _string_arg(args, "path", "file_path") or "(workspace)"
    if tool_name in {"terminal", "terminal_run"}:
        return _string_arg(args, "command") or "(none)"
    if tool_name.startswith("web"):
        return _string_arg(args, "url", "query") or "(none)"
    return _first_arg(args, "path", "file_path", "query", "url", "command") or "(none)"


def tool_activity_explanation(tool_name: str) -> str:
    if tool_name == "file_read":
        return "读取文件上下文"
    if tool_name == "grep":
        return "搜索工作区匹配"
    if tool_name == "list":
        return "列出目录以定位候选文件"
    if tool_name in {"file_write", "file_patch"}:
        return "写入目标文件，属于执行动作"
    if tool_name in {"terminal", "terminal_run"}:
        return "运行终端命令以验证或收集证据"
    if tool_name.startswith("web"):
        return "读取网页或网络上下文"
    if tool_name.startswith("memory"):
        return "读取或更新长期记忆上下文"
    return "执行工具动作"


def pop_elapsed(started: dict[str, float], call_id: str) -> float | None:
    started_at = started.pop(call_id, None)
    if started_at is None:
        return None
    return perf_counter() - started_at


def format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "elapsed=unknown"
    return f"{seconds:.2f}s"


def convergence_pause_detail(reason: str) -> str:
    text = reason.casefold()
    if "convergence" not in text and "read-only" not in text:
        return ""
    return "read-only streak reached convergence pause; review the original goal.\n"


def _process_result(is_error: bool, error_category: str | None) -> str:
    if not is_error:
        return "ok"
    if error_category:
        return f"error {error_category}"
    return "error"


def _grep_target(args: Mapping[str, object]) -> str:
    path = _string_arg(args, "path") or "."
    query = _string_arg(args, "query", "pattern")
    if not query:
        return path
    return f"{path} {_quote(query)}"


def _string_arg(args: Mapping[str, object], *names: str) -> str:
    value = _first_arg(args, *names)
    return _shorten(value, ARG_PREVIEW_LIMIT)


def _first_arg(args: Mapping[str, object], *names: str) -> str:
    for name in names:
        value = args.get(name)
        if isinstance(value, str) and value:
            return value
    return ""


def _first_non_empty_line(output: str) -> str:
    for line in output.splitlines():
        text = line.strip()
        if text:
            return text
    return ""


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + " …"


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)
