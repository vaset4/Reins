"""工具原始结果归一化与模型投影；作者：xxx；时间：2026-09-28 18:00:00。"""

from __future__ import annotations
import json
from collections.abc import Mapping
from dataclasses import replace
from runtime.types import RunToolsRequest, RunToolsResult
from tools.types import ToolError

MAX_TOOL_OUTPUT_CHARS = 40000
_TRUNCATE_HEAD = 30000
_TRUNCATE_TAIL = 10000

EXECUTION_DISPLAY_FIELDS = (
    "execution_state",
    "tool_error_category",
    "approval_state",
    "partial_state",
)


def execution_display_details(meta: Mapping[str, object]) -> dict[str, str]:
    """复制模型可见的执行状态，不暴露诊断字段；传参：实际结果元数据；返回：显示字段。"""
    return {
        key: str(meta[key])
        for key in EXECUTION_DISPLAY_FIELDS
        if meta.get(key) is not None
    }


def saved_execution_details(
    text: str, *, tool_name: str, call_id: str
) -> dict[str, str]:
    """从已保存工具回执读取执行状态；传参：规范正文及工具身份；返回：已有字段，普通正文无附加状态。"""
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(envelope, dict) or envelope.get("tool_name") != tool_name:
        return {}
    meta = envelope.get("meta")
    if not isinstance(meta, dict) or meta.get("call_id", call_id) != call_id:
        return {}
    return execution_display_details(meta)


def _to_run_tools_result(
    request: RunToolsRequest, raw_result: object
) -> RunToolsResult:
    """统一后端结果并保留错误、诊断和部分输出；传参：请求、真实结果；返回：工具结果。"""
    tool_name = request.tool_name or request.action
    if isinstance(raw_result, RunToolsResult):
        return replace(
            raw_result,
            action=request.action,
            tool_name=tool_name,
            target_scope=request.target_scope,
        )
    if isinstance(raw_result, ToolError):
        details = {
            key: value
            for key, value in raw_result.details.items()
            if key not in {"stdout", "stderr"}
        }
        partial_output = "\n".join(
            str(raw_result.details[key])
            for key in ("stdout", "stderr")
            if raw_result.details.get(key)
        )
        return RunToolsResult.error_result(
            action=request.action,
            tool_name=tool_name,
            error=f"{raw_result.category.value}: {raw_result.message}"
            + (f"\n{partial_output}" if partial_output else ""),
            summary="tool execution failed",
            target_scope=request.target_scope,
            meta={
                **details,
                "partial_state": raw_result.partial_state,
                "retryable": raw_result.retryable,
                "tool_error_category": raw_result.category.value,
            },
            diagnostics=raw_result.diagnostics,
        )
    if isinstance(raw_result, dict):
        content = raw_result.get(
            "content",
            {key: value for key, value in raw_result.items() if key != "diagnostics"},
        )
        rendered = (
            content
            if isinstance(content, str)
            else json.dumps(_jsonable(content), ensure_ascii=False, sort_keys=True)
        )
        return RunToolsResult.ok(
            action=request.action,
            tool_name=tool_name,
            content=rendered,
            summary=str(raw_result.get("summary", "tool executed")),
            target_scope=request.target_scope,
            meta=dict(raw_result.get("meta", {}))
            if isinstance(raw_result.get("meta"), dict)
            else {},
            diagnostics=dict(raw_result.get("diagnostics", {}))
            if isinstance(raw_result.get("diagnostics"), dict)
            else {},
        )
    rendered = (
        raw_result
        if isinstance(raw_result, str)
        else json.dumps(_jsonable(raw_result), ensure_ascii=False, sort_keys=True)
    )
    return RunToolsResult.ok(
        action=request.action,
        tool_name=tool_name,
        content=rendered,
        summary="tool executed",
        target_scope=request.target_scope,
    )


def _render_tool_conversation(result: RunToolsResult) -> str:
    """生成可回查原文的模型结果视图；传参：工具结果；返回：JSON文本。"""
    output = result.model_view if result.model_view is not None else result.output
    original_len = len(output)
    prompt_truncated = _tool_prompt_truncated(result)
    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        marker = f"\n...[truncated {original_len} chars]{_continue_hint(result)}...\n"
        output = output[:_TRUNCATE_HEAD] + marker + output[-_TRUNCATE_TAIL:]
    payload = _tool_result_envelope(
        result,
        output_preview=output,
        prompt_truncated=prompt_truncated,
    )
    if result.model_view is not None:
        payload["derived_view"] = True
    if result.annotations:
        payload["annotations"] = list(result.annotations)
    return json.dumps(
        payload,
        ensure_ascii=False,
    )


def _jsonable(value: object) -> object:
    """把后端数据转成持久化可读值；传参：原值；返回：JSON兼容投影。"""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tool_prompt_truncated(result: RunToolsResult) -> bool:
    """判断原文或模型视图是否被截断；传参：工具结果；返回：是否截断。"""
    return len(result.output or "") > MAX_TOOL_OUTPUT_CHARS


def _continue_hint(result: RunToolsResult) -> str:
    """Human-readable continuation marker appended when a tool output is
    truncated, so the model can resume via offset without parsing the
    envelope JSON."""
    meta = result.meta or {}
    next_offset = meta.get("next_offset")
    total = meta.get("total_count")
    if next_offset in (None, "", 0):
        return ""
    hint = f" | use offset={next_offset} to continue"
    if total not in (None, ""):
        hint += f" (of {total})"
    return hint


def _tool_result_envelope(
    result: RunToolsResult,
    *,
    output_preview: str,
    prompt_truncated: bool,
) -> dict[str, object]:
    """保持状态、分页与来源完整地封装模型回执；传参：结果、预览、截断标记；返回：回执映射。"""
    meta = _tool_view_metadata(result.meta, output_preview)
    source_truncated = _is_truthy_meta(result.meta.get("truncated"))
    layers = _truncation_layers(source_truncated, prompt_truncated)
    payload: dict[str, object] = {
        "tool_name": result.tool_name,
        "status": result.status,
        "output": output_preview,
        "error": output_preview
        if result.error and len(result.error) > MAX_TOOL_OUTPUT_CHARS
        else result.error,
        "output_complete": not layers,
        "source_truncated": source_truncated,
        "prompt_truncated": prompt_truncated,
        "truncation_layers": layers,
    }
    for key in ("total_count", "returned_count", "offset", "next_offset"):
        if key in result.meta:
            payload[key] = _jsonable_value(result.meta[key])
    if meta:
        payload["meta"] = meta
    return payload


def _tool_view_metadata(meta: dict[str, object], output: str) -> object:
    """正文已包含的结构化数据不再复制到模型元数据，操作原件不变；传参：元数据与正文；返回：模型所需补充字段。"""
    try:
        structured = json.loads(output)
    except json.JSONDecodeError:
        return _jsonable_value(meta)
    if not isinstance(structured, dict):
        return _jsonable_value(meta)
    return _jsonable_value(
        {
            key: value
            for key, value in meta.items()
            if key not in structured or structured[key] != value
        }
    )


def _truncation_layers(source_truncated: bool, prompt_truncated: bool) -> list[str]:
    """区分源端和模型投影截断；传参：结果元信息与状态；返回：截断层列表。"""
    layers: list[str] = []
    if source_truncated:
        layers.append("source")
    if prompt_truncated:
        layers.append("prompt")
    return layers


def _is_truthy_meta(value: object) -> bool:
    """读取结果元信息中的明确真值；传参：元信息值；返回：布尔标记。"""
    return value is True or str(value).lower() == "true"


def _jsonable_value(value: object) -> object:
    """递归投影模型回执元数据；传参：原值；返回：可序列化数据。"""
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_error_category(result: RunToolsResult) -> str | None:
    """Recover the `ToolErrorCategory.value` that `_to_run_tools_result`
    folded into `RunToolsResult.error` as `f"{category}: {message}"`."""
    if result.status == "ok" or not result.error:
        return None
    text = result.error
    if ":" in text:
        candidate, _ = text.split(":", maxsplit=1)
        candidate = candidate.strip()
        if candidate:
            return candidate
    return None
