"""模型响应进入解析与持久证据前保护敏感文件候选。

作者：xxx
时间：2026-09-24 22:00:00
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import path_security
from llm.messages import (
    AssistantContentPart,
    AssistantMessage,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    freeze_json_object,
    thaw_json_value,
)
from llm.provider_result import ProviderCallResult
from llm.types import LLMPlan
from runtime.lease import Lease
from runtime.types import RunToolsRequest
from tools.redacted_files import RedactedFiles
from tools.config_syntax import is_private_key


def protect_provider_result(
    result: ProviderCallResult,
    *,
    files: RedactedFiles,
    context: Mapping[str, object],
    request_id: str,
) -> ProviderCallResult:
    """成功和失败尝试共用秘密接纳边界；传参：响应、宿主、上下文和请求；返回：可解析且可保存的响应。"""
    session_id = str(context.get("session_id") or request_id)
    lease = context.get("capability_lease")
    message = result.message or result.partial_message
    protected = (
        _protect_message(
            message,
            files=files,
            session_id=session_id,
            request_id=request_id,
            lease=lease if isinstance(lease, Lease) else None,
        )
        if message is not None
        else None
    )
    error = result.error
    if error is not None:
        error = replace(
            error, summary=files.sanitize_text(error.summary, session_id=session_id)
        )
    return replace(
        result,
        message=protected if result.message is not None else None,
        partial_message=protected if result.partial_message is not None else None,
        error=error,
    )


def protect_plan(
    plan: LLMPlan,
    *,
    files: RedactedFiles,
    session_id: str,
    request_id: str,
    lease: Lease,
) -> LLMPlan:
    """各类模型客户端返回均经公共证据边界；传参：计划、宿主与身份；返回：可保存的计划副本。"""
    request = plan.run_tools_request
    if request is not None:
        request = _protect_request(
            request,
            files=files,
            session_id=session_id,
            request_id=request_id,
            lease=lease,
        )
    prompt_context = dict(plan.prompt_context)
    pending = prompt_context.get("pending_tool_calls")
    if isinstance(pending, list):
        prompt_context["pending_tool_calls"] = [
            _protect_request(
                item,
                files=files,
                session_id=session_id,
                request_id=request_id,
                lease=lease,
            )
            if isinstance(item, RunToolsRequest)
            else item
            for item in pending
        ]
    message = plan.assistant_message
    if message is not None:
        message = _protect_message(
            message,
            files=files,
            session_id=session_id,
            request_id=request_id,
            lease=lease,
        )
    error = plan.model_error
    if error is not None:
        error = replace(
            error,
            summary=files.sanitize_text(error.summary, session_id=session_id),
            raw_summary=files.sanitize_text(error.raw_summary, session_id=session_id),
        )
    attempts = tuple(
        replace(
            attempt,
            response=_sanitize_mapping(
                attempt.response, files=files, session_id=session_id
            ),
        )
        for attempt in plan.model_attempts
    )
    return replace(
        plan,
        run_tools_request=request,
        assistant_message=message,
        model_error=error,
        model_attempts=attempts,
        prompt_context=prompt_context,
        final_output=files.sanitize_text(plan.final_output, session_id=session_id)
        if plan.final_output is not None
        else None,
        reasoning_content=files.sanitize_text(
            plan.reasoning_content, session_id=session_id
        ),
        render_text_to_model=files.sanitize_text(
            plan.render_text_to_model, session_id=session_id
        ),
        raw_model_response=_sanitize_mapping(
            plan.raw_model_response, files=files, session_id=session_id
        ),
    )


def _protect_request(
    request: RunToolsRequest,
    *,
    files: RedactedFiles,
    session_id: str,
    request_id: str,
    lease: Lease,
) -> RunToolsRequest:
    """保护公共计划中的单次工具调用；传参：请求与宿主身份；返回：脱敏请求副本。"""
    args = protect_arguments(
        request.tool_name or request.action,
        request.arguments,
        files=files,
        session_id=session_id,
        request_id=request_id,
        call_id=request.call_id or "plan",
        lease=lease,
    )
    return replace(
        request,
        arguments=args,
        payload=files.sanitize_text(request.payload, session_id=session_id),
        validation_error=files.sanitize_text(
            request.validation_error, session_id=session_id
        )
        if request.validation_error
        else None,
    )


def _protect_message(
    message: AssistantMessage,
    *,
    files: RedactedFiles,
    session_id: str,
    request_id: str,
    lease: Lease | None,
) -> AssistantMessage:
    """先接纳工具秘密，再过滤同消息副本；传参：消息及宿主身份；返回：安全消息。"""
    parts: list[AssistantContentPart] = []
    for part in message.content:
        if isinstance(part, ToolCallPart):
            args = protect_arguments(
                part.tool_name,
                cast(dict[str, object], thaw_json_value(part.arguments)),
                files=files,
                session_id=session_id,
                request_id=request_id,
                call_id=part.call_id,
                lease=lease,
            )
            parts.append(
                replace(
                    part, arguments=freeze_json_object(args, path="protected.arguments")
                )
            )
        elif isinstance(part, TextPart) and '"secret_replacements"' in part.text:
            parts.append(
                replace(
                    part,
                    text=_protect_text_json(
                        part.text,
                        files=files,
                        session_id=session_id,
                        request_id=request_id,
                        lease=lease if isinstance(lease, Lease) else None,
                    ),
                )
            )
        elif isinstance(part, TextPart) and part.text.lstrip().startswith("{"):
            parts.append(
                replace(
                    part,
                    text=_protect_text_json(
                        part.text,
                        files=files,
                        session_id=session_id,
                        request_id=request_id,
                        lease=lease if isinstance(lease, Lease) else None,
                    ),
                )
            )
        else:
            parts.append(part)
    # 【模型响应】【秘密隔离】先收齐工具中的秘密，再移除同一消息正文和思考中的副本
    safe_parts = tuple(
        replace(part, text=files.sanitize_text(part.text, session_id=session_id))
        if isinstance(part, (TextPart, ThinkingPart))
        else part
        for part in parts
    )
    protected = replace(message, content=safe_parts)
    if protected.provider_state is not None:
        protected = replace(
            protected,
            provider_state=replace(
                protected.provider_state,
                payload=_sanitize_mapping(
                    protected.provider_state.payload, files=files, session_id=session_id
                ),
            ),
        )
    return protected


def protect_arguments(
    tool: str,
    args: dict[str, object],
    *,
    files: RedactedFiles,
    session_id: str,
    request_id: str,
    call_id: str,
    lease: Lease | None,
) -> dict[str, object]:
    """保留工具协议和非法参数的失败语义；传参：原调用及权限身份；返回：秘密值替换为内存引用的参数。"""
    from tools.tool_registry import _normalize_tool_arguments

    args = _normalize_tool_arguments(tool_name=tool.strip(), arguments=args)
    protected = files.protect_arguments(
        args, session_id=session_id, request_id=request_id, call_id=call_id
    )
    path = args.get("path")
    sensitive = (
        isinstance(path, str)
        and lease is not None
        and path_security.uses_redacted_files(Path(path), lease)
    )
    private_key = any(
        isinstance(args.get(name), str) and is_private_key(str(args[name]))
        for name in ("content", "old_text", "new_text")
    )
    if tool.strip() in {"file_write", "file_patch"} and (
        sensitive or private_key or args.get("view_id")
    ):
        protected = files.protect_file_text(
            protected, session_id=session_id, request_id=request_id, call_id=call_id
        )
    return protected


def _protect_text_json(
    text: str,
    *,
    files: RedactedFiles,
    session_id: str,
    request_id: str,
    lease: Lease | None,
) -> str:
    """保护旧文本协议中的结构化候选，破损JSON仍保持失败；传参：正文及调用身份；返回：安全正文。"""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        if not any(
            f'"{name}"' in text
            for name in ("secret_replacements", "file_write", "file_patch")
        ):
            return text
        reference = files.stage_text(
            text,
            session_id=session_id,
            identity=(request_id, "text", "invalid_response"),
        )
        return '{"protected_invalid_response":' + reference
    if not isinstance(value, dict) or value.get("type") != "run_tools":
        return text
    args = value.get("arguments")
    if not isinstance(args, dict):
        return text
    protected = protect_arguments(
        str(value.get("tool", value.get("action", ""))),
        args,
        files=files,
        session_id=session_id,
        request_id=request_id,
        call_id="text",
        lease=lease,
    )
    return json.dumps({**value, "arguments": protected}, ensure_ascii=False)


def _sanitize_mapping(
    value: Mapping[str, Any], *, files: RedactedFiles, session_id: str
) -> dict[str, Any]:
    """过滤供应商连续状态中的已知秘密副本；传参：状态、宿主和会话；返回：独立安全状态。"""
    return cast(
        dict[str, Any],
        _sanitize_value(thaw_json_value(value), files=files, session_id=session_id),
    )


def _sanitize_value(value: Any, *, files: RedactedFiles, session_id: str) -> Any:
    """在既有JSON形状内过滤文本，不改变字段类型；传参：值、宿主和会话；返回：安全副本。"""
    if isinstance(value, str):
        return files.sanitize_text(value, session_id=session_id)
    if isinstance(value, dict):
        return {
            files.sanitize_text(key, session_id=session_id): _sanitize_value(
                item, files=files, session_id=session_id
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _sanitize_value(item, files=files, session_id=session_id) for item in value
        ]
    return value
