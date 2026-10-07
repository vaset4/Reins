from __future__ import annotations

import json
from collections.abc import Collection, Mapping

from llm.messages import JsonValue, ToolCallPart, thaw_json_value
from llm.types import LLMPlan, ModelError
from runtime.types import RunToolsRequest
from tools.tool_registry import ToolRegistry, get_default_tool_registry

# 模型这一轮一个工具都没点名时回喂给它的错误文案
_EMPTY_TOOL_CALL_SUMMARY = "empty native tool call list"


def parse_tool_call_parts(
    parts: tuple[ToolCallPart, ...],
    *,
    allowed_tool_names: Collection[str],
    registry: ToolRegistry | None = None,
) -> LLMPlan:
    """把 typed Provider 栈交来的工具调用内容块解析成执行计划。

    作者：xxx
    时间：2026-08-30 15:20:00
    传参：parts 为 AssistantMessage 里的 ToolCallPart 序列；allowed_tool_names 为本轮准许的工具名；registry 为工具注册表
    返回：承载首条工具请求的 LLMPlan；工具不准许或参数校验失败时返回协议错误
    """
    registry = registry or get_default_tool_registry()
    allowed = frozenset(allowed_tool_names)
    # 1. 模型这一轮走了 tool_call 停止原因却没交出任何调用块，按空响应回喂
    if not parts:
        return _protocol_error(
            summary=_EMPTY_TOOL_CALL_SUMMARY,
            category="empty_response",
            raw_summary=_EMPTY_TOOL_CALL_SUMMARY,
        )

    # 2. 逐块解析；ToolCallPart 已保证 call_id/tool_name 非空、arguments 是 JSON object，故此处只做准许校验与参数校验
    parsed_plans = [
        _parse_tool_call_part(
            part=part,
            registry=registry,
            allowed_tool_names=allowed,
        )
        for part in parts
    ]

    # 3. 参数或权限错误属于对应调用，保留原call_id让执行器逐项回填
    for index, plan in enumerate(parsed_plans):
        if plan.model_error is not None:
            part = parts[index]
            parsed_plans[index] = LLMPlan(
                run_tools_request=RunToolsRequest(
                    action=part.tool_name,
                    tool_name=part.tool_name,
                    arguments=_thaw_tool_arguments(part.arguments),
                    call_id=part.call_id,
                    validation_error=plan.model_error.raw_summary
                    or plan.model_error.summary,
                )
            )

    if len(parsed_plans) == 1:
        return parsed_plans[0]

    # 4. 同批调用由共同执行边界按后端并发能力安排，结果身份不随完成顺序改变
    first_plan = parsed_plans[0]
    first_plan.prompt_context["pending_tool_calls"] = [
        plan.run_tools_request
        for plan in parsed_plans
        if plan.run_tools_request is not None
    ]
    return first_plan


def parse_llm_response(
    raw_text: str,
    *,
    protocol_mode: str = "text_json",
    allowed_tool_names: Collection[str],
    registry: ToolRegistry | None = None,
) -> LLMPlan:
    """按回复协议区分最终正文与动作请求。

    传参：raw_text为模型正文，protocol_mode为协议模式，allowed_tool_names为本轮工具范围，registry为注册表
    返回：最终答案、合法工具请求或明确协议错误
    """
    if protocol_mode == "native_tool_calls":
        stripped = raw_text.strip()
        if stripped and not stripped.startswith("{"):
            return LLMPlan(final_output=stripped)

    registry = registry or get_default_tool_registry()
    data = _parse_json_object(raw_text)
    if isinstance(data, LLMPlan):
        return data

    # 1. 原生工具协议下，无顶层动作类型的合法JSON属于用户答案，保留完整正文
    if protocol_mode == "native_tool_calls" and "type" not in data:
        return LLMPlan(final_output=raw_text.strip())

    response_type = data.get("type")
    if response_type == "final":
        return _parse_final_response(data)
    if response_type == "run_tools" and protocol_mode == "native_tool_calls":
        return _protocol_error(
            summary="text JSON run_tools is not allowed in native_tool_calls mode",
            category="invalid_model_protocol",
            raw_summary="native text response attempted run_tools",
        )
    if response_type == "run_tools":
        return _parse_run_tools_response(
            data=data,
            registry=registry,
            allowed_tool_names=allowed_tool_names,
        )

    if response_type == "error":
        message = str(data.get("message", "")).strip() or "unknown model error"
        # 【Agent运行】【停止归因】模型明确报错必须进入失败路径，不能用错误正文伪装成功答案
        return LLMPlan(
            model_error=ModelError.create(
                category="model_reported_error",
                summary=message,
                stage="response",
            )
        )
    return _protocol_error(
        summary="invalid response type",
        category="invalid_model_protocol",
        raw_summary=f"type={response_type}",
    )


def _parse_tool_call_part(
    *,
    part: ToolCallPart,
    registry: ToolRegistry,
    allowed_tool_names: Collection[str],
) -> LLMPlan:
    """校验单个 typed 工具调用块并回填其调用标识。

    作者：xxx
    时间：2026-08-30 15:20:00
    传参：part 为单个 ToolCallPart；registry 为工具注册表；allowed_tool_names 为本轮准许的工具名
    返回：承载该工具请求的 LLMPlan；不准许或参数校验失败时返回协议错误
    """
    # 1. 工具名与注册表口径对齐：注册表按去空白后的名字查定义，这里先去空白再判准许，避免同一个名字两边判断不一致
    tool_name = part.tool_name.strip()
    plan = _parse_structured_tool_request(
        tool_name=tool_name,
        arguments=_thaw_tool_arguments(part.arguments),
        registry=registry,
        allowed_tool_names=allowed_tool_names,
        error_summary="invalid native tool_call request",
        validation_detail=f"tool={tool_name}",
    )
    # 2. 回填 call_id，工具结果要靠它跟本次调用配对
    if plan.run_tools_request is not None:
        plan.run_tools_request.call_id = part.call_id
    return plan


def _thaw_tool_arguments(arguments: Mapping[str, JsonValue]) -> dict[str, object]:
    """把 ToolCallPart 深冻结的参数解冻成普通可变 dict。

    作者：xxx
    时间：2026-08-30 15:20:00
    传参：arguments 为 ToolCallPart 深冻结后的只读参数
    返回：不与内容块共享容器的普通 dict
    注：RunToolsRequest.arguments 声明是 dict，嵌套的 mappingproxy 传下去不会当场报错，
        但下游序列化工具参数时会 TypeError，写入类工具改参数时会 item assignment 失败
    """
    return {key: thaw_json_value(value) for key, value in arguments.items()}


def _parse_json_object(raw_text: str) -> dict[str, object] | LLMPlan:
    try:
        data = json.loads(raw_text)
    except Exception as exc:
        return _protocol_error(
            summary=f"invalid json: {exc}",
            category="invalid_model_protocol",
            raw_summary=str(exc),
        )
    if isinstance(data, dict):
        return data
    return _protocol_error(
        summary="invalid json object",
        category="invalid_model_protocol",
        raw_summary="response must be an object",
    )


def _parse_final_response(data: dict[str, object]) -> LLMPlan:
    content = str(data.get("content", "")).strip()
    if content:
        return LLMPlan(final_output=content)
    return _protocol_error(
        summary="empty final content",
        category="empty_response",
        raw_summary="empty final content",
    )


def _parse_run_tools_response(
    *,
    data: dict[str, object],
    registry: ToolRegistry,
    allowed_tool_names: Collection[str],
) -> LLMPlan:
    tool_name = str(data.get("tool", "")).strip()
    arguments = data.get("arguments")
    if tool_name and isinstance(arguments, dict):
        return _parse_structured_tool_request(
            tool_name=tool_name,
            arguments=dict(arguments),
            registry=registry,
            allowed_tool_names=allowed_tool_names,
            error_summary="invalid run_tools request",
            validation_detail=(
                f"tool={tool_name}; structured run_tools validation failed"
            ),
        )
    return _parse_legacy_tool_request(data, registry, allowed_tool_names)


def _parse_structured_tool_request(
    *,
    tool_name: str,
    arguments: dict[str, object],
    registry: ToolRegistry,
    allowed_tool_names: Collection[str],
    error_summary: str,
    validation_detail: str,
) -> LLMPlan:
    if not _is_allowed_tool(registry, tool_name, allowed_tool_names):
        return _protocol_error(
            summary=error_summary,
            category="invalid_tool_arguments",
            raw_summary=f"tool={tool_name}; not allowed in current request",
        )
    request = registry.validate_model_request(
        tool_name=tool_name,
        arguments=dict(arguments),
    )
    if isinstance(request, RunToolsRequest):
        return LLMPlan(run_tools_request=request)
    return _protocol_error(
        summary=error_summary,
        category="invalid_tool_arguments",
        raw_summary=f"{validation_detail}; {request.message}",
    )


def _parse_legacy_tool_request(
    data: dict[str, object],
    registry: ToolRegistry,
    allowed_tool_names: Collection[str],
) -> LLMPlan:
    action = str(data.get("action", "")).strip()
    payload = str(data.get("payload", "")).strip()
    if not action or not payload:
        return _protocol_error(
            summary="invalid run_tools request",
            category="invalid_model_protocol",
            raw_summary="legacy run_tools request missing action or payload",
        )
    if registry.get(action) is None:
        return _protocol_error(
            summary="invalid run_tools request",
            category="invalid_model_protocol",
            raw_summary=f"unknown action: {action}",
        )
    return _validate_legacy_tool_request(
        action=action,
        payload=payload,
        registry=registry,
        allowed_tool_names=allowed_tool_names,
    )


def _validate_legacy_tool_request(
    *,
    action: str,
    payload: str,
    registry: ToolRegistry,
    allowed_tool_names: Collection[str],
) -> LLMPlan:
    if not _is_allowed_tool(registry, action, allowed_tool_names):
        return _protocol_error(
            summary="invalid run_tools request",
            category="invalid_tool_arguments",
            raw_summary=f"tool={action}; not allowed in current request",
        )
    request = registry.validate_model_request(
        tool_name=action,
        arguments={"payload": payload},
    )
    if isinstance(request, RunToolsRequest):
        return LLMPlan(run_tools_request=request)
    return _protocol_error(
        summary="invalid run_tools request",
        category="invalid_model_protocol",
        raw_summary=f"legacy run_tools validation failed: {request.message}",
    )


def _is_allowed_tool(
    registry: ToolRegistry,
    tool_name: str,
    allowed_tool_names: Collection[str],
) -> bool:
    allowed = frozenset(allowed_tool_names)
    if tool_name in allowed:
        return True
    if any(item.endswith("*") and tool_name.startswith(item[:-1]) for item in allowed):
        return True
    definition = registry.get(tool_name)
    return bool(definition is not None and definition.name in allowed)


def _protocol_error(
    *,
    summary: str,
    category: str,
    raw_summary: str,
) -> LLMPlan:
    error = ModelError.create(
        category=category,
        summary=summary,
        raw_summary=raw_summary,
        stage="parse",
    )
    return LLMPlan(final_output=error.render_output(), model_error=error)
