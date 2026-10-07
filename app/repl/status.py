"""User-facing status summaries for the REPL.

This module stays deliberately small and read-only: it interprets run facts,
run errors, checkpoints, and task summary layers for display. It must not
change the runtime state machine or tool/model behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

from runtime.run_facts import latest_lifecycle_from_facts
from runtime.stream_events import ModelRetryScheduled

StatusCategory = Literal[
    "waiting_user",
    "waiting_approval",
    "recoverable_failure",
    "strategy_available",
    "hard_blocked",
    "degraded_text_mode",
    "continuable",
    "done",
]

# REPL status categories are a user-facing recovery map, not the provider retry
# policy. Some provider non-retryable errors are still continuable from the REPL
# when the user can change strategy, while configuration/capability boundaries
# are hard blocks even if the original error category came from a tool.
_HARD_ERROR_CATEGORIES = {
    "missing_config",
    "permission",
    "auth",
    "authentication",
    "authorization",
    "billing",
    "model_not_found",
    "context_overflow",
}
_RECOVERABLE_ERROR_CATEGORIES = {
    "empty_response",
    "invalid_model_protocol",
    "invalid_provider_response",
    "overloaded",
    "payload_too_large",
    "rate_limited",
    "server_error",
    "text_protocol_parse_error",
    "timeout",
    "transport",
    "transport_error",
}
_STRATEGY_ERROR_CATEGORIES = {
    "invalid_input",
    "tool_execution_error",
    "tool_error",
    "unknown",
}

_RETRY_REASONS = {
    "transport_error": "模型连接中断",
    "timeout": "模型响应超时",
    "rate_limited": "模型服务限流",
    "overloaded": "模型服务繁忙",
    "server_error": "模型服务暂时异常",
}


def format_model_retry(event: ModelRetryScheduled) -> str:
    """展示真实等待时长与下一次尝试；参数：模型重试状态；返回：中文状态提示。"""
    reason = _RETRY_REASONS.get(event.error_category, "模型请求暂时失败")
    return (
        f"{reason}，等待 {event.wait_seconds:.1f} 秒后自动重试"
        f"（第 {event.attempt_index}/{event.max_attempts} 次尝试）。输入 /stop 可停止。"
    )


@dataclass(frozen=True, slots=True)
class ReplStatusSummary:
    category: StatusCategory
    label: str
    detail: str
    next_step: str
    can_continue: bool
    run_id: str = ""
    session_id: str = ""
    evidence: str = ""


def classify_run_status(
    facts: Sequence[Mapping[str, Any]],
    *,
    errors: Sequence[Mapping[str, Any]] = (),
    checkpoint_state: str = "",
    session_status: str = "",
) -> ReplStatusSummary:
    """Classify the latest known run into a small user-facing state."""
    run_id = _first_non_empty(facts, "run_id")
    session_id = _first_non_empty(facts, "session_id")
    terminal = _latest_terminal_status(facts) or session_status.lower()
    latest_state = _latest_state(facts).upper()
    latest_error = _latest_error(facts, errors)
    error_category = _error_category(latest_error)
    evidence = _error_message(latest_error)

    if terminal == "waiting_user":
        return ReplStatusSummary(
            category="waiting_user",
            label="waiting for user",
            detail="The latest run is waiting for user input.",
            next_step="Reply with the requested information to continue the run.",
            can_continue=True,
            run_id=run_id,
            session_id=session_id,
            evidence=evidence,
        )
    if terminal == "waiting_approval" or _is_approval_state(latest_state):
        return ReplStatusSummary(
            category="waiting_approval",
            label="waiting for approval",
            detail="A tool wants permission before it changes anything.",
            next_step="Approve or deny the request; the run can continue from that point.",
            can_continue=True,
            run_id=run_id,
            session_id=session_id,
            evidence=evidence,
        )
    if _has_approval_required_event(facts):
        return ReplStatusSummary(
            category="waiting_approval",
            label="waiting for approval",
            detail="The latest run reached an approval boundary.",
            next_step="Review the requested tool action before continuing.",
            can_continue=True,
            run_id=run_id,
            session_id=session_id,
            evidence=evidence,
        )
    if _has_degraded_text_marker(facts, errors) and terminal != "failed":
        return ReplStatusSummary(
            category="degraded_text_mode",
            label="degraded text mode",
            detail="The model/tool protocol is using a text-compatible fallback.",
            next_step="You can continue, but native tool-call support should be preferred.",
            can_continue=True,
            run_id=run_id,
            session_id=session_id,
            evidence=evidence,
        )
    if terminal == "done":
        return ReplStatusSummary(
            category="done",
            label="done",
            detail="The latest run completed.",
            next_step="Send a new request or use the current task summary to continue related work.",
            can_continue=True,
            run_id=run_id,
            session_id=session_id,
            evidence=evidence,
        )
    if _latest_tool_failed(facts):
        return _tool_failure_summary(
            facts,
            error_category=error_category,
            evidence=evidence,
            run_id=run_id,
            session_id=session_id,
        )
    if terminal == "failed" or latest_error:
        if _is_hard_block(error_category, evidence):
            return ReplStatusSummary(
                category="hard_blocked",
                label="hard blocked",
                detail=_hard_block_detail(error_category, evidence),
                next_step=(
                    "Fix the configuration, permission, or unavailable capability "
                    "before retrying this path."
                ),
                can_continue=False,
                run_id=run_id,
                session_id=session_id,
                evidence=evidence,
            )
        if error_category in _RECOVERABLE_ERROR_CATEGORIES:
            return ReplStatusSummary(
                category="recoverable_failure",
                label="recoverable failure",
                detail="The run failed at a recoverable model or transport boundary.",
                next_step="Use /status for context, then continue or retry with a safer/smaller request.",
                can_continue=True,
                run_id=run_id,
                session_id=session_id,
                evidence=evidence,
            )
        return ReplStatusSummary(
            category="recoverable_failure",
            label="recoverable failure",
            detail="The run stopped with an error, but no hard block was identified.",
            next_step="Check the recovery hint and continue after adjusting the request if needed.",
            can_continue=True,
            run_id=run_id,
            session_id=session_id,
            evidence=evidence,
        )
    if terminal == "paused" or checkpoint_state:
        return ReplStatusSummary(
            category="continuable",
            label="continuable",
            detail="The latest run paused with a recovery point.",
            next_step="Use /resume or send the next prompt to continue from the saved state.",
            can_continue=True,
            run_id=run_id,
            session_id=session_id,
            evidence=evidence,
        )
    return ReplStatusSummary(
        category="continuable",
        label="ready",
        detail="No blocking status is visible for the current session.",
        next_step="Send the next prompt, or use /status after a run to see more detail.",
        can_continue=True,
        run_id=run_id,
        session_id=session_id,
        evidence=evidence,
    )


def explain_model_stop(
    stop_reason: str | None,
    content: str | None,
) -> ReplStatusSummary | None:
    if not stop_reason:
        return None
    if stop_reason == "protocol_error":
        return _recoverable_model_summary(
            category="invalid_model_protocol",
            evidence=content or "",
        )
    if stop_reason.startswith("model_error:"):
        category = stop_reason.split(":", maxsplit=1)[1]
        if _is_hard_block(category, content or ""):
            return ReplStatusSummary(
                category="hard_blocked",
                label="hard blocked",
                detail=_hard_block_detail(category, content or ""),
                next_step="Fix the model/provider configuration before retrying this path.",
                can_continue=False,
                evidence=content or "",
            )
        return _recoverable_model_summary(category=category, evidence=content or "")
    return None


def explain_tool_error(
    error_category: str | None,
    output: str,
    *,
    tool_name: str = "",
) -> ReplStatusSummary:
    category = (error_category or "unknown").lower()
    if _is_hard_block(category, output):
        if _is_browser_boundary(category, output):
            return _browser_boundary_summary(category, output, tool_name=tool_name)
        return ReplStatusSummary(
            category="hard_blocked",
            label="hard blocked",
            detail=_hard_block_detail(category, output),
            next_step="Resolve the permission/configuration issue before rerunning this tool path.",
            can_continue=False,
            evidence=output,
        )
    return ReplStatusSummary(
        category="strategy_available",
        label="strategy available",
        detail=f"The tool failed, but another safe strategy may work{f' for {tool_name}' if tool_name else ''}.",
        next_step="Continue by correcting arguments, narrowing scope, or using a read-only check first.",
        can_continue=True,
        evidence=output,
    )


def explain_exception_status(exc: BaseException) -> ReplStatusSummary:
    text = str(exc)
    lowered = text.lower()
    if "llm_client" in lowered or "missing_config" in lowered or "api key" in lowered:
        return ReplStatusSummary(
            category="hard_blocked",
            label="hard blocked",
            detail="The run cannot start until model configuration is available.",
            next_step="Configure the model/provider, then retry the request.",
            can_continue=False,
            evidence=text,
        )
    if "stopiteration" in lowered or "generator raised" in lowered:
        return ReplStatusSummary(
            category="recoverable_failure",
            label="recoverable failure",
            detail="The run failed at a recoverable model or transport boundary.",
            next_step="Use /status for context, then continue or retry with a safer/smaller request.",
            can_continue=True,
            evidence=text,
        )
    return ReplStatusSummary(
        category="recoverable_failure",
        label="recoverable failure",
        detail="The REPL run stopped before a clean terminal state.",
        next_step="Use /status to inspect the latest saved state, then continue or retry after adjusting the request.",
        can_continue=True,
        evidence=text,
    )


def read_latest_run_errors(
    data_root: Path | str,
    *,
    session_id: str,
    run_id: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """按真实会话与运行读取近期错误；参数：数据根、归属与条数；返回：有序错误，损坏明确失败。"""
    from runtime.run_evidence import RunEvidenceStore

    records = RunEvidenceStore(data_root).list_records(
        session_id=session_id, run_id=run_id, kind="error"
    )
    return [row["payload"] for row in records[-limit:]]


def format_status_hint(summary: ReplStatusSummary) -> str:
    continue_text = "yes" if summary.can_continue else "no"
    return (
        f"Status hint: {summary.label} ({summary.category}); "
        f"can continue: {continue_text}; next: {summary.next_step}"
    )


def _tool_failure_summary(
    facts: Sequence[Mapping[str, Any]],
    *,
    error_category: str,
    evidence: str,
    run_id: str,
    session_id: str,
) -> ReplStatusSummary:
    if _is_hard_block(error_category, evidence):
        if _is_browser_boundary(error_category, evidence):
            return _browser_boundary_summary(
                error_category,
                evidence,
                tool_name=_latest_tool_name(facts),
                run_id=run_id,
                session_id=session_id,
            )
        return ReplStatusSummary(
            category="hard_blocked",
            label="hard blocked",
            detail=_hard_block_detail(error_category, evidence),
            next_step="Fix the permission/configuration issue before retrying this tool path.",
            can_continue=False,
            run_id=run_id,
            session_id=session_id,
            evidence=evidence,
        )
    return ReplStatusSummary(
        category="strategy_available",
        label="strategy available",
        detail="The last tool attempt failed, but the run can try a different safe approach.",
        next_step="Continue by correcting tool arguments, reading context first, or narrowing the request.",
        can_continue=True,
        run_id=run_id,
        session_id=session_id,
        evidence=evidence or _latest_tool_output(facts),
    )


def _recoverable_model_summary(*, category: str, evidence: str) -> ReplStatusSummary:
    detail = "The model response failed in a recoverable way."
    if category == "empty_response":
        detail = "The model returned an empty response."
    elif category in {"invalid_model_protocol", "invalid_provider_response"}:
        detail = "The model response did not match the expected protocol."
    elif category == "rate_limited":
        detail = "API 限流，正在等待后重试..."
    elif category == "overloaded":
        detail = "供应商过载，正在等待后重试..."
    elif category == "server_error":
        detail = "供应商服务端错误，正在重试..."
    elif category in {"timeout", "transport_error"}:
        detail = "连接中断，正在重试..."
    return ReplStatusSummary(
        category="recoverable_failure",
        label="recoverable failure",
        detail=detail,
        next_step="Continue or retry after the system records the error; use a smaller request if it repeats.",
        can_continue=True,
        evidence=evidence,
    )


def _latest_terminal_status(facts: Sequence[Mapping[str, Any]]) -> str:
    lifecycle = latest_lifecycle_from_facts(facts)
    if lifecycle:
        return str(lifecycle["lifecycle"])
    for fact in reversed(facts):
        if fact.get("event") == "state:transition":
            state = str(fact.get("to_state", "")).lower()
            if state in {"done", "paused", "failed"}:
                return state
    return ""


def _latest_state(facts: Sequence[Mapping[str, Any]]) -> str:
    lifecycle = latest_lifecycle_from_facts(facts)
    if lifecycle:
        return str(lifecycle["lifecycle"])
    for fact in reversed(facts):
        if fact.get("event") == "state:transition":
            return str(fact.get("to_state", "")).upper()
    return ""


def _is_approval_state(latest_state: str) -> bool:
    # approval 判定只认运行事实里的 lifecycle：真实审批边界会把 run_facts 的
    # latest_state 写成 WAITING_APPROVAL。checkpoint 存储里的 state 快照虽然
    # 真实审批时也同源写成 waiting_approval，但那是冗余镜像——两者由
    # _record_lifecycle_boundary 同一次调用同值写入，靠 latest_state 即可兜住，
    # 显示层不再从 checkpoint 存储推导 approval 状态。
    approval_states = {"AWAITING_APPROVAL", "WAITING_APPROVAL"}
    return latest_state in approval_states


def _has_approval_required_event(facts: Sequence[Mapping[str, Any]]) -> bool:
    for fact in reversed(facts):
        event = str(fact.get("event", ""))
        if event == "approval:required":
            return True
        if event == "approval:decision":
            return False
        if event == "run:lifecycle":
            lifecycle = str(fact.get("lifecycle", "")).lower()
            return lifecycle == "waiting_approval"
        if event == "state:transition" and str(fact.get("to_state", "")).upper() in {
            "DONE",
            "PAUSED",
            "FAILED",
        }:
            return False
    return False


def _has_degraded_text_marker(
    facts: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
) -> bool:
    joined = " ".join(_flatten_text([*facts, *errors])).lower()
    return any(
        marker in joined
        for marker in (
            "degraded_text_mode",
            "text_protocol_parse_error",
            "text compatible",
            "text-compatible",
            "text_compat",
        )
    )


def _latest_tool_failed(facts: Sequence[Mapping[str, Any]]) -> bool:
    for fact in reversed(facts):
        if fact.get("event") != "tool:response":
            continue
        tool = fact.get("tool")
        if isinstance(tool, Mapping):
            return str(tool.get("status", "ok")) != "ok" or bool(tool.get("error"))
        return str(fact.get("status", "ok")) != "ok" or bool(fact.get("error"))
    return False


def _latest_tool_output(facts: Sequence[Mapping[str, Any]]) -> str:
    for fact in reversed(facts):
        if fact.get("event") == "tool:response" and isinstance(
            fact.get("tool"), Mapping
        ):
            tool = fact["tool"]
            return str(tool.get("error") or tool.get("output_summary") or "")
        if fact.get("event") == "tool:response":
            return str(fact.get("error") or fact.get("output") or "")
    return ""


def _latest_tool_name(facts: Sequence[Mapping[str, Any]]) -> str:
    for fact in reversed(facts):
        if fact.get("event") != "tool:response":
            continue
        tool = fact.get("tool")
        if isinstance(tool, Mapping):
            return str(tool.get("name") or tool.get("tool_name") or "")
        return str(fact.get("tool_name", ""))
    return ""


def _latest_error(
    facts: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """选取最近的错误回执；传参：运行事实与错误记录；返回：错误映射，无错误时为空。"""
    if errors:
        return errors[-1]
    fact: Mapping[str, Any]
    for fact in reversed(facts):
        if fact.get("event") == "tool:response" and isinstance(
            fact.get("tool"), Mapping
        ):
            tool = fact["tool"]
            if str(tool.get("status", "ok")) != "ok" or tool.get("error"):
                return cast(Mapping[str, Any], tool)
        if fact.get("event") == "tool:response" and (
            str(fact.get("status", "ok")) != "ok" or fact.get("error")
        ):
            return fact
        summary = fact.get("summary")
        if isinstance(summary, Mapping) and summary.get("error"):
            error = summary.get("error")
            if isinstance(error, Mapping):
                return error
            return {"message": str(error)}
    return {}


def _error_category(error: Mapping[str, Any]) -> str:
    for key in ("category", "error_category"):
        value = error.get(key)
        if value:
            return str(value).lower()
    message = _error_message(error).lower()
    for category in (
        _HARD_ERROR_CATEGORIES
        | _RECOVERABLE_ERROR_CATEGORIES
        | _STRATEGY_ERROR_CATEGORIES
    ):
        if category in message:
            return category
    if "mcp" in message and ("unavailable" in message or "not configured" in message):
        return "mcp_unavailable"
    if "browser" in message and (
        "unavailable" in message or "not configured" in message
    ):
        return "browser_unavailable"
    return ""


def _error_message(error: Mapping[str, Any]) -> str:
    for key in ("message", "error", "summary", "output_summary"):
        value = error.get(key)
        if value:
            return str(value)
    return ""


def _is_hard_block(category: str, evidence: str) -> bool:
    lowered = f"{category} {evidence}".lower()
    if category in _HARD_ERROR_CATEGORIES:
        return True
    return any(
        marker in lowered
        for marker in (
            "approval_denied",
            "requires_permanent_grant",
            "tool_risk_denied",
            "path_security_deny",
            "network_disabled",
            "mcp_unavailable",
            "mcp_not_configured",
            "browser_unavailable",
            "browser_domain_denied",
            "browser_file_path_denied",
            "not configured",
            "missing config",
            "api key",
            "insufficient balance",
            "insufficient quota",
            "quota exceeded",
            "model not found",
            "unauthorized",
            "forbidden",
            "401",
            "403",
            "permission denied",
        )
    )


def _is_browser_boundary(category: str, evidence: str) -> bool:
    lowered = f"{category} {evidence}".lower()
    return "browser_" in lowered or "browser capability" in lowered


def _browser_boundary_summary(
    category: str,
    evidence: str,
    *,
    tool_name: str = "",
    run_id: str = "",
    session_id: str = "",
) -> ReplStatusSummary:
    lowered = f"{category} {evidence}".lower()
    if "browser_domain_denied" in lowered:
        detail = "The browser capability refused this domain by policy."
        next_step = "Use a different source, ask for a permitted URL, or continue without browser navigation."
    elif "browser_file_path_denied" in lowered:
        detail = "The browser capability refused this local file because it is outside the read lease."
        next_step = (
            "Use an allowed fixture path or continue with a normal file-read strategy."
        )
    else:
        detail = "The requested browser capability is unavailable in this environment."
        next_step = "Install or enable the browser runtime, or continue with web fetch/read-only evidence instead."
    suffix = f" for {tool_name}" if tool_name else ""
    return ReplStatusSummary(
        category="hard_blocked",
        label="browser capability boundary",
        detail=f"{detail}{suffix}.",
        next_step=next_step,
        can_continue=False,
        run_id=run_id,
        session_id=session_id,
        evidence=evidence,
    )


def _hard_block_detail(category: str, evidence: str) -> str:
    lowered = f"{category} {evidence}".lower()
    if category == "auth" or "unauthorized" in lowered or "forbidden" in lowered:
        return "需要检查 API 配置：认证失败或密钥无效。请检查密钥来源和供应商配置。"
    if category == "billing":
        return "需要检查 API 配置：额度不足或账单问题。请检查供应商账户余额。"
    if category == "model_not_found":
        return "模型不存在或不可用，请检查模型名称和供应商配置（/model show）。"
    if category == "missing_config":
        return "缺少必要的模型配置。请使用 /model set 或检查环境变量。"
    if "mcp" in lowered:
        return "The requested MCP capability is not configured or is unavailable."
    if "browser" in lowered:
        return "The requested browser capability is not configured or is unavailable."
    if "permission" in lowered or "approval" in lowered:
        return "The requested action is blocked by permission or approval rules."
    if "config" in lowered or "api key" in lowered:
        return "Required provider or capability configuration is missing."
    return (
        "The latest run reached a hard boundary that needs user/configuration action."
    )


def _first_non_empty(facts: Sequence[Mapping[str, Any]], key: str) -> str:
    for fact in facts:
        value = str(fact.get(key, "")).strip()
        if value:
            return value
    return ""


def _flatten_text(values: Sequence[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            for key, item in value.items():
                result.append(str(key))
                result.extend(_flatten_text([item]))
        elif isinstance(value, (list, tuple)):
            result.extend(_flatten_text(value))
        else:
            result.append(str(value))
    return result


__all__ = [
    "ReplStatusSummary",
    "classify_run_status",
    "explain_exception_status",
    "explain_model_stop",
    "explain_tool_error",
    "format_status_hint",
    "read_latest_run_errors",
]
