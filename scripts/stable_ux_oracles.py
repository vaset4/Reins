"""Stable UX 七场景的纯证据判定器

作者：xxx

本模块只消费场景收集器传入的 observed evidence，不读取文件、不启动进程，
也不把模型最终回复中的“成功”表述当作验收证据。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Final, Literal

OracleStatus = Literal["pass", "fail"]


@dataclass(frozen=True, slots=True)
class OracleAssertion:
    """表示一条基于 observed evidence 的不可变验收断言"""

    id: str
    passed: bool
    actual: object


@dataclass(frozen=True, slots=True)
class OracleResult:
    """表示单个场景的纯判定结果和首个稳定错误码"""

    status: OracleStatus
    assertions: tuple[OracleAssertion, ...]
    error_code: str


_ERROR_CODES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "s1_html_content": "S1_HTML_CONTENT_INVALID",
        "s1_html_integrity": "S1_HTML_INTEGRITY_INVALID",
        "s1_approval_before_write": "S1_APPROVAL_SEQUENCE_INVALID",
        "s1_fact_chain": "S1_FACT_CHAIN_INCOMPLETE",
        "s1_session_state": "S1_SESSION_STATE_MISMATCH",
        "s2_session_continuity": "S2_SESSION_CONTINUITY_INVALID",
        "s2_page_correction": "S2_PAGE_CORRECTION_INVALID",
        "s2_html_integrity": "S2_HTML_INTEGRITY_INVALID",
        "s2_conversation": "S2_CONVERSATION_INCOMPLETE",
        "s2_summary": "S2_SUMMARY_STALE",
        "s2_approval": "S2_APPROVAL_MISSING",
        "s3_raw_response": "S3_RAW_RESPONSE_MISSING",
        "s3_model_visible_error": "S3_ERROR_NOT_MODEL_VISIBLE",
        "s3_error_fact": "S3_ERROR_FACT_MISSING",
        "s3_terminal_lifecycle": "S3_TERMINAL_LIFECYCLE_MISSING",
        "s3_summary": "S3_SUMMARY_POLLUTED",
        "s4_retryable_error": "S4_RETRYABLE_ERROR_MISSING",
        "s4_strategy_change": "S4_STRATEGY_UNCHANGED",
        "s4_error_feedback": "S4_ERROR_NOT_MODEL_VISIBLE",
        "s4_budget": "S4_TOOL_BUDGET_EXCEEDED",
        "s4_approval": "S4_APPROVAL_BYPASSED",
        "s4_terminal_lifecycle": "S4_TERMINAL_LIFECYCLE_MISSING",
        "s4_fact_chain": "S4_TOOL_FACT_CHAIN_INCOMPLETE",
        "s5_pending_preserved": "S5_PENDING_EVIDENCE_CHANGED",
        "s5_query_evidence": "S5_QUERY_EVIDENCE_INVALID",
        "s5_retry_denied": "S5_RETRY_PERMISSION_NOT_ENFORCED",
        "s5_resume_run": "S5_RESUME_RUN_MISSING",
        "s5_no_early_side_effect": "S5_SIDE_EFFECT_BEFORE_DECISION",
        "s5_terminal_lifecycle": "S5_TERMINAL_LIFECYCLE_MISSING",
        "s6_transport_success": "S6_MCP_TRANSPORT_RESULT_MISSING",
        "s6_risk": "S6_MCP_RISK_INVALID",
        "s6_denied_path": "S6_MCP_DENIED_PATH_MISSING",
        "s6_approval": "S6_MCP_APPROVAL_MISSING",
        "s6_fact_chain": "S6_MCP_FACT_CHAIN_INCOMPLETE",
        "s6_process_exit": "S6_MCP_PROCESS_NOT_CLOSED",
        "s7_extract": "S7_BROWSER_EXTRACT_INVALID",
        "s7_screenshot": "S7_SCREENSHOT_INVALID",
        "s7_artifact_reference": "S7_SCREENSHOT_UNREFERENCED",
        "s7_fact_chain": "S7_BROWSER_FACT_CHAIN_INCOMPLETE",
        "s7_terminal_lifecycle": "S7_TERMINAL_LIFECYCLE_MISSING",
        "common_isolation": "SCENARIO_ISOLATION_EVIDENCE_INVALID",
    }
)


def evaluate(scenario_id: str, observed: Mapping[str, object]) -> OracleResult:
    """按场景 ID 计算纯 evidence 判定

    参数：scenario_id 为 scenario_1 至 scenario_7；observed 为收集器生成的只读证据映射
    返回：不可变 OracleResult；未知场景直接抛出 ValueError
    """
    oracle = _ORACLES.get(scenario_id)
    if oracle is None:
        raise ValueError(f"unknown stable UX scenario: {scenario_id}")
    return oracle(observed)


def _scenario_1(observed: Mapping[str, object]) -> OracleResult:
    """判定创建 Reins index.html 的文件、审批、facts 与 session 证据"""
    digest = _text(observed.get("html_sha256"))
    approvals = _records(observed.get("approval_records"))
    facts = _records(observed.get("facts"))
    state = _mapping(observed.get("session_state"))
    run_ids = _strings(observed.get("run_ids"))
    valid_approvals = tuple(
        row
        for row in approvals
        if _text(row.get("tool")) == "file_write"
        and _text(row.get("risk")) == "confirm"
        and _text(row.get("target")).replace("\\", "/").endswith("/index.html")
        and row.get("target_existed") is False
    )
    approval_ok = (
        len(valid_approvals) == 1
        and observed.get("approval_count") == 1
        and observed.get("approval_before_exists") is True
    )
    required = {"run:start", "llm:response", "tool:request", "tool:response"}
    events = _strings(observed.get("fact_events"))
    assertions = (
        OracleAssertion(
            "s1_html_content",
            observed.get("html_exists") is True
            and observed.get("html_contains_reins") is True,
            (observed.get("html_exists"), observed.get("html_contains_reins")),
        ),
        OracleAssertion("s1_html_integrity", _valid_sha256(digest), digest),
        OracleAssertion("s1_approval_before_write", approval_ok, len(valid_approvals)),
        OracleAssertion(
            "s1_fact_chain",
            required.issubset(set(events))
            and "run:lifecycle" in events
            and _latest_lifecycle(facts) == "done",
            events,
        ),
        OracleAssertion(
            "s1_session_state",
            bool(run_ids)
            and _text(state.get("last_run_id")) == run_ids[-1]
            and _text(state.get("last_run_status")) == "done",
            (_text(state.get("last_run_id")), _text(state.get("last_run_status"))),
        ),
        _isolation_assertion(observed),
    )
    return _result(assertions)


def _scenario_2(observed: Mapping[str, object]) -> OracleResult:
    """判定同 session 的页面纠正、conversation 与 summary 证据"""
    run_ids = _strings(observed.get("run_ids"))
    page_text = _text(observed.get("page_text"))
    before_hash = _text(observed.get("html_before_sha256"))
    after_hash = _text(observed.get("html_after_sha256"))
    user_inputs = _strings(observed.get("user_inputs"))
    conversation = _conversation_texts(observed.get("conversation_rows"))
    conversation_text = _text(observed.get("conversation_text"))
    summary = _text(observed.get("summary"))
    personal_markers = ("个人介绍", "about me", "个人简历")
    correction_ok = (
        observed.get("page_contains_reins") is True
        and observed.get("page_personal_primary") is False
        and "reins" in page_text.casefold()
        and not any(marker in page_text.casefold() for marker in personal_markers)
    )
    identity_ok = (
        observed.get("session_continuous") is True
        and len(run_ids) == 2
        and run_ids[0] != run_ids[1]
    )
    assertions = (
        OracleAssertion("s2_session_continuity", identity_ok, run_ids),
        OracleAssertion("s2_page_correction", correction_ok, len(page_text)),
        OracleAssertion(
            "s2_html_integrity",
            _valid_sha256(before_hash)
            and _valid_sha256(after_hash)
            and before_hash != after_hash,
            (before_hash, after_hash),
        ),
        OracleAssertion(
            "s2_conversation",
            len(user_inputs) == 2
            and all(
                item in conversation or item in conversation_text
                for item in user_inputs
            ),
            conversation or (conversation_text,),
        ),
        OracleAssertion(
            "s2_summary",
            "reins" in summary.casefold()
            and bool(user_inputs)
            and summary.strip() != user_inputs[0].strip(),
            summary,
        ),
        OracleAssertion(
            "s2_approval",
            observed.get("approval_count") == 1,
            observed.get("approval_count"),
        ),
        _isolation_assertion(observed),
    )
    return _result(assertions)


def _scenario_3(observed: Mapping[str, object]) -> OracleResult:
    """判定协议错误 raw evidence、模型可见恢复和明确终态"""
    invalid_raw = _text(observed.get("bare_protocol_error"))
    raw_files = _strings(observed.get("raw_response_files"))
    raw_refs = _strings(observed.get("raw_response_refs"))
    raw_text = _text(observed.get("raw_response_text"))
    model_requests = observed.get("model_requests")
    facts = _records(observed.get("facts"))
    summary = _text(observed.get("summary"))
    raw_ok = (
        bool(invalid_raw)
        and invalid_raw in raw_text
        and bool(raw_files)
        and len(set(raw_refs)) >= 2
        and all(reference.startswith("evidence:") for reference in raw_refs)
        and all(_is_relative_path(path) for path in raw_files)
    )
    visible_ok = observed.get(
        "protocol_error_visible_to_model"
    ) is True and _contains_text(model_requests, "invalid_model_protocol")
    error_fact_ok = any(
        fact.get("event") == "llm:response"
        and _contains_text(fact.get("summary"), "invalid_model_protocol")
        and any(
            _contains_text(fact.get("summary"), reference) for reference in raw_refs
        )
        for fact in facts
    )
    lifecycle = _text(observed.get("final_lifecycle"))
    assertions = (
        OracleAssertion("s3_raw_response", raw_ok, raw_files),
        OracleAssertion(
            "s3_model_visible_error",
            visible_ok,
            observed.get("protocol_error_visible_to_model"),
        ),
        OracleAssertion("s3_error_fact", error_fact_ok, _fact_events(facts)),
        OracleAssertion(
            "s3_terminal_lifecycle", lifecycle in {"done", "paused"}, lifecycle
        ),
        OracleAssertion(
            "s3_summary",
            bool(summary)
            and summary.strip() != invalid_raw.strip()
            and not summary.startswith("MODEL_PROTOCOL_ERROR"),
            summary,
        ),
        _isolation_assertion(observed),
    )
    return _result(assertions)


def _scenario_4(observed: Mapping[str, object]) -> OracleResult:
    """判定 retryable 工具失败回灌、改策、预算和审批边界"""
    calls = _records(observed.get("tool_calls"))
    responses = _records(observed.get("tool_responses"))
    model_requests = observed.get("model_requests")
    facts = _records(observed.get("facts"))
    budget = _integer(observed.get("max_steps"))
    first = responses[0] if responses else {}
    second = responses[1] if len(responses) > 1 else {}
    first_error = _text(first.get("error"))
    retryable_ok = (
        len(responses) == 2
        and _text(first.get("status")) == "error"
        and _mapping(first.get("meta")).get("retryable") is True
        and bool(first_error)
    )
    strategy_ok = (
        len(calls) >= 2
        and observed.get("strategy_changed") is True
        and calls[0] != calls[-1]
        and _text(second.get("status")) in {"success", "ok"}
    )
    feedback_ok = bool(first_error) and _contains_text(model_requests, first_error)
    lifecycle = _text(observed.get("final_lifecycle"))
    assertions = (
        OracleAssertion("s4_retryable_error", retryable_ok, len(responses)),
        OracleAssertion("s4_strategy_change", strategy_ok, tuple(map(dict, calls))),
        OracleAssertion("s4_error_feedback", feedback_ok, bool(model_requests)),
        OracleAssertion(
            "s4_budget",
            budget > 0
            and len(calls) == observed.get("attempt_count")
            and len(calls) <= budget,
            (len(calls), budget),
        ),
        OracleAssertion(
            "s4_approval",
            observed.get("confirm_side_effect_bypassed") is False,
            observed.get("confirm_side_effect_bypassed"),
        ),
        OracleAssertion("s4_terminal_lifecycle", lifecycle == "done", lifecycle),
        OracleAssertion(
            "s4_fact_chain",
            _tool_chain_present(facts, "acceptance_strategy"),
            _tool_fact_names(facts),
        ),
        _isolation_assertion(observed),
    )
    return _result(assertions)


def _scenario_5(observed: Mapping[str, object]) -> OracleResult:
    """核对自主查询与结束、原件保留和真实恢复拒绝；参数：观测证据；返回：验收判定。"""
    source = _mapping(observed.get("source_checkpoint"))
    operation = _mapping(observed.get("source_operation"))
    after = _mapping(observed.get("source_operation_after"))
    pending = _mapping(source.get("pending_tool_call"))
    runs = _strings(observed.get("run_ids"))
    query_facts, retry_facts = (
        _records(observed.get("query_facts")),
        _records(observed.get("retry_facts")),
    )
    preserved = [
        bool(pending),
        source == _mapping(observed.get("source_checkpoint_after")),
        pending.get("tool_name") == "file_write",
        bool(_text(pending.get("call_id"))),
        bool(_text(_mapping(pending.get("args")).get("path"))),
        _mapping(operation.get("call")) == _mapping(after.get("call")),
        {
            key: _mapping(operation.get("call")).get(key)
            for key in ("tool_name", "args", "call_id")
        }
        == pending,
        all(
            operation.get(key) == after.get(key)
            for key in ("session_id", "run_id", "operation_id")
        ),
        operation.get("state") == after.get("state") == "not_started",
        "retry_operation_id" not in after,
    ]
    digest = _text(observed.get("before_sha256"))
    identity = [
        bool(source.get("checkpoint_id")),
        bool(source.get("run_id")),
        bool(operation.get("operation_id")),
        len(runs) == len(set(runs)) == 2,
        source.get("run_id") not in runs,
        source.get("session_id") == observed.get("session_id"),
        operation.get("session_id") == source.get("session_id"),
        operation.get("run_id") == source.get("run_id"),
    ]
    terminal = [
        _latest_lifecycle(query_facts) == "done",
        _latest_lifecycle(retry_facts) == "done",
        any(
            _mapping(row.get("summary")).get("has_final") is True
            for row in query_facts
            if row.get("event") == "llm:response"
        ),
        _mapping(observed.get("checkpoint_after_query")).get("run_id")
        == (runs[0] if runs else None),
        _mapping(observed.get("checkpoint_after_retry")).get("run_id")
        == (runs[-1] if runs else None),
        all(
            _mapping(observed.get(key)).get("session_id") == source.get("session_id")
            for key in ("checkpoint_after_query", "checkpoint_after_retry")
        ),
    ]
    hashes = [observed.get("after_query_sha256"), observed.get("after_retry_sha256")]
    assertions = (
        OracleAssertion("s5_pending_preserved", all(preserved), preserved),
        OracleAssertion(
            "s5_query_evidence",
            _s5_query_evidence(observed, pending, operation),
            observed.get("query_operations"),
        ),
        OracleAssertion(
            "s5_retry_denied",
            _s5_retry_denied(observed, pending, operation),
            observed.get("retry_operations"),
        ),
        OracleAssertion("s5_resume_run", all(identity), identity),
        OracleAssertion(
            "s5_no_early_side_effect",
            _valid_sha256(digest)
            and hashes == [digest, digest]
            and _mapping(pending.get("args")).get("expected_sha256") == digest
            and "file_write" not in _tool_fact_names(query_facts),
            hashes,
        ),
        OracleAssertion("s5_terminal_lifecycle", all(terminal), terminal),
        _isolation_assertion(observed),
    )
    return _result(assertions)


def _s5_query_evidence(
    observed: Mapping[str, object],
    pending: Mapping[str, object],
    operation: Mapping[str, object],
) -> bool:
    """查询成功回执和后续模型请求均携带原调用；参数：观测、待决和原操作；返回：是否完整。"""
    rows = _records(observed.get("query_operations"))
    if len(rows) != 1:
        return False
    row = rows[0]
    result = _mapping(row.get("result"))
    try:
        payload = _mapping(json.loads(_text(result.get("content"))))
    except (ValueError, TypeError):
        return False
    returned = _records(payload.get("operations"))
    if len(returned) != 1:
        return False
    call = _mapping(returned[0].get("call"))
    keys = ("tool_name", "args", "call_id")
    model = _records(observed.get("model_requests"))
    runs = _strings(observed.get("run_ids"))
    return all(
        (
            result.get("status") == "ok",
            _mapping(row.get("call")).get("tool_name") == "operation_status",
            _mapping(row.get("call")).get("args")
            == {"operation_id": operation.get("operation_id")},
            row.get("run_id") == (runs[0] if runs else None),
            row.get("session_id") == operation.get("session_id"),
            all(
                returned[0].get(key) == operation.get(key)
                for key in ("run_id", "session_id")
            ),
            returned[0].get("operation_id") == operation.get("operation_id"),
            {key: call.get(key) for key in keys}
            == {key: pending.get(key) for key in keys},
            len(model) >= 2,
            _s5_feedback(model[-1] if model else {}, row, payload),
            _s5_receipt_matches(_records(observed.get("query_facts")), row),
        )
    )


def _s5_retry_denied(
    observed: Mapping[str, object],
    pending: Mapping[str, object],
    operation: Mapping[str, object],
) -> bool:
    """显式重试绑定原操作并经过目标工具审批拒绝；参数：观测、待决和原操作；返回：是否禁止副作用。"""
    rows, batches = (
        _records(observed.get("retry_operations")),
        _records(observed.get("approval_batches")),
    )
    if len(rows) != 1 or len(batches) != 1:
        return False
    row, batch = rows[0], batches[0]
    requests = _records(batch.get("requests"))
    if len(requests) != 1:
        return False
    call, result = _mapping(row.get("call")), _mapping(row.get("result"))
    meta, authorization = (
        _mapping(result.get("meta")),
        _mapping(row.get("authorization")),
    )
    runs = _strings(observed.get("run_ids"))
    checks = [
        call.get("tool_name") == "resume_operation",
        call.get("args")
        == {"action": "retry", "operation_id": operation.get("operation_id")},
        row.get("run_id") == (runs[-1] if runs else None),
        row.get("session_id") == operation.get("session_id"),
        row.get("state") == "not_started",
        result.get("status") == "error",
        meta.get("execution_state") == "not_started",
        meta.get("tool_error_category") == "permission",
        meta.get("approval_state") == "denied",
        authorization.get("decision") == "deny",
        authorization.get("source") == "user_action",
        authorization.get("batch_id") == batch.get("batch_id"),
        requests[0].get("operation_id") == row.get("operation_id"),
        requests[0].get("tool") == pending.get("tool_name"),
        requests[0].get("args") == pending.get("args"),
        _s5_receipt_matches(_records(observed.get("retry_facts")), row),
    ]
    return all(checks)


def _s5_feedback(
    request: Mapping[str, object],
    operation: Mapping[str, object],
    payload: Mapping[str, object],
) -> bool:
    """核对下一请求实际回灌的本次查询结果；参数：请求、查询操作及结果正文；返回：是否同源。"""
    call = _mapping(operation.get("call"))
    messages = _records(_mapping(request.get("sources")).get("messages"))
    rows = [
        row
        for row in messages
        if row.get("kind") == "tool_result"
        and row.get("call_id") == call.get("call_id")
    ]
    if len(rows) != 1:
        return False
    feedback = _mapping(rows[0].get("fed_back"))
    content = _records(feedback.get("content"))
    try:
        envelope = _mapping(
            json.loads("".join(_text(part.get("text")) for part in content))
        )
        output = _mapping(json.loads(_text(envelope.get("output"))))
    except (ValueError, TypeError):
        return False
    return all(
        (
            feedback.get("status") == "success",
            feedback.get("tool_name") == "operation_status",
            feedback.get("call_id") == call.get("call_id"),
            envelope.get("status") == "ok",
            output == payload,
            _mapping(envelope.get("meta")).get("operation_id")
            == operation.get("operation_id"),
            request.get("run_id") == operation.get("run_id"),
            request.get("session_id") == operation.get("session_id"),
        )
    )


def _s5_receipt_matches(
    facts: Sequence[Mapping[str, object]], operation: Mapping[str, object]
) -> bool:
    """核对唯一工具请求与回执的真实操作身份和状态；参数：事实及操作；返回：是否配对。"""
    rows = [
        row for row in facts if row.get("event") in {"tool:request", "tool:response"}
    ]
    if [row.get("event") for row in rows] != ["tool:request", "tool:response"]:
        return False
    call = _mapping(operation.get("call"))
    return all(
        (
            all(
                _mapping(row.get("tool")).get("name") == call.get("tool_name")
                for row in rows
            ),
            all(
                _mapping(row.get("tool")).get("call_id") == call.get("call_id")
                for row in rows
            ),
            all(row.get("run_id") == operation.get("run_id") for row in rows),
            all(row.get("session_id") == operation.get("session_id") for row in rows),
            rows[1].get("operation_id") == operation.get("operation_id"),
            _mapping(rows[1].get("tool")).get("status")
            == _mapping(operation.get("result")).get("status"),
        )
    )


def _scenario_6(observed: Mapping[str, object]) -> OracleResult:
    """判定本地 echo MCP 的真实 transport、风险、拒绝与退出证据"""
    approvals = _records(observed.get("approval_records"))
    facts = _records(observed.get("facts"))
    allowed = _mapping(observed.get("mcp_allowed_result"))
    denied = _mapping(observed.get("mcp_denied_result"))
    risk = _text(observed.get("mcp_risk")).casefold()
    allowed_ok = (
        observed.get("mcp_transport_used") is True
        and _text(allowed.get("status")) in {"success", "ok"}
        and _contains_text(allowed, "reins-echo")
    )
    denied_ok = _text(denied.get("status")) in {"denied", "error"} and (
        _contains_text(denied, "permission")
        or _contains_text(denied, "unavailable")
        or _contains_text(denied, "not available")
        or _contains_text(denied, "not allowed")
    )
    approvals_ok = any(
        _text(row.get("tool")) == "mcp_echo_echo"
        and _text(row.get("risk")) == "confirm"
        for row in approvals
    )
    mcp_fact_names = tuple(
        name
        for name in _tool_fact_names(facts)
        if name.startswith("mcp_") or name == "capabilities"
    )
    assertions = (
        OracleAssertion("s6_transport_success", allowed_ok, dict(allowed)),
        OracleAssertion("s6_risk", risk == "confirm", risk),
        OracleAssertion("s6_denied_path", denied_ok, dict(denied)),
        OracleAssertion("s6_approval", approvals_ok, len(approvals)),
        OracleAssertion(
            "s6_fact_chain",
            len(mcp_fact_names) >= 4 and _latest_lifecycle(facts) == "done",
            mcp_fact_names,
        ),
        OracleAssertion(
            "s6_process_exit",
            observed.get("mcp_process_stopped") is True,
            observed.get("mcp_process_stopped"),
        ),
        _isolation_assertion(observed),
    )
    return _result(assertions)


def _scenario_7(observed: Mapping[str, object]) -> OracleResult:
    """判定 Chromium extract、真实截图内容和 artifact 引用链"""
    extracted_text = _text(observed.get("browser_text"))
    screenshot_hash = _text(observed.get("screenshot_sha256"))
    bytes_hash = _text(observed.get("screenshot_bytes_sha256"))
    screenshot_path = _text(observed.get("screenshot_path"))
    artifact_files = _strings(observed.get("artifact_files"))
    observed_refs = tuple(
        artifact_id
        for row in _records(observed.get("artifact_refs"))
        if (artifact_id := _text(row.get("artifact_id")))
    )
    facts = _records(observed.get("facts"))
    screenshot_ok = (
        observed.get("screenshot_exists") is True
        and _integer(observed.get("screenshot_size")) > 0
        and _valid_sha256(screenshot_hash)
        and screenshot_hash == bytes_hash
        and _is_relative_path(screenshot_path)
    )
    tool_names = _tool_fact_names(facts)
    refs = _artifact_refs(facts)
    required_tools = {"browser_navigate", "browser_extract", "browser_screenshot"}
    assertions = (
        OracleAssertion(
            "s7_extract",
            all(
                text in extracted_text.casefold()
                for text in ("hello", "email", "send", "docs")
            ),
            extracted_text,
        ),
        OracleAssertion(
            "s7_screenshot",
            screenshot_ok,
            (observed.get("screenshot_size"), screenshot_hash, screenshot_path),
        ),
        OracleAssertion(
            "s7_artifact_reference",
            observed.get("screenshot_referenced") is True
            and bool(observed_refs)
            and set(observed_refs).issubset(set(refs))
            and screenshot_path in artifact_files,
            (observed_refs, refs, artifact_files),
        ),
        OracleAssertion(
            "s7_fact_chain", required_tools.issubset(set(tool_names)), tool_names
        ),
        OracleAssertion(
            "s7_terminal_lifecycle",
            _text(observed.get("final_lifecycle")) == "done"
            and _latest_lifecycle(facts) == "done",
            observed.get("final_lifecycle"),
        ),
        _isolation_assertion(observed),
    )
    return _result(assertions)


def _result(assertions: tuple[OracleAssertion, ...]) -> OracleResult:
    """把断言序列收敛为 pass/fail，并返回首个失败断言的稳定错误码"""
    failed = next((assertion for assertion in assertions if not assertion.passed), None)
    if failed is None:
        return OracleResult(status="pass", assertions=assertions, error_code="")
    return OracleResult(
        status="fail",
        assertions=assertions,
        error_code=_ERROR_CODES[failed.id],
    )


def _mapping(value: object) -> Mapping[str, object]:
    """把合法映射原样作为只读视图返回，畸形值返回空映射以触发失败断言"""
    return value if isinstance(value, Mapping) else MappingProxyType({})


def _records(value: object) -> tuple[Mapping[str, object], ...]:
    """把全为映射的序列转为 tuple；任一畸形项都让该组证据整体无效"""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    if any(not isinstance(item, Mapping) for item in value):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _strings(value: object) -> tuple[str, ...]:
    """把全为字符串的序列转为 tuple；畸形序列返回空值并由 oracle fail closed"""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    if any(not isinstance(item, str) for item in value):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _text(value: object) -> str:
    """返回原始字符串证据；非字符串不做隐式转换"""
    return value if isinstance(value, str) else ""


def _integer(value: object) -> int:
    """返回原始整数证据；布尔值和其他类型视为无效"""
    return value if type(value) is int else -1


def _valid_sha256(value: str) -> bool:
    """确认 observed digest 是完整的小写 SHA-256，而不是成功占位文本"""
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _isolation_assertion(observed: Mapping[str, object]) -> OracleAssertion:
    """校验场景根和所有 evidence path 都保持隔离且使用相对路径"""
    roots = tuple(
        _text(observed.get(key)) for key in ("project_root", "data_root", "home_root")
    )
    fact_files = _strings(observed.get("fact_files"))
    artifact_files = _strings(observed.get("artifact_files"))
    paths = (*roots, *fact_files, *artifact_files)
    passed = (
        observed.get("roots_are_isolated") is True
        and len(set(roots)) == 3
        and all(_is_relative_path(path) for path in paths)
        and bool(fact_files)
    )
    return OracleAssertion(
        "common_isolation", passed, (roots, fact_files, artifact_files)
    )


def _fact_events(facts: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """返回 facts 中观察到的稳定排序事件名集合"""
    return tuple(sorted({_text(fact.get("event")) for fact in facts}))


def _latest_lifecycle(facts: Sequence[Mapping[str, object]]) -> str:
    """从 facts 末尾向前读取最后一个显式 run lifecycle"""
    for fact in reversed(facts):
        if fact.get("event") == "run:lifecycle":
            return _text(fact.get("lifecycle"))
    return ""


def _contains_text(value: object, needle: str) -> bool:
    """递归检查结构化 observed evidence 是否包含指定原始文本"""
    if isinstance(value, str):
        return needle in value
    if isinstance(value, Mapping):
        return any(_contains_text(item, needle) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return any(_contains_text(item, needle) for item in value)
    return False


def _conversation_texts(value: object) -> tuple[str, ...]:
    """从 conversation 行中提取原始 content 字符串"""
    return tuple(_text(row.get("content")) for row in _records(value))


def _event_index(facts: Sequence[Mapping[str, object]], event: str) -> int:
    """返回指定 event 的首个 facts 下标，不存在时返回 -1"""
    return next(
        (index for index, fact in enumerate(facts) if fact.get("event") == event),
        -1,
    )


def _fact_index(facts: Sequence[Mapping[str, object]], text: str) -> int:
    """返回首个包含指定文本的 facts 下标，不存在时返回 -1"""
    return next(
        (index for index, fact in enumerate(facts) if _contains_text(fact, text)),
        -1,
    )


def _tool_fact_names(facts: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """从 tool request/response facts 提取工具名，并保留实际事件顺序"""
    names: list[str] = []
    for fact in facts:
        if fact.get("event") not in {"tool:request", "tool:response"}:
            continue
        names.append(_text(_mapping(fact.get("tool")).get("name")))
    return tuple(names)


def _tool_chain_present(facts: Sequence[Mapping[str, object]], tool_name: str) -> bool:
    """确认指定工具留下两组 request/response 且 run 明确进入 done"""
    requests = 0
    responses = 0
    for fact in facts:
        name = _text(_mapping(fact.get("tool")).get("name"))
        if name != tool_name:
            continue
        requests += fact.get("event") == "tool:request"
        responses += fact.get("event") == "tool:response"
    return requests == 2 and responses == 2 and _latest_lifecycle(facts) == "done"


def _artifact_refs(facts: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """从 tool response facts 提取实际 artifact_id 引用"""
    refs: list[str] = []
    for fact in facts:
        if fact.get("event") != "tool:response":
            continue
        tool = _mapping(fact.get("tool"))
        for row in _records(tool.get("artifact_refs")):
            artifact_id = _text(row.get("artifact_id"))
            if artifact_id:
                refs.append(artifact_id)
    return tuple(refs)


def _is_relative_path(value: str) -> bool:
    """同时按 POSIX 与 Windows 语义确认 evidence path 为非空相对路径"""
    return (
        bool(value)
        and not PurePosixPath(value).is_absolute()
        and not PureWindowsPath(value).is_absolute()
    )


_ORACLES: Final[Mapping[str, Callable[[Mapping[str, object]], OracleResult]]] = (
    MappingProxyType(
        {
            "scenario_1": _scenario_1,
            "scenario_2": _scenario_2,
            "scenario_3": _scenario_3,
            "scenario_4": _scenario_4,
            "scenario_5": _scenario_5,
            "scenario_6": _scenario_6,
            "scenario_7": _scenario_7,
        }
    )
)


__all__ = ["OracleAssertion", "OracleResult", "OracleStatus", "evaluate"]
