from __future__ import annotations

from pathlib import Path

import pytest

from app.repl.status import (
    classify_run_status,
    explain_exception_status,
    explain_model_stop,
    explain_tool_error,
    read_latest_run_errors,
)
from runtime.run_evidence import RunEvidenceStore


def _state(to_state: str) -> dict[str, object]:
    return {
        "event": "state:transition",
        "session_id": "session-1",
        "run_id": "run-1",
        "to_state": to_state,
    }


def _lifecycle(lifecycle: str) -> dict[str, object]:
    return {
        "event": "run:lifecycle",
        "session_id": "session-1",
        "run_id": "run-1",
        "segment_id": "segment-1",
        "lifecycle": lifecycle,
        "reason": "test boundary",
    }


def test_classifies_waiting_approval() -> None:
    summary = classify_run_status([_state("AWAITING_APPROVAL")])

    assert summary.category == "waiting_approval"
    assert summary.can_continue is True


def test_classifies_lifecycle_waiting_approval() -> None:
    summary = classify_run_status([_lifecycle("waiting_approval")])

    assert summary.category == "waiting_approval"
    assert summary.can_continue is True


def test_classifies_lifecycle_waiting_user() -> None:
    summary = classify_run_status([_lifecycle("waiting_user")])

    assert summary.category == "waiting_user"
    assert summary.can_continue is True
    assert "user input" in summary.detail


def test_lifecycle_fact_takes_precedence_over_historical_state() -> None:
    summary = classify_run_status([_state("DONE"), _lifecycle("paused")])

    assert summary.category == "continuable"
    assert summary.can_continue is True


def test_lifecycle_done_closes_prior_approval_required_marker() -> None:
    summary = classify_run_status(
        [
            {"event": "approval:required"},
            _lifecycle("done"),
        ]
    )

    assert summary.category == "done"


def test_classifies_recoverable_model_failure() -> None:
    summary = classify_run_status(
        [_state("FAILED")],
        errors=[
            {
                "category": "invalid_model_protocol",
                "message": "MODEL_PROTOCOL_ERROR: invalid JSON",
            }
        ],
    )

    assert summary.category == "recoverable_failure"
    assert summary.can_continue is True


def test_classifies_strategy_available_for_tool_failure() -> None:
    summary = classify_run_status(
        [
            {
                "event": "tool:response",
                "session_id": "session-1",
                "run_id": "run-1",
                "tool": {
                    "status": "error",
                    "error_category": "invalid_input",
                    "error": "missing path",
                },
            }
        ]
    )

    assert summary.category == "strategy_available"
    assert summary.can_continue is True


def test_classifies_mcp_flat_tool_failure_as_hard_blocked() -> None:
    summary = classify_run_status(
        [
            {
                "event": "tool:response",
                "session_id": "session-1",
                "run_id": "run-1",
                "tool_name": "mcp_echo_echo",
                "status": "error",
                "error_category": "invalid_input",
                "error": "invalid_input: mcp_not_configured: MCP config not found",
            }
        ]
    )

    assert summary.category == "hard_blocked"
    assert summary.can_continue is False
    assert "MCP" in summary.detail


def test_classifies_browser_unavailable_as_capability_boundary() -> None:
    summary = classify_run_status(
        [
            {
                "event": "tool:response",
                "session_id": "session-1",
                "run_id": "run-1",
                "tool": {
                    "name": "browser_extract",
                    "status": "error",
                    "error_category": "unknown",
                    "error": "unknown: browser_unavailable: Playwright is not installed",
                },
            }
        ]
    )

    assert summary.category == "hard_blocked"
    assert summary.can_continue is False
    assert "browser capability" in summary.detail
    assert "install" in summary.next_step.lower()


def test_classifies_browser_domain_denied_as_explainable_boundary() -> None:
    summary = classify_run_status(
        [
            {
                "event": "tool:response",
                "session_id": "session-1",
                "run_id": "run-1",
                "tool": {
                    "name": "browser_navigate",
                    "status": "error",
                    "error_category": "permission",
                    "error": "permission: browser_domain_denied",
                },
            }
        ]
    )

    assert summary.category == "hard_blocked"
    assert summary.can_continue is False
    assert "browser" in summary.detail.lower()
    assert "different source" in summary.next_step.lower()


def test_classifies_hard_blocked_configuration_failure() -> None:
    summary = classify_run_status(
        [_state("FAILED")],
        errors=[
            {
                "category": "missing_config",
                "message": "provider API key is not configured",
            }
        ],
    )

    assert summary.category == "hard_blocked"
    assert summary.can_continue is False


def test_classifies_plain_provider_error_as_recoverable() -> None:
    summary = classify_run_status(
        [_state("FAILED")],
        errors=[
            {
                "category": "provider_error",
                "message": "provider returned empty response",
            }
        ],
    )

    assert summary.category == "recoverable_failure"
    assert summary.can_continue is True


def test_repl_status_mapping_is_user_facing_not_provider_retry_policy() -> None:
    recoverable = classify_run_status(
        [_state("FAILED")],
        errors=[{"category": "server_error", "message": "provider 500"}],
    )
    hard = classify_run_status(
        [_state("FAILED")],
        errors=[{"category": "context_overflow", "message": "too many tokens"}],
    )

    assert recoverable.category == "recoverable_failure"
    assert recoverable.can_continue is True
    assert hard.category == "hard_blocked"
    assert hard.can_continue is False


def test_classifies_provider_quota_error_as_hard_blocked() -> None:
    summary = classify_run_status(
        [_state("FAILED")],
        errors=[
            {
                "category": "provider_error",
                "message": "insufficient quota for this model",
            }
        ],
    )

    assert summary.category == "hard_blocked"
    assert summary.can_continue is False


def test_classifies_degraded_text_mode() -> None:
    summary = classify_run_status(
        [
            {
                "event": "llm:response",
                "session_id": "session-1",
                "run_id": "run-1",
                "summary": {"mode": "degraded_text_mode"},
            }
        ]
    )

    assert summary.category == "degraded_text_mode"
    assert summary.can_continue is True


def test_classifies_continuable_pause() -> None:
    summary = classify_run_status([_state("PAUSED")])

    assert summary.category == "continuable"
    assert summary.can_continue is True


def test_classifies_done() -> None:
    summary = classify_run_status([_state("DONE")])

    assert summary.category == "done"
    assert summary.can_continue is True


def test_model_and_tool_error_explainers() -> None:
    model = explain_model_stop(
        "model_error:missing_config",
        "MODEL_PROVIDER_ERROR: missing_config",
    )
    assert model is not None
    assert model.category == "hard_blocked"

    protocol = explain_model_stop("protocol_error", "MODEL_PROTOCOL_ERROR: bad")
    assert protocol is not None
    assert protocol.category == "recoverable_failure"

    tool = explain_tool_error(
        "invalid_input", "missing argument", tool_name="file_read"
    )
    assert tool.category == "strategy_available"

    exc = explain_exception_status(
        RuntimeError("llm_client is required for run_stream")
    )
    assert exc.category == "hard_blocked"


def test_read_latest_run_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同秒错误按实际提交顺序取最近两条；参数：隔离目录和替换器；返回：无。"""
    source_ids = iter(f"error-{idx}" for idx in range(6))
    monkeypatch.setattr(
        "runtime.run_evidence.utc_now", lambda: "2026-10-06T00:00:00+00:00"
    )
    monkeypatch.setattr("runtime.run_evidence.new_ulid", lambda: next(source_ids))
    store = RunEvidenceStore(tmp_path)
    for idx in range(6):
        store.append_error(
            session_id="session-1",
            run_id="run-1",
            error={"category": "invalid_model_protocol", "message": f"err {idx}"},
        )

    errors = read_latest_run_errors(
        tmp_path,
        session_id="session-1",
        run_id="run-1",
        limit=2,
    )

    assert [item["message"] for item in errors] == ["err 4", "err 5"]
