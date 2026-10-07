from __future__ import annotations

from frontends.shared.run_lifecycle import build_lifecycle_observation
from frontends.shared.run_observation import build_run_observation


def test_run_observation_collects_context_checkpoint_and_raw_gaps() -> None:
    observation = build_run_observation(
        [
            {
                "event": "checkpoint:saved",
                "ts": "2026-06-03T00:00:00Z",
                "checkpoint": {
                    "checkpoint_id": "ck-1",
                    "state": "PAUSED",
                    "reason": "pre_tool",
                    "pending_tool_call": {"tool_name": "file_write"},
                },
            },
            {"event": "context:built", "summary": {"tool_history_count": 2}},
            {
                "event": "context:segments",
                "segments": [
                    {"name": "identity", "tokens_est": 10},
                    {"name": "tool_results", "tokens_est": 5},
                ],
            },
            {
                "event": "llm:response",
                "summary": {
                    "evidence": {
                        "model_request": "sessions/s/runs/r/raw/model_request_1.json",
                        "parsed_plan": "(not written, trace_level == off)",
                    }
                },
            },
        ]
    )

    assert observation.checkpoints[0].reason == "pre_tool"
    assert observation.checkpoints[0].pending_tool_call == {"tool_name": "file_write"}
    assert observation.context.build_count == 1
    assert observation.context.total_tokens_est == 15
    assert observation.context.last_tool_history_count == 2
    assert observation.raw_evidence[0].status == "path_recorded"
    assert observation.raw_evidence[1].status == "missing"
    assert observation.raw_evidence[2].status == "not_available"


def test_run_observation_collects_memory_compression_and_watchdog() -> None:
    observation = build_run_observation(
        [
            {
                "event": "memory:injection_explain",
                "explain": {"round_id": "round-1", "injected": [{"id": "m1"}]},
            },
            {
                "event": "memory:score_breakdown",
                "round_id": "round-1",
                "skipped": [{"memory_id": "m2", "type": "fact"}],
            },
            {
                "event": "trim:delta",
                "reason": "overflow",
                "tokens_before": 100,
                "tokens_after": 80,
                "removed_sections": ["conversation"],
            },
            {
                "event": "llm:response",
                "summary": {"error": {"category": "timeout"}},
            },
            {
                "event": "tool:response",
                "tool": {"status": "error", "error_category": "not_found"},
            },
            {
                "event": "run:lifecycle",
                "lifecycle": "paused",
                "reason": "lease step limit reached",
            },
        ]
    )

    assert observation.memory.injections[0]["round_id"] == "round-1"
    assert observation.memory.score_breakdowns[0]["skipped"][0]["type"] == "fact"
    assert observation.compression.events[0]["removed_sections"] == ["conversation"]
    assert observation.watchdog.llm_failures == 1
    assert observation.watchdog.tool_failures == 1
    assert observation.watchdog.unique_failure_patterns == 1
    assert observation.watchdog.paused is True


def test_lifecycle_observation_prefers_run_lifecycle_over_legacy_transition() -> None:
    observation = build_lifecycle_observation(
        [
            {
                "event": "state:transition",
                "ts": "2026-06-03T00:00:00Z",
                "to_state": "DONE",
            },
            {
                "event": "run:lifecycle",
                "ts": "2026-06-03T00:00:01Z",
                "lifecycle": "waiting_approval",
                "reason": "tool requires approval",
                "checkpoint_id": "ck-approval",
                "resumable": True,
            },
        ]
    )

    assert observation.lifecycle == "waiting_approval"
    assert observation.reason == "tool requires approval"
    assert observation.source == "run:lifecycle"
    assert observation.checkpoint_id == "ck-approval"
    assert observation.resumable is True


def test_lifecycle_observation_reads_legacy_terminal_transition() -> None:
    observation = build_lifecycle_observation(
        [
            {
                "event": "state:transition",
                "ts": "2026-06-03T00:00:00Z",
                "to_state": "PAUSED",
                "close_reason": "paused",
            },
        ]
    )

    assert observation.lifecycle == "paused"
    assert observation.reason == "paused"
    assert observation.source == "legacy_state_transition"


def test_lifecycle_observation_maps_legacy_approval_wait() -> None:
    observation = build_lifecycle_observation(
        [
            {
                "event": "state:transition",
                "ts": "2026-06-03T00:00:00Z",
                "to_state": "AWAITING_APPROVAL",
            },
        ]
    )

    assert observation.lifecycle == "waiting_approval"
    assert observation.source == "legacy_state_transition"
