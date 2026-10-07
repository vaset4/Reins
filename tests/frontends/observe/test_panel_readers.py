from __future__ import annotations

from frontends.observe.panels.approval_audit import _extract_approval_decisions
from frontends.observe.panels.compression_lane import _extract_compression_events
from frontends.observe.panels.memory_cards import _extract_injections
from frontends.observe.panels.state_heatmap import _extract_lifecycles


def test_memory_cards_reads_explain_payload() -> None:
    injections = _extract_injections(
        [
            {
                "event": "memory:injection_explain",
                "ts": "2026-06-03T00:00:00Z",
                "explain": {
                    "round_id": "task-1:round",
                    "injected": [{"memory_id": "m-1"}],
                    "skipped": [
                        {
                            "memory_id": "m-2",
                            "type": "fact",
                            "reason": "stale",
                        }
                    ],
                },
            }
        ]
    )

    assert injections == [
        {
            "ts": "2026-06-03T00:00:00Z",
            "injected": [{"memory_id": "m-1"}],
            "skipped": [{"memory_id": "m-2", "type": "fact", "reason": "stale"}],
            "round_id": "task-1:round",
        }
    ]


def test_compression_lane_uses_trim_delta_and_removed_sections() -> None:
    events = _extract_compression_events(
        [
            {
                "event": "context:trim",
                "reason": "legacy",
                "tokens_before": 100,
                "tokens_after": 80,
            },
            {
                "event": "trim:delta",
                "ts": "2026-06-03T00:00:00Z",
                "reason": "overflow",
                "tokens_before": 1000,
                "tokens_after": 700,
                "removed_sections": ["conversation", "tool_results"],
            },
        ]
    )

    assert events == [
        {
            "ts": "2026-06-03T00:00:00Z",
            "event": "trim:delta",
            "reason": "overflow",
            "tokens_before": 1000,
            "tokens_after": 700,
            "removed_sections": ["conversation", "tool_results"],
        }
    ]


def test_state_heatmap_reads_lifecycle_boundaries() -> None:
    lifecycles = _extract_lifecycles(
        [
            {
                "event": "state:transition",
                "ts": "2026-06-03T00:00:00Z",
                "to_state": "DONE",
            },
            {
                "event": "run:lifecycle",
                "ts": "2026-06-03T00:00:01Z",
                "lifecycle": "done",
                "reason": "model final",
            },
        ]
    )

    assert lifecycles == [
        {
            "ts": "2026-06-03T00:00:01Z",
            "lifecycle": "done",
            "reason": "model final",
            "checkpoint_id": "",
            "resumable": None,
        }
    ]


def test_approval_audit_reads_lifecycle_waiting_approval() -> None:
    decisions = _extract_approval_decisions(
        [
            {
                "event": "run:lifecycle",
                "ts": "2026-06-03T00:00:00Z",
                "lifecycle": "waiting_approval",
                "reason": "tool requires approval",
                "checkpoint_id": "ck-1",
            }
        ]
    )

    assert decisions == [
        {
            "ts": "2026-06-03T00:00:00Z",
            "source": "run:lifecycle",
            "lifecycle": "waiting_approval",
            "reason": "tool requires approval",
            "checkpoint_id": "ck-1",
        }
    ]
