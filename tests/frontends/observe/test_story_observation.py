from __future__ import annotations

from frontends.observe.api.story import build_run_story


def test_story_exposes_shared_observation_tracks() -> None:
    story = build_run_story(
        [
            {
                "event": "checkpoint:saved",
                "ts": "2026-06-03T00:00:00Z",
                "checkpoint": {
                    "checkpoint_id": "ck-1",
                    "state": "PAUSED",
                    "reason": "pre_tool",
                },
            },
            {
                "event": "run:lifecycle",
                "ts": "2026-06-03T00:00:01Z",
                "lifecycle": "waiting_user",
                "reason": "model requested input",
            },
            {
                "event": "context:segments",
                "segments": [{"name": "identity", "tokens_est": 10}],
            },
            {
                "event": "llm:response",
                "summary": {"evidence": {"model_request": "raw/request.json"}},
            },
            {
                "event": "trim:delta",
                "reason": "overflow",
                "removed_sections": ["conversation"],
            },
        ],
        summary=None,
        state=None,
    )

    assert story["overview"]["status"] == "waiting_user"
    assert story["lifecycle"]["source"] == "run:lifecycle"
    assert story["lifecycle"]["lifecycle"] == "waiting_user"
    assert story["checkpoints"][0]["reason"] == "pre_tool"
    assert story["context"]["segments"][0]["name"] == "identity"
    assert story["raw_evidence"][0]["path"] == "raw/request.json"
    assert story["compression"]["events"][0]["removed_sections"] == ["conversation"]
