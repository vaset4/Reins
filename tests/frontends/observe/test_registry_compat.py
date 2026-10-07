"""Unit tests for panel registry and backward compatibility."""

from __future__ import annotations

from typing import Any

import pytest

from frontends.observe.panels.base import Panel, RunContext
from frontends.observe.registry import PanelRegistry, build_default_registry
from runtime.run_facts import RunFactStore


class _TestPanel(Panel):
    id = "test_panel"
    title = "Test"
    section = "test"
    phase = "X"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        return {"test": True}


class TestPanelRegistry:
    def test_register_and_get(self):
        reg = PanelRegistry()
        panel = _TestPanel()
        reg.register(panel)
        assert reg.get("test_panel") is panel

    def test_duplicate_id_rejected(self):
        reg = PanelRegistry()
        reg.register(_TestPanel())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_TestPanel())

    def test_empty_id_rejected(self):
        class BadPanel(Panel):
            id = ""
            title = "Bad"
            section = "x"
            phase = "X"

            def build(self, ctx: RunContext) -> dict[str, Any]:
                return {}

        reg = PanelRegistry()
        with pytest.raises(ValueError, match="non-empty"):
            reg.register(BadPanel())

    def test_descriptors(self):
        reg = PanelRegistry()
        reg.register(_TestPanel())
        descs = reg.descriptors()
        assert len(descs) == 1
        assert descs[0].id == "test_panel"
        assert descs[0].phase == "X"

    def test_default_registry_has_panels(self):
        reg = build_default_registry()
        assert len(reg.all()) >= 12


class TestBackwardCompat:
    """Existing RunFactStore readers must work unchanged when new fact types
    are present in committed run facts."""

    def test_read_run_ignores_unknown_types(self, tmp_path):
        payloads = [
            {
                "type": "run_fact",
                "event": "run:start",
                "ts": "2026-05-24T00:00:00Z",
                "session_id": "session-abc",
                "run_id": "run-xyz",
                "task_id": "t1",
            },
            {
                "type": "run_fact",
                "event": "context:segments",
                "ts": "2026-05-24T00:00:01Z",
                "session_id": "session-abc",
                "run_id": "run-xyz",
                "segments": [{"name": "identity", "tokens_est": 100}],
            },
            {
                "type": "run_fact",
                "event": "memory:score_breakdown",
                "ts": "2026-05-24T00:00:02Z",
                "session_id": "session-abc",
                "run_id": "run-xyz",
                "injected": [],
                "skipped": [],
            },
            {
                "type": "run_fact",
                "event": "skill:activation",
                "ts": "2026-05-24T00:00:03Z",
                "session_id": "session-abc",
                "run_id": "run-xyz",
                "skills": [],
            },
            {
                "type": "run_fact",
                "event": "llm:cache_usage",
                "ts": "2026-05-24T00:00:04Z",
                "session_id": "session-abc",
                "run_id": "run-xyz",
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 50,
            },
            {
                "type": "run_fact",
                "event": "trim:delta",
                "ts": "2026-05-24T00:00:05Z",
                "session_id": "session-abc",
                "run_id": "run-xyz",
                "tokens_before": 5000,
                "tokens_after": 3000,
            },
            {
                "type": "run_fact",
                "event": "state:transition",
                "ts": "2026-05-24T00:00:06Z",
                "session_id": "session-abc",
                "run_id": "run-xyz",
                "from_state": "RUNNING",
                "to_state": "DONE",
            },
        ]
        store = RunFactStore(tmp_path)
        for payload in payloads:
            store.append(payload)
        facts = store.read_run("run-xyz")
        assert len(facts) == 7

        runs = store.list_runs_for_session("session-abc")
        assert len(runs) == 1
        assert runs[0].run_id == "run-xyz"
        assert runs[0].status == "done"


def test_cache_usage_panel_handles_null_and_unreported(tmp_path) -> None:
    """面板收到 null 和旧格式（无该键）不抛异常，且正确计入"未上报次数"。"""
    from frontends.observe.panels.cache_usage import CacheUsagePanel

    payloads = [
        {
            "type": "run_fact",
            "event": "llm:cache_usage",
            "ts": "2026-09-01T00:00:01Z",
            "session_id": "session-abc",
            "run_id": "run-xyz",
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 50,
        },
        {
            "type": "run_fact",
            "event": "llm:cache_usage",
            "ts": "2026-09-01T00:00:02Z",
            "session_id": "session-abc",
            "run_id": "run-xyz",
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
        },
        {
            "type": "run_fact",
            "event": "llm:cache_usage",
            "ts": "2026-09-01T00:00:03Z",
            "session_id": "session-abc",
            "run_id": "run-xyz",
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        {
            "type": "run_fact",
            "event": "llm:cache_usage",
            "ts": "2026-09-01T00:00:04Z",
            "session_id": "session-abc",
            "run_id": "run-xyz",
        },
    ]
    store = RunFactStore(tmp_path / ".reins" / "data")
    for payload in payloads:
        store.append(payload)
    facts = store.read_run("run-xyz")

    class FakeCtx:
        def __init__(self, facts: list[dict[str, Any]]) -> None:
            self._facts = facts

        def facts(self) -> list[dict[str, Any]]:
            return self._facts

    panel = CacheUsagePanel()
    result = panel.build(FakeCtx(facts))

    assert "totals" in result
    totals = result["totals"]
    assert totals["total_cache_creation"] == 100, "只计非 null 的数值"
    assert totals["total_cache_read"] == 50
    assert totals["unreported_count"] == 2, "null 与缺键各计一次"
