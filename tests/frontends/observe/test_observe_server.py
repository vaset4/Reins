"""Tests for frontends.observe — the Agent Flight Recorder dashboard."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from app.run_task import run_task
from scripts.testing.llm import from_test_sequence
from runtime.run_facts import RunFactStore
from runtime.run_evidence import RunEvidenceStore
from runtime.session_state import SessionState, SessionStateStore

import frontends.observe.server as observe_server
from frontends.observe.api.session_files import build_session_file_inventory
from frontends.observe.readers.evidence_reader import EvidenceReader

PORT = 18765
BASE = f"http://127.0.0.1:{PORT}"
REQUEST_TIMEOUT_SECONDS = 5


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """在隔离原件上启动真实观察服务；传参：临时目录工厂；返回：服务进程。"""
    root = tmp_path_factory.mktemp("observe")
    data_root = root / "data"
    response = run_task(
        "查看运行事实",
        root,
        data_root=data_root,
        llm_client=from_test_sequence(["已经完成"]),
        session_id="session-observe",
    )
    assert response.status == "done"
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "frontends.observe.server",
                "--data-root",
                str(data_root),
                "--port",
                str(PORT),
            ],
            stdout=out,
            stderr=err,
        )
        try:
            _wait_for_server()
            yield proc
        finally:
            proc.terminate()
            proc.wait(timeout=5)


def _wait_for_server() -> None:
    deadline = time.monotonic() + 10
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            _get("/api/health")
            return
        except Exception as exc:  # pragma: no cover - diagnostic path
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"observe server did not become ready: {last_error}")


def _get(path: str) -> Any:
    with urlopen(f"{BASE}{path}", timeout=REQUEST_TIMEOUT_SECONDS) as resp:
        return json.loads(resp.read())


def _get_status(path: str) -> int:
    try:
        with urlopen(f"{BASE}{path}", timeout=REQUEST_TIMEOUT_SECONDS):
            pass
        return 200
    except HTTPError as e:
        return e.code


def _first_run_id() -> str | None:
    sessions = _get("/api/sessions")["sessions"]
    sid = sessions[0]["session_id"]
    runs = _get(f"/api/sessions/{sid}/runs")["runs"]
    return runs[0]["run_id"] if runs else None


class TestHealth:
    def test_health_ok(self, server):
        data = _get("/api/health")
        assert data["ok"] is True
        assert "data_root" in data

    def test_health_prices_loaded(self, server):
        data = _get("/api/health")
        assert "prices_loaded" in data


class TestPanels:
    def test_panel_list(self, server):
        panels = _get("/api/panels")
        assert len(panels) >= 12
        ids = [p["id"] for p in panels]
        assert "overview" in ids
        assert "timeline" in ids
        assert "context_inspector" in ids

    def test_panel_descriptor_shape(self, server):
        panels = _get("/api/panels")
        for p in panels:
            assert "id" in p
            assert "title" in p
            assert "section" in p
            assert "phase" in p


class TestSessions:
    def test_session_file_inventory_browser_contract(self, tmp_path):
        SessionStateStore(tmp_path).save(
            SessionState(
                session_id="session-test",
                original_user_goal="inspect session files",
                last_run_id="run-test",
                last_checkpoint_state="DONE",
                summary="summary body",
                writeback_targets={"session": True},
            )
        )
        evidence_ref = RunEvidenceStore(tmp_path).write_record(
            session_id="session-test",
            run_id="run-test",
            kind="model_request",
            source_id="request-1",
            payload={"ok": True},
        )
        facts = RunFactStore(tmp_path)
        for payload in [
            {
                "event": "run:lifecycle",
                "lifecycle": "done",
                "reason": "model final",
                "ts": "2026-05-27T00:00:00+00:00",
            },
            {
                "event": "state:transition",
                "from_state": "IDLE",
                "to_state": "DONE",
                "ts": "2026-05-27T00:00:00+00:00",
            },
            {
                "event": "checkpoint:saved",
                "checkpoint": {"state": "DONE", "reason": "enter_done"},
                "ts": "2026-05-27T00:00:00+00:00",
            },
            {
                "event": "context:built",
                "summary": {"tool_history_count": 0},
                "ts": "2026-05-27T00:00:00+00:00",
            },
            {
                "event": "context:segments",
                "segments": [{"tokens_est": 3}],
                "ts": "2026-05-27T00:00:00+00:00",
            },
        ]:
            facts.append(
                {**payload, "session_id": "session-test", "run_id": "run-test"}
            )

        data = build_session_file_inventory(tmp_path, "session-test")

        assert data["summary"]["lifecycle_events"] == 1
        assert data["summary"]["legacy_state_transitions"] == 1
        assert data["summary_file"]["text"] == "summary body"
        assert data["state"]["groups"][0]["fields"][0]["value"] == "session-test"
        assert data["runs"][0]["files"][1]["path"] == evidence_ref
        assert data["runs"][0]["errors"]["present"] is False
        assert len(data["runs"][0]["event_track"]) == 5

    def test_session_file_inventory_rejects_similar_prefix_sibling(self, tmp_path):
        (tmp_path / "sessions-evil").mkdir()

        data = build_session_file_inventory(tmp_path, "..\\sessions-evil")

        assert data == {"status": "rejected", "session_id": "..\\sessions-evil"}

    def test_session_file_inventory_missing_inside_root(self, tmp_path):
        (tmp_path / "sessions").mkdir()

        data = build_session_file_inventory(tmp_path, "session-missing")

        assert data == {"status": "missing", "session_id": "session-missing"}

    def test_list_sessions(self, server):
        data = _get("/api/sessions")
        assert "sessions" in data
        assert "count" in data
        assert data["count"] > 0

    def test_session_runs(self, server):
        sessions = _get("/api/sessions")["sessions"]
        sid = sessions[0]["session_id"]
        data = _get(f"/api/sessions/{sid}/runs")
        assert "runs" in data

    def test_session_files_inventory(self, server):
        sessions = _get("/api/sessions")["sessions"]
        sid = sessions[0]["session_id"]
        data = _get(f"/api/sessions/{sid}/files")
        assert data["status"] == "ok"
        assert data["summary"]["runs"] >= 0
        assert "state_fields" in data
        assert "not_displayed" in data
        assert "summary_file" in data
        assert "state" in data
        if data["runs"]:
            assert "files" in data["runs"][0]
            assert "event_track" in data["runs"][0]


class TestRunPanels:
    def _first_run(self, server):
        return _first_run_id()

    def test_overview_panel(self, server):
        rid = self._first_run(server)
        assert rid is not None
        data = _get(f"/api/runs/{rid}/panels/overview")
        assert "session_id" in data
        assert "total_facts" in data

    def test_all_panels_return_json(self, server):
        rid = self._first_run(server)
        assert rid is not None
        panels = _get("/api/panels")
        for p in panels:
            data = _get(f"/api/runs/{rid}/panels/{p['id']}")
            assert isinstance(data, dict)


class TestRunStory:
    def test_story_payload_shape(self, server):
        rid = _first_run_id()
        assert rid is not None
        data = _get(f"/api/runs/{rid}/story")
        story = data["story"]
        assert "session_state" in data
        assert "overview" in story
        assert "model_calls" in story
        assert "tool_calls" in story
        assert "fact_stream" in story
        assert "run_files" in story
        assert "warnings" in story
        if story["fact_stream"]:
            assert "raw" in story["fact_stream"][0]

    def test_story_preserves_model_evidence_paths(self, server):
        rid = _first_run_id()
        assert rid is not None
        calls = _get(f"/api/runs/{rid}/story")["story"]["model_calls"]
        assert calls
        evidence = calls[0]["evidence"]
        assert "model_request" in evidence
        assert "model_response" in evidence


class TestTraversal:
    def test_dotdot_rejected(self, server):
        data = _get("/api/raw?path=../../etc/passwd")
        assert data["status"] == "invalid_reference" and data["data"] == ""

    def test_absolute_path_rejected(self, server):
        data = _get("/api/raw?path=/etc/passwd")
        assert data["status"] == "invalid_reference" and data["data"] == ""

    def test_missing_path_param(self, server):
        status = _get_status("/api/raw?path=")
        assert status == 400

    def test_valid_missing_file(self, server):
        data = _get("/api/raw?path=session:nonexist")
        assert data["status"] == "missing"

    def test_evidence_reader_rejects_similar_prefix_sibling(self, tmp_path):
        data_root = tmp_path / "data"
        data_root.mkdir()
        (tmp_path / "data-evil").mkdir()
        (tmp_path / "data-evil" / "secret.json").write_text(
            '{"secret":"must-not-read"}', encoding="utf-8"
        )
        reader = EvidenceReader(data_root)

        data = reader.read_raw_file("..\\data-evil\\secret.json")

        assert data["status"] == "invalid_reference" and data["data"] == ""

    def test_evidence_reader_allows_missing_file_inside_root(self, tmp_path):
        data_root = tmp_path / "data"
        data_root.mkdir()
        reader = EvidenceReader(data_root)

        data = reader.read_raw_file("session:missing")

        assert data["status"] == "missing"

    def test_static_path_rejects_similar_prefix_sibling(self, tmp_path, monkeypatch):
        static_root = tmp_path / "static"
        static_root.mkdir()
        (tmp_path / "static-evil").mkdir()
        monkeypatch.setattr(observe_server, "STATIC_ROOT", static_root)

        assert observe_server._safe_static("..\\static-evil\\app.js") is None

    def test_static_path_allows_missing_file_inside_root(self, tmp_path, monkeypatch):
        static_root = tmp_path / "static"
        static_root.mkdir()
        monkeypatch.setattr(observe_server, "STATIC_ROOT", static_root)

        assert observe_server._safe_static("missing.css") == static_root / "missing.css"


class TestCrossRun:
    def test_provider_model_distribution(self, server):
        data = _get("/api/cross/provider-model-distribution")
        assert "distribution" in data
        assert "total_calls" in data

    def test_failure_pivot(self, server):
        data = _get("/api/cross/failure-pivot")
        assert "pivot" in data

    def test_run_comparison_missing_params(self, server):
        status = _get_status("/api/cross/run-comparison?run_a=&run_b=")
        assert status == 400


class TestStatic:
    def test_index_html(self, server):
        with urlopen(f"{BASE}/", timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            html = resp.read().decode()
        assert "智能体运行记录器" in html

    def test_css_served(self, server):
        with urlopen(
            f"{BASE}/static/styles.css", timeout=REQUEST_TIMEOUT_SECONDS
        ) as resp:
            assert resp.status == 200

    def test_new_css_modules_served(self, server):
        for path in ["base.css", "story.css", "sessionFiles.css", "evidence.css"]:
            with urlopen(
                f"{BASE}/static/{path}", timeout=REQUEST_TIMEOUT_SECONDS
            ) as resp:
                assert resp.status == 200

    def test_js_served(self, server):
        with urlopen(f"{BASE}/static/app.js", timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            assert resp.status == 200
