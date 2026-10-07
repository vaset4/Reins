from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from context.engine import build_context_sections_for_tests
from runtime.ledger import LedgerStore
from runtime.run_evidence import RunEvidenceStore
from runtime.run_facts import RunFactStore
from tasks.store import TaskStore


def test_corrected_intent_is_written_to_layered_summary(tmp_path: Path) -> None:
    task_id = "2026-05-09-corrected-intent"
    store = TaskStore(tmp_path)
    store.create_task("make a personal intro page", task_id=task_id)
    store.update_summary_layers(
        task_id,
        intent="介绍当前目录 xiangmu 里的 Reins 项目，不是个人介绍。",
        progress="index.html still needs to be corrected.",
        resume_hint="Read project files, then update index.html as a Reins project page.",
    )

    layers = store.read_summary_layers(task_id)

    assert "Reins 项目" in layers.intent
    assert "个人介绍" in layers.intent
    assert "Read project files" in layers.resume_hint
    assert "make a personal intro page" not in layers.intent
    assert "Intent" in layers.summary
    assert "Resume hint" in layers.summary


def test_layered_summary_keeps_compatible_aggregate_file(tmp_path: Path) -> None:
    task_id = "2026-05-09-compatible-summary"
    store = TaskStore(tmp_path)
    store.create_task("describe Reins", task_id=task_id)
    store.update_summary_layers(
        task_id,
        intent="介绍 Reins 项目。",
        progress="index.html 已生成，仍需修正。",
        resume_hint="继续修改 index.html。",
    )

    summary = store.read_summary_layers(task_id).summary

    assert "Intent" in summary
    assert "Progress" in summary
    assert "继续修改 index.html" in summary


def test_layered_summary_writes_ledger_events(tmp_path: Path) -> None:
    task_id = "2026-05-09-ledger-summary"
    store = TaskStore(tmp_path)
    store.create_task("describe Reins", task_id=task_id)

    store.update_summary_layers(
        task_id,
        intent="介绍 Reins 项目。",
        progress="index.html 已生成。",
        resume_hint="继续修改 index.html。",
    )

    events = LedgerStore(tmp_path).read_task_events(task_id)
    summary_events = [event for event in events if event.event == "summary.updated"]

    assert [event.payload["summary_kind"] for event in summary_events] == [
        "intent",
        "progress",
        "resume_hint",
        "summary",
    ]
    assert summary_events[-1].payload["content"]


def test_error_log_does_not_pollute_summary_layers(tmp_path: Path) -> None:
    task_id = "2026-05-09-error-isolated"
    store = TaskStore(tmp_path)
    store.create_task("describe Reins project", task_id=task_id)
    store.update_summary_layers(
        task_id,
        intent="介绍当前目录的 Reins 项目。",
        progress="Need to inspect the repository before writing.",
        resume_hint="Continue by reading project files.",
    )

    RunEvidenceStore(tmp_path).append_error(
        session_id="session-1",
        run_id="run-1",
        error={
            "category": "invalid_model_protocol",
            "message": "MODEL_PROTOCOL_ERROR: invalid JSON",
            "stage": "parse",
        },
    )

    layers = store.read_summary_layers(task_id)
    assert layers.intent == "介绍当前目录的 Reins 项目。"
    assert layers.progress == "Need to inspect the repository before writing."
    assert layers.resume_hint == "Continue by reading project files."
    assert "MODEL_PROTOCOL_ERROR" not in layers.summary


def test_continue_context_prioritizes_resume_hint_before_errors(
    tmp_path: Path,
) -> None:
    task_id = "2026-05-09-continue-priority"
    store = TaskStore(tmp_path)
    store.create_task("correct Reins project page", task_id=task_id)
    store.update_summary_layers(
        task_id,
        intent="介绍当前目录 xiangmu 的 Reins 项目。",
        progress="Already corrected the mistaken personal-page goal.",
        resume_hint="Open project files and finish correcting index.html.",
    )
    facts = RunFactStore(tmp_path)
    facts.append(
        {
            "event": "run:start",
            "session_id": "session-1",
            "run_id": "run-1",
            "task_id": task_id,
            "focus_task_id": task_id,
        }
    )
    facts.append(
        {
            "event": "tool:response",
            "session_id": "session-1",
            "run_id": "run-1",
            "task_id": task_id,
            "tool": {
                "call_id": "call-1",
                "name": "file_read",
                "status": "ok",
                "output_summary": "index.html currently looks personal.",
            },
        }
    )
    RunEvidenceStore(tmp_path).append_error(
        session_id="session-1",
        run_id="run-1",
        error={
            "category": "invalid_model_protocol",
            "message": "MODEL_PROTOCOL_ERROR should stay below intent.",
            "stage": "parse",
        },
    )

    sections = build_context_sections_for_tests(
        task_id,
        "user",
        {"latest": "继续"},
        lease=SimpleNamespace(max_tokens=1000),
        task_relevant=True,
        session_id="session-1",
        run_id="run-1",
        data_root=tmp_path,
    )
    names = [section.name for section in sections]
    resume_priority = sections[names.index("resume_priority")].content

    assert names.index("resume_priority") < names.index("intent")
    assert names.index("intent") < names.index("progress")
    assert "resume_hint" in resume_priority
    assert "unfinished_runs" in resume_priority
    assert "index.html currently looks personal" in resume_priority
    assert "MODEL_PROTOCOL_ERROR" not in resume_priority
