from __future__ import annotations

from pathlib import Path

from context.artifact_ref import store_large_output
from tasks.store import TaskStore
from tools.read_artifact import read_artifact


def test_task_summary_update_writes_progress_layer(tmp_path: Path) -> None:
    task_id = "2026-05-04-summary"
    store = TaskStore(tmp_path)
    store.create_task("summarize", task_id=task_id)

    store.update_summary(task_id, "updated")

    layers = store.read_summary_layers(task_id)
    assert layers.progress == "updated"
    assert layers.resume_hint == ""
    assert layers.summary == "updated"


def test_large_output_becomes_artifact_ref_and_readable(tmp_path: Path) -> None:
    task_id = "2026-05-04-artifact"
    TaskStore(tmp_path).create_task("artifact", task_id=task_id)
    content = "x" * 5000

    ref = store_large_output(tmp_path, task_id, content, summary="large output")

    assert ref is not None
    assert ref.summary == "large output"
    assert read_artifact(tmp_path, ref.artifact_id, mode="summary") == "large output"
    assert read_artifact(tmp_path, ref.artifact_id, mode="head", head_chars=3) == "xxx"
    assert read_artifact(tmp_path, ref.artifact_id, mode="full") == content
