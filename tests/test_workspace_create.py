from __future__ import annotations

from pathlib import Path

from tasks.store import TaskStore


def test_task_create_does_not_precreate_project_workspace(tmp_path: Path) -> None:
    project = tmp_path / "project"
    store = TaskStore(project / ".reins" / "data")

    task = store.create_task("write outputs", task_id="task-1")

    workspace = project / ".reins" / "workspace" / task.task_id
    assert not workspace.exists()


def test_promote_inbox_does_not_precreate_project_workspace(tmp_path: Path) -> None:
    project = tmp_path / "project"
    store = TaskStore(project / ".reins" / "data")

    inbox = store.create_task("chat", task_id="inbox-1", is_inbox=True)
    task = store.promote_inbox_to_task(inbox.task_id, new_task_id="task-1")

    workspace = project / ".reins" / "workspace" / task.task_id
    assert not workspace.exists()


def test_workspace_cleanup_stub_does_not_remove_files(tmp_path: Path) -> None:
    project = tmp_path / "project"
    store = TaskStore(project / ".reins" / "data")
    task = store.create_task("write outputs", task_id="task-1")
    output = project / ".reins" / "workspace" / task.task_id / "outputs" / "keep.txt"
    output.parent.mkdir(parents=True)
    output.write_text("keep", encoding="utf-8")

    store.cleanup_workspace(task.task_id)

    assert output.read_text(encoding="utf-8") == "keep"
