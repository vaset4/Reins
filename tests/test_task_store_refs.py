from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from runtime.persistence import RuntimeStore

from context.engine import build_context_sections_for_tests
from skills.store import SkillStore, build_skill_markdown
from tasks.records import TaskRecord, record_payload
from tasks.store import TaskStore


def test_task_record_retires_current_run_field() -> None:
    record = TaskRecord.from_dict(
        {
            "task_id": "2026-06-03-retired",
            "goal": "retire dead run field",
            "current_run": {"run_id": "run-old"},
        }
    )

    assert not hasattr(record, "current_run")
    assert "current_run" not in record_payload(record)


def test_task_store_refs_feed_build_context_and_preserve_unknown_yaml(
    tmp_path: Path,
) -> None:
    task_id = "2026-06-03-refs"
    store = TaskStore(tmp_path)
    store.create_task("use exact refs", task_id=task_id)
    _write_yaml_field(tmp_path, task_id, "future_field", {"keep": True})
    (tmp_path / "spec-ref.md").write_text("Task ref spec body.", encoding="utf-8")
    SkillStore(tmp_path).create_skill(
        "task-ref-skill",
        build_skill_markdown(
            name="Task Ref Skill",
            body="Task ref skill body.",
        ),
        meta={},
    )

    updated = store.update_task_refs(
        task_id,
        spec_refs=["spec-ref.md"],
        skill_refs=["task-ref-skill"],
    )
    sections = build_context_sections_for_tests(
        task_id,
        "user",
        {"latest": "continue"},
        lease=SimpleNamespace(max_tokens=1000),
        task_relevant=True,
        data_root=tmp_path,
        project_root=tmp_path,
    )
    payload = store.load_task_payload(task_id)

    assert updated.spec_refs == ["spec-ref.md"]
    assert updated.skill_refs == ["task-ref-skill"]
    assert payload["future_field"] == {"keep": True}
    assert _section(sections, "specs").find("Task ref spec body.") >= 0
    assert _section(sections, "skills").find("Task ref skill body.") >= 0


def _write_yaml_field(data_root: Path, task_id: str, key: str, value: object) -> None:
    """模拟新版本持久扩展字段；传参：根、目标、字段和值；返回：无。"""
    with RuntimeStore(data_root).transaction() as batch:
        payload = batch.get("task", task_id)
        assert payload is not None
        batch.put("task", task_id, {**payload, key: value})


def _section(sections: list[object], name: str) -> str:
    for section in sections:
        if getattr(section, "name", "") == name:
            return str(getattr(section, "content", ""))
    raise AssertionError(name)
