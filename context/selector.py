from __future__ import annotations

from pathlib import Path

import yaml

from runtime.run_facts import RunFactStore
from tasks.records import TaskRecord
from tasks.store import TaskStore


def default_data_root() -> Path:
    return Path.home() / ".reins" / "data"


def task_dir(data_root: Path | str, task_id: str) -> Path:
    """定位实际工作材料；传参：根与目标；返回：工作目录。"""
    return TaskStore(data_root).task_dir(task_id)


def load_task_record(data_root: Path | str, task_id: str) -> TaskRecord:
    """读取规范目标状态；传参：根与身份；返回：目标记录。"""
    return TaskStore(data_root).require_task(task_id)


def read_text_file(data_root: Path | str, task_id: str, name: str) -> str:
    """读取目标派生正文视图；传参：根、目标和材料名；返回：可展示文字。"""
    store = TaskStore(data_root)
    if name == "journal.md":
        return store.read_journal(task_id)
    if name == "todo.md":
        from tools.todo_tool import list_todos

        return "\n".join(
            f"- [{'x' if item.status == 'done' else ' '}] {item.content} <!-- status:{item.status} -->"
            for item in list_todos(task_id, data_root=data_root)
        )
    layers = store.read_summary_layers(task_id)
    summaries = {
        "summary.md": layers.summary,
        "intent.md": layers.intent,
        "progress.md": layers.progress,
        "resume_hint.md": layers.resume_hint,
    }
    if name not in summaries:
        raise ValueError(f"unknown task material: {name}")
    return summaries[name]


def read_pending_todo(data_root: Path | str, task_id: str) -> str:
    """提取未完成待办；传参：根和目标；返回：待办正文。"""
    from tools.todo_tool import list_todos

    return "\n".join(
        f"- [ ] {item.content}"
        for item in list_todos(task_id, data_root=data_root)
        if item.status in {"pending", "in_progress"}
    )


def read_recent_trajectory(
    data_root: Path | str, task_id: str, *, limit: int = 5
) -> list[dict[str, object]]:
    """读取目标最近运行事实；传参：根、目标和数量；返回：已提交事实。"""
    facts = RunFactStore(data_root).read_task_facts(task_id)
    return [{str(key): value for key, value in fact.items()} for fact in facts[-limit:]]


def read_profile_preferences(data_root: Path | str) -> dict[str, object]:
    path = Path(data_root) / "profile" / "preferences.yaml"
    if not path.is_file():
        return {"language": "zh"}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    result = data if isinstance(data, dict) else {}
    result.setdefault("language", "zh")
    return {str(key): value for key, value in result.items()}


def read_spec_ref(ref: str, *, project_root: Path | None = None) -> str:
    path = Path(ref)
    if not path.is_absolute() and project_root is not None:
        path = project_root / ref
    if not path.is_file():
        return f"{ref}: missing"
    text = path.read_text(encoding="utf-8")
    return text[:1200].strip()


__all__ = [
    "default_data_root",
    "load_task_record",
    "read_pending_todo",
    "read_profile_preferences",
    "read_recent_trajectory",
    "read_spec_ref",
    "read_text_file",
    "task_dir",
]
