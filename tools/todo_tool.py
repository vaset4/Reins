from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from runtime.persistence import RuntimeStore, record_key

TODO_PENDING = "pending"
TODO_IN_PROGRESS = "in_progress"
TODO_DONE = "done"
TODO_BLOCKED = "blocked"
TODO_STATUSES = {TODO_PENDING, TODO_IN_PROGRESS, TODO_DONE, TODO_BLOCKED}


@dataclass(slots=True)
class TodoItem:
    idx: int
    content: str
    status: str


def add_todo(task_id: str, content: str, *, data_root: Path | str) -> TodoItem:
    """原子追加待办并分配顺序；传参：目标与正文；返回：新项。"""
    with RuntimeStore(data_root).transaction() as batch:
        rows = batch.list_raw("task_todo", filters={"task_id": task_id})
        index = max((int(row.payload["idx"]) for row in rows), default=-1) + 1
        batch.put(
            "task_todo",
            record_key(task_id, str(index)),
            {
                "task_id": task_id,
                "idx": index,
                "status": TODO_PENDING,
                "content": content,
            },
            expected_revision=0,
        )
        return TodoItem(index, content, TODO_PENDING)


def update_todo(
    task_id: str, idx: int, status: str, *, data_root: Path | str
) -> TodoItem:
    """只改变指定待办状态；传参：目标、序号和状态；返回：新视图。"""
    if status not in TODO_STATUSES:
        raise ValueError(status)
    with RuntimeStore(data_root).transaction() as batch:
        identity = record_key(task_id, str(idx))
        row = batch.get("task_todo", identity)
        if row is None:
            raise IndexError(idx)
        batch.put("task_todo", identity, {**row, "status": status})
        return TodoItem(idx, row["content"], status)


def list_todos(
    task_id: str, filter: str | None = None, *, data_root: Path | str
) -> list[TodoItem]:
    """按原始顺序列待办；传参：目标及状态过滤；返回：独立视图。"""
    filters = {"task_id": task_id, **({"status": filter} if filter is not None else {})}
    with RuntimeStore(data_root).snapshot() as source:
        rows = source.list("task_todo", filters=filters)
        return [
            TodoItem(row["idx"], row["content"], row["status"])
            for row in sorted(rows, key=lambda row: row["idx"])
        ]
