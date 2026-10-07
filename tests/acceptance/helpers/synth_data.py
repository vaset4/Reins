from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(slots=True)
class SyntheticTask:
    task_id: str
    status: str
    goal: str
    tags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.task_id or not self.status or not self.goal:
            raise ValueError("task_id, status, and goal are required")


@dataclass(slots=True)
class SyntheticMemory:
    memory_id: str
    type: str
    content: str
    tags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if not self.memory_id or not self.type or not self.content:
            raise ValueError("memory_id, type, and content are required")


def make_task(
    task_id: str | None = None,
    status: str = "active",
    goal: str = "synthetic task",
    tags: list[str] | None = None,
) -> SyntheticTask:
    task = SyntheticTask(
        task_id=task_id or f"task_{uuid4().hex[:8]}",
        status=status,
        goal=goal,
        tags=list(tags or []),
    )
    task.validate()
    return task


def make_similar_tasks(n: int = 3, theme: str = "research") -> list[SyntheticTask]:
    return [
        make_task(goal=f"{theme} task {index + 1}", tags=[theme]) for index in range(n)
    ]


def make_memory(
    type: str = "fact",
    content: str = "synthetic memory",
    tags: list[str] | None = None,
) -> SyntheticMemory:
    memory = SyntheticMemory(
        memory_id=f"memory_{uuid4().hex[:8]}",
        type=type,
        content=content,
        tags=list(tags or []),
    )
    memory.validate()
    return memory
