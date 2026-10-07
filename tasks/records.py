from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

TASK_STATUS_ACTIVE = "active"
TASK_STATUS_DONE = "done"
TASK_STATE_VERSION = 2


@dataclass(slots=True)
class TaskRecord:
    task_id: str
    goal: str
    status: str = TASK_STATUS_ACTIVE
    created_at: str = ""
    updated_at: str = ""
    tags: list[str] = field(default_factory=list)
    spec_refs: list[str] = field(default_factory=list)
    skill_refs: list[str] = field(default_factory=list)
    grants: list[dict[str, Any]] = field(default_factory=list)
    done_at: str | None = None
    is_inbox: bool = False
    sediment_done: bool = False
    state_version: int = TASK_STATE_VERSION
    revision: int = 1
    status_source: str = "goal_created"
    completion: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRecord":
        return cls(
            task_id=str(data.get("task_id", "")),
            goal=str(data.get("goal", "")),
            status=str(data.get("status", TASK_STATUS_ACTIVE)),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            tags=[str(item) for item in data.get("tags", [])],
            spec_refs=[str(item) for item in data.get("spec_refs", [])],
            skill_refs=[str(item) for item in data.get("skill_refs", [])],
            grants=_load_grants(data.get("grants")),
            done_at=str(data["done_at"]) if data.get("done_at") is not None else None,
            is_inbox=bool(data.get("is_inbox", False)),
            sediment_done=bool(data.get("sediment_done", False)),
            **_state_metadata(data),
        )


@dataclass(frozen=True, slots=True)
class TaskSummaryLayers:
    intent: str = ""
    progress: str = ""
    resume_hint: str = ""
    summary: str = ""


def record_payload(record: TaskRecord) -> dict[str, Any]:
    return asdict(record)


def render_compatible_summary(layers: TaskSummaryLayers) -> str:
    values = [
        value.strip()
        for value in (layers.intent, layers.resume_hint, layers.progress)
        if value.strip()
    ]
    if len(set(values)) == 1:
        return values[0]
    rows = [
        ("Intent", layers.intent),
        ("Resume hint", layers.resume_hint),
        ("Progress", layers.progress),
    ]
    rendered = [
        f"## {title}\n\n{content.strip()}" for title, content in rows if content.strip()
    ]
    if rendered:
        return "\n\n".join(rendered)
    return layers.summary.strip()


def _load_grants(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _state_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """读取有版本的目标状态，旧 done 只保留历史结论，不补造完成依据。

    传参：data 为 task.yaml 内容；返回：状态版本、修订及完成来源
    """
    version = data.get("state_version", 1)
    if type(version) is not int or version not in (1, TASK_STATE_VERSION):
        raise ValueError(f"unsupported task state version: {version}")
    revision = data.get("revision", 0 if version == 1 else None)
    if type(revision) is not int or revision < 0:
        raise ValueError("task revision must be a non-negative integer")
    completion = data.get("completion")
    if completion is not None and (
        not isinstance(completion, dict)
        or completion.get("version") != 1
        or not isinstance(completion.get("evidence"), list)
        or not completion["evidence"]
    ):
        raise ValueError("invalid task completion evidence")
    return {
        "state_version": version,
        "revision": revision,
        "status_source": str(data.get("status_source", "legacy_unverified")),
        "completion": dict(completion) if completion is not None else None,
    }
