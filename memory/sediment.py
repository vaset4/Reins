from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from memory.reflection import (
    ReflectionProposal,
    normalize_reflection,
    proposal_to_dict,
)
from memory.sediment_sinks import sink_statuses, write_memory_sink, write_skill_sink
from runtime.run_facts import RunFactStore
from tasks.store import TaskStore
from skills.store import SkillStore

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SedimentConfig:
    enabled: bool = True
    skip_inbox: bool = True
    min_trajectory_steps: int = 5
    on_failed: Literal["lesson_only"] = "lesson_only"
    propose_skill: bool = False
    propose_memory: bool = True
    bypass_done_check: bool = False


@dataclass(frozen=True, slots=True)
class SedimentDecision:
    should_run: bool
    reason: str
    memory_type: str | None = None


@dataclass(frozen=True, slots=True)
class SedimentInput:
    task_id: str
    status: str
    task: dict[str, object]
    summary: str
    journal: str
    trajectory: list[dict[str, object]]
    memory_type: str
    skill_versions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SedimentResult:
    status: str
    reason: str
    memory_id: str | None = None
    skill_id: str | None = None
    sink_statuses: dict[str, str] | None = None


# 反思 proposer 产 ReflectionProposal 结构体或扁平 dict（Q6 保留 dict 兼容），
# 两者统一由 run_sediment 经 normalize_reflection 规整为 ReflectionProposal
ReflectionProposer = Callable[[SedimentInput], ReflectionProposal | dict[str, object]]


def should_run_sediment(
    data_root: Path | str,
    task_id: str,
    config: SedimentConfig | None = None,
) -> SedimentDecision:
    cfg = config or SedimentConfig()
    task = _read_task(data_root, task_id)
    if not cfg.enabled:
        return SedimentDecision(False, "disabled")
    if cfg.skip_inbox and bool(task.get("is_inbox", False)):
        return SedimentDecision(False, "skip_inbox")
    if bool(task.get("sediment_done", False)) and not cfg.bypass_done_check:
        return SedimentDecision(False, "already_done")

    steps = len(_read_trajectory(data_root, task_id))
    if steps < cfg.min_trajectory_steps:
        return SedimentDecision(False, "short_task")
    status = str(task.get("status", ""))
    if status == "failed":
        return SedimentDecision(True, "failed_task", "lesson")
    if status == "done":
        return SedimentDecision(True, "done_task", "fact")
    return SedimentDecision(False, "status_not_eligible")


def run_sediment(
    data_root: Path | str,
    task_id: str,
    proposer: ReflectionProposer,
    config: SedimentConfig | None = None,
) -> SedimentResult:
    cfg = config or SedimentConfig()
    decision = should_run_sediment(data_root, task_id, cfg)
    # 1. 白名单判定未命中（非终态/短任务/收件箱等）直接 skip，不记失败
    if not decision.should_run or decision.memory_type is None:
        return SedimentResult("skipped", decision.reason)
    # 2. memory/skill 两个 sink 都关闭时不产出任何经验，skip 且不标 done
    if not cfg.propose_memory and not cfg.propose_skill:
        return SedimentResult("skipped", "no_sinks_enabled")

    try:
        return _run_sinks(data_root, task_id, proposer, cfg, decision.memory_type)
    except Exception as exc:
        _record_failure(data_root, task_id)
        return SedimentResult("failed", str(exc))


def _run_sinks(
    data_root: Path | str,
    task_id: str,
    proposer: ReflectionProposer,
    cfg: SedimentConfig,
    memory_type: str,
) -> SedimentResult:
    # 1. 取任务材料并把 proposer 产物（结构体或扁平 dict）统一规整为 ReflectionProposal
    draft_input = _build_input(data_root, task_id, memory_type)
    proposal = normalize_reflection(
        proposer(draft_input), memory_type, skill_versions=draft_input.skill_versions
    )
    # 2. 按开关分别落 memory / skill sink；缺对应 proposal 时 sink 抛错触发失败
    memory_id = write_memory_sink(data_root, task_id, proposal, cfg.propose_memory)
    skill_id = write_skill_sink(data_root, task_id, proposal, cfg.propose_skill)
    # 3. 两个 sink 都成功后才落 draft 快照并标 done，任一失败不留快照
    _write_draft(data_root, task_id, proposal, force=cfg.bypass_done_check)
    TaskStore(data_root).update_sediment_status(task_id, done=True)
    return SedimentResult(
        "written",
        "ok",
        memory_id=memory_id,
        skill_id=skill_id,
        sink_statuses=sink_statuses(
            cfg.propose_memory, cfg.propose_skill, memory_id, skill_id
        ),
    )


def _write_draft(
    data_root: Path | str,
    task_id: str,
    proposal: ReflectionProposal,
    force: bool = False,
) -> None:
    """保留每次整理快照而非复制备份文件；传参：目标及材料；返回：无。"""
    TaskStore(data_root).append_reflection(task_id, proposal_to_dict(proposal))


def _build_input(
    data_root: Path | str, task_id: str, memory_type: str
) -> SedimentInput:
    return SedimentInput(
        task_id=task_id,
        status=str(_read_task(data_root, task_id).get("status", "")),
        task=_read_task(data_root, task_id),
        summary=_read_text(data_root, task_id, "summary.md"),
        journal=_read_text(data_root, task_id, "journal.md"),
        trajectory=_read_trajectory(data_root, task_id),
        memory_type=memory_type,
        skill_versions={
            skill.skill_id: skill.version
            for skill in SkillStore(data_root).list_skills()
        },
    )


def _record_failure(data_root: Path | str, task_id: str) -> None:
    task = _read_task(data_root, task_id)
    raw_attempts = task.get("sediment_attempts", 0)
    attempts = int(raw_attempts) + 1 if isinstance(raw_attempts, int | str) else 1
    store = TaskStore(data_root)
    if attempts >= 4:
        store.update_sediment_status(task_id, attempts=attempts, failed=True)
        return
    store.update_sediment_status(task_id, attempts=attempts)


def _read_task(data_root: Path | str, task_id: str) -> dict[str, object]:
    return TaskStore(data_root).load_task_payload(task_id)


def _read_text(data_root: Path | str, task_id: str, filename: str) -> str:
    store = TaskStore(data_root)
    return (
        store.read_journal(task_id)
        if filename == "journal.md"
        else store.read_summary_layers(task_id).summary
    )


def _read_trajectory(data_root: Path | str, task_id: str) -> list[dict[str, object]]:
    facts = RunFactStore(data_root).read_task_facts(task_id)
    return [{str(key): value for key, value in fact.items()} for fact in facts]


__all__ = [
    "ReflectionProposer",
    "SedimentConfig",
    "SedimentDecision",
    "SedimentInput",
    "SedimentResult",
    "run_sediment",
    "should_run_sediment",
]
