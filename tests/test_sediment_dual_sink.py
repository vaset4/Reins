from __future__ import annotations

import logging
import json
from pathlib import Path

import pytest

from memory.reflection import (
    MemoryDraftProposal,
    ReflectionProposal,
    SkillCandidateProposal,
)
from memory.sediment import SedimentConfig, SedimentInput, run_sediment
from memory.store import MemoryStore
from runtime.run_facts import RunFactStore
from runtime.persistence import RuntimeStore
from skills.store import SkillStore, build_skill_markdown
from tasks.store import TaskStore


def test_sediment_writes_memory_and_active_skill_by_default(tmp_path: Path) -> None:
    # AC2：默认复核模式（auto_with_notify）→ 沉淀技能直接上架 active、可被召回，
    # meta 落 source_task_id/description/validation 但不盖 draft_source（D8）
    task_id = _task(tmp_path, "dual", "done")

    result = run_sediment(
        tmp_path,
        task_id,
        _dual_proposal,
        SedimentConfig(propose_skill=True),
    )

    assert result.status == "written"
    assert result.memory_id == "memory-dual"
    assert result.skill_id == "skill-dual"
    assert result.sink_statuses == {"memory": "written", "skill": "written"}
    assert _indexed_sediment_done(tmp_path, task_id) is True

    memory = MemoryStore(tmp_path).load_memory("memory-dual")
    assert memory.content == "Focused validation commands should be recorded."
    skill = SkillStore(tmp_path).load_skill("skill-dual")
    assert skill.frontmatter.state == "active"
    assert skill.frontmatter.required_capabilities == ["terminal"]
    assert skill.frontmatter.applicable_task_tags == ["validation"]
    assert "python -m pytest tests/test_example.py -q" in skill.body
    assert "## Validation" in skill.body
    assert "draft_source" not in skill.meta
    assert skill.meta["source_task_id"] == task_id
    assert (
        skill.meta["description"]
        == "Use when a task needs focused validation before commit."
    )
    assert skill.meta["validation"] == "Run focused pytest before commit."
    active_ids = [
        item.skill_id for item in SkillStore(tmp_path).list_skills(active_only=True)
    ]
    assert "skill-dual" in active_ids


def test_sediment_skill_body_hitting_safety_scan_is_refused(tmp_path: Path) -> None:
    # AC4：技能 body 命中安全扫描 → 不分模式一律拒写（不落盘），skill 记 skipped、不拖垮沉淀
    task_id = _task(tmp_path, "unsafe-skill", "done")

    result = run_sediment(
        tmp_path,
        task_id,
        _unsafe_skill_proposal,
        SedimentConfig(propose_skill=True),
    )

    assert result.status == "written"
    assert result.skill_id is None
    assert result.sink_statuses == {"memory": "written", "skill": "skipped"}
    assert not (tmp_path / "skills" / "unsafe-skill").exists()


def test_sediment_skill_body_holding_real_credential_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # 技能正文带真密钥值（不是 API_KEY 这类常量名）时同样拒写——补规则前这种会落盘
    task_id = _task(tmp_path, "leaky-skill", "done")

    with caplog.at_level(logging.ERROR):
        result = run_sediment(
            tmp_path,
            task_id,
            _credential_skill_proposal,
            SedimentConfig(propose_skill=True),
        )

    assert result.skill_id is None
    assert result.sink_statuses == {"memory": "written", "skill": "skipped"}
    assert not (tmp_path / "skills" / "leaky-skill").exists()
    assert "【沉淀】【技能安全拦截】" in caplog.text


def test_sediment_existing_skill_creates_revision_without_overwriting_original(
    tmp_path: Path,
) -> None:
    """同ID的新经验产生修订，保留旧版内容及来源；传参：隔离目录；返回：无。"""
    task_id = _task(tmp_path, "dual", "done")
    existing = SkillStore(tmp_path).create_skill(
        "skill-dual",
        build_skill_markdown(name="Existing", body="Pre-existing real skill."),
    )
    assert existing.body == "Pre-existing real skill."

    result = run_sediment(
        tmp_path,
        task_id,
        _dual_proposal,
        SedimentConfig(propose_skill=True),
    )

    assert result.status == "written"
    assert result.skill_id == "skill-dual"
    assert result.sink_statuses == {"memory": "written", "skill": "written"}
    store = SkillStore(tmp_path)
    assert (
        store.load_skill("skill-dual", version=existing.version).body
        == "Pre-existing real skill."
    )
    assert store.load_skill("skill-dual").previous_version == existing.version
    assert store.load_skill("skill-dual").evaluation_status == "not_evaluated"


def test_sediment_memory_only_sink_skips_skill_write(tmp_path: Path) -> None:
    memory_only = _task(tmp_path, "memory-only", "done")

    memory_result = run_sediment(
        tmp_path,
        memory_only,
        _dual_proposal,
        SedimentConfig(propose_skill=False),
    )

    assert memory_result.sink_statuses == {"memory": "written", "skill": "disabled"}
    assert memory_result.skill_id is None
    assert not (tmp_path / "skills" / "skill-dual").exists()


def test_sediment_no_sinks_enabled_skips_without_done_flag(tmp_path: Path) -> None:
    no_sinks = _task(tmp_path, "no-sinks", "done")

    skipped = run_sediment(
        tmp_path,
        no_sinks,
        _dual_proposal,
        SedimentConfig(propose_memory=False, propose_skill=False),
    )

    assert skipped.status == "skipped"
    assert skipped.reason == "no_sinks_enabled"
    with RuntimeStore(tmp_path).snapshot() as source:
        assert not source.list("task_reflection", filters={"task_id": no_sinks})
    assert _task_payload(tmp_path, no_sinks)["sediment_done"] is False


def test_sediment_skill_only_sink_writes_skill(tmp_path: Path) -> None:
    skill_only = _task(tmp_path, "skill-only", "done")

    skill_result = run_sediment(
        tmp_path,
        skill_only,
        _skill_only_proposal,
        SedimentConfig(propose_memory=False, propose_skill=True),
    )

    assert skill_result.sink_statuses == {"memory": "disabled", "skill": "written"}
    assert skill_result.memory_id is None
    assert _indexed_sediment_done(tmp_path, skill_only) is True
    assert MemoryStore(tmp_path).list_memories() == []


def test_requested_skill_sink_missing_proposal_skips(tmp_path: Path) -> None:
    # skill 可选语义：propose_skill=True 但 LM 判定无可复用经验、只产 memory
    # → memory 照落、skill 跳过（非失败）、标 done；不因缺可选 skill 拖垮沉淀
    task_id = _task(tmp_path, "missing-skill", "done")

    result = run_sediment(
        tmp_path,
        task_id,
        _memory_only_proposal,
        SedimentConfig(propose_skill=True),
    )

    payload = _task_payload(tmp_path, task_id)
    assert result.status == "written"
    assert result.memory_id == "memory-only"
    assert result.skill_id is None
    assert result.sink_statuses == {"memory": "written", "skill": "skipped"}
    assert payload["sediment_done"] is True
    assert not list((tmp_path / "skills").glob("*"))


def test_skill_only_sink_no_skill_produced_marks_done_without_write(
    tmp_path: Path,
) -> None:
    # degenerate 边界：propose_memory=False + propose_skill=True + LM 未产 skill
    # → 两侧皆无产出，标 done、不重试（沉淀跑过、LM 判定无可存，非假成功）
    task_id = _task(tmp_path, "skill-only-empty", "done")

    result = run_sediment(
        tmp_path,
        task_id,
        _memory_only_proposal,
        SedimentConfig(propose_memory=False, propose_skill=True),
    )

    assert result.status == "written"
    assert result.memory_id is None
    assert result.skill_id is None
    assert result.sink_statuses == {"memory": "disabled", "skill": "skipped"}
    assert _indexed_sediment_done(tmp_path, task_id) is True


def test_invalid_skill_proposal_fails_without_done_flag(tmp_path: Path) -> None:
    task_id = _task(tmp_path, "invalid-skill", "done")

    result = run_sediment(
        tmp_path,
        task_id,
        _invalid_skill_proposal,
        SedimentConfig(propose_memory=False, propose_skill=True),
    )

    payload = _task_payload(tmp_path, task_id)
    assert result.status == "failed"
    assert result.reason == "skill body is required"
    assert payload["sediment_done"] is False
    assert payload["sediment_attempts"] == 1
    assert not (tmp_path / "skills" / "skill-invalid").exists()


def test_sediment_no_id_rerun_same_content_fails_with_conflict_reason(
    tmp_path: Path,
) -> None:
    # AC11：沉淀 memory 无 id、重跑相同 content → 第二遍撞写入冲突被拒 →
    # _write_memory 抛诚实错、原因串含撞上的 memory_id（非误标「backlog is full」）
    first_task = _task(tmp_path, "sed-conflict-1", "done")
    first = run_sediment(
        tmp_path,
        first_task,
        _no_id_memory_proposal,
        SedimentConfig(propose_memory=True, propose_skill=False),
    )
    assert first.status == "written"
    assert first.memory_id is not None
    existing_id = first.memory_id

    second_task = _task(tmp_path, "sed-conflict-2", "done")
    second = run_sediment(
        tmp_path,
        second_task,
        _no_id_memory_proposal,
        SedimentConfig(propose_memory=True, propose_skill=False),
    )

    assert second.status == "failed"
    assert second.reason is not None
    assert existing_id in second.reason
    assert "backlog" not in second.reason


def _task(data_root: Path, task_id: str, status: str) -> str:
    store = TaskStore(data_root)
    store.create_task(task_id, task_id=task_id)
    store.update_task_status(task_id, status)
    fact_store = RunFactStore(data_root)
    for index in range(5):
        fact_store.append(
            {
                "event": "step",
                "ts": f"2026-05-30T00:00:{index:02d}+00:00",
                "session_id": f"session-{task_id}",
                "run_id": f"run-{task_id}",
                "task_id": task_id,
                "index": index,
            }
        )
    store.update_summary(task_id, "summary")
    store.append_journal(task_id, "journal")
    return task_id


def _indexed_sediment_done(data_root: Path, task_id: str) -> bool:
    """从已追平原件的派生索引读取整理状态；参数：数据根和目标身份；返回：完成标记。"""
    with RuntimeStore(data_root).index_connection() as connection:
        row = connection.execute(
            "SELECT payload FROM records WHERE kind = ? AND record_id = ?",
            ("task", task_id),
        ).fetchone()
    assert row is not None
    return json.loads(row[0])["sediment_done"]


def _task_payload(data_root: Path, task_id: str) -> dict[str, object]:
    """读取包含扩展字段的已提交任务；参数：数据根和目标身份；返回：任务原件内容。"""
    return TaskStore(data_root).load_task_payload(task_id)


def _dual_proposal(draft: SedimentInput) -> ReflectionProposal:
    return ReflectionProposal(
        memory=MemoryDraftProposal(
            memory_id="memory-dual",
            type=draft.memory_type,
            content="Focused validation commands should be recorded.",
            tags=["validation"],
        ),
        skill=_skill("skill-dual"),
    )


def _skill_only_proposal(_draft: SedimentInput) -> ReflectionProposal:
    return ReflectionProposal(memory=None, skill=_skill("skill-only"))


def _no_id_memory_proposal(draft: SedimentInput) -> ReflectionProposal:
    # 无 memory_id：底层每次 new_ulid()，重跑同 content 触发写入冲突（AC11）
    return ReflectionProposal(
        memory=MemoryDraftProposal(
            memory_id=None,
            type=draft.memory_type,
            content="Duplicate sediment content without an id.",
            tags=["validation"],
        ),
        skill=None,
    )


def _memory_only_proposal(draft: SedimentInput) -> ReflectionProposal:
    return ReflectionProposal(
        memory=MemoryDraftProposal(
            memory_id="memory-only",
            type=draft.memory_type,
            content="Keep focused tests before commit.",
            tags=["validation"],
        ),
        skill=None,
    )


def _invalid_skill_proposal(_draft: SedimentInput) -> ReflectionProposal:
    return ReflectionProposal(memory=None, skill=_skill("skill-invalid", body=""))


def _unsafe_skill_proposal(draft: SedimentInput) -> ReflectionProposal:
    # 技能 body 混入密钥常量名，用于验证输出侧安全扫描拦截（AC4/R3）
    return ReflectionProposal(
        memory=MemoryDraftProposal(
            memory_id="memory-dual",
            type=draft.memory_type,
            content="Focused validation commands should be recorded.",
            tags=["validation"],
        ),
        skill=_skill(
            "unsafe-skill",
            body="Export the API_KEY value before running the tests.",
        ),
    )


def _credential_skill_proposal(draft: SedimentInput) -> ReflectionProposal:
    # 技能 body 混入真实密钥值（非常量名），验证密钥值规则在技能侧同样生效
    return ReflectionProposal(
        memory=MemoryDraftProposal(
            memory_id="memory-leaky",
            type=draft.memory_type,
            content="Focused validation commands should be recorded.",
            tags=["validation"],
        ),
        skill=_skill(
            "leaky-skill",
            body=f"Authenticate with ghp_{'A' * 36} before running the suite.",
        ),
    )


def _skill(
    skill_id: str,
    body: str = "Run `python -m pytest tests/test_example.py -q` before commit.",
) -> SkillCandidateProposal:
    return SkillCandidateProposal(
        skill_id=skill_id,
        name="Focused Validation",
        description="Use when a task needs focused validation before commit.",
        body=body,
        required_capabilities=["terminal"],
        validation="Run focused pytest before commit.",
        applicable_task_tags=["validation"],
    )
