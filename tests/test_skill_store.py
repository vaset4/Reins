from __future__ import annotations

from pathlib import Path

from context.skill_recall import recall_skills
from skills.store import SkillStore, build_skill_markdown


def test_skill_store_round_trip_and_stats(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)
    skill = store.create_skill(
        "inspect-workspace",
        build_skill_markdown(
            name="Inspect Workspace",
            body="# Inspect\n\nCheck the workspace before edits.",
            trigger_keywords=["inspect"],
            applicable_task_tags=["code"],
        ),
        script="def main(args):\n    return {'ok': args}\n",
        meta={
            "script_entry": "script.py:main",
            "input_schema": {},
            "output_schema": {},
        },
    )

    assert skill.frontmatter.name == "Inspect Workspace"
    assert skill.script_path is not None
    updated = store.update_skill_stats("inspect-workspace", success=True)
    assert updated.frontmatter.use_count == 1
    assert updated.frontmatter.success_count == 1


def test_skill_store_filters_and_archives(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)
    store.create_skill(
        "delegate-research",
        build_skill_markdown(
            name="Delegate Research",
            body="Run a subagent.",
            role="subagent",
            applicable_task_tags=["research"],
        ),
        meta={"role": "subagent"},
    )

    assert [item.skill_id for item in store.list_skills(role="subagent")] == [
        "delegate-research"
    ]
    assert store.archive_skill("delegate-research").frontmatter.state == "archived"
    assert store.restore_skill("delegate-research").frontmatter.state == "active"


def test_skill_health_warning_after_repeated_failures(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)
    store.create_skill(
        "fragile",
        build_skill_markdown(name="Fragile", body="Usually fails."),
        meta={},
    )
    for index in range(10):
        store.update_skill_stats("fragile", success=index < 4)

    assert (
        store.health_warning("fragile")
        == "skill script exit-zero rate is low; inspect script evidence"
    )


def test_archived_skill_excluded_from_active_listing_and_recall(tmp_path: Path) -> None:
    # archived 技能不进 active 列表、不被召回——manual 候选草稿退役后
    # 这条契约由 archive_skill 这条路继续守
    store = SkillStore(tmp_path)
    store.create_skill(
        "active-validation",
        build_skill_markdown(
            name="Active Validation",
            body="Run focused validation tests before commit.",
            applicable_task_tags=["validation"],
        ),
        meta={},
    )
    store.create_skill(
        "archived-validation",
        build_skill_markdown(
            name="Archived Validation",
            body="Run focused validation tests before commit.",
            applicable_task_tags=["validation"],
        ),
        meta={},
    )
    store.archive_skill("archived-validation")

    active_ids = [item.skill_id for item in store.list_skills(active_only=True)]
    assert active_ids == ["active-validation"]

    recalled = recall_skills(
        tmp_path,
        task_summary="run focused validation tests before commit",
        task_tags=["validation"],
    )
    assert [item.skill.skill_id for item in recalled] == ["active-validation"]
