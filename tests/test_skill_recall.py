from __future__ import annotations

from pathlib import Path

from context.skill_recall import recall_skills
from skills.store import SkillStore, build_skill_markdown


def test_chinese_skill_body_has_text_score(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)
    store.create_skill(
        "meal-planning",
        build_skill_markdown(
            name="meal-planning",
            body="根据用户饮食偏好安排晚餐，保持清淡。",
            trigger_keywords=[],
            applicable_task_tags=[],
        ),
        meta={},
    )

    results = recall_skills(
        tmp_path, task_summary="饮食偏好", task_tags=[], recent_user_text="晚餐"
    )

    assert results
    assert results[0].text_score > 0


def test_skill_recall_prioritizes_trigger_keyword_hits(tmp_path: Path) -> None:
    store = SkillStore(tmp_path)
    for skill_id, keyword, body in [
        ("pytest-runner", "pytest", "Run focused tests."),
        ("git-helper", "commit", "Prepare git commit."),
        ("docs-helper", "docs", "Update docs."),
        ("memory-helper", "memory", "Write memory."),
        ("browser-helper", "browser", "Use browser."),
    ]:
        store.create_skill(
            skill_id,
            build_skill_markdown(
                name=skill_id,
                body=body,
                trigger_keywords=[keyword],
                applicable_task_tags=[keyword],
            ),
            meta={},
        )

    results = recall_skills(
        tmp_path,
        task_summary="run tests before commit",
        task_tags=["pytest"],
        recent_user_text="please run pytest",
    )

    assert [item.skill.skill_id for item in results] == [
        "pytest-runner",
        "git-helper",
    ]
    assert SkillStore(tmp_path).load_skill("pytest-runner").frontmatter.last_used_at


def test_unrelated_skills_are_not_injected_or_marked_used(tmp_path: Path) -> None:
    """没有相关证据的技能保持可发现但不自动注入；传参：临时根；返回：无。"""
    store = SkillStore(tmp_path)
    store.create_skill(
        "css-layout",
        build_skill_markdown(name="css-layout", body="CSS grid layout"),
        meta={},
    )
    result = recall_skills(
        tmp_path, task_summary="整理报销发票", task_tags=[], recent_user_text="核对金额"
    )
    assert result == []
    assert not store.load_skill("css-layout").frontmatter.last_used_at
