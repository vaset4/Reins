from __future__ import annotations

from pathlib import Path

from skills.store import SkillStore, build_skill_markdown
from tools.skill_tool import skill_search


def test_skill_search_returns_recalled_skills(tmp_path: Path) -> None:
    SkillStore(tmp_path).create_skill(
        "pytest-runner",
        build_skill_markdown(
            name="Pytest Runner",
            body="Run focused pytest tests.",
            trigger_keywords=["pytest"],
        ),
        meta={},
    )

    results = skill_search(tmp_path, "pytest")

    assert results[0]["skill_id"] == "pytest-runner"
