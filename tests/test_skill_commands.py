from __future__ import annotations

import sys
from pathlib import Path

import pytest

from skills.store import SkillStore, build_skill_markdown


def test_cli_skill_health_command_is_retired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from app import cli

    project = tmp_path / "project"
    data_root = project / ".reins" / "data"
    project.mkdir()
    store = SkillStore(data_root)
    store.create_skill(
        "fragile",
        build_skill_markdown(
            name="Fragile",
            body="Usually fails.",
            trigger_keywords=["Fragile skill."],
        ),
        meta={},
    )
    for _index in range(10):
        store.update_skill_stats("fragile", success=False)
    monkeypatch.setenv("REINS_PROJECT_ROOT", str(project))
    monkeypatch.setattr(sys, "argv", ["reins", "skill", "health"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert "invalid choice: 'skill'" in capsys.readouterr().err


def test_cli_skill_archive_command_is_retired_without_state_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from app import cli

    project = tmp_path / "project"
    data_root = project / ".reins" / "data"
    project.mkdir()
    SkillStore(data_root).create_skill(
        "toggle",
        build_skill_markdown(
            name="Toggle",
            body="Body.",
            trigger_keywords=["Toggle skill state."],
        ),
        meta={},
    )
    monkeypatch.setenv("REINS_PROJECT_ROOT", str(project))
    monkeypatch.setattr(sys, "argv", ["reins", "skill", "archive", "toggle"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert SkillStore(data_root).load_skill("toggle").frontmatter.state == "active"
    assert "invalid choice: 'skill'" in capsys.readouterr().err
