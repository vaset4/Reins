from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from skills.store import SkillStore, build_skill_markdown
from tools.skill_tool import read_skill


def test_invalid_frontmatter_cannot_be_loaded_as_valid_metadata(tmp_path: Path) -> None:
    """无效作者元数据和损坏已发布指南均明确失败；参数：隔离目录；返回：无。"""
    store = SkillStore(tmp_path)
    valid = store.create_skill(
        "valid",
        build_skill_markdown(name="Valid", body="Use it."),
        meta={},
    )
    invalid = "---\nname: Bad\n---\nBody\n"
    with pytest.raises(ValidationError, match="created_at"):
        store.create_skill("bad", invalid)
    assert not store.skill_exists("bad")
    assert [skill.skill_id for skill in store.list_skills()] == ["valid"]

    (valid.root / "SKILL.md").write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError, match="skill resource differs"):
        store.list_skills()
    with pytest.raises(ValueError, match="skill resource differs"):
        read_skill(tmp_path, "valid")


def test_metadata_loading_keeps_script_as_data(tmp_path: Path) -> None:
    """正式目录、元数据和指南读取不得执行脚本顶层代码；参数：隔离目录；返回：无。"""
    marker = tmp_path / "script-executed.txt"
    script = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
    store = SkillStore(tmp_path)
    created = store.create_skill(
        "read-only",
        build_skill_markdown(name="Read only", body="Read the guide."),
        script=script,
        meta={"script_entry": "script.py:main"},
    )

    assert [skill.skill_id for skill in store.list_skills()] == ["read-only"]
    loaded = store.load_skill("read-only")
    assert loaded.version == created.version
    assert loaded.script_path is not None
    assert loaded.script_path.read_text(encoding="utf-8") == script
    assert read_skill(tmp_path, "read-only")["body"] == "Read the guide."
    assert not marker.exists()
