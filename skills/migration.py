from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from skills.store import (
    SkillFrontmatter,
    _read_frontmatter,
    _read_yaml,
    _write_frontmatter,
)


@dataclass(frozen=True, slots=True)
class SkillMigrationReport:
    moved_scripts: int = 0
    updated_descriptions: int = 0
    unchanged: int = 0
    failed: int = 0


def migrate_legacy_skills(data_root: Path | str) -> SkillMigrationReport:
    skills_root = Path(data_root) / "skills"
    moved = updated = unchanged = failed = 0
    for root in sorted(skills_root.glob("*")):
        if not root.is_dir():
            continue
        try:
            moved_now, updated_now = _migrate_skill_root(root)
        except Exception:
            failed += 1
            continue
        moved += int(moved_now)
        updated += int(updated_now)
        unchanged += int(not moved_now and not updated_now)
    return SkillMigrationReport(moved, updated, unchanged, failed)


def _migrate_skill_root(root: Path) -> tuple[bool, bool]:
    _ensure_standard_dirs(root)
    moved = _migrate_script(root)
    frontmatter, body = _read_frontmatter(root / "SKILL.md")
    updated = _migrate_description(root, frontmatter, body)
    return moved, updated


def _ensure_standard_dirs(root: Path) -> None:
    (root / "scripts").mkdir(exist_ok=True)
    (root / "references").mkdir(exist_ok=True)
    (root / "assets").mkdir(exist_ok=True)


def _migrate_script(root: Path) -> bool:
    legacy_script = root / "script.py"
    if not legacy_script.is_file():
        return False
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    legacy_script.replace(scripts_dir / "script.py")
    meta = _read_yaml(root / "meta.yaml")
    meta["script_entry"] = "scripts/script.py:main"
    (root / "meta.yaml").write_text(
        yaml.safe_dump(meta, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return True


def _migrate_description(root: Path, frontmatter: dict[str, object], body: str) -> bool:
    if str(frontmatter.get("description", "")).strip():
        return False
    keywords = _keyword_list(frontmatter.get("trigger_keywords", []))
    if not keywords:
        SkillFrontmatter.model_validate(frontmatter)
        return False
    frontmatter["description"] = "Activation keywords: " + ", ".join(keywords)
    _write_frontmatter(
        root / "SKILL.md", SkillFrontmatter.model_validate(frontmatter), body
    )
    return True


def _keyword_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item]


__all__ = ["SkillMigrationReport", "migrate_legacy_skills"]
