"""提供可继续翻页的技能发现和有版本的正文读取。

作者：xxx
时间：2026-09-14 21:00:00
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import cast

from skills.store import SKILL_STATE_ACTIVE, Skill, SkillStore
from tools.catalog import DEFAULT_CATALOG_PAGE_SIZE, catalog_page, matches_query
from tools.types import ToolError, ToolErrorCategory

SKILL_PREVIEW_CHARS = 160


def skill_version(skill: Skill) -> str:
    """返回指南、元数据及脚本共同决定的固定版本；传参：技能；返回：内容SHA256。"""
    return skill.version


def _skill_entries(data_root: Path | str, query: str) -> list[dict[str, object]]:
    """在有效技能中检索正文与触发词，不把发现计为使用；传参：存储根和查询；返回：有加载入口的目录。"""
    result: list[dict[str, object]] = []
    for skill in SkillStore(data_root).list_skills(active_only=True):
        searchable = f"{skill.skill_id} {skill.frontmatter.name} {skill.body} {' '.join(skill.frontmatter.trigger_keywords)}"
        if not matches_query(query, searchable):
            continue
        version = skill_version(skill)
        result.append(
            {
                "skill_id": skill.skill_id,
                "name": skill.frontmatter.name,
                "preview": skill.body[:SKILL_PREVIEW_CHARS],
                "version": version,
                "resource_ref": f"skill:{skill.skill_id}@{version}",
                "read_action": {"skill_id": skill.skill_id, "version": version},
            }
        )
    return result


def skill_search(
    data_root: Path | str, query: str, *, k: int = 3
) -> list[dict[str, object]]:
    """保留程序内搜索入口，并返回可读引用；传参：存储根、查询与数量；返回：匹配条目。"""
    return _skill_entries(data_root, query)[:k]


def read_skill(
    data_root: Path | str,
    skill_id: str,
    *,
    version: str | None = None,
    trial: bool = False,
) -> dict[str, object]:
    """读取有效方法的完整正文及要求；传参：存储根、身份及期望版本；返回：方法材料，不执行脚本。"""
    store = SkillStore(data_root)
    skill = store.load_skill(skill_id, version=version)
    current = skill_version(skill)
    if skill.frontmatter.state != SKILL_STATE_ACTIVE and not (
        trial and version is not None and skill.frontmatter.state == "draft"
    ):
        raise ValueError(
            f"skill is {skill.frontmatter.state}; select an available version or explicitly trial a draft"
        )
    skill = store.note_guide_load(skill_id, current)
    return {
        "skill_id": skill.skill_id,
        "name": skill.frontmatter.name,
        "version": current,
        "resource_ref": f"skill:{skill.skill_id}@{current}",
        "body": skill.body,
        "required_capabilities": skill.frontmatter.required_capabilities,
        "metadata": skill.meta,
        "state": skill.frontmatter.state,
        "previous_version": skill.previous_version,
        "sources": [asdict(source) for source in skill.sources],
        "reason": skill.reason,
        "resources": [
            {
                "name": name,
                "read_action": {"tool": "file_read", "path": str(skill.root / name)},
            }
            for name in skill.resources
        ],
        "evaluation_status": skill.evaluation_status,
        "evidence_summary": skill.evidence_summary,
    }


def executor(args: dict[str, object]) -> dict[str, object] | ToolError:
    """执行分页搜索，空query枚举全部；传参：注入存储根、查询及游标；返回：有版本的目录页。"""
    data_root = args.get("__data_root__")
    if not isinstance(data_root, (str, Path)):
        return ToolError(
            ToolErrorCategory.INVALID_INPUT,
            "skill_search_no_data_root",
            retryable=False,
        )
    query = str(args.get("query", ""))
    try:
        return catalog_page(
            _skill_entries(data_root, query),
            kind="skills",
            query=query,
            cursor=cast(str | None, args.get("cursor")),
            limit=cast(int, args.get("limit", DEFAULT_CATALOG_PAGE_SIZE)),
        )
    except ValueError as exc:
        return ToolError(ToolErrorCategory.INVALID_INPUT, str(exc), retryable=False)


def read_executor(args: dict[str, object]) -> dict[str, object] | ToolError:
    """接通模型发现后的正文读取；传参：身份、版本与注入根；返回：正文或可定位的读取错误。"""
    data_root = args.get("__data_root__")
    if not isinstance(data_root, (str, Path)):
        return ToolError(
            ToolErrorCategory.INVALID_INPUT, "skill_read_no_data_root", retryable=False
        )
    try:
        return read_skill(
            data_root,
            str(args["skill_id"]),
            version=cast(str | None, args.get("version")),
            trial=args.get("trial") is True,
        )
    except (FileNotFoundError, ValueError) as exc:
        return ToolError(ToolErrorCategory.INVALID_INPUT, str(exc), retryable=False)


__all__ = ["skill_search", "read_skill", "skill_version"]
