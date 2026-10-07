"""方法内容和版本快照，发布状态与效果证据不混入内容摘要。

作者：xxx
时间：2026-09-15 02:25:05
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, Field

from memory.records import MemorySource
from schedules.persistence import immutable_bytes, read_record

SKILL_FORMAT = 2
SKILL_STATE_ACTIVE: Literal["active"] = "active"
SKILL_STATE_ARCHIVED: Literal["archived"] = "archived"
_NON_CONTENT_FIELDS = {
    "created_at",
    "last_used_at",
    "use_count",
    "success_count",
    "state",
}
_MANIFEST = "revision.json"


class SkillFrontmatter(BaseModel):
    """保留方法元数据接口；状态和统计在读取时叠加，不重写方法正文。"""

    name: str
    trigger_keywords: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    role: Literal["tool", "subagent"] = "tool"
    applicable_task_tags: list[str] = Field(default_factory=list)
    created_at: str
    last_used_at: str | None = None
    use_count: int = 0
    success_count: int = 0
    state: Literal["active", "archived", "draft", "withdrawn"] = SKILL_STATE_ACTIVE


@dataclass(frozen=True, slots=True)
class Skill:
    """固定一个可追溯的方法版本；root 指向它的资源快照，不随新版发布改变。"""

    skill_id: str
    frontmatter: SkillFrontmatter
    body: str
    meta: dict[str, Any]
    root: Path
    version: str = ""
    previous_version: str | None = None
    sources: tuple[MemorySource, ...] = ()
    reason: str = "legacy author content"
    change_id: str | None = None
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    resources: tuple[str, ...] = ()

    @property
    def script_path(self) -> Path | None:
        """定位当前版本内的脚本，拒绝资源引用越界；传参：无；返回：已存在脚本或空。"""
        script_name = str(self.meta.get("script_entry", "")).split(":", 1)[0]
        if not script_name:
            return None
        if script_name not in self.resources:
            raise ValueError("skill script is not part of the selected content version")
        path = resource_path(self.root, script_name)
        return path if path.is_file() else None

    @property
    def evaluation_status(self) -> str:
        """已评估仅表示有任务结果记录，不等同所有任务都有效；传参：无；返回：评估状态。"""
        return (
            "evaluated"
            if self.evidence_summary.get("task_outcomes")
            else "not_evaluated"
        )


@dataclass(frozen=True, slots=True)
class SkillContent:
    """待发布或已读入的完整方法原件，包括参与版本摘要的所有资源。"""

    frontmatter: SkillFrontmatter
    body: str
    meta: dict[str, Any]
    files: dict[str, bytes]
    version: str


def prepare_content(
    skill_md: str,
    *,
    meta: dict[str, Any] | None = None,
    resources: dict[str, bytes] | None = None,
) -> SkillContent:
    """校验作者材料并计算包含脚本的内容版本；传参：指南、元数据和资源；返回：完整快照。"""
    files = dict(resources or {})
    if {"SKILL.md", "meta.yaml", _MANIFEST} & files.keys():
        raise ValueError("skill resources cannot replace content metadata")
    files["SKILL.md"] = skill_md.encode("utf-8")
    files["meta.yaml"] = yaml.safe_dump(
        meta or {}, sort_keys=False, allow_unicode=True
    ).encode("utf-8")
    return content_from_files(files)


def content_from_files(files: dict[str, bytes]) -> SkillContent:
    """按原始字节读取内容，日期和旧统计不参与版本；传参：原件集合；返回：内容快照。"""
    text = files["SKILL.md"].decode("utf-8").replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise ValueError("skill is missing frontmatter")
    _, header, body = text.split("---\n", 2)
    frontmatter = SkillFrontmatter.model_validate(yaml.safe_load(header))
    meta = yaml.safe_load(files["meta.yaml"].decode("utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("skill metadata must be an object")
    resources = {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in files.items()
        if name not in {"SKILL.md", "meta.yaml"}
    }
    metadata = frontmatter.model_dump(exclude=_NON_CONTENT_FIELDS)
    payload = json.dumps(
        [metadata, body.strip(), meta, resources], ensure_ascii=False, sort_keys=True
    )
    version = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return SkillContent(frontmatter, body.strip(), meta, files, version)


def read_legacy_content(root: Path) -> SkillContent:
    """读取旧目录的真实原件供维护转换或历史查看；传参：旧目录；返回：完整内容，不写盘。"""
    files: dict[str, bytes] = {}
    excluded = {"versions", "evidence", "__pycache__"}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if excluded.intersection(relative.parts) or path.name in {
            "publication.json",
            "usage.json",
            ".write.lock",
            ".usage.lock",
        }:
            continue
        if path.is_file():
            files[relative.as_posix()] = resource_path(
                root, relative.as_posix()
            ).read_bytes()
    return content_from_files(files)


def validate_published_scripts(content: SkillContent) -> None:
    """发布前仅编译Python资源，语法错误不能替换可用方法，也不执行脚本；传参：完整版本；返回：无。"""
    for name, raw in content.files.items():
        if not name.endswith(".py"):
            continue
        try:
            compile(raw, name, "exec", dont_inherit=True)
        except SyntaxError as exc:
            raise ValueError(
                f"script syntax invalid: {name}:{exc.lineno}: {exc.msg}; publication was not changed"
            ) from exc


def write_snapshot(
    root: Path, content: SkillContent, revision: Mapping[str, object]
) -> None:
    """先持久完整资源，最后写清单，发布指针由调用者提交；传参：版本目录、内容、修订出处；返回：无。"""
    for name, raw in content.files.items():
        immutable_bytes(resource_path(root, name), raw)
    manifest = {
        **revision,
        "format_version": SKILL_FORMAT,
        "version": content.version,
        "files": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in content.files.items()
        },
    }
    immutable_bytes(
        root / _MANIFEST,
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def read_snapshot(root: Path) -> tuple[SkillContent, dict[str, Any]]:
    """每次取用核对指南与脚本原件，不能用旧摘要掩盖改写；传参：版本目录；返回：内容和出处。"""
    manifest = read_record(root / _MANIFEST)
    if manifest.get("format_version") != SKILL_FORMAT:
        raise ValueError("unsupported skill version manifest")
    hashes = manifest["files"]
    if not isinstance(hashes, dict):
        raise ValueError("skill file manifest must be an object")
    files = {name: resource_path(root, name).read_bytes() for name in hashes}
    if any(
        hashlib.sha256(raw).hexdigest() != hashes[name] for name, raw in files.items()
    ):
        raise ValueError("skill resource differs from its immutable version")
    content = content_from_files(files)
    if content.version != manifest["version"]:
        raise ValueError("skill version differs from its manifest")
    return content, manifest


def resource_path(root: Path, name: str) -> Path:
    """将资源固定在版本目录内；传参：目录与相对名称；返回：安全路径。"""
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or ":" in name:
        raise ValueError("skill resource must be a relative path inside its version")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("skill resource escapes its version")
    return path
