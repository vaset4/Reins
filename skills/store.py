"""方法版本、发布状态和效果证据的唯一写者。

作者：xxx
时间：2026-09-15 02:25:05
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from memory.records import MemorySource, validate_sources
from schedules.persistence import immutable_bytes, record_path
from runtime.persistence import ContentReference, RuntimeStore, record_key
from skills.records import (
    SKILL_FORMAT,
    SKILL_STATE_ACTIVE,
    SKILL_STATE_ARCHIVED,
    Skill,
    SkillContent,
    SkillFrontmatter,
    prepare_content,
    content_from_files,
    resource_path,
    validate_published_scripts,
)
from tasks.ids import new_ulid, utc_now

_MIN_SCRIPT_SAMPLES = 10
_LOW_EXIT_ZERO_RATIO = 0.5


class SkillStore:
    """以完整内容快照为事实源，发布和撤回只改变后续取用。"""

    def __init__(self, data_root: Path | str, *, maintenance: bool = False) -> None:
        """绑定技能目录；传参：数据根；返回：无。"""
        self._data_root = Path(data_root).resolve()
        self._maintenance = maintenance
        self._skills_root = self._data_root / "skills"
        self._db = RuntimeStore(data_root)

    @contextmanager
    def _locked(self, skill_id: str, *, usage: bool = False) -> Iterator[Path]:
        """串行化同一方法的发布与反馈；传参：身份；返回：受锁目录。"""
        if not self._maintenance and (self._data_root / ".message_owner.lock").exists():
            raise RuntimeError("skills are locked for maintenance")
        root = self._skill_root(skill_id)
        with self._db.transaction():
            yield root

    def create_skill(
        self,
        skill_id: str,
        skill_md: str,
        *,
        script: str | None = None,
        meta: dict[str, Any] | None = None,
        resources: dict[str, bytes] | None = None,
        sources: tuple[MemorySource, ...] = (),
        reason: str = "created by author",
        publish: bool = True,
        change_id: str | None = None,
    ) -> Skill:
        """保存初始方法，重复 ID 不覆盖既有正文；传参：指南、资源、出处和发布意图；返回：固定版本。"""
        validate_sources(sources)
        files = dict(resources or {})
        if script is not None:
            files["script.py"] = script.encode("utf-8")
        content = prepare_content(skill_md, meta=meta, resources=files)
        with self._locked(skill_id) as root:
            if self.skill_exists(skill_id):
                old = self.load_skill(skill_id)
                if old.version == content.version:
                    return old
                raise ValueError(
                    "skill already exists; revise it using expected_version and a reason"
                )
            state: dict[str, Any] = {
                "format_version": SKILL_FORMAT,
                "current_version": None,
                "latest_version": content.version,
                "archived": False,
                "versions": {},
                "events": [],
            }
            state = self._commit_version(
                root,
                content,
                state=state,
                previous=None,
                sources=sources,
                reason=reason,
                change_id=change_id,
                publish=publish,
            )
            self._save_publication(root, state)
        return self.load_skill(skill_id, version=content.version)

    def revise_skill(
        self,
        skill_id: str,
        skill_md: str,
        *,
        expected_version: str,
        reason: str,
        sources: tuple[MemorySource, ...],
        script: str | None = None,
        meta: dict[str, Any] | None = None,
        resources: dict[str, bytes] | None = None,
        publish: bool = False,
        change_id: str | None = None,
    ) -> Skill:
        """修订已有方法并关联前版，重投返回原版；传参：新材料、原版本、原因和证据；返回：候选或已发布版本。"""
        if not reason.strip() or not sources:
            raise ValueError("skill revision requires a reason and sources")
        validate_sources(sources)
        with self._locked(skill_id) as root:
            state = self._ensure_versioned(skill_id)
            base = self.load_skill(skill_id, version=expected_version)
            original, _manifest = self._read_snapshot(base.root)
            files = {
                name: raw
                for name, raw in original.files.items()
                if name not in {"SKILL.md", "meta.yaml"}
            }
            files.update(resources or {})
            if script is not None:
                files["script.py"] = script.encode("utf-8")
            content = prepare_content(
                skill_md, meta=base.meta if meta is None else meta, resources=files
            )
            prior = next(
                (
                    item
                    for item in state["events"]
                    if change_id is not None and item.get("change_id") == change_id
                ),
                None,
            )
            if prior is not None:
                if prior["version"] != content.version:
                    raise ValueError(
                        "skill change identity reused with different content"
                    )
                return self.load_skill(skill_id, version=content.version)
            if state["latest_version"] != expected_version:
                raise ValueError(
                    f"skill version conflict: latest={state['latest_version']}"
                )
            state = self._commit_version(
                root,
                content,
                state=state,
                previous=expected_version,
                sources=sources,
                reason=reason,
                change_id=change_id,
                publish=publish,
            )
            self._save_publication(root, state)
        return self.load_skill(skill_id, version=content.version)

    def load_skill(self, skill_id: str, *, version: str | None = None) -> Skill:
        """读取具体内容版本并叠加当前发布及证据状态；传参：身份和可选版本；返回：方法。"""
        root = self._skill_root(skill_id)
        state = self._publication(root)
        selected = version or state["current_version"] or state["latest_version"]
        if selected not in state["versions"]:
            raise FileNotFoundError(f"skill version not found: {selected}")
        version_root = record_path(root / "versions", selected, suffix="")
        content, manifest = self._read_snapshot(version_root)
        availability = state["versions"][selected]
        status = "active" if availability == "published" else availability
        if state["archived"] and status != "withdrawn":
            status = "archived"
        summary = self._evidence_summary(root, selected)
        frontmatter = content.frontmatter.model_copy(
            update={
                "state": status,
                "last_used_at": summary.get(
                    "last_used_at", content.frontmatter.last_used_at
                ),
                "use_count": content.frontmatter.use_count
                + summary["script_run_count"],
                "success_count": content.frontmatter.success_count
                + summary["script_exit_zero_count"],
            }
        )
        return Skill(
            skill_id,
            frontmatter,
            content.body,
            content.meta,
            version_root,
            version=content.version,
            previous_version=manifest.get("previous_version"),
            reason=str(manifest["reason"]),
            sources=tuple(MemorySource(**item) for item in manifest.get("sources", [])),
            change_id=manifest.get("change_id"),
            evidence_summary=summary,
            resources=tuple(content.files),
        )

    def list_skills(
        self,
        *,
        role: str | None = None,
        applicable_task_tags: list[str] | None = None,
        active_only: bool = False,
    ) -> list[Skill]:
        """枚举已提交方法，未完成快照不进入目录；传参：角色、标签和有效性；返回：方法列表。"""
        skills: list[Skill] = []
        with self._db.snapshot() as source:
            identities = sorted(
                str(row["skill_id"]) for row in source.list("skill_record")
            )
        for identity in identities:
            skill = self.load_skill(identity)
            if role is not None and skill.frontmatter.role != role:
                continue
            if active_only and skill.frontmatter.state != SKILL_STATE_ACTIVE:
                continue
            if not set(applicable_task_tags or []).issubset(
                skill.frontmatter.applicable_task_tags
            ):
                continue
            skills.append(skill)
        return skills

    def list_versions(self, skill_id: str) -> list[dict[str, Any]]:
        """列出内容谱系和当前可用性，不混淆版本与验证；传参：身份；返回：版本视图。"""
        root = self._skill_root(skill_id)
        state = self._publication(root)
        result = []
        for version, status in state["versions"].items():
            skill = self.load_skill(skill_id, version=version)
            result.append(
                {
                    "version": version,
                    "state": status,
                    "current": state["current_version"] == version,
                    "latest": state["latest_version"] == version,
                    "previous_version": skill.previous_version,
                    "reason": skill.reason,
                    "sources": [asdict(source) for source in skill.sources],
                    "evaluation_status": skill.evaluation_status,
                }
            )
        return result

    def publish_version(
        self, skill_id: str, version: str, *, reason: str, change_id: str | None = None
    ) -> Skill:
        """明确启用指定版本，未评估仍保持未评估；传参：身份、版本与原因；返回：发布后的方法。"""
        return self._set_publication(
            skill_id, version, status="published", reason=reason, change_id=change_id
        )

    def withdraw_version(
        self, skill_id: str, version: str, *, reason: str, change_id: str | None = None
    ) -> Skill:
        """撤回错误版本，不隐式回退到旧方法；传参：身份、版本与原因；返回：保留原文的撤回版本。"""
        return self._set_publication(
            skill_id, version, status="withdrawn", reason=reason, change_id=change_id
        )

    def archive_skill(self, skill_id: str) -> Skill:
        """暂时停用整个方法，保留所有版本；传参：身份；返回：归档方法。"""
        return self._set_archived(skill_id, True)

    def restore_skill(self, skill_id: str) -> Skill:
        """取消整体归档，不恢复已撤回内容；传参：身份；返回：实际可用状态。"""
        return self._set_archived(skill_id, False)

    def touch_skill(self, skill_id: str, *, version: str | None = None) -> Skill:
        """只记录目录选用，不推断指南已执行；传参：身份和版本；返回：带使用统计的方法。"""
        return self._record_usage(
            skill_id, version=version, counters={"selection_count": 1}
        )

    def note_guide_load(self, skill_id: str, version: str) -> Skill:
        """记录指南正文读取完成，不推断模型采用或任务效果；传参：身份和版本；返回：统计视图。"""
        return self._record_usage(
            skill_id, version=version, counters={"guide_load_count": 1}
        )

    def update_skill_stats(
        self, skill_id: str, success: bool, *, version: str | None = None
    ) -> Skill:
        """保留脚本调用统计，success 仅指退出零；传参：身份、退出状态和版本；返回：统计视图。"""
        return self._record_usage(
            skill_id,
            version=version,
            counters={"script_run_count": 1, "script_exit_zero_count": int(success)},
        )

    def record_outcome(
        self,
        skill_id: str,
        version: str,
        *,
        outcome: str,
        case_id: str,
        reason: str,
        sources: tuple[MemorySource, ...],
        event_id: str,
    ) -> Skill:
        """按真实案例记录效果评价，独立于脚本退出；传参：方法、版本、案例及来源；返回：含评价的方法。"""
        if (
            outcome not in {"achieved", "failed", "unknown"}
            or not reason.strip()
            or not case_id
        ):
            raise ValueError(
                "skill outcome requires a case, reason and achieved/failed/unknown result"
            )
        validate_sources(sources, verification=True)
        with self._locked(skill_id):
            self.load_skill(skill_id, version=version)
            payload = {
                "kind": "task_outcome",
                "skill_id": skill_id,
                "version": version,
                "case_id": case_id,
                "outcome": outcome,
                "reason": reason,
                "sources": [asdict(source) for source in sources],
                "event_id": event_id,
            }
            with self._db.transaction() as batch:
                old = batch.get("skill_outcome", event_id)
                if old is not None and old != payload:
                    raise ValueError("skill outcome identity has different content")
                if old is None:
                    batch.put("skill_outcome", event_id, payload)
        return self.load_skill(skill_id, version=version)

    def health_warning(self, skill_id: str) -> str | None:
        """低脚本退出率只提示脚本问题，不代表任务成功率；传参：身份；返回：提示或空。"""
        summary = self.load_skill(skill_id).evidence_summary
        count = summary["script_run_count"]
        if (
            count >= _MIN_SCRIPT_SAMPLES
            and summary["script_exit_zero_count"] / count < _LOW_EXIT_ZERO_RATIO
        ):
            return "skill script exit-zero rate is low; inspect script evidence"
        return None

    def skill_exists(self, skill_id: str) -> bool:
        """只认已提交技能身份；传参：技能编号；返回：是否存在。"""
        self._skill_root(skill_id)
        with self._db.snapshot() as source:
            return source.get("skill_record", skill_id) is not None

    def _ensure_versioned(self, skill_id: str) -> dict[str, Any]:
        """读取规范发布状态；传参：技能身份；返回：当前指针。"""
        return self._publication(self._skill_root(skill_id))

    def _commit_version(
        self,
        root: Path,
        content: SkillContent,
        *,
        state: dict[str, Any],
        previous: str | None,
        sources: tuple[MemorySource, ...],
        reason: str,
        change_id: str | None,
        publish: bool,
    ) -> dict[str, Any]:
        """完整快照先落盘，再构造单次发布状态；传参：目录、内容、前驱和出处；返回：待提交状态。"""
        if publish:
            validate_published_scripts(content)
        version_root = root / "versions" / content.version
        if content.version not in state["versions"]:
            self._write_snapshot(
                version_root,
                content,
                {
                    "previous_version": previous,
                    "reason": reason,
                    "sources": [asdict(source) for source in sources],
                    "change_id": change_id,
                    "created_at": utc_now(),
                },
            )
        else:
            existing, _ = self._read_snapshot(version_root)
            if existing.version != content.version:
                raise ValueError(
                    "existing skill snapshot differs from proposed content"
                )
        versions = {
            **state["versions"],
            content.version: "published"
            if publish
            else state["versions"].get(content.version, "draft"),
        }
        event = {
            "action": "publish" if publish else "propose",
            "version": content.version,
            "previous_version": previous,
            "reason": reason,
            "sources": [asdict(source) for source in sources],
            "change_id": change_id,
            "event_id": change_id or new_ulid(),
            "at": utc_now(),
        }
        return {
            **state,
            "latest_version": content.version,
            "versions": versions,
            "current_version": content.version if publish else state["current_version"],
            "events": [*state["events"], event],
        }

    def _set_publication(
        self,
        skill_id: str,
        version: str,
        *,
        status: str,
        reason: str,
        change_id: str | None,
    ) -> Skill:
        """在锁内原子改变指定版本的发布状态；传参：身份、版本、状态和依据；返回：实际版本。"""
        if not reason.strip():
            raise ValueError("skill publication requires a reason")
        with self._locked(skill_id) as root:
            state = self._ensure_versioned(skill_id)
            skill = self.load_skill(skill_id, version=version)
            if any(
                item.get("change_id") == change_id
                for item in state["events"]
                if change_id is not None
            ):
                return self.load_skill(skill_id, version=version)
            current = state["current_version"]
            if status == "published":
                content, _ = self._read_snapshot(skill.root)
                validate_published_scripts(content)
                current = version
            elif current == version:
                current = None
            event = {
                "action": status,
                "version": version,
                "reason": reason,
                "change_id": change_id,
                "event_id": change_id or new_ulid(),
                "at": utc_now(),
            }
            self._save_publication(
                root,
                {
                    **state,
                    "current_version": current,
                    "versions": {**state["versions"], version: status},
                    "events": [*state["events"], event],
                },
            )
        return self.load_skill(skill_id, version=version)

    def _set_archived(self, skill_id: str, archived: bool) -> Skill:
        """整体归档不修改任何内容版本；传参：身份及归档标志；返回：实际方法。"""
        with self._locked(skill_id) as root:
            state = self._ensure_versioned(skill_id)
            self._save_publication(root, {**state, "archived": archived})
        return self.load_skill(skill_id)

    def _record_usage(
        self, skill_id: str, *, version: str | None, counters: dict[str, int]
    ) -> Skill:
        """统计写到独立文件，已取用旧版本不会被新版抢走归属；传参：身份、版本、增量；返回：视图。"""
        with self._locked(skill_id, usage=True):
            skill = self.load_skill(skill_id, version=version)
            with self._db.transaction() as batch:
                key = record_key(skill_id, skill.version)
                previous = batch.get("skill_usage", key) or {}
                updated = {
                    **previous,
                    **{
                        name: previous.get(name, 0) + value
                        for name, value in counters.items()
                    },
                    "skill_id": skill_id,
                    "version": skill.version,
                    "last_used_at": utc_now(),
                }
                batch.put("skill_usage", key, updated)
        return self.load_skill(skill_id, version=skill.version)

    def _evidence_summary(self, root: Path, version: str) -> dict[str, Any]:
        """读取某版本的使用与效果证据；传参：技能目录和版本；返回：真实统计。"""
        with self._db.snapshot() as source:
            usage = source.get("skill_usage", record_key(root.name, version)) or {}
            outcomes = sorted(
                (
                    item
                    for item in source.list("skill_outcome")
                    if item["skill_id"] == root.name and item["version"] == version
                ),
                key=lambda item: item["event_id"],
            )
        return {
            "selection_count": 0,
            "guide_load_count": 0,
            "script_run_count": 0,
            "script_exit_zero_count": 0,
            **usage,
            "task_outcomes": outcomes,
        }

    def _publication(self, root: Path) -> dict[str, Any]:
        """读取唯一发布指针；传参：技能目录；返回：状态，缺失明确失败。"""
        with self._db.snapshot() as source:
            row = source.get("skill_record", root.name)
            if row is None:
                raise FileNotFoundError(root.name)
            state = row["publication"]
        if state.get("format_version") != SKILL_FORMAT or not isinstance(
            state.get("versions"), dict
        ):
            raise ValueError("unsupported skill publication state")
        return cast(dict[str, Any], state)

    def _save_publication(self, root: Path, state: dict[str, Any]) -> None:
        """资源完成后发布版本指针；传参：技能目录和状态；返回：无。"""
        with self._db.transaction() as batch:
            batch.put(
                "skill_record", root.name, {"skill_id": root.name, "publication": state}
            )

    def _write_snapshot(
        self, root: Path, content: SkillContent, revision: dict[str, Any]
    ) -> None:
        """先持久资源，再提交不可变修订元数据；传参：版本目录、材料和出处；返回：无。"""
        for name, raw in content.files.items():
            immutable_bytes(resource_path(root, name), raw)
        manifest = {
            **revision,
            "format_version": SKILL_FORMAT,
            "version": content.version,
            "files": {
                name: hashlib.sha256(raw).hexdigest()
                for name, raw in content.files.items()
            },
        }
        with self._db.transaction() as batch:
            skill_id = root.parent.parent.name
            # 【技能】【发布原件】资源与修订属于同一次提交，重建不能跳过已发布资源缺失
            for name, raw in content.files.items():
                path = resource_path(root, name)
                batch.reference_content(
                    ContentReference(
                        path.relative_to(self._data_root).as_posix(),
                        hashlib.sha256(raw).hexdigest(),
                        len(raw),
                    )
                )
            batch.put(
                "skill_revision",
                record_key(skill_id, content.version),
                {
                    "skill_id": skill_id,
                    "version": content.version,
                    "manifest": manifest,
                },
            )

    def _read_snapshot(self, root: Path) -> tuple[SkillContent, dict[str, Any]]:
        """每次执行前核验真实资源字节；传参：版本目录；返回：材料及出处。"""
        with self._db.snapshot() as source:
            row = source.get(
                "skill_revision", record_key(root.parent.parent.name, root.name)
            )
            if row is None:
                raise FileNotFoundError(f"skill revision missing: {root.name}")
            manifest = row["manifest"]
        hashes = manifest["files"]
        files = {name: resource_path(root, name).read_bytes() for name in hashes}
        if any(
            hashlib.sha256(raw).hexdigest() != hashes[name]
            for name, raw in files.items()
        ):
            raise ValueError("skill resource differs from its immutable version")
        content = content_from_files(files)
        if content.version != manifest["version"]:
            raise ValueError("skill version differs from its manifest")
        return content, manifest

    def _skill_root(self, skill_id: str) -> Path:
        """把外部身份限制在技能存储内；传参：身份；返回：安全目录。"""
        return record_path(self._skills_root, skill_id, suffix="")


def build_skill_markdown(
    *,
    name: str,
    body: str,
    trigger_keywords: list[str] | None = None,
    role: Literal["tool", "subagent"] = "tool",
    applicable_task_tags: list[str] | None = None,
    required_capabilities: list[str] | None = None,
) -> str:
    """构造作者指南，不宣称经过执行验证；传参：方法信息和正文；返回：Markdown。"""
    frontmatter = SkillFrontmatter(
        name=name,
        trigger_keywords=trigger_keywords or [],
        required_capabilities=required_capabilities or [],
        role=role,
        applicable_task_tags=applicable_task_tags or [],
        created_at=utc_now(),
    )
    header = yaml.safe_dump(
        frontmatter.model_dump(), sort_keys=False, allow_unicode=True
    ).strip()
    return f"---\n{header}\n---\n{body.rstrip()}\n"


__all__ = [
    "SKILL_STATE_ACTIVE",
    "SKILL_STATE_ARCHIVED",
    "Skill",
    "SkillFrontmatter",
    "SkillStore",
    "build_skill_markdown",
]
