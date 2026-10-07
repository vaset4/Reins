from __future__ import annotations

import logging
import hashlib
import json
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from memory.reflection import (
    MemoryDraftProposal,
    ReflectionProposal,
    SkillCandidateProposal,
)
from memory.safety_scan import scan as safety_scan
from memory.records import MemoryDetails, MemorySource
from memory.writer import MemoryWriteResult, MemoryWriter
from skills.store import SkillStore, build_skill_markdown

_LOG = logging.getLogger(__name__)


def write_memory_sink(
    data_root: Path | str,
    task_id: str,
    proposal: ReflectionProposal,
    enabled: bool,
) -> str | None:
    if not enabled:
        return None
    if proposal.memory is None:
        raise ValueError("missing memory proposal")
    return _write_memory(data_root, task_id, proposal.memory)


def write_skill_sink(
    data_root: Path | str,
    task_id: str,
    proposal: ReflectionProposal,
    enabled: bool,
) -> str | None:
    if not enabled:
        return None
    # skill 可选：LM 判定本任务无跨任务可复用经验时省略 skill，跳过落库而非判失败
    # （区别于 memory 必产——缺 memory 仍在 write_memory_sink 抛错）
    if proposal.skill is None:
        return None
    return _write_skill_candidate(data_root, task_id, proposal.skill)


def sink_statuses(
    memory_enabled: bool,
    skill_enabled: bool,
    memory_id: str | None,
    skill_id: str | None,
) -> dict[str, str]:
    return {
        "memory": "written"
        if memory_id is not None
        else _disabled_or_missing(memory_enabled),
        "skill": _skill_status(skill_enabled, skill_id),
    }


def _skill_status(enabled: bool, skill_id: str | None) -> str:
    # skill 可选：已写=written；开启但 LM 未产=skipped（正常）；开关关=disabled
    if skill_id is not None:
        return "written"
    return "skipped" if enabled else "disabled"


def _write_memory(
    data_root: Path | str, task_id: str, proposal: MemoryDraftProposal
) -> str:
    memory_type = proposal.type or "fact"
    content = _required_text(proposal.content, "sediment memory content")
    # 复用 MemoryWriter 的 safety_scan+review_mode 编排，安全逻辑单一来源（Q3）
    with closing(
        MemoryWriter(data_root, config_path=Path(data_root) / "config.yaml")
    ) as writer:
        result = writer.write_memory(
            memory_type,
            content,
            proposal.tags,
            applicable_task_tags=[task_id],
            memory_id=proposal.memory_id,
            details=MemoryDetails(
                sources=(MemorySource("model_inference", f"task:{task_id}"),)
            ),
        )
    # 命中隐私闸门或近重复冲突被拒写时无 memory_id
    # 显式抛错交给 run_sediment 记失败，不伪装成功、不静默标 done（R6）
    if result.memory_id is None:
        raise ValueError(_blocked_reason(result))
    return result.memory_id


def _blocked_reason(result: MemoryWriteResult) -> str:
    # 把 Writer 的拒写原因转成沉淀失败原因，便于交接排查
    if result.blocked_by_safety_scan:
        return "sediment memory blocked by safety scan"
    # 撞写入冲突：本条已存在于库中，原因串带撞上的 memory_id
    if result.blocked_by_conflict:
        return (
            f"sediment memory skipped: duplicates existing memory "
            f"{result.conflict_with}"
        )
    # manual 积压闸已退役；走到这里说明 Writer 拒写但没标已知原因，保底不伪装成功
    return "sediment memory skipped: no memory_id"


@dataclass(frozen=True, slots=True)
class _ValidatedSkill:
    """校验后的技能落库载荷，收口多处传参并保证必填非空

    字段:
        skill_id: 技能唯一标识兼落盘目录名
        name: 技能名称，写入 frontmatter
        description: 技能用途说明，落 meta（frontmatter 无此字段）
        body: 已拼好 ## Validation 小节的技能正文
        validation: 验证说明，落 meta 供溯源
        proposal: 原始候选载荷，取 required_capabilities/applicable_task_tags 用
    """

    skill_id: str
    name: str
    description: str
    body: str
    validation: str
    proposal: SkillCandidateProposal


def _write_skill_candidate(
    data_root: Path | str, task_id: str, proposal: SkillCandidateProposal
) -> str | None:
    """显式旧沉淀入口也通过版本写者保存，不跳过同ID新经验；传参：目录、任务、候选；返回：方法ID或安全拒写。"""
    # 1. 校验必填字段，空 body/id/name/validation 仍诚实抛错（R6，不因自动转正放松）
    validated = _validate_skill(proposal)
    store = SkillStore(data_root)
    # 3. 输出侧安全扫描：模型产出的 skill body 落盘前过 safety_scan（补齐 E7 输出侧缺口）
    #    不安全内容一律拒写（不分模式，R3/D3 两分支收窄），敏感串不落可召回池
    if not safety_scan(validated.body).is_safe:
        _LOG.error(
            "【沉淀】【技能安全拦截】skill_id=%s 命中安全扫描，拒写不落库",
            validated.skill_id,
        )
        return None
    # 【沉淀】【方法发布】显式补扫沿用启用策略，但保存、脚本退出和任务效果始终分别表达
    return _write_active_skill(store, task_id, validated)


def _validate_skill(proposal: SkillCandidateProposal) -> _ValidatedSkill:
    # 校验并拼装技能落库载荷，空必填字段在此诚实抛错（R6）
    validation = _required_text(proposal.validation, "skill validation")
    body = _skill_body_with_validation(
        _required_text(proposal.body, "skill body"), validation
    )
    return _ValidatedSkill(
        skill_id=_required_text(proposal.skill_id, "skill id"),
        name=_required_text(proposal.name, "skill name"),
        description=_required_text(proposal.description, "skill description"),
        body=body,
        validation=validation,
        proposal=proposal,
    )


def _write_active_skill(
    store: SkillStore, task_id: str, validated: _ValidatedSkill
) -> str:
    """以冻结的原版本承接显式补扫，新经验不覆盖原件；传参：写者、任务、候选；返回：方法ID。"""
    # 自拼 frontmatter：trigger_keywords 无上游来源留空，召回靠 body+tags 打分
    skill_md = build_skill_markdown(
        name=validated.name,
        body=validated.body,
        required_capabilities=validated.proposal.required_capabilities,
        applicable_task_tags=validated.proposal.applicable_task_tags,
    )
    sources = (MemorySource("model_inference", f"task:{task_id}"),)
    meta = {
        "source_task_id": task_id,
        "description": validated.description,
        "validation": validated.validation,
    }
    reason = validated.proposal.change_reason or f"new experience from task {task_id}"
    digest = hashlib.sha256(
        json.dumps(
            [task_id, validated.body, meta], sort_keys=True, ensure_ascii=False
        ).encode()
    ).hexdigest()
    change_id = f"sediment-{digest}"
    if store.skill_exists(validated.skill_id):
        expected = validated.proposal.expected_version
        if expected is None:
            raise ValueError(
                "existing skill revision requires the version read before reflection"
            )
        previous = store.load_skill(validated.skill_id, version=expected)
        skill = store.revise_skill(
            validated.skill_id,
            skill_md,
            expected_version=expected,
            reason=reason,
            sources=sources,
            meta={**previous.meta, **meta},
            change_id=change_id,
            publish=True,
        )
    else:
        skill = store.create_skill(
            validated.skill_id,
            skill_md,
            meta=meta,
            sources=sources,
            reason=reason,
            change_id=change_id,
            publish=True,
        )
    return skill.skill_id


def _skill_body_with_validation(body: str, validation: str) -> str:
    if "## Validation" in body:
        return body
    return f"{body}\n\n## Validation\n\n{validation}"


def _disabled_or_missing(enabled: bool) -> str:
    return "missing" if enabled else "disabled"


def _required_text(value: str, name: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text


__all__ = [
    "sink_statuses",
    "write_memory_sink",
    "write_skill_sink",
]
