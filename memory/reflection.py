from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class MemoryDraftProposal:
    memory_id: str | None
    type: str | None
    content: str
    tags: list[str]


@dataclass(frozen=True, slots=True)
class SkillCandidateProposal:
    skill_id: str
    name: str
    description: str
    body: str
    required_capabilities: list[str]
    validation: str
    applicable_task_tags: list[str]
    expected_version: str | None = None
    change_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ReflectionProposal:
    memory: MemoryDraftProposal | None
    skill: SkillCandidateProposal | None


RawReflectionProposal = ReflectionProposal | Mapping[str, object]


def normalize_reflection(
    raw: RawReflectionProposal,
    default_memory_type: str,
    *,
    skill_versions: Mapping[str, str] | None = None,
) -> ReflectionProposal:
    if isinstance(raw, ReflectionProposal):
        proposal = raw
    elif isinstance(raw, Mapping):
        proposal = _proposal_from_mapping(raw, default_memory_type)
    else:
        raise TypeError("reflection proposal must be ReflectionProposal or dict")
    if proposal.skill is not None and proposal.skill.expected_version is None:
        proposal = replace(
            proposal,
            skill=replace(
                proposal.skill,
                expected_version=(skill_versions or {}).get(proposal.skill.skill_id),
            ),
        )
    return _with_memory_type(proposal, default_memory_type)


def proposal_to_dict(proposal: ReflectionProposal) -> dict[str, object]:
    return {
        "memory": _memory_to_dict(proposal.memory),
        "skill": _skill_to_dict(proposal.skill),
    }


def _proposal_from_mapping(
    raw: Mapping[str, object], default_memory_type: str
) -> ReflectionProposal:
    if "memory" in raw or "skill" in raw:
        return ReflectionProposal(
            memory=_optional_memory(raw.get("memory"), default_memory_type),
            skill=_optional_skill(raw.get("skill")),
        )
    return ReflectionProposal(
        memory=_memory_from_mapping(raw, default_memory_type),
        skill=None,
    )


def _optional_memory(
    value: object, default_memory_type: str
) -> MemoryDraftProposal | None:
    if value is None:
        return None
    return _memory_from_mapping(_require_mapping(value, "memory"), default_memory_type)


def _optional_skill(value: object) -> SkillCandidateProposal | None:
    if value is None:
        return None
    return _skill_from_mapping(_require_mapping(value, "skill"))


def _memory_from_mapping(
    raw: Mapping[str, object], default_memory_type: str
) -> MemoryDraftProposal:
    content = _required_text(raw.get("content"), "memory content")
    memory_type = _optional_text(raw.get("type")) or default_memory_type
    return MemoryDraftProposal(
        memory_id=_optional_text(raw.get("memory_id")),
        type=memory_type,
        content=content,
        tags=_string_list(raw.get("tags")),
    )


def _skill_from_mapping(raw: Mapping[str, object]) -> SkillCandidateProposal:
    return SkillCandidateProposal(
        skill_id=_required_text(raw.get("skill_id"), "skill id"),
        name=_required_text(raw.get("name"), "skill name"),
        description=_required_text(raw.get("description"), "skill description"),
        body=_required_text(raw.get("body"), "skill body"),
        required_capabilities=_string_list(raw.get("required_capabilities")),
        validation=_required_text(raw.get("validation"), "skill validation"),
        applicable_task_tags=_string_list(raw.get("applicable_task_tags")),
        expected_version=_optional_text(raw.get("expected_version")),
        change_reason=_optional_text(raw.get("change_reason")),
    )


def _with_memory_type(
    proposal: ReflectionProposal, default_memory_type: str
) -> ReflectionProposal:
    memory = proposal.memory
    if memory is None:
        return proposal
    memory_type = "lesson" if default_memory_type == "lesson" else memory.type
    return replace(proposal, memory=replace(memory, type=memory_type))


def _memory_to_dict(proposal: MemoryDraftProposal | None) -> dict[str, object] | None:
    if proposal is None:
        return None
    return {
        "memory_id": proposal.memory_id,
        "type": proposal.type,
        "content": proposal.content,
        "tags": proposal.tags,
    }


def _skill_to_dict(proposal: SkillCandidateProposal | None) -> dict[str, object] | None:
    if proposal is None:
        return None
    return {
        "skill_id": proposal.skill_id,
        "name": proposal.name,
        "description": proposal.description,
        "body": proposal.body,
        "required_capabilities": proposal.required_capabilities,
        "validation": proposal.validation,
        "applicable_task_tags": proposal.applicable_task_tags,
        "expected_version": proposal.expected_version,
        "change_reason": proposal.change_reason,
    }


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} proposal must be an object")
    return value


def _required_text(value: object, name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise ValueError(f"{name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item).strip())]


__all__ = [
    "MemoryDraftProposal",
    "RawReflectionProposal",
    "ReflectionProposal",
    "SkillCandidateProposal",
    "normalize_reflection",
    "proposal_to_dict",
]
