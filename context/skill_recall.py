from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from context.memory_recall import jaccard
from memory.index import tokenize_search_text
from skills.store import SKILL_STATE_ACTIVE, Skill, SkillStore


@dataclass(frozen=True, slots=True)
class SkillRecallResult:
    skill: Skill
    score: float
    text_score: float
    tag_jaccard: float
    trigger_hits: int


def recall_skills(
    data_root: Path | str,
    *,
    task_summary: str,
    task_tags: list[str],
    recent_user_text: str = "",
    k: int = 3,
) -> list[SkillRecallResult]:
    """仅召回有相关证据的技能；传参：目标、标签、近期输入与数量；返回：按关联度排序的技能。"""
    store = SkillStore(data_root)
    query_tokens = _tokens(f"{recent_user_text} {task_summary}")
    results = [
        _score_skill(skill, query_tokens, task_tags, recent_user_text=recent_user_text)
        for skill in store.list_skills(active_only=True)
        if skill.frontmatter.state == SKILL_STATE_ACTIVE
    ]
    selected = sorted(
        (item for item in results if item.score > 0),
        key=lambda item: (item.score, item.skill.skill_id),
        reverse=True,
    )[:k]
    for item in selected:
        current = store.touch_skill(item.skill.skill_id, version=item.skill.version)
        if current.frontmatter.state != SKILL_STATE_ACTIVE:
            raise ValueError(
                f"skill publication changed before use: {item.skill.skill_id}@{item.skill.version}"
            )
    return selected


def _score_skill(
    skill: Skill,
    query_tokens: set[str],
    task_tags: list[str],
    *,
    recent_user_text: str,
) -> SkillRecallResult:
    """用正文、标签与触发词计算相关度；传参：技能和查询；返回：总分与分项证据。"""
    body_tokens = _tokens(skill.body)
    text_score = len(query_tokens & body_tokens) / max(len(query_tokens), 1)
    tag_score = jaccard(task_tags, skill.frontmatter.applicable_task_tags)
    trigger_hits = sum(
        1
        for keyword in skill.frontmatter.trigger_keywords
        if keyword.lower() in recent_user_text.lower()
    )
    score = text_score + tag_score + trigger_hits
    return SkillRecallResult(skill, score, text_score, tag_score, trigger_hits)


def _tokens(text: str) -> set[str]:
    return set(tokenize_search_text(text))


__all__ = ["SkillRecallResult", "recall_skills"]
