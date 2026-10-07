from __future__ import annotations

from collections.abc import Sequence

from llm.types import CacheTier, PromptSection, ProviderHint


def to_prompt_sections(sections: Sequence[object]) -> list[PromptSection]:
    prompt_sections: list[PromptSection] = []
    for section in sections:
        name = str(getattr(section, "name"))
        content = str(getattr(section, "content"))
        prompt_sections.append(
            PromptSection(name, content, ProviderHint(_cache_tier(section)))
        )
    return prompt_sections


def cache_tier_counts(sections: Sequence[object]) -> dict[CacheTier, int]:
    counts = {tier: 0 for tier in CacheTier}
    for section in sections:
        counts[_cache_tier(section)] += 1
    return counts


def _cache_tier(section: object) -> CacheTier:
    value = getattr(section, "cache_tier")
    if isinstance(value, CacheTier):
        return value
    return CacheTier(str(value))


__all__ = ["cache_tier_counts", "to_prompt_sections"]
