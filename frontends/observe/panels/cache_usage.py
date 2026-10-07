from __future__ import annotations

from typing import Any

from frontends.observe.panels.base import Panel, RunContext


class CacheUsagePanel(Panel):
    id = "cache_usage"
    title = "Cache Usage"
    section = "model"
    phase = "C"

    def build(self, ctx: RunContext) -> dict[str, Any]:
        facts = ctx.facts()
        entries = _extract_cache_usage(facts)
        totals = _compute_totals(entries)
        return {"entries": entries, "totals": totals}


def _extract_cache_usage(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """挑出缓存用量 fact，保留"未上报"的 None 语义。

    作者：xxx
    时间：2026-09-01 00:00:00
    传参：facts 为一次运行的全部 run fact
    返回：每条缓存用量的投影；未上报的字段为 None，不折成 0
    """
    results = []
    for fact in facts:
        if fact.get("event") != "llm:cache_usage":
            continue
        # 缺键（旧 fact）与显式 null（新 fact）都取到 None，两者语义一致：厂商没报
        results.append(
            {
                "ts": fact.get("ts", ""),
                "call_id": fact.get("call_id", ""),
                "cache_creation_input_tokens": fact.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": fact.get("cache_read_input_tokens"),
            }
        )
    return results


def _compute_totals(entries: list[dict[str, Any]]) -> dict[str, int]:
    """汇总缓存用量，把"未上报"与"报了 0"分开计。

    作者：xxx
    时间：2026-09-01 00:00:00
    传参：entries 为 _extract_cache_usage 的输出
    返回：两项总量加未上报次数；总量只累加真实上报的数值
    """
    # 1. 只累加厂商真实报过的数值，None 不参与求和（否则 sum 会抛 TypeError）
    creation = sum(
        value
        for e in entries
        if isinstance(value := e.get("cache_creation_input_tokens"), int)
    )
    read = sum(
        value
        for e in entries
        if isinstance(value := e.get("cache_read_input_tokens"), int)
    )
    # 2. 两项都没报的调用单独计数，让读者能分清"缓存没省下"和"这家厂商不报缓存用量"
    unreported = sum(
        1
        for e in entries
        if e.get("cache_creation_input_tokens") is None
        and e.get("cache_read_input_tokens") is None
    )
    return {
        "total_cache_creation": creation,
        "total_cache_read": read,
        "unreported_count": unreported,
    }


__all__ = ["CacheUsagePanel"]
