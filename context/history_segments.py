"""【上下文】【历史片段】同源目标片段及四级表示的结构和完整交互边界。

作者：xxx
时间：2026-10-01 12:00:00
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

from llm.messages import AgentMessage, agent_message_to_mapping, group_tool_call_units

HISTORY_LEVELS = ("P1", "P2", "P3", "P4")


@dataclass(frozen=True, slots=True)
class HistorySegment:
    """保存同一目标段的原件范围和独立表示；参数：身份、主题、消息和四级正文；返回：不可变片段。"""

    segment_id: str
    title: str
    message_ids: tuple[str, ...]
    p1: str
    p2: str
    p3: str
    p4: str
    scope: str = ""

    def render(self, level: str = "P2") -> str:
        """按已有档位展开，不调用模型；参数：P1–P4；返回：含相同身份及主题的独立正文。"""
        if level not in HISTORY_LEVELS:
            raise ValueError("unknown historical representation level")
        body = getattr(self, level.lower())
        scope = f" scope={self.scope}" if self.scope else ""
        return f"[{self.segment_id} {level}{scope}] {self.title}\n{body}".rstrip()


def segment_from_mapping(value: Mapping[str, Any]) -> HistorySegment:
    """严格读取模型或持久片段；参数：结构对象；返回：完整四级片段，缺字段直接失败。"""
    expected = {"segment_id", "title", "message_ids", "p1", "p2", "p3", "p4", "scope"}
    if set(value) - expected or expected - {"scope"} - set(value):
        raise ValueError(
            "history segment requires identity, source and all four levels"
        )
    for name in expected - {"message_ids"}:
        text = value.get(name, "")
        if not isinstance(text, str) or (
            name not in {"p4", "scope"} and not text.strip()
        ):
            raise ValueError(f"history segment {name} is invalid")
    ids = value["message_ids"]
    if (
        not isinstance(ids, (list, tuple))
        or not ids
        or any(not isinstance(item, str) or not item for item in ids)
    ):
        raise ValueError("history segment requires original message identities")
    if len(set(ids)) != len(ids):
        raise ValueError("history segment source contains duplicate identities")
    return HistorySegment(**{**value, "message_ids": tuple(ids)})


def merge_segments(
    previous: tuple[HistorySegment, ...],
    values: Sequence[Mapping[str, Any]],
    sources: Mapping[str, object],
) -> tuple[HistorySegment, ...]:
    """更新本次尚未发布片段，禁止移换已有来源；参数：候选、局部核对结果和原件；返回：新片段集合。"""
    result = {item.segment_id: item for item in previous}
    seen: set[str] = set()
    for value in values:
        segment = segment_from_mapping(value)
        if segment.segment_id in seen:
            raise ValueError("history segment identity repeated in one delta")
        seen.add(segment.segment_id)
        if any(identity not in sources for identity in segment.message_ids):
            raise ValueError("history segment refers outside frozen originals")
        existing = result.get(segment.segment_id)
        if existing is not None and existing.message_ids != segment.message_ids:
            raise ValueError("history segment cannot change its frozen source range")
        result[segment.segment_id] = segment
    return tuple(result.values())


def validate_segment_coverage(
    segments: Sequence[HistorySegment], messages: Sequence[AgentMessage]
) -> None:
    """核对连续、无重叠、无遗漏的完整交互覆盖；参数：语义分段和原件；返回：无，错误拒绝发布。"""
    expected = tuple(message.message_id for message in messages)
    actual = tuple(identity for segment in segments for identity in segment.message_ids)
    if actual != expected:
        raise ValueError(
            "historical segments must cover new originals exactly and in order"
        )
    boundaries = {0}
    position = 0
    for group in group_tool_call_units(messages):
        position += len(group)
        boundaries.add(position)
    position = 0
    for segment in segments:
        position += len(segment.message_ids)
        if position not in boundaries:
            raise ValueError("historical segment splits a tool interaction")


def segment_digest(segment: HistorySegment) -> str:
    """将四档正文和同源身份绑定到核对版本；参数：完整片段；返回：稳定内容指纹。"""
    text = json.dumps(
        asdict(segment), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_segment_audits(
    segments: Sequence[HistorySegment],
    checks: Sequence[Mapping[str, object]],
    messages: Sequence[AgentMessage] = (),
) -> None:
    """核对最终档位版本与已读原文，不能借用旧版本检查；参数：片段、核对记录及可选原文；返回：无。"""
    for segment in segments:
        version = segment_digest(segment)
        matching = []
        checked: set[str] = set()
        for check in checks:
            versions = check.get("segment_versions")
            if (
                not isinstance(versions, Mapping)
                or versions.get(segment.segment_id) != version
            ):
                continue
            identities, message_ids = check.get("segment_ids"), check.get("message_ids")
            if not isinstance(identities, (list, tuple)) or not isinstance(
                message_ids, (list, tuple)
            ):
                raise ValueError("historical audit requires explicit source identities")
            if segment.segment_id in identities:
                checked.update(message_ids)
                matching.append(check)
        if not set(segment.message_ids).issubset(checked):
            raise ValueError(
                "historical representations require an original audit of their final version"
            )
        # 1. 【上下文】【历史核对】同一消息分页多次仍须覆盖完整正文，不能只凭消息身份声称读完
        originals = [
            message for message in messages if message.message_id in segment.message_ids
        ]
        _validate_audit_spans(originals, matching)


def _validate_audit_spans(
    messages: Sequence[AgentMessage], checks: Sequence[Mapping[str, object]]
) -> None:
    """合并同版本核对实际读到的页，拒绝缺口和伪造范围；参数：原文和对应检查；返回：无。"""
    spans: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
    for check in checks:
        values = check.get("source_spans", ())
        if not isinstance(values, (list, tuple)):
            raise ValueError("historical audit source spans are invalid")
        for value in values:
            if not isinstance(value, Mapping) or not isinstance(
                value.get("message_id"), str
            ):
                raise ValueError("historical audit source span has no message identity")
            if any(
                type(value.get(key)) is not int
                for key in ("part_index", "start", "end", "total")
            ):
                raise ValueError(
                    "historical audit source span requires integer positions"
                )
            spans.setdefault((value["message_id"], value["part_index"]), []).append(
                (value["start"], value["end"], value["total"])
            )
    for message in messages:
        parts = cast(list[dict[str, Any]], agent_message_to_mapping(message)["content"])
        for index, part in enumerate(parts):
            if not isinstance(part.get("text"), str):
                continue
            total, position = len(part["text"]), 0
            for start, end, declared_total in sorted(
                spans.get((message.message_id, index), ())
            ):
                if (
                    declared_total != total
                    or not 0 <= start <= end <= total
                    or start > position
                ):
                    raise ValueError(
                        "historical audit source spans do not cover the complete original"
                    )
                position = max(position, end)
            if position != total:
                raise ValueError(
                    "historical audit source spans do not cover the complete original"
                )
