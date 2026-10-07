"""维护带原文出处的接续条目；未提及的条目原样继承。

作者：xxx
时间：2026-09-25 12:00:00
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from context.history_segments import (
    HistorySegment,
    merge_segments,
    segment_from_mapping,
    validate_segment_audits,
)

ENTRY_KINDS = frozenset(
    {
        "requirement",
        "goal",
        "decision",
        "result",
        "observation",
        "conclusion",
        "open",
        "history",
    }
)


@dataclass(frozen=True, slots=True)
class SummaryCitation:
    """保存已核对的简短原话；传参：消息身份、引文、真实来源和结果状态；返回：不可变出处。"""

    message_id: str
    quote: str
    source_kind: str
    result_status: str = ""


@dataclass(frozen=True, slots=True)
class SummaryEntry:
    """表达本次工作所需信息；传参：身份、语义类别、正文、出处和版本；返回：派生条目。"""

    entry_id: str
    kind: str
    text: str
    sources: tuple[SummaryCitation, ...]
    revision: int = 1
    rewrite_count: int = 0
    scope: str = ""
    active: bool = True
    retired_reason: str = ""
    previous_ref: str | None = None


@dataclass(frozen=True, slots=True)
class SummaryTopic:
    """保存本片段的话题线索；传参：身份、标题、历史说明及消息范围；返回：可回查目录条目。"""

    topic_id: str
    title: str
    text: str
    message_ids: tuple[str, ...]
    scope: str = ""


@dataclass(frozen=True, slots=True)
class SummaryContent:
    """一次发布的条目、增量目录与核对证据；传参：工作视图及审计数据；返回：不可变候选。"""

    entries: tuple[SummaryEntry, ...] = ()
    topics: tuple[SummaryTopic, ...] = ()
    checks: tuple[Mapping[str, object], ...] = ()
    segments: tuple[HistorySegment, ...] = ()

    def render(self) -> str:
        """只渲染有依据的实际内容，不填充空节；传参：无；返回：紧凑接续正文。"""
        rows = []
        for entry in self.entries:
            if not entry.active:
                continue
            refs = "; ".join(
                f"{source.message_id}/{source.source_kind}"
                + (f"/{source.result_status}" if source.result_status else "")
                + f":{source.quote}"
                for source in entry.sources
            )
            scope = f" scope={entry.scope}" if entry.scope else ""
            rows.append(
                f"[{entry.entry_id}@{entry.revision} {entry.kind}{scope}] {entry.text}\nsource: {refs}"
            )
        return "\n".join(rows)


def parse_delta(text: str) -> dict[str, Any]:
    """解析单次增量响应而不修补无效模型输出；传参：最终回答；返回：严格对象。"""
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) - {
        "base_summary_id",
        "add",
        "revise",
        "retire",
        "topics",
        "dispositions",
        "findings",
        "segments",
    }:
        raise ValueError("summary delta has invalid fields")
    for name in (
        "add",
        "revise",
        "retire",
        "topics",
        "dispositions",
        "findings",
        "segments",
    ):
        items = value.get(name, [])
        if not isinstance(items, list) or any(
            not isinstance(item, dict) for item in items
        ):
            raise ValueError(f"summary delta {name} must be an object array")
    return value


def apply_delta(
    content: SummaryContent,
    delta: Mapping[str, Any],
    *,
    base_summary_id: str | None,
    sources: Mapping[str, Mapping[str, str]],
    required_messages: Sequence[str] = (),
    published_content: SummaryContent | None = None,
    new_evidence: Sequence[str] | None = None,
) -> SummaryContent:
    """核验引用和版本后应用局部变更；传参：前版、差量、冻结来源和新增消息；返回：独立候选。"""
    if delta.get("base_summary_id") != base_summary_id:
        raise ValueError("summary delta refers to another base snapshot")
    entries = {entry.entry_id: entry for entry in content.entries}
    published = {
        entry.entry_id: entry
        for entry in (
            published_content if published_content is not None else content
        ).entries
    }
    changed: set[str] = set()
    for action in ("add", "revise", "retire"):
        for item in delta.get(action, []):
            identity = _text(item, "entry_id")
            if identity in changed:
                raise ValueError("summary delta changes an identity more than once")
            changed.add(identity)
            if action == "add":
                if identity in entries:
                    raise ValueError("summary entry identity already exists")
                entries[identity] = _new_entry(item, sources)
                continue
            old = entries.get(identity)
            if old is None or item.get("expected_revision") != old.revision:
                raise ValueError("summary entry is unknown or has a revision conflict")
            anchor = published.get(identity)
            previous_ref = (
                f"{base_summary_id}/{identity}@{anchor.revision}"
                if base_summary_id and anchor
                else None
            )
            entries[identity] = _change_entry(
                old,
                item,
                sources,
                action=action,
                previous_ref=previous_ref,
                required_messages=required_messages
                if new_evidence is None
                else new_evidence,
            )
    topics = _merge_topics(content.topics, delta.get("topics", []), sources)
    segments = merge_segments(content.segments, delta.get("segments", []), sources)
    _validate_dispositions(
        delta.get("dispositions", []),
        required_messages,
        entries,
        topics,
        segments=segments,
    )
    return SummaryContent(tuple(entries.values()), topics, content.checks, segments)


def _change_entry(
    old: SummaryEntry,
    item: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, str]],
    *,
    action: str,
    previous_ref: str | None,
    required_messages: Sequence[str],
) -> SummaryEntry:
    """有新增原话才允许改变旧要求，并保留实际改写链；传参：旧条目、候选和来源边界；返回：新版本。"""
    if old.kind == "requirement":
        citations = _citations(item.get("sources"), sources)
        if not any(
            source.source_kind == "user_input"
            and source.message_id in required_messages
            for source in citations
        ):
            raise ValueError(
                "changing an existing user requirement needs new user input evidence"
            )
    if action == "retire":
        return _retire_entry(old, item, sources, previous_ref)
    entry = _new_entry(item, sources)
    return replace(
        entry,
        revision=old.revision + 1,
        rewrite_count=old.rewrite_count + 1,
        previous_ref=previous_ref,
    )


def _text(item: Mapping[str, Any], key: str) -> str:
    """读取必要文本字段；传参：模型对象和字段名；返回：非空文本，非法值直接报错。"""
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"summary {key} must be non-empty text")
    return value


def _citations(
    value: object, sources: Mapping[str, Mapping[str, str]]
) -> tuple[SummaryCitation, ...]:
    """引文必须逐字存在于冻结原文，来源角色由程序给出；传参：引用数组和原文；返回：已核对出处。"""
    if not isinstance(value, list) or not value:
        raise ValueError("summary entries require original citations")
    result = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("summary citation must be an object")
        identity, quote = _text(item, "message_id"), _text(item, "quote")
        original = sources.get(identity)
        if original is None or quote not in original["text"]:
            raise ValueError("summary citation is not present in its frozen original")
        citation = SummaryCitation(
            identity, quote, original["source_kind"], original.get("result_status", "")
        )
        if citation not in result:
            result.append(citation)
    return tuple(result)


def _new_entry(
    item: Mapping[str, Any], sources: Mapping[str, Mapping[str, str]]
) -> SummaryEntry:
    """构造有来源的条目，防止系统恢复或工具结果冒充用户要求；传参：对象和来源；返回：条目。"""
    kind = _text(item, "kind")
    if kind not in ENTRY_KINDS:
        raise ValueError("unknown summary entry kind")
    citations = _citations(item.get("sources"), sources)
    if kind == "requirement" and not any(
        source.source_kind == "user_input" for source in citations
    ):
        raise ValueError("user requirement needs actual user input")
    if kind == "conclusion" and not any(
        source.source_kind == "assistant" for source in citations
    ):
        raise ValueError("working conclusion needs a public assistant statement")
    scope = item.get("scope", "")
    if not isinstance(scope, str):
        raise ValueError("summary scope must be text")
    return SummaryEntry(
        _text(item, "entry_id"), kind, _text(item, "text"), citations, scope=scope
    )


def _retire_entry(
    old: SummaryEntry,
    item: Mapping[str, Any],
    sources: Mapping[str, Mapping[str, str]],
    previous_ref: str | None,
) -> SummaryEntry:
    """显式移出工作视图但保留原件和继承链；传参：条目、去向与依据；返回：历史状态条目。"""
    reason = _text(item, "reason")
    if item.get("destination") != "history":
        raise ValueError("retired summary entry must retain a history destination")
    citations = _citations(item.get("sources"), sources)
    if old.kind == "requirement" and not any(
        source.source_kind == "user_input" for source in citations
    ):
        raise ValueError("retiring a user requirement requires user input evidence")
    return replace(
        old,
        active=False,
        retired_reason=reason,
        revision=old.revision + 1,
        previous_ref=previous_ref,
        sources=tuple(dict.fromkeys((*old.sources, *citations))),
    )


def _merge_topics(
    previous: tuple[SummaryTopic, ...],
    values: list[dict[str, Any]],
    sources: Mapping[str, Mapping[str, str]],
) -> tuple[SummaryTopic, ...]:
    """追加有精确原文范围的增量线索；传参：本次已有线索、新线索、来源；返回：合并目录。"""
    topics = {topic.topic_id: topic for topic in previous}
    for item in values:
        identity = _text(item, "topic_id")
        ids = item.get("message_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(value, str) or value not in sources for value in ids)
        ):
            raise ValueError("topic requires existing source messages")
        if identity in topics:
            raise ValueError("duplicate summary topic identity")
        topics[identity] = SummaryTopic(
            identity,
            _text(item, "title"),
            _text(item, "text"),
            tuple(dict.fromkeys(ids)),
            scope=str(item.get("scope", "")),
        )
    return tuple(topics.values())


def _validate_dispositions(
    values: list[dict[str, Any]],
    required: Sequence[str],
    entries: Mapping[str, SummaryEntry],
    topics: Sequence[SummaryTopic],
    *,
    segments: Sequence[HistorySegment],
) -> None:
    """检查新增原文的目的地身份与片段归属，不代替语义核对；传参：声明及条目/话题/片段；返回：无。"""
    seen: set[str] = set()
    topic_ids = {topic.topic_id for topic in topics}
    segment_sources = {
        segment.segment_id: frozenset(segment.message_ids) for segment in segments
    }
    # 1. 【上下文】【原文去向】各类目的地共用字符串身份，片段不能借同名条目或原件保留词绕过来源检查
    if set(segment_sources).intersection(set(entries) | topic_ids | {"original"}):
        raise ValueError(
            "summary segment identity conflicts with entry, topic or original destination"
        )
    target_ids = set(entries) | topic_ids | set(segment_sources) | {"original"}
    for item in values:
        identity = _text(item, "message_id")
        if identity in seen or identity not in required:
            raise ValueError(
                "summary message disposition is duplicate or outside the new material"
            )
        seen.add(identity)
        destinations = item.get("destinations")
        if (
            not isinstance(destinations, list)
            or not destinations
            or any(not isinstance(value, str) for value in destinations)
        ):
            raise ValueError(
                "summary disposition destinations must be a non-empty text array"
            )
        _text(item, "reason")
        if any(destination not in target_ids for destination in destinations):
            raise ValueError(
                "summary disposition points to a missing entry, topic or segment"
            )
        if any(
            identity not in segment_sources[destination]
            for destination in destinations
            if destination in segment_sources
        ):
            raise ValueError(
                "summary segment destination does not cover its source message"
            )
    if set(required) != seen:
        raise ValueError("summary did not account for every new source message")


def content_from_mapping(value: object) -> SummaryContent:
    """读取已发布的结构化摘要，不给旧摘要伪造核对记录；传参：持久对象；返回：内容记录。"""
    if not isinstance(value, dict) or set(value) not in (
        {"entries", "topics", "checks"},
        {"entries", "topics", "checks", "segments"},
    ):
        raise ValueError("invalid structured summary content")
    if any(
        not isinstance(value[key], list)
        or any(not isinstance(item, dict) for item in value[key])
        for key in value
    ):
        raise ValueError("structured summary fields must be object arrays")
    entries = tuple(
        SummaryEntry(
            **{
                **item,
                "sources": tuple(
                    SummaryCitation(**source) for source in item["sources"]
                ),
            }
        )
        for item in value["entries"]
    )
    topics = tuple(
        SummaryTopic(**{**item, "message_ids": tuple(item["message_ids"])})
        for item in value["topics"]
    )
    segments = tuple(segment_from_mapping(item) for item in value.get("segments", []))
    content = SummaryContent(entries, topics, tuple(value["checks"]), segments)
    validate_content(content)
    return content


def validate_content(content: SummaryContent) -> None:
    """发布与读取共享身份、来源角色和版本合同；传参：摘要内容；返回：无，非法内容明确拒绝。"""
    entries, topics = content.entries, content.topics
    if len({segment.segment_id for segment in content.segments}) != len(
        content.segments
    ):
        raise ValueError("historical segments contain duplicate identities")
    for segment in content.segments:
        segment_from_mapping(asdict(segment))
    validate_segment_audits(content.segments, content.checks)
    if len({entry.entry_id for entry in entries}) != len(entries) or len(
        {topic.topic_id for topic in topics}
    ) != len(topics):
        raise ValueError("structured summary contains duplicate identities")
    for entry in entries:
        _validate_saved_entry(entry)
    for topic in topics:
        if (
            any(
                not isinstance(item, str) or not item.strip()
                for item in (
                    topic.topic_id,
                    topic.title,
                    topic.text,
                    *topic.message_ids,
                )
            )
            or not topic.message_ids
        ):
            raise ValueError(
                "saved summary topic requires text and original message identities"
            )
        if not isinstance(topic.scope, str):
            raise ValueError("saved summary topic scope must be text")


def _validate_saved_entry(entry: SummaryEntry) -> None:
    """读取时验证来源类别和版本，不信任手工损坏的持久字段；传参：条目；返回：无，非法内容明确失败。"""
    if (
        not isinstance(entry.entry_id, str)
        or not entry.entry_id.strip()
        or entry.kind not in ENTRY_KINDS
    ):
        raise ValueError("saved summary entry has invalid identity or kind")
    if (
        not isinstance(entry.text, str)
        or not entry.text.strip()
        or not isinstance(entry.scope, str)
    ):
        raise ValueError("saved summary entry text and scope are invalid")
    if (
        type(entry.revision) is not int
        or entry.revision < 1
        or type(entry.rewrite_count) is not int
        or entry.rewrite_count < 0
    ):
        raise ValueError("saved summary entry revision is invalid")
    if type(entry.active) is not bool or not entry.sources:
        raise ValueError("saved summary entry requires state and citations")
    for source in entry.sources:
        if source.source_kind not in {
            "user_input",
            "assistant",
            "agent",
            "tool_result",
        } or source.result_status not in {"", "success", "error", "partial"}:
            raise ValueError(
                "saved summary citation has an invalid source kind or status"
            )
        if any(
            not isinstance(item, str) or not item.strip()
            for item in (source.message_id, source.quote)
        ):
            raise ValueError("saved summary citation requires original text")
    expected = {"requirement": "user_input", "conclusion": "assistant"}.get(entry.kind)
    if expected is not None and not any(
        source.source_kind == expected for source in entry.sources
    ):
        raise ValueError("saved summary entry changed the authority of its source")


def content_to_mapping(content: SummaryContent) -> dict[str, object]:
    """输出已冻结内容；传参：摘要候选；返回：供持久化及模型核对的独立对象。"""
    return asdict(content)
