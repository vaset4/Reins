"""【上下文】【片段去向】离线重放真实核对响应，验证目的地身份及原文归属。

作者：xxx
时间：2026-10-01 15:54:03
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from context.history_segments import HistorySegment
from context.summary_entries import (
    SummaryCitation,
    SummaryContent,
    SummaryEntry,
    SummaryTopic,
    apply_delta,
)

SOURCE_TEXT = "预算不得超过500，禁止上传"
SOURCES = {
    identity: {"text": SOURCE_TEXT, "source_kind": "user_input"}
    for identity in ("user-1", "user-2")
}


def segment_mapping(identity="seg-1", message_id="user-1"):
    """构造可独立定位的同源片段；参数：片段与消息身份；返回：模型差量结构。"""
    return {
        "segment_id": identity,
        "title": "本地资料整理",
        "message_ids": [message_id],
        "p1": SOURCE_TEXT,
        "p2": SOURCE_TEXT,
        "p3": SOURCE_TEXT,
        "p4": "500/禁止上传",
    }


def disposition(destination):
    """构造单条新原文的明确去向；参数：目的地身份；返回：未经修改的去向声明。"""
    return {
        "message_id": "user-1",
        "destinations": [destination],
        "reason": "本条原文由对应历史片段保留",
    }


@pytest.mark.parametrize("case", ["conditions", "correction"])
def test_real_audit_can_reference_an_existing_segment_without_changing_meaning(case):
    """真实生成和核对都保持原样离线执行；参数：原始场景；返回：无，不调用模型。"""
    path = (
        Path(__file__).parent
        / "fixtures"
        / "context"
        / f"stage9_{case}_segment_destinations.json"
    )
    fixture = json.loads(path.read_text(encoding="utf-8"))
    sources = {}
    for group in fixture["original_groups"]:
        for message in group:
            assert all(part["kind"] == "text" for part in message["content"])
            sources[message["message_id"]] = {
                "text": "\n".join(part["text"] for part in message["content"]),
                "source_kind": message["source_kind"],
                "result_status": message.get("status", ""),
            }
    candidate = apply_delta(
        SummaryContent(),
        fixture["generation"],
        base_summary_id=fixture["base_summary_id"],
        sources=sources,
        required_messages=fixture["required_message_ids"],
    )
    checked = apply_delta(
        candidate,
        fixture["audit"],
        base_summary_id=fixture["base_summary_id"],
        sources=sources,
        required_messages=fixture["required_message_ids"],
    )
    assert (
        checked.entries == candidate.entries and checked.segments == candidate.segments
    )
    segments = {segment.segment_id: segment for segment in checked.segments}
    mapped = [
        (item["message_id"], target)
        for item in fixture["audit"]["dispositions"]
        for target in item["destinations"]
        if target in segments
    ]
    assert mapped and all(
        message_id in segments[target].message_ids for message_id, target in mapped
    )


def test_new_segment_is_a_destination_in_the_same_delta():
    """新片段和去向同批返回时先合并再核对归属；参数：无；返回：无。"""
    delta = {
        "base_summary_id": None,
        "segments": [segment_mapping()],
        "dispositions": [disposition("seg-1")],
    }
    result = apply_delta(
        SummaryContent(),
        delta,
        base_summary_id=None,
        sources=SOURCES,
        required_messages=("user-1",),
    )
    assert result.segments[0].message_ids == ("user-1",)


@pytest.mark.parametrize("existing", [False, True])
def test_segment_destination_must_contain_the_declared_source(existing):
    """存在的片段不能替另一条原文冒领去向；参数：是否为先前候选片段；返回：无。"""
    value = segment_mapping(message_id="user-2")
    prior = (
        SummaryContent(
            segments=(HistorySegment(**{**value, "message_ids": ("user-2",)}),)
        )
        if existing
        else SummaryContent()
    )
    delta = {"base_summary_id": None, "dispositions": [disposition("seg-1")]}
    if not existing:
        delta["segments"] = [value]
    with pytest.raises(ValueError, match="segment destination does not cover"):
        apply_delta(
            prior,
            delta,
            base_summary_id=None,
            sources=SOURCES,
            required_messages=("user-1",),
        )


@pytest.mark.parametrize("collision", ["entry", "topic", "original"])
def test_segment_identity_cannot_collide_with_other_destinations(collision):
    """片段不得借条目、话题或原件保留词混淆目的地；参数：冲突类别；返回：无。"""
    citation = SummaryCitation("user-1", SOURCE_TEXT, "user_input")
    identity = "original" if collision == "original" else "taken"
    prior = SummaryContent(
        entries=(SummaryEntry(identity, "requirement", SOURCE_TEXT, (citation,)),)
        if collision == "entry"
        else (),
        topics=(SummaryTopic(identity, "预算", SOURCE_TEXT, ("user-1",)),)
        if collision == "topic"
        else (),
    )
    delta = {
        "base_summary_id": None,
        "segments": [segment_mapping(identity)],
        "dispositions": [disposition(identity)],
    }
    with pytest.raises(ValueError, match="segment identity conflicts"):
        apply_delta(
            prior,
            delta,
            base_summary_id=None,
            sources=SOURCES,
            required_messages=("user-1",),
        )


def test_new_entry_cannot_hide_an_existing_segment_identity():
    """后来新增条目也不能覆盖先前片段的目的地含义；参数：无；返回：无。"""
    value = segment_mapping()
    prior = SummaryContent(
        segments=(HistorySegment(**{**value, "message_ids": ("user-1",)}),)
    )
    delta = {
        "base_summary_id": None,
        "add": [
            {
                "entry_id": "seg-1",
                "kind": "requirement",
                "text": SOURCE_TEXT,
                "sources": [{"message_id": "user-1", "quote": SOURCE_TEXT}],
            }
        ],
        "dispositions": [disposition("seg-1")],
    }
    with pytest.raises(ValueError, match="segment identity conflicts"):
        apply_delta(
            prior,
            delta,
            base_summary_id=None,
            sources=SOURCES,
            required_messages=("user-1",),
        )


def test_unknown_segment_destination_is_still_rejected():
    """未创建的片段不能用宽松过滤假装已有覆盖；参数：无；返回：无。"""
    with pytest.raises(ValueError, match="missing entry"):
        apply_delta(
            SummaryContent(),
            {"base_summary_id": None, "dispositions": [disposition("missing-segment")]},
            base_summary_id=None,
            sources=SOURCES,
            required_messages=("user-1",),
        )
