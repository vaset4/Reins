"""绑定会话分支与原文版本的历史快照分页。

作者：xxx
时间：2026-09-14 20:00:00
"""

from __future__ import annotations

import json
from typing import Any, cast

from context.cursors import decode_cursor, encode_cursor
from context.history_segments import HISTORY_LEVELS
from llm.messages import (
    AssistantMessage,
    ToolCallPart,
    agent_message_to_mapping,
    content_part_to_mapping,
    group_tool_call_units,
    model_visible_text,
)
from memory.index import tokenize_search_text
from runtime.session_compaction import SessionCompactionStore, source_digest
from runtime.session_message_store import MaterializedSession

DEFAULT_HISTORY_PAGE_SIZE = 20
MAX_HISTORY_PAGE_SIZE = 50
QUERY_CURSOR_PREFIX = "h2:"


def read_history_page(
    view: MaterializedSession,
    *,
    call_id: str,
    before: str | None = None,
    cursor: str | None = None,
    limit: int = DEFAULT_HISTORY_PAGE_SIZE,
    summaries: SessionCompactionStore | None = None,
    summary_id: str | None = None,
    source_ref: str | None = None,
    query: str | None = None,
    view_kind: str | None = None,
    segment_id: str | None = None,
    level: str | None = None,
) -> dict[str, object]:
    """向前补读当前分支原文，追加消息不改变已开始的快照；传参：分支与位置；返回：页面及续读游标。"""
    if type(limit) is not int or not 0 < limit <= MAX_HISTORY_PAGE_SIZE:
        raise ValueError("history page limit is invalid")
    if before is not None and cursor is not None:
        raise ValueError("history before and cursor cannot be combined")
    options = {
        "view": view_kind,
        "summary_id": summary_id,
        "source_ref": source_ref,
        "query": query,
        "segment_id": segment_id,
        "level": level,
    }
    if any(value is not None for value in options.values()) or (
        cursor is not None and cursor.startswith(QUERY_CURSOR_PREFIX)
    ):
        if before is not None:
            raise ValueError(
                "history queries and source references cannot be combined with before"
            )
        return _read_query(
            view, summaries, options, call_id=call_id, cursor=cursor, limit=limit
        )
    snapshot = _history_snapshot(view, call_id=call_id, before=before, cursor=cursor)
    end = cast(int, snapshot["offset"])
    start = max(0, end - limit)
    page = view.messages[start:end]
    next_cursor = (
        encode_cursor("history", {**snapshot, "offset": start}) if start else None
    )
    return {
        "messages": [agent_message_to_mapping(message) for message in page],
        "next_before": page[0].message_id if start else None,
        "next_cursor": next_cursor,
        "branch_entry_id": snapshot["branch_entry_id"],
        "content_sha256": snapshot["sha256"],
        "offset": start,
        "end_offset": end,
        "returned_count": len(page),
        "total_count": snapshot["count"],
    }


def _history_snapshot(
    view: MaterializedSession,
    *,
    call_id: str,
    before: str | None,
    cursor: str | None,
) -> dict[str, object]:
    """创建或核实游标所指的消息快照；传参：当前分支、调用和位置；返回：来源与结束位置。"""
    if cursor is not None:
        snapshot = decode_cursor(
            cursor,
            "history",
            fields={
                "session_id": str,
                "branch_entry_id": str,
                "count": int,
                "offset": int,
                "sha256": str,
            },
        )
        if snapshot["session_id"] != view.session_id:
            raise ValueError("history cursor belongs to another session")
        if snapshot["branch_entry_id"] not in {
            entry.entry_id for entry in view.entries
        }:
            raise ValueError("history cursor is not in the current branch")
        count, offset = cast(int, snapshot["count"]), cast(int, snapshot["offset"])
        if not 0 <= offset <= count <= len(view.messages):
            raise ValueError("history cursor position is invalid")
        if source_digest(view.messages[:count]) != snapshot["sha256"]:
            raise ValueError("history content version changed; restart pagination")
        return {
            key: value
            for key, value in snapshot.items()
            if key not in {"kind", "version"}
        }
    # 1. 【会话】【历史补读】首屏排除本次补读公告；其后消息只影响下一次新快照
    if before is None:
        before = next(
            (
                message.message_id
                for message in view.messages
                if isinstance(message, AssistantMessage)
                and any(
                    isinstance(part, ToolCallPart) and part.call_id == call_id
                    for part in message.content
                )
            ),
            None,
        )
    end = len(view.messages)
    if before is not None:
        position = next(
            (
                index
                for index, message in enumerate(view.messages)
                if message.message_id == before
            ),
            None,
        )
        if position is None:
            raise ValueError("history cursor is not in the current branch")
        end = position
    return {
        "session_id": view.session_id,
        "branch_entry_id": view.leaf_id,
        "count": end,
        "offset": end,
        "sha256": source_digest(view.messages[:end]),
    }


def _read_query(
    view: MaterializedSession,
    summaries: SessionCompactionStore | None,
    options: dict[str, str | None],
    *,
    call_id: str,
    cursor: str | None,
    limit: int,
) -> dict[str, object]:
    """按冻结条件扫描当前分支，不把未扫描完解释为没有历史；传参：来源与查询；返回：页面和扫描证据。"""
    snapshot = _query_snapshot(view, summaries, options, call_id=call_id, cursor=cursor)
    kind = snapshot["options"]["view"]
    if kind == "segments":
        rows = _segment_rows(view, summaries, snapshot)
    else:
        rows = (
            _topic_rows(view, summaries, snapshot)
            if kind == "topics"
            else _message_rows(view, snapshot)
        )
    start = snapshot["offset"]
    if start > len(rows):
        raise ValueError("history scan cursor is beyond its frozen range")
    end = min(len(rows), start + limit)
    terms = set(tokenize_search_text(snapshot["options"]["query"] or ""))
    scanned = rows[start:end]
    selected = [
        item
        for item in scanned
        if not terms or terms.issubset(set(tokenize_search_text(item["search_text"])))
    ]
    next_cursor = (
        QUERY_CURSOR_PREFIX
        + encode_cursor("history_query", {**snapshot, "offset": end})
        if end < len(rows)
        else None
    )
    result: dict[str, object] = {
        "view": kind,
        "next_before": None,
        "next_cursor": next_cursor,
        "branch_entry_id": snapshot["branch_entry_id"],
        "content_sha256": snapshot["sha256"],
        "scan_start": start,
        "scan_end": end,
        "scanned_count": len(scanned),
        "total_scan_items": len(rows),
        "scan_complete": end == len(rows),
        "query": snapshot["options"]["query"],
        "notice": "Keyword matches are candidates, not semantic proof. Continue scanning or explicitly query original messages if needed.",
    }
    if kind in {"topics", "segments"}:
        result[kind] = [
            {key: value for key, value in item.items() if key != "search_text"}
            for item in selected
        ]
        result["returned_count"] = len(selected)
        records = (
            [summaries.read(identity, view) for identity in snapshot["summary_ids"]]
            if summaries
            else []
        )
        missing = [
            record.summary_id
            for record in records
            if record.content is None
            or (kind == "segments" and not record.content.segments)
        ]
        result["directory_state"] = (
            "available" if len(records) > len(missing) else "not_provided"
        )
        result["summaries_without_directory"] = missing
    else:
        indexes = sorted(
            {index for item in selected for index in item["message_indexes"]}
        )
        result["messages"] = [
            agent_message_to_mapping(view.messages[index]) for index in indexes
        ]
        result["ranges"] = [item["range"] for item in selected]
        result["returned_count"] = len(indexes)
    return result


def _query_snapshot(
    view: MaterializedSession,
    summaries: SessionCompactionStore | None,
    options: dict[str, str | None],
    *,
    call_id: str,
    cursor: str | None,
) -> dict[str, Any]:
    """创建或核对查询身份、来源和扫描位置；传参：当前分支与参数；返回：固定快照。"""
    if cursor is not None:
        if not cursor.startswith(QUERY_CURSOR_PREFIX):
            raise ValueError("plain history cursor cannot be mixed with query options")
        snapshot: dict[str, Any] = decode_cursor(
            cursor[len(QUERY_CURSOR_PREFIX) :],
            "history_query",
            fields={
                "session_id": str,
                "branch_entry_id": str,
                "count": int,
                "sha256": str,
                "offset": int,
                "options": dict,
                "ranges": list,
                "summary_ids": list,
            },
        )
        for key, value in options.items():
            if value is not None and value != cast(
                dict[str, object], snapshot["options"]
            ).get(key):
                raise ValueError(
                    "history cursor query or source cannot change during pagination"
                )
        _validate_query_source(view, snapshot)
        return {
            key: value
            for key, value in snapshot.items()
            if key not in {"kind", "version"}
        }
    kind = options["view"] or (
        "segments" if options["segment_id"] or options["level"] else "messages"
    )
    if kind not in {"messages", "topics", "segments"}:
        raise ValueError("history view must be messages, topics or segments")
    if options["level"] is not None and options["level"] not in HISTORY_LEVELS:
        raise ValueError("history level must be P1, P2, P3 or P4")
    if kind != "segments" and (
        options["segment_id"] is not None or options["level"] is not None
    ):
        raise ValueError("segment identity and level require the segments view")
    if options["source_ref"] is not None and (
        kind != "messages"
        or options["summary_id"] is not None
        or options["query"] is not None
    ):
        raise ValueError(
            "source_ref cannot be combined with topics, summary_id or query"
        )
    snapshot = _history_snapshot(view, call_id=call_id, before=None, cursor=None)
    count = cast(int, snapshot["count"])
    identities: list[str] = []
    ranges = [[0, count]]
    if options["summary_id"] is not None:
        if summaries is None:
            raise ValueError("summary reads require the session summary owner")
        record = summaries.read(options["summary_id"], view)
        count = len(record.message_ids)
        ranges, identities = [[0, count]], [record.summary_id]
    elif kind in {"topics", "segments"}:
        if summaries is None:
            raise ValueError("topic reads require the session summary owner")
        identities = [record.summary_id for record in summaries.chain(view)]
    if options["source_ref"] is not None:
        source = decode_cursor(
            options["source_ref"],
            "history_source",
            fields={
                "session_id": str,
                "branch_entry_id": str,
                "count": int,
                "sha256": str,
                "ranges": list,
            },
        )
        snapshot.update(
            {
                key: value
                for key, value in source.items()
                if key not in {"kind", "version"}
            }
        )
        count, ranges = (
            cast(int, source["count"]),
            cast(list[list[int]], source["ranges"]),
        )
    snapshot.update(
        count=count,
        offset=0,
        ranges=ranges,
        summary_ids=identities,
        options={**options, "view": kind},
        sha256=snapshot["sha256"]
        if options["source_ref"]
        else source_digest(view.messages[:count]),
    )
    _validate_query_source(view, snapshot)
    return snapshot


def _validate_query_source(view: MaterializedSession, snapshot: dict[str, Any]) -> None:
    """每次读取仍验证来源凭据，不把游标当权限；传参：分支与快照；返回：无，变更明确失败。"""
    if snapshot["session_id"] != view.session_id or snapshot["branch_entry_id"] not in {
        entry.entry_id for entry in view.entries
    }:
        raise ValueError("history query source belongs to another session or branch")
    count = snapshot["count"]
    if (
        type(count) is not int
        or not 0 <= count <= len(view.messages)
        or type(snapshot["offset"]) is not int
        or snapshot["offset"] < 0
    ):
        raise ValueError("history query position is invalid")
    if source_digest(view.messages[:count]) != snapshot["sha256"]:
        raise ValueError("history query source content version changed")
    if not isinstance(snapshot["ranges"], list) or any(
        not isinstance(span, list)
        or len(span) != 2
        or any(type(value) is not int for value in span)
        or not 0 <= span[0] <= span[1] <= count
        for span in snapshot["ranges"]
    ):
        raise ValueError("history query ranges are invalid")
    if not isinstance(snapshot["options"], dict) or set(snapshot["options"]) not in (
        {"view", "query", "summary_id", "source_ref"},
        {"view", "query", "summary_id", "source_ref", "segment_id", "level"},
    ):
        raise ValueError("history query options are invalid")
    if snapshot["options"]["view"] not in {"topics", "messages", "segments"} or any(
        value is not None and not isinstance(value, str)
        for value in snapshot["options"].values()
    ):
        raise ValueError("history query options have invalid types")
    identities = snapshot["summary_ids"]
    if not isinstance(identities, list) or any(
        not isinstance(value, str) or not value for value in identities
    ):
        raise ValueError("history query summary identities are invalid")


def _message_rows(
    view: MaterializedSession, snapshot: dict[str, Any]
) -> list[dict[str, Any]]:
    """按近到远返回完整交互，查询命中时附相邻指代上下文；传参：消息与范围；返回：稳定扫描行。"""
    rows: list[dict[str, Any]] = []
    position = 0
    previous: list[int] = []
    for group in group_tool_call_units(view.messages[: snapshot["count"]]):
        end = position + len(group)
        indexes = list(range(position, end))
        if any(position < span[1] and end > span[0] for span in snapshot["ranges"]):
            # 1. 【上下文】【原文搜索】工具参数也是原件，可定位调用但不能据此推断执行成功
            calls = [
                json.dumps(content_part_to_mapping(part), ensure_ascii=False)
                for message in group
                for part in message.content
                if isinstance(part, ToolCallPart)
            ]
            rows.append(
                {
                    "search_text": "\n".join(
                        [*(model_visible_text(message) for message in group), *calls]
                    ),
                    "message_indexes": [*previous, *indexes]
                    if snapshot["options"]["query"]
                    else indexes,
                    "range": [position, end],
                }
            )
        previous, position = indexes, end
    return list(reversed(rows))


def _topic_rows(
    view: MaterializedSession,
    summaries: SessionCompactionStore | None,
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    """沿冻结快照读取增量线索，来源范围仍指向原消息；传参：分支与摘要写者；返回：稳定目录。"""
    if summaries is None:
        raise ValueError("topic reads require the session summary owner")
    positions = {
        message.message_id: index
        for index, message in enumerate(view.messages[: snapshot["count"]])
    }
    timestamps = {
        entry.message.message_id: entry.timestamp
        for entry in view.entries
        if entry.message is not None
    }
    rows = []
    for identity in snapshot["summary_ids"]:
        record = summaries.read(identity, view)
        if record.content is None:
            continue
        for topic in record.content.topics:
            if any(identity not in positions for identity in topic.message_ids):
                raise ValueError("topic refers to missing original messages")
            ranges = _topic_ranges(
                view, set(topic.message_ids), len(record.message_ids)
            )
            reference = encode_cursor(
                "history_source",
                {
                    "session_id": view.session_id,
                    "branch_entry_id": record.branch_entry_id,
                    "count": len(record.message_ids),
                    "sha256": record.source_sha256,
                    "ranges": ranges,
                },
            )
            rows.append(
                {
                    "summary_id": record.summary_id,
                    "topic_id": topic.topic_id,
                    "title": topic.title,
                    "text": topic.text,
                    "scope": topic.scope,
                    "source_ref": reference,
                    "source_position": max(
                        positions[identity] for identity in topic.message_ids
                    ),
                    "source_time": timestamps.get(topic.message_ids[-1]),
                    "search_text": f"{topic.title} {topic.text} {topic.scope}",
                }
            )
    return sorted(
        rows,
        key=lambda item: (
            item["source_position"],
            item["summary_id"],
            item["topic_id"],
        ),
        reverse=True,
    )


def _segment_rows(
    view: MaterializedSession,
    summaries: SessionCompactionStore | None,
    snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    """按档位读取已核对表示和精确原件入口；参数：冻结查询；返回：片段，不重新生成摘要。"""
    if summaries is None:
        raise ValueError("segment reads require the session summary owner")
    rows = []
    positions = {
        message.message_id: index
        for index, message in enumerate(view.messages[: snapshot["count"]])
    }
    requested = snapshot["options"].get("segment_id")
    level = snapshot["options"].get("level") or "P4"
    for identity in snapshot["summary_ids"]:
        record = summaries.read(identity, view)
        for segment in record.content.segments if record.content is not None else ():
            if requested is not None and segment.segment_id != requested:
                continue
            start, end = (
                positions[segment.message_ids[0]],
                positions[segment.message_ids[-1]] + 1,
            )
            reference = encode_cursor(
                "history_source",
                {
                    "session_id": view.session_id,
                    "branch_entry_id": record.branch_entry_id,
                    "count": len(record.message_ids),
                    "sha256": record.source_sha256,
                    "ranges": [[start, end]],
                },
            )
            rows.append(
                {
                    "segment_id": segment.segment_id,
                    "summary_id": record.summary_id,
                    "title": segment.title,
                    "scope": segment.scope,
                    "text": segment.render(level),
                    "level": level,
                    "available_levels": HISTORY_LEVELS,
                    "message_ids": segment.message_ids,
                    "source_ref": reference,
                    "source_position": start,
                    "request_ids": record.request_ids,
                    "search_text": " ".join(
                        (
                            segment.title,
                            segment.scope,
                            segment.p1,
                            segment.p2,
                            segment.p3,
                            segment.p4,
                        )
                    ),
                }
            )
    return sorted(
        rows,
        key=lambda item: (item["source_position"], item["summary_id"]),
        reverse=True,
    )


def _topic_ranges(
    view: MaterializedSession, identities: set[str], count: int
) -> list[list[int]]:
    """将话题引用扩为完整调用范围和必要前文；传参：消息身份；返回：原文区间。"""
    ranges, position, previous = [], 0, 0
    for group in group_tool_call_units(view.messages[:count]):
        end = position + len(group)
        if any(message.message_id in identities for message in group):
            ranges.append([previous, end])
        previous, position = position, end
    return ranges
