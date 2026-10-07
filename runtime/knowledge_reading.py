"""【知识维护】【操作原件】冻结来源的紧凑目录与精确分页读取。

作者：xxx
时间：2026-10-02 11:37:00
"""

from typing import Any, cast

from context.cursors import decode_cursor, encode_cursor
from llm.messages import ToolResultMessage
from runtime.evidence_content import content_page, evidence_node
from runtime.history_reader import DEFAULT_HISTORY_PAGE_SIZE, MAX_HISTORY_PAGE_SIZE
from runtime.persistence import record_key
from runtime.persistence import RuntimeStore
from runtime.file_records import StoredRecord
from runtime.session_compaction import source_digest
from runtime.session_message_store import MaterializedSession, SessionMessageStore

OPERATION_PREVIEW_CHARACTERS = 320
OPERATION_PAGE_CHARACTERS = 8000
MAX_OPERATION_PAGE_CHARACTERS = 32768


def read_source_operations(
    messages: SessionMessageStore,
    view: MaterializedSession,
    arguments: dict[str, object],
) -> dict[str, Any]:
    """读取冻结分支可引用的真实操作；参数：依赖、来源及分页选择；返回：目录或指定原件正文页。"""
    database = messages.database
    receipts = {
        message.call_id: message
        for message in view.messages
        if isinstance(message, ToolResultMessage)
    }
    with database.snapshot() as source:
        anchor = source.raw(
            "session_entry", record_key(view.session_id, str(view.leaf_id))
        )
    if anchor is None or anchor.location is None:
        raise ValueError("knowledge operation source anchor is missing")
    with database.snapshot(sequence=anchor.location.sequence) as source:
        rows = [
            row
            for row in source.list_raw("tool_operation", session_id=view.session_id)
            if row.payload["call"]["call_id"] in receipts
        ]
    rows.sort(key=lambda row: (row.payload["updated_at"], row.record_id))
    identity = {
        "source_session_id": view.session_id,
        "source_entry_id": view.leaf_id,
        "source_mode": "origin_results",
    }
    if arguments.get("action") == "operation":
        return {**identity, **source_operation_page(database, rows, arguments)}
    return {
        **identity,
        **source_operation_directory(
            rows, view, arguments, sequence=anchor.location.sequence
        ),
    }


def source_operation_page(
    database: RuntimeStore, rows: list[StoredRecord], arguments: dict[str, object]
) -> dict[str, Any]:
    """按已授权操作身份展开完整原件的字符页；参数：原件存储、冻结操作及位置；返回：正文和精确续读位置。"""
    row = next(
        (row for row in rows if row.record_id == arguments.get("operation_id")), None
    )
    if row is None:
        raise ValueError(
            "operation is not part of the frozen knowledge source; use operations to find its identity"
        )
    offset, size = (
        arguments.get("offset", 0),
        arguments.get("max_chars", OPERATION_PAGE_CHARACTERS),
    )
    if (
        type(offset) is not int
        or type(size) is not int
        or not 0 < size <= MAX_OPERATION_PAGE_CHARACTERS
    ):
        raise ValueError("operation offset/max_chars must be valid character positions")
    return {
        "operation_id": row.record_id,
        "original": True,
        **content_page(database, evidence_node(row), offset=offset, limit=size),
    }


def source_operation_directory(
    rows: list[StoredRecord],
    view: MaterializedSession,
    arguments: dict[str, object],
    *,
    sequence: int,
) -> dict[str, Any]:
    """分页列出冻结操作及出处，游标不可混用到消息页或另一来源；参数：操作、来源、位置与提交；返回：紧凑目录。"""
    receipts = {
        message.call_id: message
        for message in view.messages
        if isinstance(message, ToolResultMessage)
    }
    size = arguments.get("limit", DEFAULT_HISTORY_PAGE_SIZE)
    if type(size) is not int or not 0 < size <= MAX_HISTORY_PAGE_SIZE:
        raise ValueError("knowledge operations limit must be a valid item count")
    snapshot = {
        "session_id": view.session_id,
        "entry_id": str(view.leaf_id),
        "sha256": source_digest(view.messages),
        "sequence": sequence,
    }
    start = 0
    if arguments.get("cursor") is not None:
        cursor = decode_cursor(
            cast(str, arguments["cursor"]),
            "knowledge_operations",
            fields={
                "session_id": str,
                "entry_id": str,
                "sha256": str,
                "sequence": int,
                "offset": int,
            },
        )
        if any(cursor[key] != value for key, value in snapshot.items()):
            raise ValueError(
                "knowledge operations cursor belongs to a different frozen source"
            )
        start = cast(int, cursor["offset"])
    if not 0 <= start <= len(rows):
        raise ValueError("knowledge operations cursor position is invalid")
    end = min(start + size, len(rows))
    return {
        "operations": [
            operation_summary(
                row.payload, receipts[row.payload["call"]["call_id"]].message_id
            )
            for row in rows[start:end]
        ],
        "total_count": len(rows),
        "returned_count": end - start,
        "next_cursor": encode_cursor(
            "knowledge_operations", {**snapshot, "offset": end}
        )
        if end < len(rows)
        else None,
        "complete": end == len(rows),
        "message_coverage_advanced": False,
    }


def operation_summary(row: dict[str, Any], message_id: str) -> dict[str, Any]:
    """目录只携带身份、真实状态和短预览，不冒充完整回执；参数：操作/消息身份；返回：可精确展开的目录项。"""
    result = row.get("result") or {}
    preview = result.get("output") or result.get("error") or ""
    return {
        "operation_id": row["operation_id"],
        "session_id": row["session_id"],
        "run_id": row["run_id"],
        "call_id": row["call"]["call_id"],
        "tool_name": row["call"]["tool_name"],
        "state": row["state"],
        "result_status": result.get("status"),
        "source_message_id": message_id,
        "path": row["call"]["args"].get("path"),
        "source_mode": "origin_results",
        "preview": str(preview)[:OPERATION_PREVIEW_CHARACTERS],
        "preview_only": True,
        "read_action": {
            "tool": "knowledge_read",
            "action": "operation",
            "operation_id": row["operation_id"],
        },
    }
