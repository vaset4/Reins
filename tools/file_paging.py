"""【文件工具】【分页原件】将查询与文本范围绑定到同一资源和观察版本。

作者：xxx
时间：2026-10-02 14:15:00
"""

from hashlib import sha256
import json
from typing import Any

from context.cursors import decode_cursor, encode_cursor
from runtime.types import ReadOnlyInspectionRequest, ReadOnlyInspectionResult

MAX_DIRECTORY_PAGE_ITEMS = 200


class FilePageChanged(ValueError):
    """查询资源或表示版本变化，旧页不能继续拼接。"""


def page_identity(request: ReadOnlyInspectionRequest, meta: dict[str, Any]) -> str:
    """绑定资源、动作、查询与类型；参数：读取请求、实际资源信息；返回：稳定查询身份。"""
    value = [request.action, meta["resolved_path"], request.query, request.path_kind]
    return sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def cursor_position(
    request: ReadOnlyInspectionRequest, meta: dict[str, Any]
) -> dict[str, Any] | None:
    """核对游标所属查询和内容版本；参数：当前请求与资源版本；返回：原选择位置，无游标时为空。"""
    if request.cursor is None:
        return None
    position = decode_cursor(
        request.cursor,
        "file_view",
        fields={"identity": str, "source_version": str, "offset": int, "end": int},
    )
    if (
        position["identity"] != page_identity(request, meta)
        or position["source_version"] != meta["representation_version"]
    ):
        raise FilePageChanged(
            "file source, query or representation changed; begin a new read"
        )
    return dict(position)


def next_cursor(
    request: ReadOnlyInspectionRequest, meta: dict[str, Any], offset: int, end: int
) -> str | None:
    """在原范围内生成续页位置；参数：请求、版本、下个位置和选择末尾；返回：游标或读取结束。"""
    if offset >= end:
        return None
    return encode_cursor(
        "file_view",
        {
            "identity": page_identity(request, meta),
            "source_version": meta["representation_version"],
            "offset": offset,
            "end": end,
        },
    )


def entry_page(
    entries: list[dict[str, Any]],
    request: ReadOnlyInspectionRequest,
    meta: dict[str, Any],
    default_limit: int,
) -> ReadOnlyInspectionResult:
    """对完整查询结果按条目分页，保留类型和读取入口；参数：结果、请求、版本、默认页长；返回：实际页面。"""
    version = sha256(
        json.dumps([entries, meta], ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    details = {
        **meta,
        "representation": "path_entries",
        "representation_version": version,
        "unit": "entries",
    }
    position = cursor_position(request, details)
    start = position["offset"] if position is not None else 0
    limit = request.limit if request.limit is not None else default_limit
    if (
        type(limit) is not int
        or not 1 <= limit <= MAX_DIRECTORY_PAGE_ITEMS
        or not 0 <= start <= len(entries)
    ):
        raise ValueError("invalid entry page size or position")
    selected = entries[start : start + limit]
    after = start + len(selected)
    lines = [str(row["display"]) for row in selected]
    return ReadOnlyInspectionResult(
        request.action,
        "ok",
        "\n".join(lines),
        meta={
            **details,
            "entries": [
                {key: value for key, value in row.items() if key != "display"}
                for row in selected
            ],
            "offset": start,
            "returned_count": len(selected),
            "total_count": len(entries),
            "complete": after == len(entries),
            "truncated": after < len(entries),
            "next_cursor": next_cursor(request, details, after, len(entries)),
        },
    )


def text_page(
    content: str,
    request: ReadOnlyInspectionRequest,
    meta: dict[str, Any],
    max_chars: int,
) -> ReadOnlyInspectionResult:
    """按一开始行选择原文，超长单行以字符游标继续；参数：正文、请求、来源、页长；返回：无遗漏正文页。"""
    representation = str(meta.get("representation", "utf8_text"))
    version = sha256(
        json.dumps(
            [representation, meta.get("content_sha256"), content], ensure_ascii=False
        ).encode()
    ).hexdigest()
    details = {
        **meta,
        "representation": representation,
        "representation_version": version,
        "unit": "characters",
    }
    position = cursor_position(request, details)
    lines = content.splitlines(keepends=True)
    if position is not None:
        if request.start_line is not None or request.line_count is not None:
            raise ValueError("cursor cannot be combined with a new line selection")
        start, stop = position["offset"], position["end"]
    else:
        first = request.start_line if request.start_line is not None else 1
        count = request.line_count
        if type(first) is not int or first < 1 or first > len(lines) + 1:
            raise ValueError(
                "start_line must identify an existing line or the end of the file"
            )
        if count is not None and (type(count) is not int or count < 1):
            raise ValueError("line_count must be a positive integer")
        start = sum(map(len, lines[: first - 1]))
        stop = (
            len(content)
            if count is None
            else start + sum(map(len, lines[first - 1 : first - 1 + count]))
        )
    if not 0 <= start <= stop <= len(content):
        raise ValueError("text cursor is outside its original selection")
    end = min(start + max_chars, stop)
    following = next_cursor(request, details, end, stop)
    return ReadOnlyInspectionResult(
        "read_file",
        "ok",
        content[start:end],
        meta={
            **details,
            "offset": start,
            "returned_count": end - start,
            "total_count": len(content),
            "total_lines": len(lines),
            "start_line": content.count("\n", 0, start) + 1,
            "selection_end_offset": stop,
            "selection_complete": end == stop,
            "complete": start == 0 and end == len(content),
            "truncated": end < len(content),
            "next_offset": end if end < stop else None,
            "next_cursor": following,
        },
    )
