"""工具与技能目录共用的查询及版本分页。

作者：xxx
时间：2026-09-14 21:00:00
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import cast

from context.cursors import decode_cursor, encode_cursor

DEFAULT_CATALOG_PAGE_SIZE = 20
MAX_CATALOG_PAGE_SIZE = 50


def matches_query(query: str, text: str) -> bool:
    """按名称及说明检索，空查询枚举全部；传参：查询和可搜索文字；返回：是否命中。"""
    return all(term in text.casefold() for term in query.casefold().split())


def catalog_page(
    items: Sequence[Mapping[str, object]],
    *,
    kind: str,
    query: str,
    cursor: str | None = None,
    limit: int = DEFAULT_CATALOG_PAGE_SIZE,
    source_version: str = "",
) -> dict[str, object]:
    """对已通过可见性/权限过滤的目录分页；传参：条目、查询及游标；返回：页面，变更使旧游标明确失效。"""
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_CATALOG_PAGE_SIZE
    ):
        raise ValueError(f"catalog limit must be between 1 and {MAX_CATALOG_PAGE_SIZE}")
    encoded = json.dumps(
        [source_version, items],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    version = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    offset = 0
    if cursor is not None:
        position = decode_cursor(
            cursor, kind, fields={"catalog_version": str, "query": str, "offset": int}
        )
        if position["catalog_version"] != version or position["query"] != query:
            raise ValueError(
                "catalog changed or query differs; start a new catalog page"
            )
        offset = cast(int, position["offset"])
        if not 0 < offset < len(items):
            raise ValueError("catalog cursor is outside the current result set")
    end = min(offset + limit, len(items))
    next_cursor = (
        encode_cursor(kind, {"catalog_version": version, "query": query, "offset": end})
        if end < len(items)
        else None
    )
    return {
        "items": list(items[offset:end]),
        "total": len(items),
        "catalog_version": version,
        "next_cursor": next_cursor,
        "has_more": next_cursor is not None,
    }
