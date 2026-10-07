"""历史与产物分页的版本化位置表达，不承担读取授权。

作者：xxx
时间：2026-09-14 20:00:00
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from typing import cast

CURSOR_VERSION = 1


def encode_cursor(kind: str, fields: Mapping[str, object]) -> str:
    """编码读取位置及来源版本；传参：游标类别与字段；返回：可原样传回的游标。"""
    payload = json.dumps(
        {**fields, "kind": kind, "version": CURSOR_VERSION}, separators=(",", ":")
    )
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def decode_cursor(
    cursor: str, kind: str, *, fields: Mapping[str, type]
) -> dict[str, object]:
    """校验外部游标的格式与字段类型；传参：游标、类别及字段声明；返回：已解析位置。"""
    try:
        encoded = base64.b64decode(cursor, altchars=b"-_", validate=True)
        value = json.loads(encoded)
    except (binascii.Error, UnicodeError, ValueError, TypeError) as exc:
        raise ValueError("invalid paging cursor") from exc
    if (
        not isinstance(value, dict)
        or value.get("kind") != kind
        or type(value.get("version")) is not int
    ):
        raise ValueError("paging cursor kind or version is invalid")
    if value["version"] != CURSOR_VERSION or set(value) != {*fields, "kind", "version"}:
        raise ValueError("unsupported paging cursor format")
    if any(type(value[name]) is not expected for name, expected in fields.items()):
        raise ValueError("paging cursor contains invalid fields")
    return cast(dict[str, object], value)
