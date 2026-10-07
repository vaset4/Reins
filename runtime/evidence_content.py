"""【请求查看】【正文读取】按文件引用流式还原实际协议内容。

作者：xxx
时间：2026-09-30 15:00:00
"""

from __future__ import annotations

import codecs
import copy
import json
from collections.abc import Generator, Iterator
from typing import Any

from runtime.file_records import ContentReference, SourceCorruptionError, StoredRecord
from runtime.persistence import RuntimeStore

TEXT_CHUNK_BYTES = 16384
DEFAULT_DETAIL_CHARS = 32768


def evidence_node(record: StoredRecord) -> dict[str, Any]:
    """保留正文引用构造读取树；参数：已提交原件；返回：仅显式引用槽含类型对象的独立JSON树。"""
    result = copy.deepcopy(record.payload)
    try:
        for item in record.references:
            pointer = item["pointer"]
            target: Any = result
            for part in pointer[:-1]:
                target = target[part]
            if not pointer or target[pointer[-1]] is not None:
                raise SourceCorruptionError("invalid evidence content reference slot")
            target[pointer[-1]] = ContentReference.from_mapping(item["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise SourceCorruptionError("invalid evidence content reference") from exc
    return result


def member(node: Any, name: str) -> Any:
    """取得协议对象字段；参数：读取树与字段；返回：子节点，未记录字段为None。"""
    if not isinstance(node, dict):
        raise ValueError("invalid persistent evidence object")
    return node.get(name)


def scalar(node: Any) -> Any:
    """读取短身份和状态；参数：读取树节点；返回：标量，禁止把正文引用当作身份。"""
    if isinstance(node, (dict, list, ContentReference)):
        raise ValueError("expected inline evidence identity")
    return node


def text_chunks(
    store: RuntimeStore, reference: ContentReference
) -> Generator[str, None, None]:
    """校验文件后分段解码UTF-8；参数：原件服务和引用；返回：不截断多字节字符的文本片段。"""
    decoder = codecs.getincrementaldecoder("utf-8")()
    for chunk in store.iter_content(reference, chunk_size=TEXT_CHUNK_BYTES):
        yield decoder.decode(chunk)
    yield decoder.decode(b"", final=True)


def json_chunks(store: RuntimeStore, node: Any) -> Iterator[str]:
    """保持协议结构流式生成JSON；参数：原件服务及读取树；返回：完整JSON片段。"""
    if isinstance(node, ContentReference):
        yield '"'
        for chunk in text_chunks(store, node):
            yield json.dumps(chunk, ensure_ascii=False)[1:-1]
        yield '"'
    elif isinstance(node, list):
        yield "["
        for index, child in enumerate(node):
            if index:
                yield ",\n"
            yield from json_chunks(store, child)
        yield "]"
    elif isinstance(node, dict):
        yield "{"
        for index, (key, child) in enumerate(node.items()):
            if index:
                yield ",\n"
            yield json.dumps(key, ensure_ascii=False) + ": "
            yield from json_chunks(store, child)
        yield "}"
    else:
        yield json.dumps(node, ensure_ascii=False, allow_nan=False)


def content_page(
    store: RuntimeStore, node: Any, *, offset: int, limit: int
) -> dict[str, Any]:
    """生成协议JSON的有限字符页；参数：来源、节点、偏移和页长；返回：正文和精确后续位置。"""
    if offset < 0 or limit < 1 or limit > DEFAULT_DETAIL_CHARS:
        raise ValueError("invalid evidence content page")
    skipped = 0
    parts: list[str] = []
    remaining = limit + 1
    for chunk in json_chunks(store, node):
        if skipped + len(chunk) <= offset:
            skipped += len(chunk)
            continue
        start = max(0, offset - skipped)
        selected = chunk[start : start + remaining]
        parts.append(selected)
        remaining -= len(selected)
        skipped += len(chunk)
        if remaining == 0:
            break
    text = "".join(parts)
    has_more = len(text) > limit
    return {
        "text": text[:limit],
        "offset": offset,
        "has_more": has_more,
        "total_chars": json_length(store, node),
        "next_offset": offset + limit if has_more else None,
    }


def json_length(store: RuntimeStore, node: Any) -> int:
    """流式统计转义后的字符长度；参数：原件服务与节点；返回：精确长度，不物化整条正文。"""
    return sum(len(chunk) for chunk in json_chunks(store, node))


def materialize_node(store: RuntimeStore, node: Any) -> Any:
    """为显式完整读取还原节点；参数：原件服务和选中节点；返回：原协议JSON值。"""
    return json.loads("".join(json_chunks(store, node)))
