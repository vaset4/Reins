"""【会话历史】【阅读投影】按轮次和真实分叉生成目录，保留原始节点身份。

作者：xxx
时间：2026-10-02 11:24:00
"""

from collections import Counter
from contextlib import closing
from typing import Any

from runtime.evidence_content import evidence_node, text_chunks
from runtime.file_records import ContentReference, StoredRecord
from runtime.persistence import RuntimeStore

TREE_LABEL_CHARACTERS = 96
ENTRY_LABELS = {
    "inbound": "已接纳输入",
    "delivery": "用户",
    "branch": "分支起点",
    "message": "消息",
}
ROLE_LABELS = {"user": "用户", "assistant": "回复", "tool_result": "工具结果"}


def tree_rows(
    payloads: list[dict[str, Any]],
    *,
    view: str,
    query: str,
    labels: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """从轻量原件元数据生成可见祖先关系；参数：节点、视图与文字筛选；返回：不含大正文的目录。"""
    if view not in {"turns", "all"}:
        raise ValueError("history tree view must be turns or all")
    if not isinstance(query, str):
        raise ValueError("history tree query must be text")
    originals = {row["entry_id"]: row for row in payloads}
    children = Counter(row["parent_id"] for row in payloads)
    preambles = {
        str(row["message"]["message_id"]).removesuffix(":tool-calls")
        for row in payloads
        if str((row.get("message") or {}).get("message_id", "")).endswith(":tool-calls")
    }
    visible: dict[str, dict[str, Any]] = {}
    for row in payloads:
        message = row.get("message") or {}
        if row["type"] == "delivery":
            message = (originals.get(row.get("input_id")) or {}).get("message") or {}
        if view != "all" and not reading_entry(
            row,
            message=message,
            originals=originals,
            children=children,
            preambles=preambles,
        ):
            continue
        label = (
            labels[row["entry_id"]] if labels is not None else tree_label(row, message)
        )
        if query.casefold() not in label.casefold():
            continue
        visible[row["entry_id"]] = {
            key: row.get(key)
            for key in ("entry_id", "parent_id", "type", "run_id", "timestamp")
        }
        visible[row["entry_id"]]["label"] = label
    return link_tree_rows(visible, originals)


def reading_entry(
    row: dict[str, Any],
    *,
    message: dict[str, Any],
    originals: dict[str, dict[str, Any]],
    children: Counter[str | None],
    preambles: set[str],
) -> bool:
    """保留用户输入、最终回复和真实分叉，思考与工具进入完整视图；参数：节点及关系；返回：是否默认可见。"""
    if row["type"] == "branch" or children[row["entry_id"]] > 1:
        return True
    if row["type"] == "delivery":
        return originals[row["input_id"]].get("input_source") != "agent"
    if row["type"] != "message" or message.get("message_id") in preambles:
        return False
    kinds = {part["kind"] for part in message.get("content", [])}
    return (
        message.get("kind") in {"user", "assistant"}
        and "text" in kinds
        and "tool_call" not in kinds
    )


def link_tree_rows(
    visible: dict[str, dict[str, Any]], originals: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """过滤中间节点后连接最近可见祖先，单链与分叉分别计数；参数：可见及原节点；返回：独立目录行。"""
    linked = []
    for row in visible.values():
        parent = row["parent_id"]
        while parent is not None and parent not in visible:
            parent = originals[parent]["parent_id"]
        linked.append({**row, "display_parent_id": parent})
    displayed_children = Counter(row["display_parent_id"] for row in linked)
    return [
        {**row, "visible_children": displayed_children[row["entry_id"]]}
        for row in linked
    ]


def tree_label(row: dict[str, Any], message: dict[str, Any]) -> str:
    """优先展示正文摘要和时间，长原文按需展开；参数：条目及消息元数据；返回：目录文字。"""
    role = ROLE_LABELS.get(str(message.get("kind", "")), ENTRY_LABELS[row["type"]])
    if row.get("input_kind") == "approval":
        role = "授权记录"
    text = " ".join(
        str(part["text"])
        for part in message.get("content", [])
        if part.get("kind") == "text" and isinstance(part.get("text"), str)
    )
    preview = " ".join(text.split())[:TREE_LABEL_CHARACTERS]
    timestamp = str(row.get("timestamp") or "").replace("T", " ")[:19]
    return f"{role} · {preview or '原文见详情'} · {timestamp}"


def original_tree_label(
    database: RuntimeStore, row: dict[str, Any], source: StoredRecord
) -> str:
    """按需读取标签所需的原文字节，不物化整条大消息；参数：原件服务/展示节点/正文来源；返回：明确摘要。"""
    message = evidence_node(source).get("message") or {}
    parts = []
    for part in message.get("content", []):
        if part.get("kind") != "text":
            continue
        value = part.get("text")
        if isinstance(value, ContentReference):
            with closing(text_chunks(database, value)) as chunks:
                value = next(chunks)
        if not isinstance(value, str):
            raise ValueError("history text preview has no original content")
        parts.append({"kind": "text", "text": value[:TREE_LABEL_CHARACTERS]})
    return tree_label(row, {**message, "content": parts})


def project_tree_page(
    payloads: list[dict[str, Any]],
    *,
    after: int,
    limit: int,
    view: str,
    query: str,
    selected_entry: str | None,
    labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """筛选后定位原选择或最近可见祖先，空结果不遗忘原选择；参数：节点与查询；返回：有界目录页。"""
    rows = tree_rows(payloads, view=view, query=query, labels=labels)
    positions = {row["entry_id"]: index for index, row in enumerate(rows)}
    parents = {row["entry_id"]: row["parent_id"] for row in payloads}
    selection = selected_entry
    if selection is not None and selection not in parents:
        raise ValueError("selected history entry does not belong to this session")
    while selection is not None and selection not in positions:
        selection = parents[selection]
    if selection is not None:
        after = positions[selection] // limit * limit
    return {
        "entries": rows[after : after + limit],
        "after": after,
        "next_after": after + limit if after + limit < len(rows) else None,
        "selected_entry_id": selection,
        "view": view,
        "query": query,
        "total_count": len(rows),
        "raw_count": len(payloads),
    }
