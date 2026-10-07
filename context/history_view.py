"""【上下文】【历史视图】从唯一摘要与消息原件投影稳定材料和当前要求。

作者：xxx
时间：2026-10-01 12:00:00
"""

from __future__ import annotations

import json

from context.cursors import encode_cursor
from context.materials import ContextMaterial
from context.summary_entries import SummaryContent
from runtime.session_compaction import SessionCompactionStore
from runtime.session_message_store import MaterializedSession


def history_context(
    view: MaterializedSession, store: SessionCompactionStore
) -> dict[str, object]:
    """分离可选历史与不可裁剪要求；参数：当前分支、摘要所有者；返回：请求组装所需材料。"""
    # 1. 【上下文】【空历史】首次请求允许尚无会话原件，空分支不能触发读取或创建会话
    if view.leaf_id is None:
        return {
            "history_materials": (),
            "context_branch_id": "",
            "history_material_levels": {},
            "effective_requirements": "",
            "history_representation_version": 1,
        }
    chain = store.chain(view)
    positions = {
        message.message_id: index for index, message in enumerate(view.messages)
    }
    materials = []
    levels: dict[str, dict[str, str]] = {}
    legacy = next(
        (
            record
            for record in chain
            if record.content is None or not record.content.segments
        ),
        None,
    )
    visible = chain[: chain.index(legacy)] if legacy is not None else chain
    represented_ids = {
        identity
        for record in visible
        if record.content is not None
        for segment in record.content.segments
        for identity in segment.message_ids
    }
    if legacy is not None and not set(legacy.message_ids).issubset(represented_ids):
        # 2. 【上下文】【旧摘要接续】尚未分离当前要求的旧正文在原文重建前不能直接降成引用
        materials.append(
            ContextMaterial(
                identity=f"summary:{legacy.summary_id}",
                source="history",
                version=legacy.summary_id,
                scope=view.session_id,
                text=legacy.text,
                protected=legacy.content is None,
                reference=json.dumps(
                    {
                        "tool": "read_history",
                        "arguments": {"summary_id": legacy.summary_id},
                    },
                    ensure_ascii=False,
                ),
            )
        )
    for record in reversed(visible):
        if record.content is None or not record.content.segments:
            # 1. 【上下文】【历史兼容】旧累计摘要仅在没有新版片段时使用，不伪造四级或核对证据
            continue
        for segment in record.content.segments:
            identity = f"segment:{record.summary_id}:{segment.segment_id}"
            levels[identity] = {
                level: segment.render(level) for level in ("P1", "P2", "P3", "P4")
            }
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
            materials.append(
                ContextMaterial(
                    identity=identity,
                    source="history",
                    version=f"{record.summary_id}/P2",
                    scope=view.session_id,
                    text=segment.render(),
                    reference=json.dumps(
                        {
                            "segment_id": segment.segment_id,
                            "title": segment.title,
                            "summary_id": record.summary_id,
                            "tool": "read_history",
                            "arguments": {"source_ref": reference},
                        },
                        ensure_ascii=False,
                    ),
                )
            )
    current = chain[0] if chain else None
    protected = list(store.effective_requirements(view))
    if current is not None and current.content is not None:
        protected.extend(
            entry
            for entry in current.content.entries
            if entry.active and entry.kind in {"goal", "open"}
        )
    branch = next(
        (entry.entry_id for entry in reversed(view.entries) if entry.type == "branch"),
        view.entries[0].entry_id if view.entries else view.session_id,
    )
    return {
        "history_materials": tuple(materials),
        "context_branch_id": branch,
        "history_material_levels": levels,
        "effective_requirements": SummaryContent(entries=tuple(protected)).render(),
        "history_representation_version": 1,
    }
