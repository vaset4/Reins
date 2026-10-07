"""【知识维护】【操作分页】默认不重复注入大回执，原件仍可精确读齐。

作者：xxx
时间：2026-10-02 11:43:00
"""

import json

import pytest

from llm.messages import AssistantMessage, TextPart, ToolCallPart, ToolResultMessage
from runtime.knowledge_reading import read_source_operations
from runtime.tool_operations import ToolOperationStore
from tests.test_stage10_knowledge_sources import delivered_source


def operation_sources(root):
    """保存三个有大正文的实际操作及匹配消息；参数：隔离根；返回：存储/冻结分支/正文。"""
    messages, session_id = delivered_source(root)
    body = "真实原件内容" * 9000
    for index in range(3):
        call_id = f"read-{index}"
        messages.append_message(
            session_id,
            AssistantMessage(
                f"assistant-{index}",
                (ToolCallPart(call_id, "file_read", {"path": f"document-{index}.md"}),),
            ),
            run_id="source-run",
        )
        ToolOperationStore(root).write(
            {
                "session_id": session_id,
                "run_id": "source-run",
                "operation_id": f"op-{index}",
            },
            {
                "call": {
                    "call_id": call_id,
                    "tool_name": "file_read",
                    "args": {"path": f"document-{index}.md"},
                },
                "state": "completed",
                "result": {"status": "ok", "output": body},
            },
        )
        messages.append_message(
            session_id,
            ToolResultMessage(
                f"result-{index}", call_id, "file_read", (TextPart(body),), "success"
            ),
            run_id="source-run",
        )
    return messages, messages.materialize(session_id), body


def test_compact_operation_pages_and_exact_original_cover_the_frozen_source(tmp_path):
    """目录分页不返回完整行，精确展开可重组原始JSON；参数：隔离根；返回：无。"""
    messages, view, body = operation_sources(tmp_path)
    first = read_source_operations(messages, view, {"action": "operations", "limit": 2})
    assert (
        len(first["operations"]) == 2 and first["next_cursor"] and not first["complete"]
    )
    assert len(json.dumps(first, ensure_ascii=False)) < 4000
    assert all(
        row["preview_only"] and row["source_mode"] == "origin_results"
        for row in first["operations"]
    )
    second = read_source_operations(
        messages,
        view,
        {"action": "operations", "limit": 2, "cursor": first["next_cursor"]},
    )
    assert (
        len(second["operations"]) == 1
        and second["next_cursor"] is None
        and second["complete"]
    )
    chunks, offset = [], 0
    while offset is not None:
        page = read_source_operations(
            messages,
            view,
            {"action": "operation", "operation_id": "op-0", "offset": offset},
        )
        chunks.append(page["text"])
        offset = page["next_offset"]
    assert json.loads("".join(chunks))["result"]["output"] == body
    with pytest.raises(ValueError, match="not part"):
        read_source_operations(
            messages, view, {"action": "operation", "operation_id": "foreign"}
        )
