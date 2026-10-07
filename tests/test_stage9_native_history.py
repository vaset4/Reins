"""【上下文】【历史导航】验证真实模型工具入口能展开已保存的多级历史。

作者：xxx
时间：2026-10-01 12:20:00
"""

from __future__ import annotations

import json

from llm.messages import ToolCallPart, ToolResultMessage
from runtime.session_compaction import SessionCompactionStore
from runtime.session_messages import materialize_messages
from scripts.testing.llm import from_test_native_tool_then_final
from tests.test_native_actions import native_registry
from tests.test_stage9_history import segment_content, source_history
from tests.test_tool_batch_execution import make_run


def test_native_history_selects_saved_level_and_returns_original_reference(tmp_path):
    """经原生声明与运行执行展开指定档位；参数：隔离数据空间；返回：无。"""
    loop, context, _ = make_run(tmp_path, native_registry(), [])
    messages, source, first = source_history(tmp_path, context.session_id)
    content = segment_content(source, first)
    saved = SessionCompactionStore(messages).publish(
        source,
        content.render(),
        content=content,
        request_ids=("generate", "audit"),
    )
    loop.llm_client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "segment-read",
                "read_history",
                {
                    "view": "segments",
                    "summary_id": saved.summary_id,
                    "segment_id": "local-study",
                    "level": "P2",
                },
            ),
        ],
        "已找到带来源的本地方案历史",
    )
    list(loop.run_stream(context))
    result = next(
        message
        for message in materialize_messages(tmp_path, context.session_id)
        if isinstance(message, ToolResultMessage) and message.call_id == "segment-read"
    )
    page = json.loads(json.loads(result.content[0].text)["output"])
    assert page["segments"][0]["level"] == "P2"
    assert page["segments"][0]["summary_id"] == saved.summary_id
    assert page["segments"][0]["source_ref"]
    assert "费用不得超过500" in json.dumps(page, ensure_ascii=False)
