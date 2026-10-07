"""【知识维护】【冻结来源】覆盖真实入站交付和未完成工具尾部的回读一致性。

作者：xxx
时间：2026-10-02 11:01:00
"""

from llm.messages import AssistantMessage, TextPart, ToolCallPart, ToolResultMessage
from runtime.knowledge_maintenance import KnowledgeMaintenance, complete_source
from runtime.knowledge_sources import frozen_knowledge_source
from runtime.lease import from_trigger
from runtime.schema_meta import ensure_current_schema
from runtime.session_compaction import source_digest
from runtime.session_message_store import SessionMessageStore
from runtime.workspaces import WorkspaceStore
from runtime.types import RunContext, Trigger
from scripts.testing.llm import from_test_stub


def delivered_source(root):
    """建立真实入站及交付记录；参数：隔离目录；返回：消息存储和来源会话。"""
    ensure_current_schema(root)
    session_id = "stage10-source"
    WorkspaceStore(root).bind_session(session_id, root)
    messages = SessionMessageStore(root)
    messages.accept_input(
        session_id, "金额保留两位小数", input_id="source-input", run_id="source-run"
    )
    messages.deliver_inputs(session_id, run_id="source-run", task_id=None)
    return messages, session_id


def test_first_delivered_input_round_trips_its_frozen_anchor(tmp_path):
    """首条用户输入冻结在交付位置，重启后摘要与消息完全一致；参数：隔离目录；返回：无。"""
    messages, session_id = delivered_source(tmp_path)
    frozen = complete_source(messages.materialize(session_id))
    replay = SessionMessageStore(tmp_path).materialize(
        session_id, at_entry_id=frozen.leaf_id
    )
    assert replay.messages == frozen.messages
    assert source_digest(replay.messages) == source_digest(frozen.messages)
    assert replay.entries[-1].type == "delivery"


def test_unfinished_tool_tail_retains_complete_delivery_prefix(tmp_path):
    """尚未返回的工具调用不进入冻结源，前面的交付原文仍可读；参数：隔离目录；返回：无。"""
    messages, session_id = delivered_source(tmp_path)
    messages.append_message(
        session_id,
        AssistantMessage(
            "pending-assistant",
            (
                TextPart("读取文件"),
                ToolCallPart("pending-call", "file_read", {"path": "amount.txt"}),
            ),
        ),
        run_id="source-run",
    )
    frozen = complete_source(messages.materialize(session_id))
    replay = messages.materialize(session_id, at_entry_id=frozen.leaf_id)
    assert [message.message_id for message in replay.messages] == ["source-input"]
    assert replay.messages == frozen.messages
    assert replay.pending_tool_calls == ()


def test_approval_record_does_not_become_a_frozen_model_message(tmp_path):
    """审批原件留在父链，但不冒充已投影用户正文导致接纳失败；参数：隔离目录；返回：无。"""
    messages, session_id = delivered_source(tmp_path)
    messages.append_message(
        session_id,
        AssistantMessage(
            "call", (ToolCallPart("read", "file_read", {"path": "rule.txt"}),)
        ),
    )
    messages.accept_input(
        session_id, "批准本次", input_id="approval-input", input_kind="approval"
    )
    messages.append_message(
        session_id,
        ToolResultMessage(
            "result", "read", "file_read", (TextPart("两位小数"),), "success"
        ),
    )
    messages.append_message(
        session_id, AssistantMessage("answer", (TextPart("读取完成"),))
    )
    before = messages.read_entries(session_id)
    context = RunContext(
        session_id=session_id,
        run_id="source-run",
        trigger=Trigger.USER,
        payload={"input_message_id": "source-input"},
        capability_lease=from_trigger(
            "user", capabilities={"background_run": {"enabled": True}}
        ),
    )
    work = KnowledgeMaintenance(tmp_path).observe(
        context, client=from_test_stub("unused")
    )
    assert work is not None and "approval-input" not in work["message_ids"]
    assert len(frozen_knowledge_source(messages, work).messages) == 4
    assert messages.read_entries(session_id) == before
