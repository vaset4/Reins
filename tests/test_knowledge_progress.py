"""验证冻结知识来源的并发接纳和持久阅读进度。

作者：xxx
时间：2026-10-01 18:20:00
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from threading import Barrier, local
from types import SimpleNamespace

import pytest

from app.scheduled_run import create_scheduler
from llm.messages import UserMessage
from memory.records import MemoryDetails
from memory.store import MemoryStore
from runtime.history_reader import read_history_page
from runtime.knowledge_jobs import KnowledgeJobs
from runtime.knowledge_maintenance import KnowledgeMaintenance
from runtime.lease import from_trigger
from runtime.memory_actions import MemoryActions
from runtime.session_messages import append_assistant_message, append_user_message
from runtime.tool_operations import ToolOperation
from runtime.types import RunToolsRequest, Trigger
from runtime.workspaces import WorkspaceStore
from scripts.testing.llm import from_test_stub
from tests import test_knowledge_validity_queries

query_context = test_knowledge_validity_queries.query_context


def admit_source(root, context):
    """把两条真实输入接纳为同一冻结工作；参数：隔离目录/运行依赖；返回：工作及来源视图。"""
    append_user_message(root, "source", "客户编号保留六位")
    append_user_message(root, "source", "金额保留两位小数")
    view = context.messages.materialize("source")
    context.run.capability_lease = from_trigger(
        "user", task_id="query", capabilities={"background_run": {"enabled": True}}
    )
    work = KnowledgeMaintenance(root).admit_sources(
        context.run,
        client=from_test_stub("unused"),
        view=view,
        message_ids={message.message_id for message in view.messages},
        reason="explicit_selection",
    )
    return work, view


def test_concurrent_pages_accumulate_read_coverage(
    tmp_path, query_context, monkeypatch
):
    """同时读取不同页不能以旧进度覆盖新进度；参数：隔离依赖/并发注入；返回：无。"""
    work, view = admit_source(tmp_path, query_context)
    WorkspaceStore(tmp_path).bind_session("worker", tmp_path)
    worker = replace(
        query_context.run,
        session_id="worker",
        trigger=Trigger.CRON,
        payload={
            "knowledge_origin": work,
            "source_session_id": "source",
            "source_run_id": work["source_run_id"],
        },
    )
    context = replace(query_context, run=worker)
    reader = KnowledgeJobs(context, data_root=tmp_path, client=from_test_stub("unused"))
    cursor = read_history_page(view, call_id="first", limit=1)["next_cursor"]
    arguments = (
        {"action": "messages", "limit": 1},
        {"action": "messages", "limit": 1, "cursor": cursor},
    )
    calls = [
        ToolOperation(
            RunToolsRequest("knowledge_read"), f"page-{index}", "knowledge_read", args
        )
        for index, args in enumerate(arguments)
    ]
    barrier, thread_state = Barrier(2), local()
    original = KnowledgeMaintenance.load

    def concurrent_load(self, work_id):
        """让两个读者取得同一旧进度后提交，真实发布锁保持不变；参数：工作；返回：读取快照。"""
        row = original(self, work_id)
        if not getattr(thread_state, "loaded", False):
            thread_state.loaded = True
            barrier.wait(timeout=5)
        return row

    with monkeypatch.context() as patch:
        patch.setattr(KnowledgeMaintenance, "load", concurrent_load)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(reader.execute, call) for call in calls]
            assert all(future.result(timeout=10).status == "ok" for future in futures)
    assert set(
        KnowledgeMaintenance(tmp_path).load(work["work_id"])["read_message_ids"]
    ) == set(work["message_ids"])


def test_claimed_work_keeps_frozen_sources_while_new_requests_accumulate(
    tmp_path, query_context
):
    """已认领工作不被追加请求改写，后续请求合并且重启保留范围；参数：隔离依赖；返回：无。"""
    first, _ = admit_source(tmp_path, query_context)
    with closing(
        create_scheduler(project_root=tmp_path, data_root=tmp_path)
    ) as scheduler:
        occurrence = scheduler.accept_due(now=datetime.now(timezone.utc))[0]
    manager = KnowledgeMaintenance(tmp_path)
    second_id = append_user_message(tmp_path, "source", "日期显示北京时间")
    third_id = append_user_message(tmp_path, "source", "客户姓名不要缩写")
    view = query_context.messages.materialize("source")
    barrier = Barrier(2)

    def accept(message_id):
        """并发接纳不同新增来源；参数：消息身份；返回：已持久工作。"""
        barrier.wait(timeout=5)
        return manager.admit_sources(
            query_context.run,
            client=from_test_stub("unused"),
            view=view,
            message_ids={message_id},
            reason="new_source",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(accept, identity) for identity in (second_id, third_id)]
        rows = [future.result(timeout=10) for future in futures]
    restarted = KnowledgeMaintenance(tmp_path)
    assert manager.load(first["work_id"])["message_ids"] == first["message_ids"]
    assert (
        occurrence.schedule_snapshot["knowledge_origin"]["message_ids"]
        == first["message_ids"]
    )
    assert rows[0]["work_id"] == rows[1]["work_id"] != first["work_id"]
    assert set(restarted.load(rows[0]["work_id"])["message_ids"]) == {
        second_id,
        third_id,
    }
    manager.update(rows[0]["work_id"], state="failed", error="model unavailable")
    failed = restarted.load(rows[0]["work_id"])
    assert failed["outcome"] is None and failed["read_message_ids"] == []
    assert (
        restarted.status()["historical_sources"]
        == "unprocessed_unless_explicitly_selected"
    )


def test_direct_markdown_edit_blocks_old_automatic_candidate(tmp_path, query_context):
    """手改原件在自动候选提交前同步成新版，旧expected_version不能覆盖；参数：隔离依赖；返回：无。"""
    work, _ = admit_source(tmp_path, query_context)
    WorkspaceStore(tmp_path).bind_session("worker", tmp_path)
    worker = replace(
        query_context.run,
        session_id="worker",
        trigger=Trigger.CRON,
        payload={
            "knowledge_origin": work,
            "source_session_id": "source",
            "source_run_id": work["source_run_id"],
        },
    )
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory(
            "fact",
            "金额保留一位小数",
            [],
            details=MemoryDetails(scope="session:source"),
        )
        old = store.load_memory(identity)
        path = store.current_path(identity)
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "金额保留一位小数", "金额保留三位小数"
            ),
            encoding="utf-8",
        )
    call = ToolOperation(
        RunToolsRequest("memory_manage"),
        "revision",
        "memory_manage",
        {
            "action": "revise",
            "memory_id": identity,
            "expected_version": old.version,
            "source_mode": "origin_inputs",
            "content": "金额保留两位小数",
            "reason": "生成于用户编辑前的旧候选",
        },
        operation_id="stale-candidate",
    )
    result = MemoryActions(
        replace(query_context, run=worker), data_root=tmp_path
    ).execute(call)
    assert result.status == "error" and "version conflict" in result.output
    with closing(MemoryStore(tmp_path)) as store:
        current = store.load_memory(identity)
        assert current.content == "金额保留三位小数"
        assert current.previous_version == old.version
        assert current.details.sources[-1].kind == "external_edit"


def test_saved_credentials_admit_previously_unaccepted_sources(
    tmp_path, query_context, monkeypatch
):
    """临时凭据未消费的来源在可持久凭据恢复后接纳；参数：隔离依赖/客户端注入；返回：无。"""
    identity = append_user_message(tmp_path, "source", "请长期保留客户编号规则")
    context = query_context.run
    context.payload["input_message_id"] = identity
    context.capability_lease = from_trigger(
        "user", task_id="query", capabilities={"background_run": {"enabled": True}}
    )
    temporary = from_test_stub("unused")
    monkeypatch.setattr(
        temporary,
        "_resolved_target",
        SimpleNamespace(credential_source="cli", api_key="test-only"),
    )
    manager = KnowledgeMaintenance(tmp_path)
    assert manager.observe(context, client=temporary) is None
    rejected = manager.status(session_id="source")
    assert (
        rejected["works"] == [] and rejected["admissions"][0]["state"] == "not_accepted"
    )
    with manager.database.snapshot() as source:
        sequence = source.sequence
    manager.observe(context, client=temporary)
    with manager.database.snapshot() as source:
        assert source.sequence == sequence
    accepted = manager.observe(context, client=from_test_stub("unused"))
    assert accepted["message_ids"] == [identity] and accepted["outcome"] is None
    assert manager.status(session_id="source")["admissions"][0]["state"] == "accepted"


@pytest.mark.parametrize("mixed_messages", [False, True])
def test_finish_requires_message_coverage_without_mandatory_read_action(
    tmp_path, query_context, mixed_messages
):
    """完成依赖全部冻结消息覆盖，不强制某一种读取动作；参数：隔离依赖/混合来源开关；返回：无。"""
    if mixed_messages:
        append_assistant_message(
            tmp_path, "source", "这只是原会话的临时分析，不形成长期知识"
        )
    work, view = admit_source(tmp_path, query_context)
    WorkspaceStore(tmp_path).bind_session("worker", tmp_path)
    worker = replace(
        query_context.run,
        session_id="worker",
        trigger=Trigger.CRON,
        payload={
            "knowledge_origin": work,
            "source_session_id": "source",
            "source_run_id": work["source_run_id"],
        },
    )
    reader = KnowledgeJobs(
        replace(query_context, run=worker),
        data_root=tmp_path,
        client=from_test_stub("unused"),
    )
    read = ToolOperation(
        RunToolsRequest("knowledge_read"),
        "read",
        "knowledge_read",
        {"action": "user_inputs", "limit": 1},
    )
    first = reader.execute(read)
    assert first.status == "ok" and first.meta["next_cursor"]
    last = reader.execute(
        replace(
            read,
            call_id="last",
            args={**read.args, "cursor": first.meta["next_cursor"]},
        )
    )
    assert last.status == "ok" and last.meta["next_cursor"] is None
    manager = KnowledgeMaintenance(tmp_path)
    user_ids = {
        message.message_id
        for message in view.messages
        if isinstance(message, UserMessage)
    }
    assert set(manager.load(work["work_id"])["read_message_ids"]) == user_ids
    # 1. 【知识维护】【来源覆盖】操作视图不把尚未读到的助手消息标成已读
    operations = reader.execute(
        replace(read, call_id="operations", args={"action": "operations"})
    )
    assert operations.status == "ok"
    assert set(manager.load(work["work_id"])["read_message_ids"]) == user_ids
    finish = ToolOperation(
        RunToolsRequest("knowledge_finish"),
        "finish",
        "knowledge_finish",
        {"outcome": "no_op", "reason": "已核对原文，没有需要新增的长期知识"},
    )
    result = reader.execute(finish)
    if mixed_messages:
        assert (
            result.status == "error"
            and "frozen sources have not all been read" in result.output
        )
        assert manager.load(work["work_id"])["outcome"] is None
        full = reader.execute(
            replace(read, call_id="full", args={"action": "messages"})
        )
        assert full.status == "ok" and full.meta["next_cursor"] is None
        assert {message["message_id"] for message in full.meta["messages"]} == set(
            work["message_ids"]
        )
        # 2. 【知识维护】【明确完成】补齐阅读本身不重放之前失败的完成动作
        assert manager.load(work["work_id"])["outcome"] is None
        result = reader.execute(replace(finish, call_id="finish-reviewed"))
    assert result.status == "ok"
    final = manager.load(work["work_id"])
    assert final["state"] == "no_op" and set(final["read_message_ids"]) == set(
        work["message_ids"]
    )


@pytest.mark.parametrize(
    "invalid_kind", ["branch_anchor", "users_to_messages", "messages_to_users"]
)
def test_invalid_knowledge_cursor_does_not_advance_coverage(
    tmp_path, query_context, invalid_kind
):
    """分支锚点和不同过滤视图的游标都明确拒绝，不推进阅读；参数：隔离依赖/错误类型；返回：无。"""
    append_assistant_message(tmp_path, "source", "保留原始分析")
    work, _ = admit_source(tmp_path, query_context)
    WorkspaceStore(tmp_path).bind_session("worker", tmp_path)
    worker = replace(
        query_context.run,
        session_id="worker",
        trigger=Trigger.CRON,
        payload={
            "knowledge_origin": work,
            "source_session_id": "source",
            "source_run_id": work["source_run_id"],
        },
    )
    reader = KnowledgeJobs(
        replace(query_context, run=worker),
        data_root=tmp_path,
        client=from_test_stub("unused"),
    )
    call = ToolOperation(
        RunToolsRequest("knowledge_read"),
        "users",
        "knowledge_read",
        {"action": "user_inputs", "limit": 1},
    )
    users = reader.execute(call)
    messages = reader.execute(
        replace(call, call_id="messages", args={"action": "messages", "limit": 1})
    )
    assert users.status == messages.status == "ok"
    assert users.meta["next_cursor"] and messages.meta["next_cursor"]
    invalid = {
        "branch_anchor": {
            "action": "messages",
            "cursor": messages.meta["branch_entry_id"],
        },
        "users_to_messages": {
            "action": "messages",
            "cursor": users.meta["next_cursor"],
        },
        "messages_to_users": {
            "action": "user_inputs",
            "cursor": messages.meta["next_cursor"],
        },
    }
    manager = KnowledgeMaintenance(tmp_path)
    before = manager.load(work["work_id"])
    result = reader.execute(
        replace(call, call_id="invalid", args=invalid[invalid_kind])
    )
    assert result.status == "error"
    assert manager.load(work["work_id"]) == before
