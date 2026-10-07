"""通过正式记忆动作保留旧包装层的有效检索与状态断言。

作者：xxx
时间：2026-10-07 00:00:00
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone

import pytest

from context.memory_recall import recall_memories
from llm.messages import UserMessage, model_visible_text
from memory.store import MemorySource, MemoryStore
from tests.test_memory_agent_tools import invoke_memory

pytest_plugins = ("tests.test_memory_agent_tools",)


@pytest.mark.parametrize("state", ["active", "archived", "draft"])
def test_explicit_search_reports_usable_states_and_preserves_usage_boundary(
    tmp_path, memory_runtime, state
):
    """显式搜索可见归档、排除草稿，只有启用记录更新使用时间；参数：目录、运行、状态；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory(
            "fact", "deploy path uses manual rsync", ["deploy"], state=state
        )
        before = store.touch_memory(identity, "2020-01-01T00:00:00+00:00")
    result = invoke_memory(
        memory_runtime,
        "memory_query",
        {"action": "search", "query": "manual rsync deploy"},
    )
    assert result.status == "ok", result.output
    assert result.meta["index"]["state"] == "current"
    records = result.meta["records"]
    assert [row["memory_id"] for row in records] == (
        [] if state == "draft" else [identity]
    )
    if records:
        assert records[0]["state"] == state and records[0]["version"] == before.version
        assert records[0]["content"] == before.content and "score" in records[0]
    with closing(MemoryStore(tmp_path)) as store:
        after = store.load_memory(identity)
        assert after.version == before.version and after.updated_at == before.updated_at
        assert (after.last_used_at != before.last_used_at) is (state == "active")
    if state != "active":
        assert not recall_memories(
            tmp_path, task_summary="manual rsync deploy", task_tags=[]
        )


def test_explicit_search_finds_long_unverified_archived_fact(tmp_path, memory_runtime):
    """归档记录不因长期未复核而失去查找和恢复入口；参数：目录与真实动作；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory(
            "fact", "the retired vault rotation policy", [], state="archived"
        )
        historic = store.verify_memory(
            identity,
            (datetime.now(timezone.utc) - timedelta(days=181)).isoformat(),
            evidence=(MemorySource("tool_result", "fixture:historic-check"),),
        )
    result = invoke_memory(
        memory_runtime,
        "memory_query",
        {"action": "search", "query": "retired vault rotation policy"},
    )
    assert result.status == "ok", result.output
    assert [row["memory_id"] for row in result.meta["records"]] == [identity]
    assert result.meta["records"][0]["last_verified_at"] == historic.last_verified_at


def test_explicit_search_orders_matches_and_exposes_original_metadata(
    tmp_path, memory_runtime
):
    """搜索保留相关性顺序和原文身份，结果可继续精确读取；参数：目录与真实动作；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        first = store.create_memory("fact", "pytest fixture uses tmp_path", ["testing"])
        archived = store.create_memory(
            "fact", "use small commits", ["git"], state="archived"
        )
        original = store.load_memory(first)
    result = invoke_memory(
        memory_runtime, "memory_query", {"action": "search", "query": "pytest tmp_path"}
    )
    assert result.status == "ok", result.output
    records = result.meta["records"]
    assert records[0]["memory_id"] == first
    assert records[0]["type"] == "fact" and records[0]["state"] == "active"
    assert (
        records[0]["content"] == original.content
        and records[0]["version"] == original.version
    )
    assert records[0]["last_verified_at"] is None
    assert records[0]["score"] > records[1]["score"]
    assert {row["memory_id"]: row["state"] for row in records} == {
        first: "active",
        archived: "archived",
    }


def test_archive_is_idempotent_and_restore_retains_durable_history(
    tmp_path, memory_runtime
):
    """重复归档不增修订，恢复保留原文、理由及真实用户出处；参数：目录与真实动作；返回：无。"""
    context = memory_runtime[1].context
    branch = context.messages.materialize(context.run.session_id)
    inputs = [
        entry for entry in branch.entries if isinstance(entry.message, UserMessage)
    ]
    assert len(inputs) == 1
    inbound = inputs[0]
    assert model_visible_text(inbound.message) == context.run.payload["message"]
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory("fact", "audit trail for deploy", ["testing"])
        original = store.load_memory(identity)
    args = {
        "action": "archive",
        "memory_id": identity,
        "expected_version": original.version,
        "reason": "用户收起旧部署办法",
    }
    archived = invoke_memory(memory_runtime, "memory_manage", args)
    assert archived.status == "ok", archived.output
    archived_record = archived.meta["record"]
    assert archived_record["state"] == "archived"
    repeated = invoke_memory(
        memory_runtime,
        "memory_manage",
        {**args, "expected_version": archived_record["version"]},
    )
    assert repeated.status == "ok", repeated.output
    assert repeated.meta["record"]["version"] == archived_record["version"]
    restored = invoke_memory(
        memory_runtime,
        "memory_manage",
        {
            "action": "restore",
            "memory_id": identity,
            "expected_version": archived_record["version"],
            "reason": "用户恢复部署办法",
        },
    )
    assert restored.status == "ok", restored.output
    with closing(MemoryStore(tmp_path)) as store:
        current = store.load_memory(identity)
        archived_version = store.load_memory(
            identity, version=archived_record["version"]
        )
        assert current.state == "active" and current.revision == original.revision + 2
        assert current.previous_version == archived_version.version
        assert (
            current.reason == "用户恢复部署办法"
            and archived_version.reason == args["reason"]
        )
        assert current.content == archived_version.content == original.content
        assert (
            store.load_memory(identity, version=original.version).content
            == original.content
        )
        for record in (archived_version, current):
            assert (
                record.details.sources
                and record.details.sources[0].kind == "user_input"
            )
            source = record.details.sources[0]
            assert (
                source.reference == inbound.entry_id
                and source.session_id == branch.session_id
            )
            assert (
                source.run_id == (inbound.run_id or "")
                and source.observed_at == inbound.timestamp
            )


@pytest.mark.parametrize("private", [False, True])
def test_native_create_preserves_metadata_and_blocks_private_content(
    tmp_path, memory_runtime, private
):
    """正式创建回执指向真实版本，隐私拦截不落草稿或启用记录；参数：目录、运行、隐私场景；返回：无。"""
    content = "deploy token ghp_" + "A" * 36 if private else "pytest passed"
    result = invoke_memory(
        memory_runtime,
        "memory_manage",
        {
            "action": "create",
            "kind": "fact",
            "memory_scope": "global",
            "subject": "testing",
            "fact_key": "result",
            "type": "fact",
            "content": content,
            "tags": ["testing"],
        },
    )
    with closing(MemoryStore(tmp_path)) as store:
        if private:
            assert result.status == "error" and "safety scan" in result.error
            assert not store.list_memories()
            return
        assert result.status == "ok", result.output
        assert result.meta["committed"] is True
        saved = result.meta["record"]
        assert saved["memory_id"] and saved["version"] and saved["state"] == "active"
        original = store.load_memory(saved["memory_id"], version=saved["version"])
        assert original.content == content and original.tags == ["testing"]
        assert original.details.sources[0].kind == "user_input"


@pytest.mark.parametrize("state", ["active", "archived"])
def test_explicit_search_never_exposes_private_originals(
    tmp_path, memory_runtime, state
):
    """历史原件即使已有隐私文本，显式搜索仍明确过滤；参数：目录、运行、原状态；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory(
            "fact", "deploy token ghp_" + "A" * 36, [], state=state
        )
    result = invoke_memory(
        memory_runtime, "memory_query", {"action": "search", "query": "deploy token"}
    )
    assert result.status == "ok", result.output
    assert not result.meta["records"]
    assert any(
        row["memory_id"] == identity and row["reason"] == "blocked_by_safety_scan"
        for row in result.meta["skipped"]
    )
