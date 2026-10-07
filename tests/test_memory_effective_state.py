"""检验记忆替代的原子发布、有效读取和失败回执来源。

作者：xxx
时间：2026-09-25 12:00:00
"""

import sqlite3
from contextlib import closing
from dataclasses import replace

import pytest

from context.engine import recall_context_body
from context.memory_recall import recall_memories_with_outcome
from memory.records import MemoryDetails, MemoryReplacement, MemorySource
from memory.store import MemoryIndexUpdateError, MemoryStore, rebuild_memory_index
from runtime.persistence import RuntimeStore
from tests.test_memory_native_actions import run_action
from tools.builtin_tools import build_tool_registry


def old_decision(store, *, kind="fact"):
    """创建同对象的可追溯旧记录；传参：存储与类别；返回：已读旧版本。"""
    identity = store.create_memory(
        kind,
        "服务端口8000",
        ["端口"],
        memory_id="old",
        details=MemoryDetails(
            subject="服务",
            fact_key="端口",
            sources=(MemorySource("user_input", "input-old"),),
        ),
    )
    return store.load_memory(identity)


def replacement_details(old, *, source_kind="user_input"):
    """给新内容附上精确替代关系；传参：旧版本与证据类别；返回：候选元数据。"""
    return replace(
        old.details,
        sources=(MemorySource(source_kind, "evidence-new"),),
        supersedes=(MemoryReplacement(old.memory_id, old.version, "同一服务的新证据"),),
    )


def test_replacement_keeps_original_and_withdrawal_never_revives_it(tmp_path):
    """撤销新事实不恢复旧事实，查询和索引保持同一状态；传参：隔离目录；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        old = old_decision(store)
        original = store.load_memory("old", version=old.version)
        store.create_memory(
            "fact",
            "服务端口9000",
            ["端口"],
            memory_id="new",
            details=replacement_details(old),
        )
        assert (
            store.record_view(store.load_memory("old"))["effective_state"]
            == "superseded"
        )
        assert [item.memory_id for item in store.list_memories(state="active")] == [
            "new"
        ]
        new = store.load_memory("new")
        store.update_memory_state(
            "new", "withdrawn", expected_version=new.version, reason="不再采用此事实"
        )
        assert store.list_memories(state="active") == []
        assert store.record_view(old)["superseded_by"] == ["new"]
        assert store.load_memory("old", version=old.version) == original
        assert store.index_status().state == "current"
    assert rebuild_memory_index(tmp_path) == 2
    outcome = recall_memories_with_outcome(
        tmp_path, task_summary="服务端口8000", task_tags=["端口"]
    )
    assert not outcome.selected and outcome.index.state == "current"


@pytest.mark.parametrize("fault", ["version", "scope", "missing", "self"])
def test_invalid_replacement_does_not_publish_any_new_fact(tmp_path, fault):
    """版本或范围不符时旧状态保持，候选不可见；传参：临时根与错误；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        old = old_decision(store)
        details = replacement_details(old)
        if fault == "scope":
            details = replace(details, scope="project:另一个环境")
        else:
            identity = {"missing": "missing", "self": "new"}.get(fault, "old")
            version = "outdated" if fault == "version" else old.version
            details = replace(
                details, supersedes=(MemoryReplacement(identity, version, "更正"),)
            )
        with pytest.raises((ValueError, FileNotFoundError)):
            store.create_memory(
                "fact", "服务端口9000", [], memory_id="new", details=details
            )
        assert [item.memory_id for item in store.list_memories(state="active")] == [
            "old"
        ]
        with pytest.raises(FileNotFoundError):
            store.load_memory("new")


def test_committed_replacement_with_stale_index_still_controls_current_requirements(
    tmp_path,
):
    """已提交事实的派生索引损坏不复活旧要求；传参：隔离目录；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        old = old_decision(store, kind="rule")
        store.create_memory(
            "rule",
            "服务必须使用9000",
            [],
            memory_id="new",
            details=replacement_details(old),
        )
        with RuntimeStore(tmp_path).index_connection() as connection:
            connection.execute("DELETE FROM memories")
        assert store.index_status().state == "stale"
        assert store.record_view(old)["effective_state"] == "superseded"
    body = recall_context_body(
        tmp_path, task_summary="完全无关的财务问题", task_tags=[], skill_refs=[]
    )
    assert "服务必须使用9000" in body and "服务端口8000" not in body
    assert MemoryStore(tmp_path).index_status().state == "current"


def test_memory_index_failure_preserves_committed_compound_change(
    tmp_path, monkeypatch
):
    """索引失败不撤回已发布正文和替代关系；参数：隔离根与故障注入；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        old = old_decision(store)

        def fail(*_args, **_kwargs):
            """注入实际索引发布失败；参数：索引写入参数；返回：抛出IO异常。"""
            raise sqlite3.OperationalError("test index unavailable")

        with monkeypatch.context() as patch:
            patch.setattr("memory.store.upsert_memory_index", fail)
            with pytest.raises(MemoryIndexUpdateError, match="memory committed"):
                store.create_memory(
                    "fact",
                    "服务端口9000",
                    [],
                    memory_id="new",
                    details=replacement_details(old),
                )
        assert store.record_view(old)["effective_state"] == "superseded"
        assert store.index_status().state == "current"
        assert store.load_memory("new").content == "服务端口9000"


def test_factual_evidence_does_not_change_a_user_requirement(tmp_path):
    """工具可以更正现状，但不能代用户改变要求；传参：临时目录；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        old = old_decision(store, kind="rule")
        with pytest.raises(ValueError, match="requires user input"):
            store.create_memory(
                "fact",
                "服务端口9000",
                [],
                memory_id="new",
                details=replacement_details(old, source_kind="tool_result"),
            )
        with pytest.raises(ValueError, match="requires user input"):
            store.update_memory_state(
                "old", "withdrawn", sources=(MemorySource("tool_result", "observed"),)
            )
        assert store.list_memories(state="active")[0].memory_id == "old"


def test_model_memory_action_exposes_replacement_in_the_next_request(
    tmp_path, monkeypatch
):
    """覆盖参数经过真实工具声明和执行，下一请求只自动采用新值；传参：目录与替换器；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        old = old_decision(store)
    requests, operations = run_action(
        tmp_path,
        monkeypatch,
        session="replace",
        message="用9000替代原端口记录",
        arguments={
            "action": "create",
            "kind": "fact",
            "content": "服务端口9000",
            "memory_scope": "global",
            "subject": "服务",
            "fact_key": "端口",
            "supersedes": [
                {"memory_id": "old", "version": old.version, "reason": "用户更正"}
            ],
        },
    )
    assert operations[0]["result"]["status"] == "ok"
    assert operations[0]["result"]["meta"]["record"]["effective_state"] == "active"
    followup, _ = run_action(
        tmp_path, monkeypatch, session="fresh", message="现在检查服务端口"
    )
    assert "服务端口9000" in str(followup[0]) and "服务端口8000" not in str(followup[0])


def test_failed_receipt_can_support_a_lesson_but_not_an_archive(tmp_path, monkeypatch):
    """失败回执可跨会话回查为失败经验，不能伪称已取得原件；传参：目录与替换器；返回：无。"""
    registry = build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    _, reads = run_action(
        tmp_path,
        monkeypatch,
        session="failed-source",
        message="读取本地缺失文件",
        registry=registry,
        tool_name="file_read",
        arguments={"path": "missing.txt"},
    )
    source = reads[0]["operation_id"]
    arguments = {
        "action": "create",
        "type": "lesson",
        "kind": "fact",
        "content": "读取missing.txt失败，不能认为取得文件内容",
        "memory_scope": "global",
        "subject": "missing.txt",
        "fact_key": "读取经验",
        "source_mode": "tool_results",
        "source_operation_ids": [source],
    }
    _, operations = run_action(
        tmp_path,
        monkeypatch,
        session="failed-source",
        message="保留这次失败经验",
        arguments=arguments,
    )
    saved = next(
        row for row in operations if row["call"]["tool_name"] == "memory_manage"
    )
    assert saved["result"]["status"] == "ok", saved["result"]
    record = saved["result"]["meta"]["record"]
    assert record["details"]["sources"][0]["result_status"] == "error"
    requests, evidence = run_action(
        tmp_path,
        monkeypatch,
        session="read-lesson",
        message="查经验出处",
        tool_name="memory_query",
        arguments={"action": "sources", "memory_id": record["memory_id"]},
    )
    assert evidence[0]["result"]["meta"]["sources"][0]["receipt"]["status"] == "error"
    assert "missing.txt" in str(requests[-1])
    _, archive_ops = run_action(
        tmp_path,
        monkeypatch,
        session="failed-source",
        message="把这次原件归档",
        arguments={**arguments, "kind": "archive"},
    )
    archive = next(
        item for item in archive_ops if item["call"]["args"].get("kind") == "archive"
    )
    assert archive["result"]["status"] == "error"
