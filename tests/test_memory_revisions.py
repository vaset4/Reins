"""验证有来源的记忆修订、索引故障与跨会话有效内容。

作者：xxx
时间：2026-09-15 01:10:00
"""

from __future__ import annotations

from contextlib import closing
import sqlite3

import pytest

from memory.store import MemoryDetails, MemorySource, MemoryStore, rebuild_memory_index
from memory.index import MemoryIndexUpdateError
from runtime.workspaces import WorkspaceStore
from memory.writer import MemoryWriter


def test_new_memory_does_not_claim_it_was_verified(tmp_path):
    """保存用户陈述不等于核验该陈述；传参：隔离目录；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory("fact", "客户甲的预算是800元", ["预算"])
        memory = store.load_memory(identity)
    assert memory.created_at
    assert memory.last_verified_at is None


def test_missing_index_row_does_not_hide_canonical_memory(tmp_path):
    """原文已落盘时，索引缺行不能使档案从枚举中消失；传参：隔离目录；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory("fact", "合同编号为A-2026-18", ["合同"])
        with sqlite3.connect(tmp_path / "index.sqlite") as conn:
            conn.execute("DELETE FROM memories WHERE memory_id = ?", (identity,))
        assert [record.memory_id for record in store.list_memories()] == [identity]


def test_correction_preserves_original_and_rejects_stale_update(tmp_path):
    """用户更正产生有出处的新版，旧请求不能覆盖它；传参：隔离目录；返回：无。"""
    source = MemorySource("user_input", "input-2", "session-1", "run-2")
    scope = f"project:{WorkspaceStore(tmp_path).register(tmp_path / 'project').workspace_id}"
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory(
            "fact",
            "预算800元",
            ["预算"],
            details=MemoryDetails(scope=scope, subject="客户", fact_key="预算"),
        )
        old = store.load_memory(identity)
        raw = store.load_memory(identity)
        new = store.revise_memory(
            identity,
            "预算600元",
            expected_version=old.version,
            reason="用户更正预算",
            sources=(source,),
            change_id="operation-2",
        )
        assert new.previous_version == old.version
        assert new.details.sources == (source,)
        assert new.details.scope == scope
        assert new.last_verified_at is None
        assert store.load_memory(identity, version=old.version).content == "预算800元"
        assert store.load_memory(identity, version=old.version) == raw
        replay = store.revise_memory(
            identity,
            "预算600元",
            expected_version=old.version,
            reason="用户更正预算",
            sources=(source,),
            change_id="operation-2",
        )
        assert replay.version == new.version
        with pytest.raises(ValueError, match="version conflict"):
            store.revise_memory(
                identity,
                "预算900元",
                expected_version=old.version,
                reason="旧请求",
                sources=(source,),
            )
        assert store.load_memory(identity).content == "预算600元"


def test_scope_and_exact_text_are_not_semantic_equivalence(tmp_path):
    """不同范围或大小写有业务差异的编号不能被相似文本合并；传参：隔离目录；返回：无。"""
    first_scope = (
        f"project:{WorkspaceStore(tmp_path).register(tmp_path / 'one').workspace_id}"
    )
    second_scope = (
        f"project:{WorkspaceStore(tmp_path).register(tmp_path / 'two').workspace_id}"
    )
    with closing(MemoryWriter(tmp_path)) as writer:
        first = writer.write_memory(
            "fact", "代号为 AbC", [], details=MemoryDetails(scope=first_scope)
        )
        second = writer.write_memory(
            "fact", "代号为 AbC", [], details=MemoryDetails(scope=second_scope)
        )
        third = writer.write_memory(
            "fact", "代号为 abc", [], details=MemoryDetails(scope=first_scope)
        )
    assert all(result.memory_id for result in (first, second, third))
    assert len({result.memory_id for result in (first, second, third)}) == 3


def test_touch_does_not_rewrite_content_or_claim_verification(tmp_path):
    """读取不产生新内容或核验事实；传参：隔离目录；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory("fact", "合同已登记", ["合同"])
        raw = store.load_memory(identity)
        old = store.load_memory(identity)
        used = store.touch_memory(identity)
        assert used.version == old.version and used.last_used_at
        assert used.last_verified_at is None
        assert store.load_memory(identity).content == raw.content
        assert store.load_memory(identity).updated_at == raw.updated_at
        assert store.index_status().state == "current"


def test_index_failure_keeps_markdown_and_rebuild_preserves_records(
    tmp_path, monkeypatch
):
    """索引失败后精确查询仍找到原件，显式重建恢复一致；传参：目录和故障注入；返回：无。"""

    def fail_index(*_args, **_kwargs):
        """模拟派生索引写失败；传参：任意；返回：抛出数据库错误。"""
        raise sqlite3.OperationalError("injected index failure")

    with closing(MemoryStore(tmp_path)) as store:
        with monkeypatch.context() as patch:
            patch.setattr("memory.store.upsert_memory_index", fail_index)
            with pytest.raises(MemoryIndexUpdateError, match="memory committed"):
                store.create_memory(
                    "fact",
                    "合同 A-18 原件",
                    ["合同"],
                    memory_id="contract",
                    details=MemoryDetails(
                        kind="archive",
                        archive_ref="artifact:original-18",
                        exact_fields={"合同编号": "A-18"},
                    ),
                )
        assert (
            store.list_memories(exact_fields={"合同编号": "A-18"})[0].content
            == "合同 A-18 原件"
        )
        assert store.index_status().state == "current"
        assert rebuild_memory_index(tmp_path) == 1


def test_old_memory_root_requires_explicit_reset(tmp_path):
    """旧文件格式不能被当作空新库；传参：隔离目录；返回：无。"""
    root = tmp_path / "memory"
    root.mkdir()
    (root / "legacy.md").write_text("legacy content", encoding="utf-8")
    with pytest.raises(ValueError, match="old or incomplete runtime data"):
        MemoryStore(tmp_path)
    assert (root / "legacy.md").read_text(encoding="utf-8") == "legacy content"


def test_damaged_index_rebuilds_without_recreating_knowledge(tmp_path):
    """索引损坏可重建，Markdown原文与身份保持；传参：隔离目录；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "归档原文可恢复", [])
    raw = store.current_path(identity).read_bytes()
    (tmp_path / "index.sqlite").write_bytes(b"broken sqlite")
    assert MemoryStore(tmp_path).load_memory(identity).content == "归档原文可恢复"
    assert store.current_path(identity).read_bytes() == raw


def test_verification_requires_evidence_and_correction_clears_it(tmp_path):
    """核验只属于原内容，更正后的结论需重新举证；传参：目录；返回：无。"""
    evidence = (MemorySource("tool_result", "operation-1"),)
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory("fact", "库存10件", [])
        with pytest.raises(ValueError, match="requires evidence"):
            store.verify_memory(identity, evidence=())
        verified = store.verify_memory(identity, evidence=evidence)
        assert verified.last_verified_at and verified.verification == evidence
        corrected = store.revise_memory(
            identity,
            "库存8件",
            expected_version=verified.version,
            reason="用户更正",
            sources=(MemorySource("user_input", "input-3"),),
        )
        assert corrected.last_verified_at is None and corrected.verification == ()
