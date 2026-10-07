"""【记忆】【Markdown验收】验证编辑、修订、索引丢失及实际请求一致性。

作者：xxx
时间：2026-09-30 17:00:00
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from context.memory_recall import recall_memories_with_outcome
from memory.index import MemoryIndexUpdateError
from memory.records import (
    MemoryDetails,
    MemoryReplacement,
    MemorySource,
    format_memory,
    parse_memory,
    seal_memory,
)
from memory.store import MemoryStore, rebuild_memory_index
from runtime.workspaces import WorkspaceStore
from tools.file_persistence import FileEditConflict


def test_direct_body_edit_seals_new_revision_without_manual_hash(
    tmp_path: Path,
) -> None:
    """正文与中文标签直接编辑后检索使用新版；参数：隔离根；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "晚餐要清淡", ["饮食"])
    verified = store.verify_memory(
        identity, evidence=(MemorySource("tool_result", "observation-1"),)
    )
    path = store.current_path(identity)
    path.write_text(
        path.read_text(encoding="utf-8")
        .replace("晚餐要清淡", "晚餐改为面条")
        .replace("- 饮食", "- 面食"),
        encoding="utf-8",
    )
    current = store.load_memory(identity)
    assert current.content == "晚餐改为面条" and current.tags == ["面食"]
    assert (
        current.previous_version == verified.version
        and current.version != verified.version
    )
    assert current.last_verified_at is None and current.verification == ()
    assert current.details.sources[-1].kind == "external_edit"
    assert store.load_memory(identity, version=verified.version).content == "晚餐要清淡"
    outcome = recall_memories_with_outcome(tmp_path, task_summary="面条", task_tags=[])
    assert outcome.selected[0].memory.content == "晚餐改为面条"
    assert outcome.selected[0].bm25 > 0
    assert store.load_memory(identity).version == current.version


def test_workspace_memory_files_keep_original_scope(tmp_path: Path) -> None:
    """同名项目和全局文件有清晰独立归属；参数：隔离根；返回：无。"""
    data = tmp_path / "data"
    first = WorkspaceStore(data).register(tmp_path / "one" / "project")
    second = WorkspaceStore(data).register(tmp_path / "two" / "project")
    store = MemoryStore(data)
    global_id = store.create_memory("fact", "通用规则", [])
    identity = store.create_memory(
        "fact",
        "项目代号蓝鲸",
        [],
        details=MemoryDetails(scope=f"project:{first.workspace_id}"),
    )
    assert store.current_path(global_id).parent == data / "global" / "memory"
    assert "workspaces" in store.current_path(identity).parts
    assert first.workspace_id != second.workspace_id
    assert (
        not recall_memories_with_outcome(
            data,
            task_summary="蓝鲸",
            task_tags=[],
            scopes=[f"project:{second.workspace_id}"],
        )
        .selected[0]
        .memory.memory_id
        == identity
    )


def test_missing_index_rebuild_preserves_archive_sources_and_replacements(
    tmp_path: Path,
) -> None:
    """重启重建保留不可逆替代关系及历史来源；参数：隔离根；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory(
        "fact", "预算800", [], details=MemoryDetails(subject="客户", fact_key="预算")
    )
    old = store.load_memory(identity)
    details = replace(
        old.details,
        sources=(MemorySource("user_input", "input-2"),),
        supersedes=(MemoryReplacement(identity, old.version, "用户更正"),),
    )
    replacement_id = store.create_memory("fact", "预算600", [], details=details)
    replacement = store.archive_memory(replacement_id)
    (tmp_path / "index.sqlite").unlink()
    assert rebuild_memory_index(tmp_path) == 2
    reopened = MemoryStore(tmp_path)
    assert (
        reopened.record_view(reopened.load_memory(identity))["effective_state"]
        == "superseded"
    )
    assert reopened.load_memory(replacement_id).state == "archived"
    assert reopened.load_memory(replacement_id).details == replacement.details


def test_deleted_markdown_is_reported_after_index_rebuild(tmp_path: Path) -> None:
    """索引丢失不能把已删除原件当作仍有效或空知识；参数：隔离根；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "禁止复活旧正文", [])
    path = store.current_path(identity)
    path.unlink()
    (tmp_path / "index.sqlite").unlink()
    with pytest.raises(FileNotFoundError, match="published memory original missing"):
        rebuild_memory_index(tmp_path)
    with pytest.raises(FileNotFoundError, match="published memory original missing"):
        recall_memories_with_outcome(tmp_path, task_summary="正文", task_tags=[])


def test_renaming_keeps_identity_and_duplicate_identity_fails(tmp_path: Path) -> None:
    """重命名不丢修订，复制同身份文件明确拒绝；参数：隔离根；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "原件可以改文件名", [])
    before = store.load_memory(identity)
    renamed = store.current_path(identity).with_name("用户命名.md")
    store.current_path(identity).rename(renamed)
    assert store.load_memory(identity).version == before.version
    renamed.with_name("副本.md").write_bytes(renamed.read_bytes())
    with pytest.raises(ValueError, match="duplicate memory identity"):
        store.list_memories()


def test_bad_frontmatter_does_not_recall_indexed_text(tmp_path: Path) -> None:
    """坏前言不能沿旧索引继续注入正文；参数：隔离根；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "不能使用的旧原文", [])
    store.current_path(identity).write_text(
        "---\nbroken: [\n---\n新内容", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid memory original"):
        recall_memories_with_outcome(tmp_path, task_summary="原文", task_tags=[])


def test_index_failure_reports_committed_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """原件发布之后索引失败仍保留新版及历史；参数：隔离根和故障注入；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "保存前", [])
    old = store.load_memory(identity)

    def fail_index(*args: object, **kwargs: object) -> None:
        """模拟派生写入故障；参数：索引调用参数；返回：抛出真实错误。"""
        raise sqlite3.OperationalError("injected write failure")

    with monkeypatch.context() as patch:
        patch.setattr("memory.store.upsert_memory_index", fail_index)
        with pytest.raises(MemoryIndexUpdateError, match="memory committed") as error:
            store.revise_memory(
                identity,
                "已经保存",
                expected_version=old.version,
                reason="更正",
                sources=(MemorySource("user_input", "input-2"),),
            )
        assert error.value.memory_id == identity
        assert (
            parse_memory(
                store.current_path(identity).read_text(encoding="utf-8")
            ).content
            == "已经保存"
        )
    assert rebuild_memory_index(tmp_path) == 1
    assert store.load_memory(identity).previous_version == old.version


def test_concurrent_user_edit_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AI发布前再次出现外部编辑时保留用户版本；参数：隔离根与故障注入；返回：无。"""
    from memory import files

    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "读取时正文", [])
    old = store.load_memory(identity)
    publish = files.publish_file

    def edit_before_publish(
        path: Path,
        original: bytes | None,
        updated: bytes,
        *,
        backup_path: Path | None = None,
    ) -> None:
        """在实际比较前模拟外部写者；参数：发布字节；返回：按真实原语发布。"""
        assert original is not None
        path.write_bytes(
            original.replace("读取时正文".encode(), "用户刚写的新正文".encode())
        )
        publish(path, original, updated, backup_path=backup_path)

    with monkeypatch.context() as patch:
        patch.setattr(files, "publish_file", edit_before_publish)
        with pytest.raises(FileEditConflict):
            store.revise_memory(
                identity,
                "AI基于旧版的内容",
                expected_version=old.version,
                reason="AI更正",
                sources=(MemorySource("user_input", "input-3"),),
            )
    assert store.load_memory(identity).content == "用户刚写的新正文"
    with pytest.raises(ValueError, match="version conflict"):
        store.revise_memory(
            identity,
            "又一次旧请求",
            expected_version=old.version,
            reason="旧请求",
            sources=(MemorySource("user_input", "input-4"),),
        )


def test_source_change_after_selection_does_not_enter_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """评分后原件再变时拒绝旧快照进入请求；参数：隔离根和替换器；返回：无。"""
    from context import memory_recall

    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "评分时正文", ["正文"])
    path = store.current_path(identity)

    def edit_after_selection(*args: object) -> None:
        """模拟选择完成后的外部编辑；参数：原回调参数；返回：无。"""
        path.write_text(
            path.read_text(encoding="utf-8").replace("评分时正文", "新正文"),
            encoding="utf-8",
        )

    monkeypatch.setattr(memory_recall, "_touch_active", edit_after_selection)
    with pytest.raises(FileEditConflict, match="changed before use"):
        recall_memories_with_outcome(tmp_path, task_summary="正文", task_tags=[])


def test_history_rollback_cannot_remove_published_replacements(tmp_path: Path) -> None:
    """手改当前文件为历史版本不能抹掉已发布链；参数：隔离根；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "第一版", [])
    old = store.load_memory(identity)
    store.revise_memory(
        identity,
        "第二版",
        expected_version=old.version,
        reason="更正",
        sources=(MemorySource("user_input", "input-2"),),
    )
    store.current_path(identity).write_text(format_memory(old), encoding="utf-8")
    with pytest.raises(ValueError, match="publication rollback"):
        store.load_memory(identity)


def test_direct_edit_enters_actual_model_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """用生产循环捕获实际ModelRequest确认编辑稿进入模型；参数：隔离根与本地客户端；返回：无。"""
    from tests.test_memory_native_actions import run_action

    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "用户晚餐吃米饭", ["晚餐"])
    path = store.current_path(identity)
    path.write_text(
        path.read_text(encoding="utf-8").replace("用户晚餐吃米饭", "用户晚餐吃面条"),
        encoding="utf-8",
    )
    requests, _ = run_action(
        tmp_path, monkeypatch, session="edited-memory", message="晚餐吃什么"
    )
    assert requests and "用户晚餐吃面条" in str(requests[0])
    assert "用户晚餐吃米饭" not in str(requests[0])


def test_scope_revision_moves_current_and_keeps_history(tmp_path: Path) -> None:
    """更正适用范围后文件随项目归属移动，旧版可追溯；参数：隔离根；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "全局预算800", [])
    old = store.load_memory(identity)
    original_path = store.current_path(identity)
    workspace = WorkspaceStore(tmp_path).register(tmp_path / "project")
    revised = store.revise_memory(
        identity,
        "当前项目预算600",
        expected_version=old.version,
        reason="更正适用范围",
        sources=(MemorySource("user_input", "input-2"),),
        details=replace(old.details, scope=f"project:{workspace.workspace_id}"),
    )
    assert not original_path.exists()
    assert "workspaces" in store.current_path(identity).parts
    assert store.load_memory(identity).version == revised.version
    assert store.load_memory(identity, version=old.version).content == old.content


def test_interrupted_scope_relocation_reconciles_published_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """新当前稿已保存但位置登记失败，重启按持久搬迁意图找到新版；参数：隔离根及故障注入；返回：无。"""
    from memory.files import MemoryPublicationError

    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "原预算", [])
    old = store.load_memory(identity)
    workspace = WorkspaceStore(tmp_path).register(tmp_path / "project")
    register = store._files.register

    def fail_target_registration(memory, path):
        """只让新目标登记失败；参数：发布记忆及路径；返回：无。"""
        if memory.details.scope.startswith("project:"):
            raise OSError("injected registration failure")
        register(memory, path)

    with monkeypatch.context() as patch:
        patch.setattr(store._files, "register", fail_target_registration)
        with pytest.raises(MemoryPublicationError, match="memory committed"):
            store.revise_memory(
                identity,
                "项目新预算",
                expected_version=old.version,
                reason="更正范围",
                sources=(MemorySource("user_input", "input-2"),),
                details=replace(old.details, scope=f"project:{workspace.workspace_id}"),
            )
    current = MemoryStore(tmp_path).load_memory(identity)
    assert current.content == "项目新预算" and current.previous_version == old.version


def test_external_edit_index_failure_reports_saved_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """用户正文已接纳后索引失败，也要区分已经保存；参数：隔离根和故障注入；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "用户旧稿", [])
    path = store.current_path(identity)
    path.write_text(
        path.read_text(encoding="utf-8").replace("用户旧稿", "用户新稿"),
        encoding="utf-8",
    )

    def fail_index(*args: object, **kwargs: object) -> None:
        """模拟派生写失败；参数：索引参数；返回：抛出错误。"""
        raise sqlite3.OperationalError("external edit index failure")

    with monkeypatch.context() as patch:
        patch.setattr("memory.store.upsert_memory_index", fail_index)
        with pytest.raises(MemoryIndexUpdateError, match="memory committed"):
            store.load_memory(identity)
    assert parse_memory(path.read_text(encoding="utf-8")).content == "用户新稿"
    assert store.load_memory(identity).content == "用户新稿"


def test_final_replace_race_keeps_user_bytes_and_newer_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """最终系统替换竞争保留被替换的用户稿，不回写覆盖第三次编辑；参数：目录及替换器；返回：无。"""
    from tools import file_persistence

    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "最初正文", [])
    old = store.load_memory(identity)
    path = store.current_path(identity)
    raw = path.read_bytes()
    replace_file = file_persistence._KERNEL.ReplaceFileW

    def external_race(*args: object) -> object:
        """在真实ReplaceFile前后分别写用户版本；参数：系统调用参数；返回：真实结果。"""
        path.write_bytes(raw.replace("最初正文".encode(), "第二次用户稿".encode()))
        result = replace_file(*args)
        path.write_bytes(
            path.read_bytes().replace("AI修改正文".encode(), "第三次用户稿".encode())
        )
        return result

    with monkeypatch.context() as patch:
        patch.setattr(file_persistence._KERNEL, "ReplaceFileW", external_race)
        with pytest.raises(FileEditConflict) as error:
            store.revise_memory(
                identity,
                "AI修改正文",
                expected_version=old.version,
                reason="更正",
                sources=(MemorySource("user_input", "input-2"),),
            )
    assert error.value.backup_path is not None
    assert "第二次用户稿" in error.value.backup_path.read_text(encoding="utf-8")
    assert "第三次用户稿" in path.read_text(encoding="utf-8")
    assert store.load_memory(identity).content == "第三次用户稿"


def test_external_edit_reason_cannot_forge_user_evidence(tmp_path: Path) -> None:
    """模型伪造外部编辑理由不能绕过用户要求来源保护；参数：隔离目录；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory(
        "rule",
        "必须人工核对",
        [],
        details=MemoryDetails(sources=(MemorySource("user_input", "input-1"),)),
    )
    old = store.load_memory(identity)
    with pytest.raises(ValueError, match="requires user input"):
        store.revise_memory(
            identity,
            "不需要人工核对",
            expected_version=old.version,
            reason="external Markdown edit",
            sources=(MemorySource("tool_result", "observation"),),
        )
    assert store.load_memory(identity).content == old.content


def test_native_receipt_preserves_committed_identity_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """生产记忆工具返回已保存但登记失败，后续查询恢复同一身份；参数：隔离根及替换器；返回：无。"""
    from memory.files import MemoryFiles
    from tests.test_memory_native_actions import run_action

    register = MemoryFiles.register
    failed = False

    def fail_once(self, memory, path):
        """只让首次身份登记失败；参数：文件存储、原件与位置；返回：无。"""
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected identity registration failure")
        register(self, memory, path)

    monkeypatch.setattr(MemoryFiles, "register", fail_once)
    _, operations = run_action(
        tmp_path,
        monkeypatch,
        session="registration-fault",
        message="请记住预算600",
        arguments={
            "action": "create",
            "kind": "fact",
            "content": "预算600",
            "memory_scope": "global",
            "subject": "用户",
            "fact_key": "预算",
            "tags": ["预算"],
        },
    )
    result = operations[0]["result"]
    assert result["status"] == "error" and result["meta"]["committed"] is True
    assert result["meta"]["publication_state"] == "registration_failed"
    assert (
        MemoryStore(tmp_path).load_memory(result["meta"]["memory_id"]).content
        == "预算600"
    )


def test_new_markdown_without_hash_and_orphan_revision_are_distinct(
    tmp_path: Path,
) -> None:
    """新建编辑稿由系统计算版本，未发布修订不能成为历史；参数：隔离目录；返回：无。"""
    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "已发布内容", [])
    old = store.load_memory(identity)
    new = replace(old, memory_id="user-created", content="用户新建内容", version="")
    path = store.current_path(identity).with_name("用户新建.md")
    path.write_text(format_memory(new), encoding="utf-8")
    created = store.load_memory("user-created")
    assert created.content == "用户新建内容" and created.version
    assert created.details.sources[-1].kind == "external_edit"
    orphan = seal_memory(
        replace(
            old,
            content="未发布的候选",
            previous_version=old.version,
            revision=old.revision + 1,
        )
    )
    orphan_path = store._files.revision_path(
        store.current_path(identity), identity, orphan.version
    )
    orphan_path.write_text(format_memory(orphan), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="memory revision not found"):
        store.load_memory(identity, version=orphan.version)
    assert store.load_memory(identity).content == "已发布内容"


@pytest.mark.parametrize(
    "failure",
    ["backup_unlink", "relocation_rename", "relocation_read", "relocation_unlink"],
)
def test_cleanup_failure_after_publication_reports_saved_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """原件发布后清理任何一步失败仍返回已保存版本；参数：隔离根、注入器和失败点；返回：无。"""
    from memory.files import MemoryPublicationError

    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "旧正文", [])
    old = store.load_memory(identity)
    old_path = store.current_path(identity)
    details = old.details
    if failure.startswith("relocation"):
        workspace = WorkspaceStore(tmp_path).register(tmp_path / "project")
        details = replace(details, scope=f"project:{workspace.workspace_id}")
    operation = failure.rsplit("_", 1)[1]
    original = getattr(Path, "read_bytes" if operation == "read" else operation)

    def fail_cleanup(path, *args, **kwargs):
        """只拒绝当前原件发布后的指定清理步骤；参数：文件及原参数；返回：原调用结果。"""
        affected = (
            path == old_path
            if operation == "rename"
            else path.parent.name == "edit-conflicts"
        )
        if affected:
            raise PermissionError("injected cleanup failure")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            Path, "read_bytes" if operation == "read" else operation, fail_cleanup
        )
        with pytest.raises(MemoryPublicationError, match="memory committed") as error:
            store.revise_memory(
                identity,
                "已保存的新正文",
                expected_version=old.version,
                reason="更正",
                sources=(MemorySource("user_input", "input-2"),),
                details=details,
            )
    current = store.load_memory(identity)
    assert (
        current.content == "已保存的新正文" and error.value.version == current.version
    )
    assert error.value.memory_id == identity
    assert error.value.phase in {"backup_cleanup", "relocation_cleanup"}
