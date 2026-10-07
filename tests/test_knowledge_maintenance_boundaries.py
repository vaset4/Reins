"""验证自动维护的版本、授权、旧来源和文件变化边界。

作者：xxx
时间：2026-10-01 15:00:00
"""

from contextlib import closing
from datetime import datetime, timezone

import pytest

from app.run_task import run_task
from app.scheduled_run import create_scheduler
from context.memory_recall import recall_memories_with_outcome
from llm.messages import ToolCallPart
from memory.records import MemoryDetails, MemorySource
from memory.store import MemoryStore
from runtime.knowledge_maintenance import KnowledgeMaintenance
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_user_message
from runtime.tool_operations import ToolOperationStore
from runtime.workspaces import WorkspaceStore
from scripts.testing.llm import from_test_native_tool_then_final, from_test_stub
from tests.test_automatic_knowledge import start_source
from tests.test_session_runtime import capture_requests
from tools.builtin_tools import build_tool_registry


def run_worker(root, client):
    """执行当前已接纳工作，模型由脚本客户端注入；参数：目录/客户端；返回：调度回执。"""
    with closing(
        create_scheduler(
            project_root=root, data_root=root, llm_factory=lambda _: client
        )
    ) as scheduler:
        return scheduler.run_due_jobs(now=datetime.now(timezone.utc))


def test_maintenance_cannot_edit_files_skills_or_foreign_memory(tmp_path, monkeypatch):
    """实际受限目录与写入边界拒绝扩权，不依赖提示词；参数：隔离根/捕获器；返回：无。"""
    _, work = start_source(tmp_path)
    calls = [
        ToolCallPart(
            "write", "file_write", {"path": "AGENTS.md", "content": "changed"}
        ),
        ToolCallPart(
            "skill",
            "skill_manage",
            {
                "action": "create",
                "skill_id": "forbidden",
                "body": "bad",
                "reason": "bad",
            },
        ),
        ToolCallPart(
            "global",
            "memory_manage",
            {
                "action": "create",
                "memory_scope": "global",
                "kind": "fact",
                "subject": "账户",
                "fact_key": "规则",
                "content": "越界规则",
                "source_mode": "origin_inputs",
            },
        ),
    ]
    client = from_test_native_tool_then_final(calls, "尝试完成")
    requests = capture_requests(client, monkeypatch)
    assert run_worker(tmp_path, client)[0].status == "failed"
    assert not (tmp_path / "AGENTS.md").exists()
    with closing(MemoryStore(tmp_path)) as store:
        assert store.list_memories() == []
    assert "accepted project" in str(requests[-1])
    assert KnowledgeMaintenance(tmp_path).load(work["work_id"])["outcome"] is None


def test_user_edit_rejects_stale_automatic_revision(tmp_path):
    """模型准备期间用户改动不被旧expected_version覆盖；参数：隔离根；返回：无。"""
    _, work = start_source(tmp_path)
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory(
            "fact",
            "小数位为一位",
            [],
            details=MemoryDetails(
                scope=f"project:{work['workspace_id']}",
                subject="金额",
                fact_key="小数位",
            ),
        )
        old = store.load_memory(identity)
        current = store.revise_memory(
            identity,
            "人工已调整为三位",
            expected_version=old.version,
            reason="用户直接修订",
            sources=(MemorySource("user_input", "real-correction"),),
        )
    client = from_test_native_tool_then_final(
        [
            ToolCallPart("read", "knowledge_read", {"action": "messages"}),
            ToolCallPart(
                "revise",
                "memory_manage",
                {
                    "action": "revise",
                    "memory_id": identity,
                    "expected_version": old.version,
                    "content": "自动改成两位",
                    "reason": "旧候选",
                    "source_mode": "origin_inputs",
                },
            ),
            ToolCallPart(
                "finish",
                "knowledge_finish",
                {"outcome": "no_op", "reason": "不能把失败写入当no-op"},
            ),
        ],
        "结束",
    )
    assert run_worker(tmp_path, client)[0].status == "failed"
    with closing(MemoryStore(tmp_path)) as store:
        assert store.load_memory(identity).version == current.version
        assert store.load_memory(identity).content == "人工已调整为三位"
    assert KnowledgeMaintenance(tmp_path).load(work["work_id"])["outcome"] is None


def test_cancellation_during_work_blocks_later_commits(tmp_path, monkeypatch):
    """取消请求在执行结束前保持cancelling，后续发布被短锁拒绝；参数：隔离根/注入器；返回：无。"""
    _, work = start_source(tmp_path)
    manager = KnowledgeMaintenance(tmp_path)
    observed = []
    original = manager.__class__.update

    def cancel_after_read(self, work_id, **changes):
        """在实际读取之后模拟用户取消；参数：进度更新；返回：真实持久结果。"""
        result = original(self, work_id, **changes)
        if changes.get("read_message_ids"):
            observed.append(self.cancel(work_id)["state"])
        return result

    monkeypatch.setattr(KnowledgeMaintenance, "update", cancel_after_read)
    client = from_test_native_tool_then_final(
        [
            ToolCallPart("read", "knowledge_read", {"action": "messages"}),
            ToolCallPart(
                "save",
                "memory_manage",
                {
                    "action": "create",
                    "memory_scope": "project",
                    "kind": "fact",
                    "subject": "金额",
                    "fact_key": "小数位",
                    "content": "两位",
                    "source_mode": "origin_inputs",
                },
            ),
        ],
        "已停止",
    )
    run_worker(tmp_path, client)
    assert observed == ["cancelling"]
    assert manager.load(work["work_id"])["state"] == "cancelled"
    with closing(MemoryStore(tmp_path)) as store:
        assert store.list_memories() == []


def test_selected_old_source_uses_public_reflection_entry(tmp_path):
    """显式选择旧消息可提炼且独立记录覆盖，首次启用不会抢跑；参数：隔离根；返回：无。"""
    WorkspaceStore(tmp_path).bind_session("old-session", tmp_path)
    append_user_message(tmp_path, "old-session", "旧项目中客户编号固定六位")
    old_id = (
        SessionMessageStore(tmp_path).materialize("old-session").messages[0].message_id
    )
    manager = KnowledgeMaintenance(tmp_path)
    manager.configure(enabled=True)
    manager.configure(enabled=False)
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "old",
                "knowledge_reflect",
                {
                    "objective": "提炼旧客户编号约束",
                    "knowledge_only": True,
                    "source_message_ids": [old_id],
                },
            )
        ],
        "已接纳",
    )
    registry = build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    definition = registry.get("knowledge_reflect")
    definition.deferred = False
    registry.replace(definition)
    result = run_task(
        "请处理选中的旧记录",
        tmp_path,
        data_root=tmp_path,
        session_id="old-session",
        llm_client=client,
        tool_registry=registry,
    )
    assert result.status == "done"
    work = manager.status()["works"][0]
    assert work["message_ids"] == [old_id]
    assert work["outcome"] is None and work["reasons"] == [
        "explicit_selection: 提炼旧客户编号约束"
    ]


def test_relevant_file_change_is_uncertain_until_actual_new_evidence(tmp_path):
    """空闲改文件不触发任务，相关使用才核验；参数：隔离根；返回：无。"""
    from runtime.schema_meta import ensure_current_schema

    ensure_current_schema(tmp_path)
    path = tmp_path / "port.txt"
    path.write_text("port=8000", encoding="utf-8")
    client = from_test_native_tool_then_final(
        [ToolCallPart("read", "file_read", {"path": str(path)})], "读取完成"
    )
    run = run_task("查看服务端口", tmp_path, data_root=tmp_path, llm_client=client)
    manager = KnowledgeMaintenance(tmp_path)
    work = manager.status()["works"][0]
    operation = ToolOperationStore(tmp_path).for_session(work["source_session_id"])[0]
    source = MemorySource(
        "tool_result",
        operation["operation_id"],
        session_id=work["source_session_id"],
        run_id=run.run_id,
    )
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory(
            "fact",
            "服务端口8000",
            ["端口"],
            details=MemoryDetails(
                scope=f"project:{work['workspace_id']}",
                sources=(source,),
                subject="服务",
                fact_key="端口",
            ),
        )
    path.write_text("port=9000", encoding="utf-8")
    assert manager.status()["works"] == [work]
    scopes = [f"session:{work['source_session_id']}", f"project:{work['workspace_id']}"]
    outcome = recall_memories_with_outcome(
        tmp_path, task_summary="服务端口", task_tags=[], scopes=scopes
    )
    assert "待核验" in outcome.selected[0].memory.content
    with closing(MemoryStore(tmp_path)) as store:
        assert store.load_memory(identity).content == "服务端口8000"
    run_task(
        "继续检查端口",
        tmp_path,
        data_root=tmp_path,
        session_id=work["source_session_id"],
        llm_client=from_test_stub("检查中"),
    )
    assert manager.status()["works"][0]["verification_targets"]


def test_unknown_selected_source_is_rejected(tmp_path):
    """选源不接受其他会话/编造的消息身份；参数：隔离根；返回：无。"""
    _, work = start_source(tmp_path)
    with pytest.raises(ValueError, match="outside the frozen branch"):
        from runtime.types import RunContext, Trigger
        from runtime.lease import from_trigger

        context = RunContext(
            session_id=work["source_session_id"],
            trigger=Trigger.USER,
            payload={},
            capability_lease=from_trigger(
                "user",
                task_id="source",
                capabilities={"background_run": {"enabled": True}},
            ),
        )
        KnowledgeMaintenance(tmp_path).admit_sources(
            context,
            client=from_test_stub("unused"),
            view=SessionMessageStore(tmp_path).materialize(work["source_session_id"]),
            message_ids={"foreign-message"},
            reason="explicit",
        )
