"""验证自动知识工作接纳、真实来源覆盖和受限后台交付。

作者：xxx
时间：2026-10-01 14:30:00
"""

from contextlib import closing
from datetime import datetime, timezone

from app.run_task import run_task
from app.scheduled_run import create_scheduler
from llm.messages import ToolCallPart, UserMessage
from memory.store import MemoryStore
from runtime.knowledge_maintenance import KnowledgeMaintenance
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_user_message
from runtime.workspaces import WorkspaceStore
from schedules.store import ScheduleStore
from scripts.testing.llm import from_test_native_tool_then_final, from_test_stub
from tests.test_session_runtime import capture_requests


def start_source(root):
    """生产用户入口无需显式提炼即可接纳新来源；参数：隔离根；返回：主运行和工作。"""
    result = run_task(
        "本项目所有金额均保留两位小数，请长期记住",
        root,
        data_root=root,
        llm_client=from_test_stub("已了解"),
    )
    works = KnowledgeMaintenance(root).status()["works"]
    assert result.status == "done" and len(works) == 1
    return result, works[0]


def test_new_source_is_automatically_processed_without_synthetic_input(
    tmp_path, monkeypatch
):
    """真实入口自动写入可溯源项目知识，不创造用户输入；参数：隔离根/捕获器；返回：无。"""
    parent, work = start_source(tmp_path)
    worker = from_test_native_tool_then_final(
        [
            ToolCallPart("read", "knowledge_read", {"action": "messages"}),
            ToolCallPart(
                "save",
                "memory_manage",
                {
                    "action": "create",
                    "type": "rule",
                    "kind": "fact",
                    "memory_scope": "project",
                    "subject": "金额",
                    "fact_key": "小数位",
                    "content": "本项目所有金额保留两位小数",
                    "source_mode": "origin_inputs",
                },
            ),
            ToolCallPart(
                "finish",
                "knowledge_finish",
                {"outcome": "completed", "reason": "已核对原用户要求并保存项目规则"},
            ),
        ],
        "维护完成",
    )
    requests = capture_requests(worker, monkeypatch)
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: worker
        )
    ) as scheduler:
        results = scheduler.run_due_jobs(now=datetime.now(timezone.utc))
    assert results[0].status == "succeeded", requests[-1]
    row = KnowledgeMaintenance(tmp_path).load(work["work_id"])
    assert row["state"] == "completed" and row["commits"]
    with closing(MemoryStore(tmp_path)) as store:
        memories = store.list_memories(state="active")
        assert len(memories) == 1
        assert memories[0].details.sources[0].session_id == work["source_session_id"]
        assert memories[0].details.scope == f"project:{work['workspace_id']}"
    assert not any(
        isinstance(entry.message, UserMessage)
        for entry in SessionMessageStore(tmp_path).read_entries(
            row["worker_session_id"]
        )
    )
    assert len(KnowledgeMaintenance(tmp_path).status()["works"]) == 1
    assert parent.run_id == work["source_run_id"]


def test_first_enable_preserves_unprocessed_history_and_merges_new_sources(tmp_path):
    """旧记录不虚报no-op，首次边界保持，排队期间新来源合并；参数：隔离根；返回：无。"""
    WorkspaceStore(tmp_path).bind_session("old-session", tmp_path)
    append_user_message(tmp_path, "old-session", "历史项目约定")
    manager = KnowledgeMaintenance(tmp_path)
    manager.configure(enabled=True)
    boundary = manager.status()["boundary"]
    assert manager.status()["works"] == []
    parent, work = start_source(tmp_path)
    run_task(
        "新增要求：日期用北京时间",
        tmp_path,
        data_root=tmp_path,
        session_id=work["source_session_id"],
        llm_client=from_test_stub("收到"),
    )
    rows = manager.status()["works"]
    assert len(rows) == 1 and set(work["message_ids"]) < set(rows[0]["message_ids"])
    manager.configure(enabled=False)
    manager.configure(enabled=True)
    assert manager.status()["boundary"] == boundary
    assert manager.status(session_id="old-session")["works"] == []
    assert parent.status == "done"


def test_plain_final_and_unread_noop_do_not_advance_coverage(tmp_path):
    """未核验的文本或no-op不记作完成；参数：隔离根；返回：无。"""
    run_task(
        "较长的原始业务材料，必须阅读原文核验。" * 600,
        tmp_path,
        data_root=tmp_path,
        llm_client=from_test_stub("已接纳材料"),
    )
    work = KnowledgeMaintenance(tmp_path).status()["works"][0]
    worker = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "finish", "knowledge_finish", {"outcome": "no_op", "reason": "未保存"}
            )
        ],
        "完成",
    )
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: worker
        )
    ) as scheduler:
        result = scheduler.run_due_jobs(now=datetime.now(timezone.utc))[0]
    row = KnowledgeMaintenance(tmp_path).load(work["work_id"])
    assert result.status == "failed" and row["state"] == "failed"
    assert row["outcome"] is None and row["read_message_ids"] == []


def test_verified_noop_survives_restart_without_second_model(tmp_path):
    """明确no-op保存覆盖，恢复重用事实不会再次调用模型；参数：隔离根；返回：无。"""
    _, work = start_source(tmp_path)
    worker = from_test_native_tool_then_final(
        [
            ToolCallPart("read", "knowledge_read", {"action": "messages"}),
            ToolCallPart(
                "finish",
                "knowledge_finish",
                {"outcome": "no_op", "reason": "内容无需形成新的长期知识"},
            ),
        ],
        "完成",
    )
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: worker
        )
    ) as scheduler:
        assert (
            scheduler.run_due_jobs(now=datetime.now(timezone.utc))[0].status
            == "succeeded"
        )
    assert KnowledgeMaintenance(tmp_path).load(work["work_id"])["state"] == "no_op"
    with closing(
        create_scheduler(
            project_root=tmp_path,
            data_root=tmp_path,
            llm_factory=lambda _: (_ for _ in ()).throw(
                AssertionError("unexpected model call")
            ),
        )
    ) as scheduler:
        assert scheduler.run_due_jobs(now=datetime.now(timezone.utc)) == []


def test_pause_and_cancel_do_not_claim_sources_processed(tmp_path):
    """暂停不撤销既有工作，取消记录失败范围，既有记忆仍可读；参数：隔离根；返回：无。"""
    _, work = start_source(tmp_path)
    manager = KnowledgeMaintenance(tmp_path)
    manager.configure(enabled=False)
    with closing(ScheduleStore(tmp_path)) as schedules:
        assert schedules.load_schedule(work["schedule_id"]).enabled
    manager.cancel(work["work_id"])
    assert manager.load(work["work_id"])["state"] == "cancelled"
    assert manager.load(work["work_id"])["outcome"] is None
    run_task(
        "暂停期间的新内容",
        tmp_path,
        data_root=tmp_path,
        llm_client=from_test_stub("收到"),
    )
    assert len(manager.status()["works"]) == 1
