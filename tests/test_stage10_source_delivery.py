"""【知识维护】【来源派发】坏锚点在模型前失败，完整直接材料才推进覆盖。

作者：xxx
时间：2026-10-02 11:48:00
"""

from contextlib import closing

from app.scheduled_run import create_scheduler
from llm.messages import ToolCallPart
from runtime.knowledge_maintenance import KnowledgeMaintenance
from runtime.session_message_store import SessionMessageStore
from schedules.store import ScheduleStore
from scripts.testing.llm import from_test_native_tool_then_final
from tests.test_automatic_knowledge import start_source
from tests.test_session_runtime import capture_requests


def test_bad_legacy_anchor_fails_before_model_and_does_not_restart(tmp_path):
    """旧inbound锚点缺交付时保留失败，重新调度不付费重试；参数：隔离根；返回：无。"""
    _, work = start_source(tmp_path)
    manager = KnowledgeMaintenance(tmp_path)
    first = (
        SessionMessageStore(tmp_path).materialize(work["source_session_id"]).entries[0]
    )
    broken = manager.update(work["work_id"], source_entry_id=first.entry_id)
    with closing(ScheduleStore(tmp_path)) as schedules:
        schedules.update(
            work["schedule_id"], source_entry_id=first.entry_id, knowledge_origin=broken
        )

    def no_model(_options):
        """预检失败不得创建模型；参数：公开配置；返回：如调用立即失败。"""
        raise AssertionError("bad source reached the model factory")

    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=no_model
        )
    ) as scheduler:
        assert scheduler.run_due_jobs()[0].status == "failed"
    row = manager.load(work["work_id"])
    assert row["state"] == "failed" and "frozen messages are missing" in row["error"]
    assert row["outcome"] is None and row["read_message_ids"] == []
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=no_model
        )
    ) as scheduler:
        assert scheduler.run_due_jobs() == []


def test_small_source_can_finish_from_actual_direct_delivery(tmp_path, monkeypatch):
    """实际请求已带完整原文时不强制一次工具回读；参数：隔离根/捕获器；返回：无。"""
    _, work = start_source(tmp_path)
    worker = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "finish",
                "knowledge_finish",
                {"outcome": "no_op", "reason": "已核验所提供原文，本次无需新增知识"},
            )
        ],
        "完成",
    )
    requests = capture_requests(worker, monkeypatch)
    with closing(
        create_scheduler(
            project_root=tmp_path, data_root=tmp_path, llm_factory=lambda _: worker
        )
    ) as scheduler:
        assert scheduler.run_due_jobs()[0].status == "succeeded"
    row = KnowledgeMaintenance(tmp_path).load(work["work_id"])
    assert set(row["read_message_ids"]) == set(work["message_ids"])
    assert row["source_delivery_request_id"]
    assert "本项目所有金额均保留两位小数" in str(requests[0])
    assert all(
        operation.tool_name != "knowledge_read"
        for message in requests[-1].messages
        for operation in message.content
        if isinstance(operation, ToolCallPart)
    )
