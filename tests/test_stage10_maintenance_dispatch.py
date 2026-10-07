"""【知识维护】【派发协调】验证轮次汇集、持久pending与实际调用优先级。

作者：xxx
时间：2026-10-02 11:18:00
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import closing
from threading import Event

import pytest

from app.run_task import run_task
from app.scheduled_run import create_scheduler
from llm.messages import ToolCallPart
from runtime.cancellation import CancellationToken, ExecutionCancelled
from runtime.knowledge_maintenance import KnowledgeMaintenance
from runtime.model_dispatch import foreground_active, foreground_turn, maintenance_call
from runtime.schema_meta import ensure_current_schema
from runtime.session_message_store import SessionMessageStore
from scripts.testing.llm import from_test_native_tool_then_final, from_test_stub
from tests.test_automatic_knowledge import start_source

WAIT_SECONDS = 3
BLOCK_CHECK_SECONDS = 0.1


def test_intermediate_requests_do_not_create_maintenance(tmp_path, monkeypatch):
    """实际模型与工具交互过程中不派维护，完整轮次只生成一个来源；参数：隔离目录/替换器；返回：无。"""
    ensure_current_schema(tmp_path)
    target = tmp_path / "rule.txt"
    target.write_text("金额两位小数", encoding="utf-8")
    client = from_test_native_tool_then_final(
        [
            ToolCallPart("first-read", "file_read", {"path": str(target)}),
            ToolCallPart("second-read", "file_read", {"path": str(target)}),
        ],
        "已核对金额规则",
    )
    adapter = client._adapter_registry.require("scripted_test")
    original = adapter.stream
    observed = []

    def capture(request, **kwargs):
        """在真实派发边界查看已接纳维护；参数：请求/连接选项；返回：原响应流。"""
        observed.append(KnowledgeMaintenance(tmp_path).status()["works"])
        yield from original(request, **kwargs)

    monkeypatch.setattr(adapter, "stream", capture)
    assert (
        run_task("核对规则", tmp_path, data_root=tmp_path, llm_client=client).status
        == "done"
    )
    assert observed and all(not rows for rows in observed)
    works = KnowledgeMaintenance(tmp_path).status()["works"]
    assert len(works) == 1
    assert len(works[0]["message_ids"]) > 1


def test_inflight_source_keeps_one_merged_durable_pending_range(tmp_path):
    """已冻结发生保持原件，两轮增量只合并未派发范围且重启可读；参数：隔离目录；返回：无。"""
    _, first = start_source(tmp_path)
    with closing(
        create_scheduler(project_root=tmp_path, data_root=tmp_path)
    ) as scheduler:
        assert len(scheduler.accept_due()) == 1
        for text in ("日期使用北京时间", "单据编号保留前导零"):
            run_task(
                text,
                tmp_path,
                data_root=tmp_path,
                session_id=first["source_session_id"],
                llm_client=from_test_stub("收到"),
            )
            assert scheduler.accept_due() == []
    works = KnowledgeMaintenance(tmp_path).status()["works"]
    assert len(works) == 2
    assert (
        KnowledgeMaintenance(tmp_path).load(first["work_id"])["message_ids"]
        == first["message_ids"]
    )
    pending = next(row for row in works if row["work_id"] != first["work_id"])
    assert len(pending["message_ids"]) == 4
    SessionMessageStore(tmp_path).branch(
        first["source_session_id"], first["source_entry_id"]
    )
    run_task(
        "另一个分支采用月度编号",
        tmp_path,
        data_root=tmp_path,
        session_id=first["source_session_id"],
        llm_client=from_test_stub("收到分支约定"),
    )
    assert len(KnowledgeMaintenance(tmp_path).status()["works"]) == 3
    assert (
        KnowledgeMaintenance(tmp_path).load(pending["work_id"])["message_ids"]
        == pending["message_ids"]
    )


def test_foreground_delays_new_maintenance_call_without_cancelling_it(tmp_path):
    """前台存活期间后续维护等待，交互结束后同一次请求继续；参数：隔离目录；返回：无。"""
    cancellation = CancellationToken()
    started = Event()

    def dispatch():
        """请求真实维护派发窗口；参数：无；返回：获准派发证据。"""
        started.set()
        with maintenance_call(tmp_path, cancellation):
            return "sent"

    with ThreadPoolExecutor() as pool:
        with foreground_turn(tmp_path, "foreground-run"):
            assert foreground_active(tmp_path)
            future = pool.submit(dispatch)
            assert started.wait(WAIT_SECONDS)
            with pytest.raises(TimeoutError):
                future.result(timeout=BLOCK_CHECK_SECONDS)
        assert future.result(timeout=WAIT_SECONDS) == "sent"
    assert not cancellation.cancelled and not foreground_active(tmp_path)


def test_explicit_cancel_interrupts_waiting_dispatch(tmp_path):
    """等待资源也能明确取消，不产生假调用且不复活；参数：隔离目录；返回：无。"""
    cancellation = CancellationToken()
    with foreground_turn(tmp_path, "foreground-run"):
        cancellation.cancel("user_stop")
        with pytest.raises(ExecutionCancelled):
            with maintenance_call(tmp_path, cancellation):
                pytest.fail("cancelled work reached model dispatch")
