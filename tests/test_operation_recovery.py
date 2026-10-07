"""验证现实效果与必要提交之间的中断窗口。

作者：xxx
时间：2026-09-14 12:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Thread

import pytest

from llm.messages import ToolCallPart, ToolResultMessage
from llm.parser import parse_tool_call_parts
from llm.types import LLMPlan
from runtime.agent_loop import AgentLoop, State
from runtime.tool_executor import ToolBatchExecutor
from runtime.extensions import RuntimeExtensions, ToolProposal
from runtime.session_messages import materialize_messages
from runtime.types import new_run_id
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk
from tools.native_actions import register_native_actions
from tests.test_tool_batch_execution import Plans, make_run


def registry_for(execute):
    """注册有现实效果且不可自动重试的后端；传参：执行函数；返回：注册表。"""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "effect",
            "执行动作",
            {},
            "agent",
            ToolRisk.SAFE,
            False,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.NO,
            executor=execute,
        )
    )
    return registry


def test_retry_claim_rechecks_effect_source_inside_storage_boundary(tmp_path):
    """授权后原操作效果变化时拒绝认领，已有其他认领仍可查询；传参：目录；返回：无。"""
    from runtime.tool_operations import ToolOperationStore, retry_source

    store = ToolOperationStore(tmp_path)
    identity = {
        "session_id": "session-source",
        "run_id": "run-source",
        "operation_id": "op-source",
    }
    row = {"call": {"tool_name": "effect", "args": {}}, "state": "not_started"}
    store.write(identity, row)
    source = retry_source(row)
    store.write(identity, {**row, "state": "unknown"})
    with pytest.raises(ValueError, match="changed after recovery authorization"):
        store.claim_retry(identity, "op-retry", expected_source=source)
    assert "retry_operation_id" not in store.load(identity)
    store.write(identity, row)
    assert store.claim_retry(identity, "op-first", expected_source=source) == "op-first"
    assert (
        store.claim_retry(identity, "op-second", expected_source=source) == "op-first"
    )


@pytest.mark.parametrize("window", ["before_result", "before_session"])
def test_recovery_does_not_replay_effects_between_commits(
    tmp_path, monkeypatch, window
):
    """原效果只发生一次，已有结果补回，缺失结果如实未知；传参：存储/故障窗口；返回：无。"""
    effects = []

    def execute(_args):
        """留下可核对的外部效果；传参：参数；返回：证据。"""
        effects.append("effect happened")
        return "durable result evidence"

    registry = registry_for(execute)
    plan = parse_tool_call_parts(
        (ToolCallPart("effect-call", "effect", {}),),
        allowed_tool_names={"effect"},
        registry=registry,
    )
    loop, context, _client = make_run(tmp_path, registry, [plan])

    def fail(*_args, **_kwargs):
        """模拟效果之后的必要提交故障；传参：提交参数；返回：不返回。"""
        raise OSError("required commit unavailable")

    if window == "before_session":
        monkeypatch.setattr(ToolBatchExecutor, "_persist_tool_exchange", fail)
    else:
        original = loop.operations.write

        def write(identity, payload):
            """只在结果提交时中断，派发身份已保存；传参：身份/事实；返回：无。"""
            if "result" in payload:
                fail()
            original(identity, payload)

        monkeypatch.setattr(loop.operations, "write", write)
    with pytest.raises(OSError, match="required commit"):
        list(loop.run_stream(context))
    # 1. 【运行恢复】【索引重建】丢掉派生索引后仍根据工具原件识别真实结果或未知效果
    (tmp_path / "index.sqlite").unlink()
    resumed_client = Plans([LLMPlan(final_output="基于现有证据继续")])
    resumed = AgentLoop(tmp_path, llm_client=resumed_client, tool_registry=registry)
    list(resumed.run_stream(replace(context, run_id=new_run_id(), segment_id="")))
    assert effects == ["effect happened"]
    expected = (
        "durable result evidence" if window == "before_session" else "execution=unknown"
    )
    assert expected in str(resumed_client.contexts[-1])
    results = [
        message
        for message in materialize_messages(tmp_path, context.session_id)
        if isinstance(message, ToolResultMessage)
    ]
    assert len(results) == 1


def test_stop_marks_undispatched_calls_and_persists_late_result(tmp_path, monkeypatch):
    """停止当前不可中断操作时后续调用不派发，迟到结果沿原身份保存；传参：临时目录；返回：无。"""
    started, release, late = Event(), Event(), Event()
    effects = []

    def execute(_args):
        """模拟不能强停的已启动效果；传参：参数；返回：真实晚到结果。"""
        effects.append("started")
        started.set()
        assert release.wait(5)
        return "late evidence"

    registry = registry_for(execute)
    plan = parse_tool_call_parts(
        tuple(ToolCallPart(f"stop-{index}", "effect", {}) for index in range(2)),
        allowed_tool_names={"effect"},
        registry=registry,
    )
    loop, context, _client = make_run(tmp_path, registry, [plan])
    original = ToolBatchExecutor._record_late_tool_result

    def record(executor, context, call, result):
        """等待真实持久提交完成；传参：原归属/结果；返回：无。"""
        original(executor, context, call, result)
        late.set()

    monkeypatch.setattr(ToolBatchExecutor, "_record_late_tool_result", record)
    worker = Thread(target=lambda: list(loop.run_stream(context)), daemon=True)
    worker.start()
    assert started.wait(5)
    loop.cancellation.cancel()
    worker.join(5)
    assert not worker.is_alive()
    assert loop.state == State.PAUSED
    records = {
        row["call"]["call_id"]: row
        for row in loop.operations.for_session(context.session_id)
    }
    assert {key: row["state"] for key, row in records.items()} == {
        "stop-0": "unknown",
        "stop-1": "not_started",
    }
    original_focus = context.focus_task_id
    context.focus_task_id = "another-focus-after-stop"
    release.set()
    assert late.wait(5)
    records = {
        row["call"]["call_id"]: row
        for row in loop.operations.for_session(context.session_id)
    }
    assert records["stop-0"]["state"] == "late_completed"
    assert records["stop-0"]["result"]["output"] == "late evidence"
    late_facts = [
        row
        for row in loop.run_facts.read_run(context.run_id)
        if row.get("event") == "tool:late_response"
    ]
    assert late_facts[-1]["focus_task_id"] == original_focus
    assert effects == ["started"]


def test_late_result_keeps_hooked_execution_identity_and_cannot_retry_as_readonly(
    tmp_path, monkeypatch
):
    """迟到写入结果仍使用实际动作身份，不能按原只读声明重试；传参：目录、替换器；返回：无。"""
    entered, release, late = Event(), Event(), Event()
    effects, incorrect_reads = [], []

    def execute(_args):
        """写入已发生，后端暂未返回；传参：参数；返回：迟到结果。"""
        effects.append("write happened")
        entered.set()
        assert release.wait(5)
        return "实际写入结果"

    registry = registry_for(execute)
    registry.register(
        ToolDefinition(
            "read_probe",
            "只读查询",
            {},
            "agent",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=lambda _args: incorrect_reads.append(True) or "read",
        )
    )
    register_native_actions(registry)
    plan = parse_tool_call_parts(
        (ToolCallPart("changed-call", "read_probe", {}),),
        allowed_tool_names={"read_probe"},
        registry=registry,
    )
    loop, context, _client = make_run(tmp_path, registry, [plan])
    loop.extensions = RuntimeExtensions(
        before_tool=(
            lambda proposal, _token: ToolProposal("effect", {}, proposal.operation_id),
        )
    )
    original = ToolBatchExecutor._record_late_tool_result

    def record(executor, context, call, result):
        """确认迟到结果已完成持久提交；传参：归属、动作、结果；返回：无。"""
        original(executor, context, call, result)
        late.set()

    monkeypatch.setattr(ToolBatchExecutor, "_record_late_tool_result", record)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(loop.run, context)
        try:
            assert entered.wait(5)
            loop.cancellation.cancel()
            assert future.result(timeout=5) is State.PAUSED
        finally:
            release.set()
    assert late.wait(5)
    row = loop.operations.for_session(context.session_id)[0]
    assert row["call"]["execution_request"]["tool"] == "effect"
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "resume-changed",
                "resume_operation",
                {"operation_id": row["operation_id"], "action": "retry"},
            ),
        ],
        "保留原效果，采用其他办法",
    )
    resumed = AgentLoop(tmp_path, llm_client=client, tool_registry=registry)
    assert (
        resumed.run(replace(context, run_id=new_run_id(), segment_id="")) is State.DONE
    )
    results = [
        message
        for message in materialize_messages(tmp_path, context.session_id)
        if isinstance(message, ToolResultMessage)
    ]
    assert results[-1].status == "error"
    assert effects == ["write happened"]
    assert incorrect_reads == []
