"""测量真实后台装配及本地请求准备；由父进程选择冻结源码后导入。

作者：xxx
时间：2026-09-28 23:00:00
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from time import perf_counter
from typing import Any
from unittest.mock import patch

from app.background.service import BackgroundService
from app.background.sessions import (
    BackgroundSessionRecords,
    SessionRecord,
    SessionServices,
)
from scripts.testing.llm import from_test_turns, ScriptedTurnOptions
from runtime.agent_loop import AgentLoop
from runtime.model_execution import ModelRequestRunner
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.types import RunContext, Trigger
from runtime.watchdog import Watchdog
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry
from scripts.benchmark_session_runtime import (
    BODY,
    FIXED_TIME,
    SESSION_ID,
    generate_fixture,
    measure_worker,
)

LOCAL_CONTEXT_WINDOW = 100_000_000
ACTIVE_STRIDE = 10
FACT_DENSITY_RUNS = 100
BASE_FACTS_PER_RUN = 3


class PreparedBoundaryReached(RuntimeError):
    """离线计时在真实可派发Bundle形成后结束，不伪造模型答复。"""


def forbidden_dispatch(*args: object, **kwargs: object) -> Any:
    """拒绝任何意外模型或网络派发；传参：边界参数；返回：无，明确失败。"""
    raise AssertionError("offline benchmark attempted model/network dispatch")


def generate_scenario(root: Path, scenario: str, size: int) -> None:
    """生成一次后冻结的合成输入；传参：新目录/场景/规模；返回：无。"""
    if scenario == "facts":
        generate_fixture(root, "runs", FACT_DENSITY_RUNS)
        facts = RunFactStore(root)
        for run in range(FACT_DENSITY_RUNS):
            run_id = f"run-{run:05d}"
            # 【会话基准】【事实密度】批量提交真实事实，成本只在生成阶段发生
            with facts._db.transaction():
                for index in range(size - BASE_FACTS_PER_RUN):
                    facts.append(
                        {
                            "session_id": SESSION_ID,
                            "run_id": run_id,
                            "ts": FIXED_TIME,
                            "event": "benchmark:observation",
                            "index": index,
                            "text": BODY,
                        }
                    )
            assert len(facts.read_run(run_id)) == size
        return
    if scenario == "prepare":
        generate_fixture(root, "messages", size, visible_inputs=True)
        with closing(TaskStore(root)) as store:
            task = store.create_task("基准请求", is_inbox=True)
        (root / "fixture.json").write_text(
            json.dumps({"task_id": task.task_id}), encoding="utf-8"
        )
        return
    root.mkdir(parents=True, exist_ok=False)
    facts, messages = RunFactStore(root), SessionMessageStore(root)
    for index in range(size):
        identity, run_id = f"session-bench-{index:05d}", f"run-bench-{index:05d}"
        active = index % ACTIVE_STRIDE == 0
        record = SessionRecord(
            identity,
            status="running" if active else "idle",
            updated_at=FIXED_TIME,
            intent={"run_id": run_id, "input_id": f"input-{index}"} if active else {},
        )
        messages.create_session(identity, created_at=FIXED_TIME)
        WorkspaceStore(root).bind_session(identity, root)
        BackgroundSessionRecords(root).save(record)
        if active:
            facts.append(
                {
                    "session_id": identity,
                    "run_id": run_id,
                    "ts": FIXED_TIME,
                    "event": "run:lifecycle",
                    "lifecycle": "done",
                }
            )


def background_rows(root: Path, size: int) -> list[dict[str, object]]:
    """测构造、重连、终态回执恢复；传参：可写夹具副本/规模；返回：独立步骤墙钟。"""
    services = SessionServices(root, root, forbidden_dispatch, ToolRegistry)
    started = perf_counter()
    service = BackgroundService(services)
    rows: list[dict[str, object]] = [
        {
            "step": "construct",
            "seconds": perf_counter() - started,
            "result_count": len(service._sessions),
        }
    ]
    try:
        records = service.records.list_recent()
        assert len(records) == size
        for phase in ("first_in_process", "repeat_in_process"):
            started = perf_counter()
            for record in records:
                assert (
                    service.attach(record.session_id)
                    is service._sessions[record.session_id]
                )
            rows.append(
                {
                    "step": "attach_all",
                    "phase": phase,
                    "seconds": perf_counter() - started,
                    "result_count": len(service._sessions),
                }
            )
        started = perf_counter()
        for session in service._sessions.values():
            session.recover()
        elapsed = perf_counter() - started
        recovered = sum(
            session.record.status == "done" for session in service._sessions.values()
        )
        assert recovered == (size + ACTIVE_STRIDE - 1) // ACTIVE_STRIDE
        assert all(not session.runtime.active for session in service._sessions.values())
        rows.append(
            {
                "step": "recover_completed_handoff",
                "seconds": elapsed,
                "result_count": recovered,
            }
        )
    finally:
        service.close()
    return rows


def fact_density_rows(root: Path, size: int) -> list[dict[str, object]]:
    """固定100个运行、改变每运行事实数；传参：输入根/事实密度；返回：真实待处理输入推导耗时。"""
    rows = measure_worker(root, "runs", FACT_DENSITY_RUNS)
    if any(row["error"] for row in rows):
        raise AssertionError(f"fact-density projection failed: {rows}")
    return [{"step": "unhandled_inputs", "facts_per_run": size, **row} for row in rows]


def preparation_rows(root: Path, size: int) -> list[dict[str, object]]:
    """测真实循环形成可派发请求前的本地路径；传参：夹具/消息数；返回：首次和重复墙钟。"""
    task_id = json.loads((root / "fixture.json").read_text(encoding="utf-8"))["task_id"]
    client = from_test_turns(
        ["forbidden"], options=ScriptedTurnOptions(context_window=LOCAL_CONTEXT_WINDOW)
    )
    registry = ToolRegistry()
    context = RunContext(
        session_id=SESSION_ID,
        compatibility_task_id=task_id,
        trigger=Trigger.USER,
        payload={"message": "基准请求", "input_message_id": "entry-00000"},
        capability_lease=from_trigger("user", task_id=task_id, capabilities={}),
    )
    loop = AgentLoop(root, llm_client=client, tool_registry=registry)
    loop.model_runner = ModelRequestRunner(
        client,
        registry=registry,
        cancellation=loop.cancellation,
        evidence=loop.model_evidence,
        facts=loop.run_facts,
        run_evidence=loop.run_evidence,
        states=loop.session_states,
        ledger=loop._ledger_writer(),
        extensions=loop.extension_execution,
    )
    watchdog = Watchdog(context.capability_lease)
    expected_messages = SessionMessageStore(root).materialize(SESSION_ID).messages
    assert len(expected_messages) == size
    captured: list[object] = []

    def capture(
        _runner: ModelRequestRunner, bundle: Any, *_args: Any, **_kwargs: Any
    ) -> Any:
        """在派发边界截取已准备Bundle；传参：真实Bundle；返回：无，以专用异常结束计时。"""
        prepared = bundle.model_context["prepared_request"]
        assert prepared.request.messages == expected_messages
        captured.append(prepared.token_estimate)
        raise PreparedBoundaryReached

    rows: list[dict[str, object]] = []
    try:
        with (
            patch.object(ModelRequestRunner, "invoke", capture),
            patch.object(
                client._adapter_registry.require("scripted_test"),
                "stream",
                forbidden_dispatch,
            ),
        ):
            for phase in ("first_in_process", "repeat_in_process"):
                started = perf_counter()
                try:
                    list(loop._call_model("基准请求", context, None, watchdog=watchdog))
                except PreparedBoundaryReached:
                    pass
                rows.append(
                    {
                        "step": "request_prepare",
                        "phase": phase,
                        "seconds": perf_counter() - started,
                        "token_estimate": captured[-1],
                    }
                )
        assert len(captured) == 2
    finally:
        registry.close()
    return rows
