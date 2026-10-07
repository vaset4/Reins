from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from context.production_builder import ProductionContextBuilder
from llm.provider_result import reported
from llm.prompt_composer import _budget_evidence_rows
from llm.model_request import compose_model_request
from llm.types import ModelAttemptEvent, ModelUsage
from runtime.agent_loop import AgentLoop
from runtime.execution_context import TurnExecutionContext
from runtime.lease import from_trigger
from runtime.types import RunContext, Trigger
from runtime.watchdog import Watchdog
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry

# ③ 甲：运行预算油表回喂。以下测试对应 PRD 的 AC1-AC5，
# 验证 watchdog 运行态能被 builder 读出、被 prompt_composer 渲染成纯数字油表，
# 且带 per-segment 限定词、不夹带劝导语、token 未知时不伪造。

_BUDGET_KEY = "_runtime_budget_evidence"


def _builder(data_root: Path) -> ProductionContextBuilder:
    return ProductionContextBuilder(
        data_root,
        system_prompt_provider=lambda: "system prompt",
    )


def _context(payload: dict[str, object]) -> RunContext:
    merged: dict[str, object] = {"message": "继续"}
    merged.update(payload)
    return RunContext(
        task_id="budget-task",
        compatibility_task_id=None,
        trigger=Trigger.USER,
        payload=merged,
        capability_lease=from_trigger("user", task_id="budget-task"),
    )


def _evidence(**overrides: object) -> dict[str, object]:
    # 模拟 loop 在 model turn 前从 watchdog 快照出的运行态
    base: dict[str, object] = {
        "steps_used": 3,
        "steps_limit": 30,
        "tokens_used": 12_000,
        "tokens_limit": 200_000,
        "has_token_usage": True,
    }
    base.update(overrides)
    return base


def test_builder_reads_budget_evidence_from_payload(tmp_path: Path) -> None:
    # AC1：build 出的 model_context 含 budget_evidence，数值来自传入的真实运行态而非硬编码
    context = _context({_BUDGET_KEY: _evidence(steps_used=7, tokens_used=55_000)})
    with closing(TaskStore(tmp_path)) as store:
        task = store.create_task("预算证据")
    context = replace(context, task_id=task.task_id, focus_task_id=task.task_id)

    bundle = _builder(tmp_path).build(
        task="继续",
        context=context,
        toolset_policy={},
        tool_registry=ToolRegistry(),
    )

    evidence = bundle.model_context["budget_evidence"]
    assert isinstance(evidence, dict)
    assert evidence["steps_used"] == 7
    assert evidence["tokens_used"] == 55_000


def test_builder_omits_budget_evidence_when_payload_absent(tmp_path: Path) -> None:
    # 无快照时不注入 budget_evidence，不伪造空油表
    with closing(TaskStore(tmp_path)) as store:
        task = store.create_task("无预算快照")
    context = replace(_context({}), task_id=task.task_id, focus_task_id=task.task_id)
    bundle = _builder(tmp_path).build(
        task="继续",
        context=context,
        toolset_policy={},
        tool_registry=ToolRegistry(),
    )

    assert "budget_evidence" not in bundle.model_context


def test_budget_rows_render_step_numbers() -> None:
    # AC2：渲染成一行进 prompt，含 step 已用/上限数字
    rows = _budget_evidence_rows(_evidence(steps_used=3, steps_limit=30))

    assert len(rows) == 1
    assert "3/30" in rows[0]


def test_budget_rows_carry_per_segment_qualifier() -> None:
    # AC3：文案含 per-segment 限定词，不出现跨段累计数字
    rows = _budget_evidence_rows(_evidence())

    assert "this segment" in rows[0]


def test_budget_rows_have_no_advisory_language() -> None:
    # AC4：纯数字油表，不含 change strategy / 收敛 / stop 等劝导措辞
    text = _budget_evidence_rows(_evidence())[0].lower()

    for banned in ("change strategy", "收敛", "stop", "converge"):
        assert banned not in text


def test_budget_rows_omit_token_line_when_usage_unknown() -> None:
    # AC5：provider 未返回 token usage 时不伪造 0/200000
    text = _budget_evidence_rows(_evidence(tokens_used=0, has_token_usage=False))[0]

    assert "0/200000" not in text
    assert "tokens" not in text.lower()


def test_context_summary_includes_budget_gauge() -> None:
    # 预算作为实际请求观察回喂，不混入指令和持久用户消息
    request = compose_model_request(
        task="继续",
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context={"budget_evidence": _evidence(steps_used=5)},
        registry=ToolRegistry(),
        context_window=30000,
    ).request
    summary = "\n".join(part.text for part in request.observations)

    assert "5/30" in summary


def _core_with_watchdog(watchdog: Watchdog) -> TurnExecutionContext:
    # 构造调 _snapshot_runtime_budget 所需的最小轮上下文，只关心 watchdog 与 context.payload
    context = _context({})
    return TurnExecutionContext(
        task_dir=Path("."),
        context=context,
        store=cast(TaskStore, None),
        watchdog=watchdog,
        storage_task_id="budget-task",
        task="继续",
    )


def test_snapshot_writes_live_watchdog_state_into_payload(tmp_path: Path) -> None:
    # AC1 端到端：快照从真实 Watchdog 读活值写进 payload，证明数值非硬编码
    lease = from_trigger(
        "user", task_id="budget-task", max_steps=30, max_tokens=200_000
    )
    watchdog = Watchdog(lease=lease, steps_taken=8, tokens_used=64_000)
    loop = AgentLoop(tmp_path, llm_client=None, tool_registry=ToolRegistry())
    core = _core_with_watchdog(watchdog)

    loop._snapshot_runtime_budget(core)

    evidence = core.context.payload[_BUDGET_KEY]
    assert isinstance(evidence, dict)
    assert evidence["steps_used"] == 8
    assert evidence["steps_limit"] == 30
    assert evidence["tokens_used"] == 64_000
    assert evidence["has_token_usage"] is True


def test_snapshot_omits_token_usage_flag_before_any_provider_usage(
    tmp_path: Path,
) -> None:
    # R4：首轮 tokens_used==0 时快照标 has_token_usage=False，渲染方据此省略 token 行
    lease = from_trigger("user", task_id="budget-task")
    watchdog = Watchdog(lease=lease, steps_taken=0, tokens_used=0)
    loop = AgentLoop(tmp_path, llm_client=None, tool_registry=ToolRegistry())
    core = _core_with_watchdog(watchdog)

    loop._snapshot_runtime_budget(core)

    evidence = core.context.payload[_BUDGET_KEY]
    assert evidence["has_token_usage"] is False


def test_snapshot_keys_match_renderer_contract(tmp_path: Path) -> None:
    # 锁死写↔读契约：快照写的 key 必须正好是 _budget_evidence_rows 消费的 key，
    # 任一端改名（如 steps_used→steps）都会让渲染产不出 step 数字而被此测试逮住
    lease = from_trigger(
        "user", task_id="budget-task", max_steps=30, max_tokens=200_000
    )
    watchdog = Watchdog(lease=lease, steps_taken=4, tokens_used=20_000)
    loop = AgentLoop(tmp_path, llm_client=None, tool_registry=ToolRegistry())
    core = _core_with_watchdog(watchdog)

    loop._snapshot_runtime_budget(core)

    rows = _budget_evidence_rows(core.context.payload[_BUDGET_KEY])
    assert len(rows) == 1
    assert "4/30" in rows[0]
    assert "20000/200000" in rows[0]


@pytest.mark.parametrize(
    "usage, expected, incomplete",
    [
        (ModelUsage(total_tokens=reported(0)), "tokens 0/200000", 0),
        (ModelUsage(), "tokens unknown/200000", 1),
        (ModelUsage(input_tokens=reported(17)), "tokens >=17/200000", 1),
        (
            ModelUsage(input_tokens=reported(3), output_tokens=reported(2)),
            "tokens 5/200000",
            0,
        ),
    ],
)
def test_attempt_usage_reaches_budget_view_without_turning_unknown_into_zero(
    tmp_path, usage, *, expected, incomplete
):
    """真实尝试计量经运行快照进入模型视图；传参：目录、用量、预期与未知数；返回：无。"""
    watchdog = Watchdog(from_trigger("user", task_id="budget-task", max_tokens=200_000))
    watchdog.reserve_model_attempt()
    watchdog.settle_model_attempt(
        ModelAttemptEvent(
            phase="finished",
            request_id="request-budget",
            attempt_id="attempt-budget",
            attempt_index=1,
            provider="fixture",
            model="fixture",
            started_at="2026-09-14T00:00:00Z",
            usage=usage,
        )
    )
    loop = AgentLoop(tmp_path, tool_registry=ToolRegistry())
    core = _core_with_watchdog(watchdog)
    loop._snapshot_runtime_budget(core)
    text = _budget_evidence_rows(core.context.payload[_BUDGET_KEY])[0]
    assert expected in text
    assert watchdog.unknown_usage_attempts == incomplete
    if incomplete:
        assert "usage_unknown_attempts 1" in text
