"""【Agent运行】【停止归因】核对供应商结束原因与真实生命周期。

作者：xxx
时间：2026-10-01 20:40:00
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm.messages import StopReason, ToolCallPart
from runtime.agent_loop import State
from runtime.run_facts import RunFactStore
from scripts.testing.llm import _ScriptedTurn
from tests.test_harness_feedback_requests import _run
from tools.tool_registry import Idempotent, ToolDefinition, ToolRisk


@pytest.mark.parametrize("with_tool", [False, True])
@pytest.mark.parametrize(
    ("reason", "category"),
    [
        (StopReason.MAX_OUTPUT_TOKENS, "output_limit_exceeded"),
        (StopReason.CONTENT_FILTER, "content_filter"),
        (StopReason.CANCELLED, "cancelled"),
        (StopReason.UNKNOWN, "invalid_provider_response"),
    ],
)
def test_incomplete_response_never_completes_or_dispatches(
    tmp_path: Path,
    reason: StopReason,
    category: str,
    with_tool: bool,
) -> None:
    """中断正文和已闭合工具块都不能冒充完成；传参：目录、原因及响应形态；返回：无。"""
    executions: list[dict[str, object]] = []
    definition = ToolDefinition(
        "probe",
        "记录执行",
        {},
        "agent",
        ToolRisk.SAFE,
        True,
        "logical_scope",
        "builtin",
        idempotent=Idempotent.YES,
        executor=executions.append,
    )
    turn = _ScriptedTurn(
        text="尚未完整返回的内容",
        calls=(ToolCallPart("partial", "probe", {}),) if with_tool else (),
        stop_reason=reason,
    )
    loop, context, adapter = _run(tmp_path, (turn,), definition=definition)
    assert loop.state is State.FAILED
    assert len(adapter.requests) == 1 and executions == []
    terminal = [
        row
        for row in RunFactStore(tmp_path).read_run(context.run_id)
        if row.get("event") == "run:lifecycle"
    ][-1]
    assert terminal["lifecycle"] == "failed" and terminal["reason"] == category


def test_model_reported_error_is_failed(tmp_path: Path) -> None:
    """模型明确报告错误时结束为失败；传参：目录；返回：无。"""
    loop, context, adapter = _run(
        tmp_path, (_ScriptedTurn(text='{"type":"error","message":"cannot comply"}'),)
    )
    assert loop.state is State.FAILED and len(adapter.requests) == 1
    terminal = [
        row
        for row in RunFactStore(tmp_path).read_run(context.run_id)
        if row.get("event") == "run:lifecycle"
    ][-1]
    assert terminal["reason"] == "model_reported_error"
    assert "cannot comply" in loop.last_output


def test_tool_stop_without_calls_is_repaired_instead_of_completed(
    tmp_path: Path,
) -> None:
    """工具停止原因缺少调用时先交回错误，不能直接采用旁白；传参：目录；返回：无。"""
    loop, _, adapter = _run(
        tmp_path,
        (
            _ScriptedTurn(text="我准备读取材料", stop_reason=StopReason.TOOL_CALL),
            _ScriptedTurn(text="没有更多材料需要读取，现有证据已足够"),
        ),
    )
    assert loop.state is State.DONE and len(adapter.requests) == 2
    assert loop.last_output == "没有更多材料需要读取，现有证据已足够"


@pytest.mark.parametrize("reason", [StopReason.END_TURN, StopReason.STOP_SEQUENCE])
def test_completed_response_keeps_normal_final(
    tmp_path: Path, reason: StopReason
) -> None:
    """有效完成原因继续采用真实回答；传参：目录和原因；返回：无。"""
    loop, _, adapter = _run(
        tmp_path, (_ScriptedTurn(text="已完成当前回答", stop_reason=reason),)
    )
    assert loop.state is State.DONE and len(adapter.requests) == 1
    assert loop.last_output == "已完成当前回答"
