"""Checkpoint 状态解绑（07-08-checkpoint-state-decouple）验收测试。

覆盖两处解耦：
- C01：显示层 approval 判定不再从 checkpoint 存储的裸 state 推导，只认 run_facts
  的 lifecycle（+ terminal/approval:required 事件）。
- B02：缺少模型的运行以及已退休审批字段均明确失败，不写出成功事实。

作者 LKX
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.repl.status import classify_run_status
from runtime.agent_loop import AgentLoop
from runtime.types import Lease, RunContext, Trigger


def _lifecycle_fact(lifecycle: str) -> dict[str, object]:
    # 构造一条 run_facts 的 lifecycle 事件；lifecycle 值须落在 RUN_LIFECYCLES 白名单
    return {
        "event": "run:lifecycle",
        "lifecycle": lifecycle,
        "run_id": "run-1",
        "session_id": "sess-1",
    }


# --- C01：approval 判定的真实来源是 run_facts lifecycle，不是 checkpoint.state ---


def test_approval_still_detected_via_latest_state() -> None:
    # run_facts 含 waiting_approval lifecycle → 仍归 waiting_approval（真实来源生效）
    # 改前改后都应绿：证明删冗余分支后真实来源没被误伤
    summary = classify_run_status([_lifecycle_fact("waiting_approval")])
    assert summary.category == "waiting_approval"


def test_checkpoint_state_alone_no_longer_forces_approval() -> None:
    # 人为错位：只有 checkpoint_state=waiting_approval，run_facts 无 approval lifecycle
    # 删冗余分支前红（旧的 checkpoint_state 分支会 fire 误归 approval），删后绿
    # 证明 checkpoint.state 不再单独驱动 approval 判定
    summary = classify_run_status([], checkpoint_state="waiting_approval")
    assert summary.category != "waiting_approval"
    # checkpoint_state 非空 → 落到 192 行恢复点判断，归 continuable
    assert summary.category == "continuable"


def test_recovery_point_still_detected_via_checkpoint_state() -> None:
    # checkpoint_state=pre_tool + terminal=paused（无 approval facts）→ 归 continuable
    # 证明 status.py:192 的恢复点真值判断未被误删
    summary = classify_run_status(
        [], checkpoint_state="pre_tool", session_status="paused"
    )
    assert summary.category == "continuable"


# --- B02：旧审批字段和缺少执行依赖都不能产生假成功 ---


def test_delegate_without_model_cannot_report_done(tmp_path: Path) -> None:
    """未接通执行依赖的委派运行明确失败；传参：临时根；返回：无。"""
    context = RunContext(
        task_id="2026-07-08-delegate",
        trigger=Trigger.DELEGATE,
        payload={"message": "执行委派目标"},
        capability_lease=Lease(),
    )
    with pytest.raises(RuntimeError, match="llm_client is required"):
        AgentLoop(tmp_path).run(context)
    assert not (tmp_path / "sessions").exists()


@pytest.mark.parametrize("field", ("needs_approval", "approval_granted"))
def test_retired_legacy_approval_payload_fails_closed(
    tmp_path: Path,
    field: str,
) -> None:
    # 1. 构造携带已退役审批字段的无 LLM 调用
    context = RunContext(
        task_id="2026-07-08-retired-approval",
        trigger=Trigger.USER,
        payload={field: True},
        capability_lease=Lease(),
    )
    loop = AgentLoop(tmp_path)

    # 2. 旧字段必须显式失败，不能落入默认 DONE
    with pytest.raises(ValueError, match=field):
        loop.run(context)

    # 3. 输入校验失败前不得写入运行事实或任务目录
    assert loop.state is None
    assert not (tmp_path / "sessions").exists()
    assert not (tmp_path / "tasks").exists()
