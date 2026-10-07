from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final, from_test_stub

from pathlib import Path

import pytest

from llm.messages import ToolCallPart
from runtime.agent_loop import AgentLoop, State
from runtime.checkpoint import list_checkpoints
from runtime.lease import Lease, from_trigger
from runtime.run_facts import RunFactStore
from runtime.types import RunContext, Trigger
from tools.builtin_tools import build_tool_registry
from triggers.user import make_run_context as make_user_context


def _ctx(trigger: Trigger = Trigger.USER) -> RunContext:
    return RunContext(
        task_id="2026-05-04-01",
        trigger=trigger,
        payload={"tool_call": {"tool_name": "noop", "args": {}, "call_id": "c1"}},
        capability_lease=Lease(),
    )


def tool_loop(tmp_path: Path) -> tuple[AgentLoop, RunContext]:
    """经模型适配器读取真实文件；传参：临时目录；返回：执行器和已保存输入的运行。"""
    (tmp_path / "evidence.txt").write_text("actual file evidence", encoding="utf-8")
    data_root = tmp_path / "data"
    context = make_user_context("读取文件", data_root=data_root)
    context.capability_lease = from_trigger(
        "user",
        task_id=context.storage_task_id,
        capabilities={
            "fs": {
                "project_root": str(tmp_path),
                "read": [str(tmp_path)],
                "write": [str(data_root)],
            }
        },
    )
    client = from_test_native_tool_then_final(
        [
            ToolCallPart(
                "read-evidence", "file_read", {"path": str(tmp_path / "evidence.txt")}
            ),
        ],
        "已根据文件回答",
    )
    return AgentLoop(
        data_root,
        llm_client=client,
        tool_registry=build_tool_registry(repo_root=tmp_path, data_root=data_root),
    ), context


def test_missing_model_cannot_report_success_or_execute_payload_tool(
    tmp_path: Path,
) -> None:
    """缺少模型时明确失败且无执行事实；传参：临时根；返回：无。"""
    context = _ctx()
    context.payload["message"] = "执行这个动作"
    with pytest.raises(RuntimeError, match="llm_client is required"):
        AgentLoop(tmp_path).run(context)
    assert RunFactStore(tmp_path).read_run(context.run_id) == []


def test_agent_loop_records_terminal_lifecycle_and_tool_facts(tmp_path: Path) -> None:
    loop, context = tool_loop(tmp_path)
    final_state = loop.run(context)

    assert final_state is State.DONE
    facts = RunFactStore(loop.data_root).read_run(context.run_id)
    assert facts[0]["event"] == "run:start"
    assert facts[0]["trigger"] == "user"
    lifecycles = [fact for fact in facts if fact.get("event") == "run:lifecycle"]
    assert lifecycles[-1]["lifecycle"] == "done"
    assert lifecycles[-1]["reason"] == "final_output"
    assert not any(fact.get("event") == "state:transition" for fact in facts)
    events = [fact.get("event") for fact in facts]
    assert "tool:request" in events
    assert "tool:response" in events
    assert not (tmp_path / "tasks" / "2026-05-04-01" / "trajectory.jsonl").exists()
    assert not (tmp_path / "tasks" / "2026-05-04-01" / "checkpoints").exists()


def test_agent_loop_rejects_illegal_transition(tmp_path: Path) -> None:
    loop = AgentLoop(tmp_path)
    with pytest.raises(RuntimeError, match="transition is retired"):
        loop.transition(State.FAILED, _ctx())


def test_agent_loop_writes_pre_and_post_tool_checkpoints(tmp_path: Path) -> None:
    loop, context = tool_loop(tmp_path)
    loop.run(context)
    checkpoints = list_checkpoints(context.storage_task_id, data_root=loop.data_root)

    reasons = [item.reason for item in checkpoints]
    assert "pre_tool" in reasons
    assert "post_tool" in reasons


def test_agent_loop_accepts_four_phase1_triggers(tmp_path: Path) -> None:
    for trigger in (Trigger.USER, Trigger.CRON, Trigger.RESUME, Trigger.DELEGATE):
        context = make_user_context("按当前消息回答", data_root=tmp_path)
        context.trigger = trigger
        context.capability_lease = from_trigger(
            trigger.value, task_id=context.storage_task_id
        )
        assert (
            AgentLoop(tmp_path, llm_client=from_test_stub("真实模型边界的答复")).run(
                context
            )
            is State.DONE
        )


def test_agent_loop_records_run_identity_for_plain_session_message(
    tmp_path: Path,
) -> None:
    context = make_user_context("plain chat", data_root=tmp_path, formal_task=False)

    assert context.task_id is None
    assert (
        AgentLoop(tmp_path, llm_client=from_test_stub("会话答复")).run(context)
        is State.DONE
    )

    facts = RunFactStore(tmp_path).read_run(context.run_id)
    assert facts[0]["session_id"] == context.session_id
    assert facts[0]["run_id"] == context.run_id
    assert facts[0]["task_id"] is None
    assert facts[0]["compatibility_task_id"] == context.compatibility_task_id
    assert not (
        tmp_path
        / "tasks"
        / "_inbox"
        / str(context.compatibility_task_id)
        / "trajectory.jsonl"
    ).exists()
