from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final, from_test_stub

from pathlib import Path

import pytest

from llm.messages import ToolCallPart, ToolResultMessage
from runtime.agent_loop import AgentLoop, State
from runtime.checkpoint import list_checkpoints
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_messages import materialize_messages
from runtime.types import RunContext, Trigger
from tools.builtin_tools import build_tool_registry
from triggers.resume import make_run_context as make_resume_context
from triggers.user import make_run_context as make_user_context


def _paused_after_read(tmp_path: Path) -> tuple[AgentLoop, RunContext]:
    """读完真实文件后停下来问一句；传参：临时目录；返回：执行器和已保存输入的运行。"""
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
            ToolCallPart(
                "ask-evidence", "ask_user", {"question": "还要我继续核对吗？"}
            ),
        ],
        "已根据文件回答",
    )
    return AgentLoop(
        data_root,
        llm_client=client,
        tool_registry=build_tool_registry(repo_root=tmp_path, data_root=data_root),
    ), context


def test_dod_1_resume_continues_task_after_restart(tmp_path: Path) -> None:
    """真实读取后停下等用户回答，重启只补说一句；传参：临时目录；返回：无。"""
    first, first_context = _paused_after_read(tmp_path)
    data_root = first.data_root

    assert first.run(first_context) is State.PAUSED

    first_facts = RunFactStore(data_root).read_run(first_context.run_id)
    first_lifecycle = [
        fact for fact in first_facts if fact.get("event") == "run:lifecycle"
    ][-1]
    # 停在等用户输入上，恢复入口才读得懂这个断点该等什么
    assert first_lifecycle["lifecycle"] == "waiting_user"
    assert first_facts[0]["trigger"] == "user"

    resumed_context = make_resume_context(
        first_context.storage_task_id, data_root=data_root
    )
    resumed_context.payload["message"] = "继续依据已有文件内容回答"
    assert resumed_context.trigger is Trigger.RESUME
    assert resumed_context.parent_segment_id == first_context.segment_id

    assert (
        AgentLoop(
            data_root,
            llm_client=from_test_stub("恢复后的答复"),
            tool_registry=first.tool_registry,
        ).run(resumed_context)
        is State.DONE
    )

    facts = RunFactStore(data_root).read_task_facts(first_context.storage_task_id)
    resume_starts = [
        fact
        for fact in facts
        if fact.get("event") == "run:start" and fact.get("trigger") == "resume"
    ]
    assert resume_starts[-1]["parent_segment_id"] == first_context.segment_id
    resume_run_id = resume_starts[-1]["run_id"]
    resume_facts = RunFactStore(data_root).read_run(resume_run_id)
    closing = [fact for fact in resume_facts if fact.get("event") == "run:lifecycle"]
    assert closing[-1]["lifecycle"] == "done"

    checkpoints = list_checkpoints(first_context.storage_task_id, data_root=data_root)
    assert checkpoints[-1].pending_tool_call is None
    results = [
        message
        for message in materialize_messages(data_root, first_context.session_id)
        if isinstance(message, ToolResultMessage)
    ]
    # 中断前真正跑完的就是读文件那一下，问题本身不算证据
    assert [message.tool_name for message in results] == ["file_read", "ask_user"]
    assert "actual file evidence" in results[0].content[0].text


def test_dod_1_resume_missing_checkpoint_rejects(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        make_resume_context("2026-05-04-missing", data_root=tmp_path)
