from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

from llm.messages import TextPart, UserMessage
from llm.model_request import compose_model_request
from llm.types import LLMPlan
from runtime.agent_loop import AgentLoop, State
from runtime.watchdog import Watchdog
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_state import SessionStateStore
from runtime.types import RunContext, RunToolsResult, Trigger
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import ToolRegistry


def test_session_state_convergence_fields_roundtrip(tmp_path: Path) -> None:
    store = SessionStateStore(tmp_path)
    with store.database.transaction() as batch:
        batch.put(
            "session_state",
            "session-old",
            {"session_id": "session-old"},
            session_id="session-old",
        )

    loaded = store.load("session-old")

    assert loaded is not None
    assert loaded.consecutive_readonly_count == 0
    assert loaded.original_user_goal == ""
    assert loaded.hint_injection_count == 0

    loaded.consecutive_readonly_count = 5
    loaded.original_user_goal = "write html"
    loaded.hint_injection_count = 1
    store.save(loaded)
    reloaded = store.load("session-old")

    assert reloaded is not None
    assert reloaded.consecutive_readonly_count == 0
    assert reloaded.original_user_goal == "write html"
    assert reloaded.hint_injection_count == 0


@pytest.mark.parametrize("stream", [False, True])
def test_legacy_counts_do_not_trigger_reminders_or_pause(tmp_path, stream):
    """旧会话次数仍可读取，但运行中不恢复旧提醒；传参：目录与入口；返回：无。"""
    data = tmp_path / "data"
    store = TaskStore(data)
    task = store.create_task("继续核对材料")
    context = RunContext(
        task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": "继续"},
        capability_lease=from_trigger("user", task_id=task.task_id),
    )
    with store._db.transaction() as batch:
        batch.put(
            "session_state",
            context.session_id,
            {
                "session_id": context.session_id,
                "consecutive_readonly_count": 15,
                "hint_injection_count": 2,
            },
            session_id=context.session_id,
        )

    client = _FinalCaptureClient()
    loop = AgentLoop(data, llm_client=client)
    if stream:
        list(loop.run_stream(context))
    else:
        loop.run(context)
    assert loop.state is State.DONE
    assert all(not item.get("system_reminder") for item in client.contexts)
    assert not any(
        row.get("event") == "convergence_stop_hint_injected"
        for row in RunFactStore(data).read_run(context.run_id)
    )
    store.close()


def test_system_reminder_inserts_after_system_message() -> None:
    bundle = compose_model_request(
        task="next step",
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context={
            "conversation_history": (UserMessage("u1", (TextPart("first"),)),),
            "system_reminder": "[system_reminder]stop reading[/system_reminder]",
        },
        registry=ToolRegistry(),
        context_window=30000,
    )

    # 提示排在系统指令之后，两者同属 instructions 而不是伪装成 system 消息
    instructions = [part.text for part in bundle.request.instructions]
    assert len(instructions) == 2
    assert instructions[1].startswith("[system_reminder]")
    assert [message.kind for message in bundle.messages] == ["user", "user"]


def test_system_prompt_estimate_import_failure_is_explicit(monkeypatch) -> None:
    """正式提示词估算依赖缺失时明确抛出导入错误；参数：替换器；返回：无。"""
    from runtime.context_preparation import system_prompt_estimate

    real_import = builtins.__import__

    def fail_prompt_composer(
        name: str,
        globals_: object | None = None,
        locals_: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "llm.prompt_composer":
            raise ImportError("prompt composer unavailable")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_prompt_composer)

    with pytest.raises(ImportError, match="prompt composer unavailable"):
        system_prompt_estimate()


def test_artifact_creation_prompt_does_not_seed_list_tool(tmp_path: Path) -> None:
    registry = build_tool_registry(repo_root=tmp_path, data_root=tmp_path / "data")
    bundle = compose_model_request(
        task="写一个 html 页面介绍 RAG",
        stage="plan",
        protocol_mode="native_tool_calls",
        model_context={"artifact_output_dir": ".reins/workspace/task-123/outputs"},
        registry=registry,
        context_window=30000,
    )

    system_prompt = "\n".join(part.text for part in bundle.request.instructions)

    assert '"tool":"list"' not in system_prompt
    assert "write it directly with file_write" in system_prompt
    # 运行上下文已从用户消息改到 instructions，不再伪装成一轮对话
    assert "artifact_output_dir=.reins/workspace/task-123/outputs" in system_prompt


class _ContextCaptureClient:
    def __init__(self) -> None:
        self.contexts: list[dict[str, Any]] = []

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        del task
        self.contexts.append(dict(context or {}))
        return LLMPlan()

    def continue_from_run_tools(
        self, task: str, run_tools_result: object, context: object | None = None
    ) -> LLMPlan:
        del task, run_tools_result
        self.contexts.append(dict(context or {}))
        return LLMPlan()


class _FinalCaptureClient:
    def __init__(self) -> None:
        self.contexts: list[dict[str, Any]] = []

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        del task
        self.contexts.append(dict(context or {}))
        return LLMPlan(final_output="model decided")

    def continue_from_run_tools(
        self, task: str, run_tools_result: object, context: object | None = None
    ) -> LLMPlan:
        del task, run_tools_result
        self.contexts.append(dict(context or {}))
        return LLMPlan(final_output="model decided")


def test_call_model_forwards_system_reminder(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    store = TaskStore(data_root)
    record = store.create_task("goal")
    capture = _ContextCaptureClient()
    loop = AgentLoop(data_root, llm_client=capture)  # type: ignore[arg-type]
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": "goal"},
        capability_lease=from_trigger("user", task_id=record.task_id),
    )

    # _call_model 现在是生成器（边生成边发增量），不排空则函数体一行都不执行
    loop._prepare_turn_core(context).store.close()
    list(
        loop._call_model(
            "goal",
            context,
            None,
            system_reminder="[system_reminder]decide[/system_reminder]",
            watchdog=Watchdog(context.capability_lease),
        )
    )

    assert capture.contexts[0]["system_reminder"] == (
        "[system_reminder]decide[/system_reminder]"
    )
    assert capture.contexts[0]["artifact_output_dir"] == (
        f".reins/workspace/{record.task_id}/outputs"
    )


def test_invalid_input_non_missing_argument_becomes_replan(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    store = TaskStore(data_root)
    record = store.create_task("goal")
    loop = AgentLoop(data_root)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": "goal"},
        capability_lease=from_trigger("user", task_id=record.task_id),
    )
    result = RunToolsResult.error_result(
        action="read_artifact",
        tool_name="read_artifact",
        error="invalid_input: artifact_id is not a file path",
    )

    observation = loop._handle_recoverable_tool_error(context, result)

    assert observation.meta["replan_required"] is True
    assert observation.meta["retryable"] is True
    assert "Tool likely misused" in observation.output
