from __future__ import annotations

from pathlib import Path
from typing import cast

from llm.messages import (
    AssistantMessage,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
)
from runtime.agent_loop import AgentLoop
from runtime.execution_context import TurnExecutionContext
from runtime.tool_operations import ToolOperation
from runtime.lease import Lease
from runtime.session_messages import materialize_messages
from runtime.types import RunContext, RunToolsRequest, RunToolsResult, Trigger
from runtime.watchdog import Watchdog
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry
from triggers.user import make_run_context

SESSION_ID = "session-cutover"


def _context(data_root: Path, *, task_id: str = "task-1") -> RunContext:
    """构造带固定 session 的最小运行上下文。

    参数：data_root 为 data 根目录；task_id 为归属任务
    返回：可直接交给持久化路径的 RunContext
    """
    del data_root
    return RunContext(
        trigger=Trigger.USER,
        payload={"message": "hi"},
        capability_lease=Lease(),
        session_id=SESSION_ID,
        run_id="run-1",
        task_id=task_id,
    )


def _core(data_root: Path, context: RunContext) -> TurnExecutionContext:
    """构造 _persist_tool_exchange 所需的最小轮上下文。

    参数：data_root 为 data 根目录；context 为本轮运行上下文
    返回：TurnExecutionContext
    """
    return TurnExecutionContext(
        task_dir=data_root,
        context=context,
        store=TaskStore(data_root),
        watchdog=cast(Watchdog, None),
        storage_task_id=context.storage_task_id,
        task="hi",
    )


def _conversation_path(data_root: Path, task_id: str) -> Path:
    """返回旧消息 owner 的文件路径。"""
    return data_root / "tasks" / task_id / "conversation.jsonl"


def _legacy_rows(data_root: Path) -> list[Path]:
    """列出旧 owner 实际落盘的文件；递归覆盖 tasks/_inbox 下的临时会话任务。"""
    return sorted((data_root / "tasks").rglob("conversation.jsonl"))


def test_user_turn_writes_only_session_entry(tmp_path: Path) -> None:
    """AC2：user turn 只产生一条 canonical Session Entry，不再写旧 conversation。"""
    context = make_run_context(
        "帮我看看", data_root=tmp_path, session_id=SESSION_ID, run_id="run-1"
    )
    messages = materialize_messages(tmp_path, SESSION_ID)
    assert len(messages) == 1
    assert isinstance(messages[0], UserMessage)
    assert _legacy_rows(tmp_path) == []
    assert context.session_id == SESSION_ID


def test_tool_exchange_writes_paired_entries_only(tmp_path: Path) -> None:
    """AC2：一次工具交换只产生 assistant tool_call + tool_result 两条 Entry。"""
    context = _context(tmp_path)
    loop = AgentLoop(tmp_path, llm_client=None, tool_registry=ToolRegistry())
    call = ToolOperation(
        request=RunToolsRequest(
            action="file_read",
            tool_name="file_read",
            arguments={"path": "a.py"},
            call_id="call-1",
        ),
        call_id="call-1",
        tool_name="file_read",
        args={"path": "a.py"},
    )
    result = RunToolsResult.ok(action="file_read", tool_name="file_read", content="ok")

    from runtime.tool_executor import ToolBatchExecutor

    executor = ToolBatchExecutor(
        tmp_path,
        registry=loop.tool_registry,
        authorizer=loop.authorizer,
        cancellation=loop.cancellation,
        extension_execution=loop.extension_execution,
        policy=loop.tool_policy,
        operations=loop.operations,
        facts=loop.run_facts,
        evidence=loop.run_evidence,
        states=loop.session_states,
        progress=loop.progress,
        history=loop.tool_history,
        client=None,
        collaboration=None,
    )
    executor._persist_tool_exchange(_core(tmp_path, context), call, result)

    messages = materialize_messages(tmp_path, SESSION_ID)
    assert len(messages) == 2
    assistant = messages[0]
    tool_result = messages[1]
    assert isinstance(assistant, AssistantMessage)
    assert isinstance(tool_result, ToolResultMessage)
    calls = [part for part in assistant.content if isinstance(part, ToolCallPart)]
    assert calls[0].call_id == tool_result.call_id == "call-1"
    assert tool_result.tool_name == "file_read"
    assert _legacy_rows(tmp_path) == []


def test_assistant_finish_writes_single_entry(tmp_path: Path) -> None:
    """AC2：成功收尾只提交一条 assistant 消息，且不写旧 conversation。"""
    context = _context(tmp_path)
    store = TaskStore(tmp_path)
    store.create_task("hi")
    loop = AgentLoop(tmp_path, llm_client=None, tool_registry=ToolRegistry())

    loop._finish_success(store, context, "最终回答")

    messages = materialize_messages(tmp_path, SESSION_ID)
    assert len(messages) == 1
    assert isinstance(messages[0], AssistantMessage)
    assert _legacy_rows(tmp_path) == []


def test_failed_finish_writes_single_entry(tmp_path: Path) -> None:
    """AC2：失败收尾同样只提交一条 assistant 消息。"""
    context = _context(tmp_path)
    store = TaskStore(tmp_path)
    store.create_task("hi")
    loop = AgentLoop(tmp_path, llm_client=None, tool_registry=ToolRegistry())

    loop._finish_failed(store, context, "出错了")

    assert len(materialize_messages(tmp_path, SESSION_ID)) == 1
    assert _legacy_rows(tmp_path) == []
