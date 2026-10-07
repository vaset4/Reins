"""Tests for `AgentLoop.run_stream` — the generator that drives the REPL.

These pin the event sequence the renderer depends on. If a future edit
silently re-orders or drops events, these tests fail and force a deliberate
spec/test update.

Each test runs the generator end-to-end via `list(loop.run_stream(ctx))` so we
can assert on the full event sequence; the side effects (conversation rows,
run facts, task status) keep working because `run_stream` reuses the
same store/registry/watchdog wiring as the synchronous path.
"""

from __future__ import annotations
from scripts.testing.llm import (
    from_test_sequence,
    from_test_streaming_turn,
    from_test_stub,
    from_test_text_json_stub,
)

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pytest

import approval
from tests.support.approval import install_approval
from llm.messages import ToolCallPart, ToolResultMessage
from llm.parser import parse_llm_response, parse_tool_call_parts
from llm.types import LLMPlan, ModelError, ModelObservation
from runtime.agent_loop import AgentLoop, State
from runtime.session_messages import append_user_message, materialize_messages
from runtime.checkpoint import load_latest_checkpoint
from runtime.ledger import LedgerStore
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.run_evidence import RunEvidenceStore
from runtime.stream_events import (
    AssistantReasoningDelta,
    AssistantTextDelta,
    AssistantTurnComplete,
    LeaseSnapshot,
    LifecycleChanged,
    SegmentPaused,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from runtime.types import RunContext, RunToolsResult, Trigger, new_run_id
from tasks.store import TaskStore
from tools.browser import playwright_adapter
from tools.builtin_tools import build_tool_registry
from tools.readonly_inspection import DEFAULT_READ_MAX_CHARS
from tools.tool_registry import (
    Idempotent,
    TARGET_SCOPE_LOGICAL,
    TOOL_SOURCE_BUILTIN,
    TOOLSET_AGENT,
    ToolDefinition,
    ToolRegistry,
    ToolRisk,
)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    tools_dir = project / "tools"
    tools_dir.mkdir(parents=True)
    (tools_dir / "alpha.py").write_text("print('hi')\n", encoding="utf-8")
    (project / ".reins" / "data").mkdir(parents=True, exist_ok=True)
    return project


def _lease(
    project: Path,
    task_id: str,
    *,
    max_steps: int = 30,
    max_tokens: int = 200000,
    now: datetime | None = None,
):
    data = project / ".reins" / "data"
    workspace = project / ".reins" / "workspace"
    return from_trigger(
        "user",
        task_id=task_id,
        capabilities={
            "fs": {
                "project_root": str(project),
                "read": [str(project), str(data), str(workspace)],
                "write": [str(data), str(workspace)],
                "deny_read": ["*.pem", "*.key", ".env"],
            },
            "terminal": {"enabled": True, "allow_commands": []},
            "browser": {
                "enabled": True,
                "profile": "default",
                "deny_domains": [],
                "headless": True,
            },
            "mouse_keyboard": {"enabled": False},
            "network": {"enabled": True, "deny_domains": []},
            "background_run": {"enabled": False},
            "mcp": {"enabled": True, "allow_servers": []},
        },
        max_steps=max_steps,
        max_tokens=max_tokens,
        now=now,
    )


def _build_loop(
    project: Path, llm_client, *, message: str = "test goal"
) -> tuple[AgentLoop, RunContext, TaskStore]:
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task(message)
    lease = _lease(project, record.task_id)
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    loop = AgentLoop(data_root, llm_client=llm_client, tool_registry=registry)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": message},
        capability_lease=lease,
        segment_id=f"user-{record.task_id}",
    )
    # 与生产入口一致：user turn 落唯一消息 owner，且必须在 RunContext 兜底出 session_id 之后
    append_user_message(
        data_root,
        context.session_id,
        message,
        run_id=context.run_id,
        task_id=record.task_id,
    )
    return loop, context, store


def _expose_loaded(registry: ToolRegistry, *names: str) -> None:
    """把按需工具直接放进本轮目录；传参：目录与工具名；返回：无。

    用例检验的是执行与事实记录，不是发现流程；按需工具要先经 capabilities 加载
    才会进入模型可见目录，这里直接按已加载发布，省掉一次无关的发现往返。
    """
    definitions = []
    for name in names:
        definition = registry.get(name)
        assert definition is not None, name
        definition.deferred = False
        definitions.append(definition)
    registry.publish(definitions, replace_names=names)


def _recording_safe_registry(effects: list[str]) -> ToolRegistry:
    """创建两个会记录真实 executor 调用的 safe 工具

    作者：LKX
    时间：2026-08-15 00:00:00
    传参：effects 为 executor 调用顺序收集器
    返回：只注册 safe_one/safe_two 的 ToolRegistry
    """
    registry = ToolRegistry()
    for tool_name in ("safe_one", "safe_two"):
        registry.register(
            ToolDefinition(
                name=tool_name,
                description=tool_name,
                parameters={},
                toolset=TOOLSET_AGENT,
                risk_level=ToolRisk.SAFE,
                readonly=True,
                target_scope_rule=TARGET_SCOPE_LOGICAL,
                source=TOOL_SOURCE_BUILTIN,
                idempotent=Idempotent.YES,
                executor=lambda _args, name=tool_name: effects.append(name) or name,
            )
        )
    return registry


def _native_budget_case(
    project: Path,
) -> tuple[AgentLoop, RunContext, list[str], Path, str]:
    """组装两个 native safe calls 与 max_steps=1 的生产运行场景
    传参：project 为隔离项目根
    返回：AgentLoop、RunContext、executor 记录、data_root 和 task_id
    """
    effects: list[str] = []
    registry = _recording_safe_registry(effects)
    parts = tuple(
        ToolCallPart(f"call-safe-{index}", name, {})
        for index, name in enumerate(("safe_one", "safe_two"), start=1)
    )
    plan = parse_tool_call_parts(
        parts,
        allowed_tool_names={"safe_one", "safe_two"},
        registry=registry,
    )
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task("native budget")
    # 预算取消后工具批次不再被拒，循环会继续要一次模型响应来收尾
    loop = AgentLoop(
        data_root,
        llm_client=_PlanClient([plan, LLMPlan(final_output="native done")]),
        tool_registry=registry,
    )
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": record.goal},
        capability_lease=_lease(project, record.task_id, max_steps=1),
        segment_id=f"user-{record.task_id}",
    )
    return loop, context, effects, data_root, record.task_id


def _event_types(events: Iterable) -> list[str]:
    return [type(e).__name__ for e in events]


def _terminal_fact(project: Path, context: RunContext) -> dict[str, object]:
    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    return [fact for fact in facts if fact.get("event") == "run:lifecycle"][-1]


def _terminal_lifecycle(events: list[object]) -> LifecycleChanged:
    lifecycles = [event for event in events if isinstance(event, LifecycleChanged)]
    return lifecycles[-1]


def _observation(total_tokens: int | None = None) -> ModelObservation:
    return ModelObservation(
        stage="plan",
        provider="test",
        model="test-model",
        started_at=datetime.now(timezone.utc).isoformat(),
        elapsed_ms=0,
        attempt_count=1,
        success=True,
        total_tokens=total_tokens,
    )


def _invalid_tool_plan() -> LLMPlan:
    return LLMPlan(
        final_output="MODEL_PROTOCOL_ERROR",
        model_error=ModelError.create(
            category="invalid_tool_arguments",
            summary="tool is not allowed",
            raw_summary="tool=list; execution policy check failed",
            stage="execute",
        ),
        observation=_observation(),
    )


# ---------------------------------------------------------------------------
# Sequence assertions


def test_final_only_run_emits_lease_then_turn_then_done(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(['{"type":"final","content":"hello world"}'])
    loop, context, _ = _build_loop(project, client)

    events = list(loop.run_stream(context))
    types = _event_types(events)

    # First event is always LeaseSnapshot.
    assert types[0] == "LeaseSnapshot"
    # Last event is the public done lifecycle boundary.
    assert types[-1] == "LifecycleChanged"
    assert events[-1].lifecycle == "done"
    # Exactly one assistant turn complete with the final content.
    completes = [e for e in events if isinstance(e, AssistantTurnComplete)]
    assert len(completes) == 1
    assert completes[0].content == "hello world"
    assert completes[0].stop_reason == "end_turn"
    assert loop.state is State.DONE


def test_run_stream_emits_reasoning_before_final_answer(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_text_json_stub(
        '<think>Need answer.</think>{"type":"final","content":"hello"}'
    )
    loop, context, _ = _build_loop(project, client)

    events = list(loop.run_stream(context))
    reasoning = [
        event for event in events if isinstance(event, AssistantReasoningDelta)
    ]
    completes = [event for event in events if isinstance(event, AssistantTurnComplete)]

    assert len(reasoning) == 1
    assert reasoning[0].text == "Need answer."
    assert completes[-1].content == "hello"
    assert events.index(reasoning[0]) < events.index(completes[-1])


def test_run_stream_streams_native_thinking_fragment_by_fragment(
    tmp_path: Path,
) -> None:
    """provider 原生思考分片要边生成边逐条上屏，不是轮末合成一整块发一条。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：tmp_path 为 pytest 临时目录
    返回：无

    改动前 client 用 assemble(list(...)) 把惰性事件流拍平，往上五帧又全是普通函数，
    所有分片只能等模型答完后合成一条发出来。三个分片对应三条增量就说明那道墙通了；
    条数变回 1 或多出第四条（轮末补发那块）都说明退回了旧行为。
    """
    project = _project(tmp_path)
    client = from_test_streaming_turn(
        ("先看清用户", "要什么", "再决定动手"),
        answer=("看完了",),
    )
    loop, context, _ = _build_loop(project, client)

    events = list(loop.run_stream(context))
    reasoning = [
        event for event in events if isinstance(event, AssistantReasoningDelta)
    ]
    completes = [event for event in events if isinstance(event, AssistantTurnComplete)]

    assert [item.text for item in reasoning] == ["先看清用户", "要什么", "再决定动手"]
    assert len(completes) == 1
    assert events.index(reasoning[-1]) < events.index(completes[0])


def test_run_stream_streams_native_answer_text_alongside_turn_complete(
    tmp_path: Path,
) -> None:
    """native 模式的答案正文同样逐片透出，且 AssistantTurnComplete 仍每轮恰好一次。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：tmp_path 为 pytest 临时目录
    返回：无

    三片正文必须对应三条增量、顺序与到达顺序一致；合成一条就说明正文那条通道
    还没真的流起来。正文增量与轮末整段内容都在，去重是渲染层的职责
    （见 test_repl_render），这里只钉住运行时把两者都交出来且没多发 turn complete。
    """
    project = _project(tmp_path)
    client = from_test_streaming_turn(
        ("想一下",), answer=("查完了，", "结论是", "两处都要改。")
    )
    loop, context, _ = _build_loop(project, client)

    events = list(loop.run_stream(context))
    text_deltas = [event for event in events if isinstance(event, AssistantTextDelta)]
    completes = [event for event in events if isinstance(event, AssistantTurnComplete)]

    assert [item.text for item in text_deltas] == ["查完了，", "结论是", "两处都要改。"]
    assert len(completes) == 1
    assert completes[0].content == "查完了，结论是两处都要改。"


def test_run_stream_text_json_mode_never_leaks_protocol_envelope_to_screen(
    tmp_path: Path,
) -> None:
    """text_json 模式下 text 块是协议信封，一个正文增量都不许上屏。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：tmp_path 为 pytest 临时目录
    返回：无

    该模式的 text 块装的是 {"type":"final",...}，逐字打出去就是把协议噪声糊给用户。
    """
    project = _project(tmp_path)
    client = from_test_text_json_stub('{"type":"final","content":"hello"}')
    loop, context, _ = _build_loop(project, client)

    events = list(loop.run_stream(context))

    assert not [event for event in events if isinstance(event, AssistantTextDelta)]
    completes = [event for event in events if isinstance(event, AssistantTurnComplete)]
    assert completes[-1].content == "hello"


def test_tool_call_run_emits_started_then_completed(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"final","content":"tools/ contains alpha.py"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, _ = _build_loop(project, client)

    events = list(loop.run_stream(context))
    types = _event_types(events)

    # Tool start must come before tool completion.
    assert "ToolExecutionStarted" in types
    assert "ToolExecutionCompleted" in types
    assert types.index("ToolExecutionStarted") < types.index("ToolExecutionCompleted")

    started = next(e for e in events if isinstance(e, ToolExecutionStarted))
    completed = next(e for e in events if isinstance(e, ToolExecutionCompleted))
    assert started.tool_name == "list"
    assert started.args == {"path": "tools"}
    assert started.risk in {"safe", "confirm", "deny"}
    assert started.call_id == completed.call_id
    assert completed.is_error is False
    assert "alpha.py" in completed.output

    # Two assistant turns: one tool_use boundary, one final.
    completes = [e for e in events if isinstance(e, AssistantTurnComplete)]
    assert len(completes) == 2
    assert completes[0].content is None
    assert completes[0].stop_reason == "tool_use"
    assert completes[1].content == "tools/ contains alpha.py"
    assert completes[1].stop_reason == "end_turn"
    assert loop.state is State.DONE


def test_approval_precedes_tool_start_and_executor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证真实 AgentLoop 顺序为 approval → start → executor

    作者：LKX
    时间：2026-08-16 00:00:00
    传参：monkeypatch 记录审批；tmp_path 为隔离项目根
    返回：无；断言两阶段 ToolRegistry 与 stream 事件的跨层顺序
    """
    project = _project(tmp_path)
    effects: list[str] = []
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="confirm_tool",
            description="confirm tool",
            parameters={},
            toolset=TOOLSET_AGENT,
            risk_level=ToolRisk.CONFIRM,
            readonly=False,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source=TOOL_SOURCE_BUILTIN,
            idempotent=Idempotent.CONDITIONAL,
            executor=lambda _args: effects.append("executor") or "ok",
        )
    )
    install_approval(
        monkeypatch,
        lambda _request: effects.append("approval") or approval.ApprovalDecision.ONCE,
    )
    plan = parse_llm_response(
        '{"type":"run_tools","tool":"confirm_tool","arguments":{}}',
        protocol_mode="text_json",
        allowed_tool_names={"confirm_tool"},
        registry=registry,
    )
    loop, context, _ = _build_loop(
        project, _PlanClient([plan, LLMPlan(final_output="done")])
    )
    loop.tool_registry = registry

    for event in loop.run_stream(context):
        if isinstance(event, ToolExecutionStarted):
            effects.append("start")

    assert effects == ["approval", "start", "executor"]


def test_repeated_successful_readonly_tools_stay_model_driven(
    tmp_path: Path,
) -> None:
    # 【Agent运行】【重复证据】第三次重复后模型收到观察，自行决定结束当前回答
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"final","content":"model stopped on its own"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, _ = _build_loop(project, client)

    events = list(loop.run_stream(context))

    started = [event for event in events if isinstance(event, ToolExecutionStarted)]
    pause_events = [event for event in events if isinstance(event, SegmentPaused)]
    lifecycles = [event for event in events if isinstance(event, LifecycleChanged)]
    assert [event.tool_name for event in started] == ["list", "list", "list", "list"]
    assert not pause_events
    assert lifecycles[-1].lifecycle == "done"
    assert loop.state is State.DONE
    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    assert any(row.get("event") == "progress:no_progress" for row in facts)
    assert not any(row.get("event") == "progress:paused" for row in facts)
    assert "model stopped on its own" in loop.last_output


def test_run_stream_continues_after_single_html_artifact_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    install_approval(
        monkeypatch,
        lambda _req: approval.ApprovalDecision.ONCE,
    )
    client = from_test_sequence(
        [
            json.dumps(
                {
                    "type": "run_tools",
                    "tool": "file_write",
                    "arguments": {
                        "path": "rag-intro.html",
                        "content": "<!doctype html><title>RAG</title>",
                    },
                }
            ),
            '{"type":"run_tools","tool":"file_read","arguments":{"path":"rag-intro.html"}}',
            '{"type":"final","content":"rag-intro.html 已写入并读取确认"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, _ = _build_loop(
        project, client, message="写一个html前端页面简单介绍下RAG技术"
    )

    events = list(loop.run_stream(context))

    started = [e for e in events if isinstance(e, ToolExecutionStarted)]
    completes = [e for e in events if isinstance(e, AssistantTurnComplete)]
    assert [event.tool_name for event in started] == ["file_write", "file_read"]
    assert (
        (project / "rag-intro.html")
        .read_text(encoding="utf-8")
        .startswith("<!doctype html>")
    )
    assert completes[-1].content == "rag-intro.html 已写入并读取确认"
    assert completes[-1].stop_reason == "end_turn"
    assert _terminal_lifecycle(events).lifecycle == "done"
    assert loop.state is State.DONE


def test_sync_run_continues_after_single_html_artifact_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    install_approval(
        monkeypatch,
        lambda _req: approval.ApprovalDecision.ONCE,
    )
    client = from_test_sequence(
        [
            json.dumps(
                {
                    "type": "run_tools",
                    "tool": "file_write",
                    "arguments": {
                        "path": "rag-intro.html",
                        "content": "<!doctype html><title>RAG</title>",
                    },
                }
            ),
            '{"type":"run_tools","tool":"file_read","arguments":{"path":"rag-intro.html"}}',
            '{"type":"final","content":"rag-intro.html 已写入并读取确认"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, _ = _build_loop(
        project, client, message="写一个html前端页面简单介绍下RAG技术"
    )

    state = loop.run(context)

    assert state is State.DONE
    assert [entry["tool_name"] for entry in loop.tool_history] == [
        "file_write",
        "file_read",
    ]
    assert loop.last_output == "rag-intro.html 已写入并读取确认"


def test_run_stream_ask_user_pauses_segment(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"ask_user","arguments":{"question":"Which page should I update?"}}',
            '{"type":"final","content":"should not be reached"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, store = _build_loop(project, client)

    events = list(loop.run_stream(context))

    pause_events = [e for e in events if isinstance(e, SegmentPaused)]
    completes = [e for e in events if isinstance(e, AssistantTurnComplete)]
    assert len(pause_events) == 1
    assert pause_events[0].reason == "awaiting user input"
    assert pause_events[0].resumable is True
    assert completes[-1].stop_reason == "tool_use"
    assert all(complete.content != "should not be reached" for complete in completes)
    assert _terminal_lifecycle(events).lifecycle == "waiting_user"
    assert loop.state is State.PAUSED
    assert "Which page should I update?" in store.read_summary(context.task_id or "")


def test_protocol_error_run_emits_failed_state(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(['{"type":"unsupported"}', '{"type":"unsupported"}'])
    loop, context, _ = _build_loop(project, client)

    events = list(loop.run_stream(context))
    types = _event_types(events)

    assert types[-1] == "LifecycleChanged"
    assert events[-1].lifecycle == "failed"
    completes = [e for e in events if isinstance(e, AssistantTurnComplete)]
    assert len(completes) >= 1
    assert completes[-1].content is not None
    assert completes[-1].stop_reason in {
        "protocol_error",
        "model_error:invalid_model_protocol",
    }


def test_segment_step_limit_no_longer_blocks_dispatch(tmp_path: Path) -> None:
    """步数上限取消后工具照常派发，不再有预算暂停；传参：临时项目；返回：无。"""
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
        ],
        protocol_mode="text_json",
    )
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task("budget exhaustion")
    lease = _lease(project, record.task_id, max_steps=1)
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    loop = AgentLoop(data_root, llm_client=client, tool_registry=registry)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": record.goal},
        capability_lease=lease,
        segment_id=f"user-{record.task_id}",
    )

    events = list(loop.run_stream(context))
    types = _event_types(events)

    pause_events = [e for e in events if isinstance(e, SegmentPaused)]
    started = [e for e in events if isinstance(e, ToolExecutionStarted)]
    #  步数超限不再拦派发，脚本里的两次只读调用全部真实执行
    assert not pause_events
    assert len(started) == 2
    results = [
        message
        for message in materialize_messages(data_root, context.session_id)
        if isinstance(message, ToolResultMessage)
    ]
    assert len(results) == 2
    assert {
        json.loads(result.content[0].text)["meta"]["execution_state"]
        for result in results
    } == {"completed"}
    #  脚本序列用尽后按失败收尾，不再是预算暂停
    assert types[-1] == "LifecycleChanged"
    assert events[-1].lifecycle == "failed"
    assert loop.state is State.FAILED


def test_native_multi_tool_batch_runs_every_call_without_budget_pause(
    tmp_path: Path,
) -> None:
    """验证 native multi-tool 批次的每个调用都实际执行

    作者：LKX
    时间：2026-08-15 00:00:00
    传参：tmp_path 为隔离项目根
    返回：无；断言两项都有 start 且都产生副作用，不再出现预算拒绝
    """
    project = _project(tmp_path)
    loop, context, effects, data_root, _task_id = _native_budget_case(project)

    events = list(loop.run_stream(context))

    started = [event for event in events if isinstance(event, ToolExecutionStarted)]
    assert [(event.tool_name, event.call_id) for event in started] == [
        ("safe_one", "call-safe-1"),
        ("safe_two", "call-safe-2"),
    ]
    assert effects == ["safe_one", "safe_two"]
    facts = RunFactStore(data_root).read_run(context.run_id)
    denied = [
        row["tool"]
        for row in facts
        if row.get("event") == "tool:response"
        and isinstance(row.get("tool"), dict)
        and row["tool"].get("status") == "denied"
    ]
    assert not denied
    assert _terminal_lifecycle(events).lifecycle == "done"


def test_resume_pending_survives_without_budget_pause(
    tmp_path: Path,
) -> None:
    """验证 token 超限不再暂停，pending 待办仍写进检查点

    作者：LKX
    时间：2026-08-15 00:00:00
    传参：tmp_path 为隔离项目根
    返回：无；断言运行正常收尾且 checkpoint 保留同一工具调用
    """
    project = _project(tmp_path)
    client = _PlanClient(
        [
            LLMPlan(final_output="still waiting", observation=_observation(11)),
        ]
    )
    loop, context, _ = _build_loop(project, client, message="resume budget pause")
    context.trigger = Trigger.RESUME
    context.capability_lease = _lease(project, context.task_id or "", max_tokens=10)
    context.payload["resume_action"] = "ask"
    context.payload["pending_tool_call"] = {
        "tool_name": "file_write",
        "args": {"path": "pending.txt", "content": "pending"},
        "call_id": "resume-pending-budget",
    }

    events = list(loop.run_stream(context))

    assert _terminal_lifecycle(events).lifecycle == "done"
    checkpoint = load_latest_checkpoint(
        context.task_id or "", data_root=project / ".reins" / "data"
    )
    assert checkpoint is not None
    assert checkpoint.pending_tool_call == context.payload["pending_tool_call"]


def test_token_usage_past_limit_is_recorded_without_pausing(tmp_path: Path) -> None:
    """token 用量超上限只记录，模型正文照常交付；传参：临时项目；返回：无。"""
    project = _project(tmp_path)
    client = _PlanClient(
        [LLMPlan(final_output="too late", observation=_observation(total_tokens=11))]
    )
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task("token budget")
    lease = _lease(project, record.task_id, max_tokens=10)
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    loop = AgentLoop(data_root, llm_client=client, tool_registry=registry)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": record.goal},
        capability_lease=lease,
        segment_id=f"user-{record.task_id}",
    )

    events = list(loop.run_stream(context))

    completes = [event for event in events if isinstance(event, AssistantTurnComplete)]
    #  token 超限不再暂停，模型正文照常交付并正常收尾
    assert not [event for event in events if isinstance(event, SegmentPaused)]
    assert _terminal_lifecycle(events).lifecycle == "done"
    assert _terminal_fact(project, context)["reason"] == "final_output"
    assert loop.state is State.DONE
    assert [event.content for event in completes] == ["too late"]


def test_llm_failure_budget_pauses_at_agent_loop_layer(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = _PlanClient(
        [_invalid_tool_plan(), _invalid_tool_plan(), _invalid_tool_plan()]
    )
    loop, context, _store = _build_loop(project, client)

    events = list(loop.run_stream(context))

    pause = next(event for event in events if isinstance(event, SegmentPaused))
    terminal = _terminal_lifecycle(events)
    assert pause.reason == "segment llm failure budget hit"
    assert terminal.lifecycle == "paused"
    assert (
        _terminal_fact(project, context)["reason"] == "segment llm failure budget hit"
    )
    assert loop.state is State.PAUSED


def test_expired_user_lease_pauses_before_model_call(tmp_path: Path) -> None:
    project = _project(tmp_path)
    expired_at = datetime.now(timezone.utc) - timedelta(hours=2)
    client = _NoCallClient()
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task("expired lease")
    lease = _lease(project, record.task_id, now=expired_at)
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    loop = AgentLoop(data_root, llm_client=client, tool_registry=registry)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": record.goal},
        capability_lease=lease,
        segment_id=f"user-{record.task_id}",
    )

    events = list(loop.run_stream(context))

    pause = next(event for event in events if isinstance(event, SegmentPaused))
    terminal = _terminal_lifecycle(events)
    assert pause.reason == "lease expired"
    assert terminal.lifecycle == "paused"
    terminal_fact = _terminal_fact(project, context)
    assert terminal_fact["reason"] == "lease expired"
    assert loop.state is State.PAUSED


def test_resume_lease_sentinel_does_not_wall_clock_pause(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(['{"type":"final","content":"ok"}'])
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task("resume sentinel")
    lease = from_trigger("resume", task_id=record.task_id)
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    loop = AgentLoop(data_root, llm_client=client, tool_registry=registry)
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.RESUME,
        payload={"message": record.goal},
        capability_lease=lease,
        segment_id=f"resume-{record.task_id}",
    )

    events = list(loop.run_stream(context))

    assert not any(isinstance(event, SegmentPaused) for event in events)
    assert _terminal_lifecycle(events).lifecycle == "done"
    assert loop.state is State.DONE


def test_manual_pause_request_pauses_on_next_budget_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    client = _NoCallClient()
    original = AgentLoop._prepare_turn_core

    def prepare_with_pause(self: AgentLoop, context: RunContext):
        core = original(self, context)
        core.watchdog.request_pause()
        return core

    monkeypatch.setattr(AgentLoop, "_prepare_turn_core", prepare_with_pause)
    loop, context, _store = _build_loop(project, client)

    events = list(loop.run_stream(context))

    pause = next(event for event in events if isinstance(event, SegmentPaused))
    terminal = _terminal_lifecycle(events)
    assert pause.reason == "manual pause requested"
    assert terminal.lifecycle == "paused"
    terminal_fact = _terminal_fact(project, context)
    assert terminal_fact["reason"] == "manual pause requested"
    assert loop.state is State.PAUSED


def test_runtime_config_pause_request_pauses_on_next_budget_check(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    client = _NoCallClient()
    loop, context, _store = _build_loop(project, client)
    loop.runtime_config["pause_requested"] = lambda: True

    events = list(loop.run_stream(context))

    pause = next(event for event in events if isinstance(event, SegmentPaused))
    terminal = _terminal_lifecycle(events)
    assert pause.reason == "manual pause requested"
    assert terminal.lifecycle == "paused"
    assert loop.state is State.PAUSED


def test_final_only_run_emits_single_done_lifecycle(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(['{"type":"final","content":"ok"}'])
    loop, context, _ = _build_loop(project, client)

    events = list(loop.run_stream(context))

    lifecycles = [e for e in events if isinstance(e, LifecycleChanged)]
    assert len(lifecycles) == 1
    assert lifecycles[0].lifecycle == "done"


def test_lease_snapshot_summarizes_capabilities(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(['{"type":"final","content":"ok"}'])
    loop, context, _ = _build_loop(project, client)

    events = list(loop.run_stream(context))
    snapshots = [e for e in events if isinstance(e, LeaseSnapshot)]
    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.trigger == "user"
    assert snap.max_steps == 30
    # fs key has read/write paths but no enabled flag — surface as counts.
    assert isinstance(snap.capabilities_summary["fs"], dict)
    assert snap.capabilities_summary["fs"]["read_paths"] >= 1
    # terminal/browser/network/mcp have enabled flags — surface as bool.
    assert snap.capabilities_summary["terminal"] is True
    assert snap.capabilities_summary["mouse_keyboard"] is False


def test_cache_usage_fact_uses_model_call_id_not_missing_call_id(
    tmp_path: Path,
) -> None:
    class CacheUsageClient:
        def plan(self, task: str, context: object | None = None) -> LLMPlan:
            del task, context
            return LLMPlan(
                final_output="cached ok",
                observation=ModelObservation(
                    stage="plan",
                    provider="stub",
                    model="stub-model",
                    started_at="2026-06-16T00:00:00+00:00",
                    elapsed_ms=1,
                    attempt_count=1,
                    success=True,
                    cache_read_input_tokens=12,
                    cache_creation_input_tokens=3,
                ),
            )

    project = _project(tmp_path)
    loop, context, _ = _build_loop(project, CacheUsageClient())

    list(loop.run_stream(context))

    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    cache_rows = [row for row in facts if row.get("event") == "llm:cache_usage"]
    assert len(cache_rows) == 1
    assert "call_id" not in cache_rows[0]
    request = next(row for row in facts if row.get("event") == "llm:request")
    assert cache_rows[0]["model_call_id"] == request["request_id"]
    assert cache_rows[0]["model_call_id_source"] == "request_id"
    assert cache_rows[0]["cache_read_input_tokens"] == 12


def test_recovery_policy_runtime_config_controls_recoverable_budget(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    data_root = project / ".reins" / "data"
    store = TaskStore(data_root)
    record = store.create_task("budget")
    lease = _lease(project, record.task_id)
    registry = build_tool_registry(repo_root=project, data_root=data_root)
    loop = AgentLoop(
        data_root,
        llm_client=from_test_sequence(['{"type":"final","content":"ok"}']),
        tool_registry=registry,
        runtime_config={"recovery_policy": {"recoverable_error_repeats": 0}},
    )
    context = RunContext(
        task_id=record.task_id,
        trigger=Trigger.USER,
        payload={"message": "budget"},
        capability_lease=lease,
        segment_id=f"user-{record.task_id}",
    )
    result = RunToolsResult.error_result(
        action="noop",
        tool_name="noop",
        error="unknown: boom",
        meta={"error_type": "unknown", "retryable": True},
    )

    recovered = loop._apply_recovery_budget(context, result)

    assert recovered.meta["budget_exhausted"] is True
    assert recovered.meta["budget_remaining"] == 0
    assert recovered.meta["retryable"] is False


@pytest.mark.parametrize("preference", ["skip", "replay"])
def test_legacy_resume_preference_is_evidence_without_automatic_replay(
    tmp_path: Path, preference: str
) -> None:
    """旧恢复偏好送入模型，但不触发现实执行；传参：临时项目、偏好；返回：无。"""
    project = _project(tmp_path)
    client = _PlanClient([LLMPlan(final_output="resumed")])
    loop, context, _ = _build_loop(project, client, message="resume")
    context.trigger = Trigger.RESUME
    context.payload["resume_action"] = preference
    context.payload["pending_tool_call"] = {
        "tool_name": "list",
        "args": {"path": "tools"},
        "call_id": "resume-call-1",
    }

    events = list(loop.run_stream(context))

    assert not any(isinstance(event, ToolExecutionStarted) for event in events)
    evidence = client.contexts[0]["resume_choice_pending_evidence"]
    assert evidence["requested_resolution"] == preference
    assert evidence["call_id"] == "resume-call-1"
    assert _terminal_lifecycle(events).lifecycle == "done"


def test_status_query_keeps_unresolved_legacy_operation_evidence(
    tmp_path: Path,
) -> None:
    """状态查询可继续执行，旧操作证据仍保留；传参：临时项目；返回：无。"""
    project = _project(tmp_path)
    status_query = parse_llm_response(
        '{"type":"run_tools","tool":"operation_status","arguments":{}}',
        protocol_mode="text_json",
        allowed_tool_names={"operation_status"},
    )
    client = _PlanClient([status_query, LLMPlan(final_output="已核对，保留待处理动作")])
    loop, context, _ = _build_loop(project, client, message="resume context")
    context.trigger = Trigger.RESUME
    context.payload["resume_action"] = "ask"
    context.payload["pending_tool_call"] = {
        "tool_name": "file_write",
        "args": {"path": "pending.txt", "content": "pending"},
        "call_id": "resume-context-call",
    }

    events = list(loop.run_stream(context))

    assert _terminal_lifecycle(events).lifecycle == "done"
    assert len(client.contexts) >= 2
    assert "resume_choice_pending_evidence" in client.contexts[0]
    assert all(
        "resume_choice_pending_evidence" in model_context
        for model_context in client.contexts[1:]
    )
    assert [
        event.tool_name for event in events if isinstance(event, ToolExecutionStarted)
    ] == ["operation_status"]
    assert not (project / "pending.txt").exists()


def test_resume_retains_operation_goal_when_current_focus_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """中断后补回原目标结果，后续回复归当前目标；传参：隔离根、故障注入；返回：无。"""
    from runtime.session_message_store import SessionMessageStore

    project = _project(tmp_path)
    client = from_test_sequence(
        ['{"type":"run_tools","tool":"safe_one","arguments":{}}'],
        protocol_mode="text_json",
    )
    loop, context, store = _build_loop(project, client, message="原工具所属事项")
    effects: list[str] = []
    registry = _recording_safe_registry(effects)
    loop.tool_registry = registry

    def fail_commit(*_args, **_kwargs):
        """已保存操作结果，但会话回填中断；传参：提交参数；返回：抛出真实IO错误。"""
        raise OSError("session commit interrupted")

    from runtime.tool_executor import ToolBatchExecutor

    with monkeypatch.context() as interrupted:
        interrupted.setattr(ToolBatchExecutor, "_persist_tool_exchange", fail_commit)
        with pytest.raises(OSError, match="session commit interrupted"):
            loop.run(context)
    current = store.create_task("当前回复所属事项")
    resumed = replace(
        context,
        trigger=Trigger.RESUME,
        run_id=new_run_id(),
        segment_id="",
        focus_task_id=current.task_id,
    )
    resumed_loop = AgentLoop(
        loop.data_root, llm_client=from_test_stub("恢复完成"), tool_registry=registry
    )
    assert resumed_loop.run(resumed) is State.DONE
    assert effects == ["safe_one"]
    facts = RunFactStore(project / ".reins/data").read_run(context.run_id)
    tool_fact = next(row for row in facts if row["event"] == "tool:request")
    assert tool_fact["operation_task_id"] == context.task_id
    entries = (
        SessionMessageStore(project / ".reins/data")
        .materialize(context.session_id)
        .entries
    )
    tool = next(
        row for row in entries if row.message and row.message.kind == "tool_result"
    )
    assert tool.task_id == context.task_id
    assert tool.run_id == context.run_id
    assert entries[-1].task_id == current.task_id


def test_run_stream_writes_durable_artifacts_like_run(tmp_path: Path) -> None:
    """Streaming must not skip the run-fact/conversation/checkpoint side
    effects the synchronous path produces."""
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"list","arguments":{"path":"tools"}}',
            '{"type":"final","content":"summary"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, _ = _build_loop(project, client)
    list(loop.run_stream(context))

    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    events = {row.get("event") for row in facts if "event" in row}
    assert "context:built" in events
    assert "tool:request" in events
    assert "tool:response" in events
    response_index = next(
        index for index, row in enumerate(facts) if row.get("event") == "tool:response"
    )
    post_checkpoint_index = next(
        index
        for index, row in enumerate(facts)
        if row.get("event") == "checkpoint:saved"
        and isinstance(row.get("checkpoint"), dict)
        and row["checkpoint"].get("reason") == "post_tool"
    )
    assert response_index < post_checkpoint_index

    messages = materialize_messages(project / ".reins" / "data", context.session_id)
    kinds = [message.kind for message in messages]
    # 唯一消息 owner：user + assistant(tool_call) + tool_result，再加最终 assistant 回答
    assert kinds == ["user", "assistant", "tool_result", "assistant"]


def test_run_stream_tool_conversation_includes_file_read_meta(tmp_path: Path) -> None:
    """截断元数据要原样流进 tool_result 会话消息，模型靠它决定接着读哪一段。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：tmp_path 为 pytest 临时目录
    返回：无

    文件必须比读取窗口大才走得到截断分支，所以尺寸和断言都跟着窗口常量走，
    以后调窗口不用回来改数字。
    """
    project = _project(tmp_path)
    total = DEFAULT_READ_MAX_CHARS + 500
    (project / "big.txt").write_text("x" * total, encoding="utf-8")
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"file_read","arguments":{"path":"big.txt"}}',
            '{"type":"final","content":"read big file"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, _ = _build_loop(project, client)

    list(loop.run_stream(context))

    messages = materialize_messages(project / ".reins" / "data", context.session_id)
    tool_result = next(
        message for message in messages if isinstance(message, ToolResultMessage)
    )
    payload = json.loads("".join(part.text for part in tool_result.content))
    assert payload["tool_name"] == "file_read"
    assert payload["meta"]["truncated"] is True
    assert payload["meta"]["offset"] == 0
    assert payload["meta"]["next_offset"] == DEFAULT_READ_MAX_CHARS
    assert payload["meta"]["total_count"] == total
    assert payload["meta"]["resolved_path"].endswith("big.txt")


def test_production_file_read_pages_a_paper_in_at_most_two_round_trips(
    tmp_path: Path,
) -> None:
    """走一遍模型真实走过的分页循环，一篇论文体量的文件不许超过 2 次往返。

    作者：LKX
    时间：2026-09-05 00:00:00
    传参：tmp_path 为 pytest 临时目录
    返回：无

    2026-09-05 真机实测：4000 字的读取窗口把 45185 字的论文拆成 12 次 file_read，
    而 provider 按分钟限流（qq1244 是 10 次/分钟），任务在读完之前就被掐断，11 次调用
    累计烧掉 109267 prompt token 也没写出一个字的答案。这里钉住的不是"窗口多大"，
    而是"一篇论文要几次往返"——窗口被谁改回 4000 这条就会红。顺带钉住 offset /
    next_offset 这套契约能收敛，不会自己绕死循环。
    """
    project = _project(tmp_path)
    paper_chars = 45185
    (project / "paper.txt").write_text("x" * paper_chars, encoding="utf-8")
    registry = build_tool_registry(
        repo_root=project, data_root=project / ".reins" / "data"
    )
    lease = _lease(project, "task-read")

    reads = 0
    cursor = None
    collected = 0
    while reads < 12:
        result = registry.execute_tool(
            "file_read",
            {"path": "paper.txt", **({"cursor": cursor} if cursor else {})},
            lease,
        )
        assert isinstance(result, dict)
        meta = result["meta"]
        assert isinstance(meta, dict)
        reads += 1
        collected += int(meta["returned_count"])
        if not meta["truncated"]:
            break
        cursor = meta["next_cursor"]

    assert collected == paper_chars
    assert reads <= 2, f"一篇论文被拆成了 {reads} 次往返"


def test_run_stream_mcp_unconfigured_failure_records_facts_and_errors(
    tmp_path: Path,
) -> None:
    """MCP 配置读不到时连接发布不出来，失败要可解释、可回溯，不能悄悄消失

    传参：隔离目录；返回：无，核对来源状态、运行终态与事实/错误记录
    """
    from tools.mcp_client.registry import attach_mcp_registry

    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"mcp_echo_echo","arguments":{"text":"hello"}}',
        ],
        protocol_mode="text_json",
    )
    loop, context, _ = _build_loop(project, client)
    context.capability_lease.capabilities["mcp"] = {
        "enabled": True,
        "allow_servers": ["echo"],
        "config_path": str(tmp_path / "missing-mcp.yaml"),
    }
    attach_mcp_registry(context.capability_lease, loop.tool_registry)

    list(loop.run_stream(context))

    # 连接没发布出来，工具就不在本轮目录里；失败原因落在来源状态上供宿主和模型查阅
    assert loop.state is State.FAILED
    assert (
        "mcp_not_configured"
        in loop.tool_registry.source_status()["mcp"]["errors"]["echo"]
    )

    facts_text = json.dumps(RunFactStore(loop.data_root).read_run(context.run_id))
    errors_text = json.dumps(
        RunEvidenceStore(loop.data_root).list_records(
            session_id=context.session_id, run_id=context.run_id, kind="error"
        )
    )
    assert "mcp_echo_echo" in facts_text
    assert "mcp_echo_echo" in errors_text
    assert "not allowed in current request" in errors_text


def test_run_stream_browser_minimal_chain_records_screenshot_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"browser_navigate","arguments":{"url":"https://example.com"}}',
            '{"type":"run_tools","tool":"browser_extract","arguments":{}}',
            '{"type":"run_tools","tool":"browser_screenshot","arguments":{"full_page":"true"}}',
            '{"type":"final","content":"captured browser evidence"}',
        ],
        protocol_mode="text_json",
    )
    loop, context, _ = _build_loop(project, client, message="browser evidence")

    class _Session:
        def navigate(self, url: str) -> dict[str, object]:
            return {"url": url, "status": 200, "summary": "browser navigated"}

        def extract_text(self, _selector: str = "") -> str:
            return "Example Domain"

        def screenshot(self, path: Path, *, full_page: bool = False) -> dict[str, int]:
            assert full_page is True
            path.write_bytes(b"png")
            return {"width": 1280, "height": 800}

    monkeypatch.setattr(playwright_adapter, "get_session", lambda _lease: _Session())
    install_approval(
        monkeypatch,
        lambda _req: approval.ApprovalDecision.ONCE,
    )
    _expose_loaded(
        loop.tool_registry, "browser_navigate", "browser_extract", "browser_screenshot"
    )

    events = list(loop.run_stream(context))

    completed = [e for e in events if isinstance(e, ToolExecutionCompleted)]
    assert [event.tool_name for event in completed] == [
        "browser_navigate",
        "browser_extract",
        "browser_screenshot",
    ]
    assert [event.is_error for event in completed] == [False, False, False]
    screenshot_output = json.loads(completed[-1].output)
    artifact_id = screenshot_output["artifact_id"]
    assert artifact_id.startswith("art-")

    facts = RunFactStore(loop.data_root).read_run(context.run_id)
    tool_facts = [fact for fact in facts if fact.get("event") == "tool:response"]
    assert [fact["tool"]["name"] for fact in tool_facts] == [
        "browser_navigate",
        "browser_extract",
        "browser_screenshot",
    ]
    screenshot_fact = tool_facts[-1]["tool"]
    assert screenshot_fact["artifact_refs"] == [
        {"artifact_id": artifact_id, "artifact_type": "screenshot"}
    ]
    assert artifact_id in screenshot_fact["output_summary"]


def test_run_stream_rejects_missing_message(tmp_path: Path) -> None:
    project = _project(tmp_path)
    client = from_test_sequence(['{"type":"final","content":"ok"}'])
    loop, context, _ = _build_loop(project, client)
    context.payload = {}

    with pytest.raises(ValueError, match="payload\\['message'\\]"):
        list(loop.run_stream(context))


def test_run_stream_rejects_missing_llm_client(tmp_path: Path) -> None:
    project = _project(tmp_path)
    loop, context, _ = _build_loop(
        project, from_test_sequence(['{"type":"final","content":"ok"}'])
    )
    loop.llm_client = None

    with pytest.raises(RuntimeError, match="llm_client"):
        list(loop.run_stream(context))


def test_run_stream_context_read_failure_stops_before_model_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = _project(tmp_path)
    client = _NoCallClient()
    loop, context, _store = _build_loop(project, client)

    def fail_task_events(self: LedgerStore, task_id: str) -> object:
        del self, task_id
        raise OSError("summary store unavailable")

    monkeypatch.setattr(LedgerStore, "read_task_events", fail_task_events)

    events = list(loop.run_stream(context))
    completes = [event for event in events if isinstance(event, AssistantTurnComplete)]
    errors = [
        row["payload"]
        for row in RunEvidenceStore(loop.data_root).list_records(
            session_id=context.session_id, run_id=context.run_id, kind="error"
        )
    ]

    assert loop.state is State.FAILED
    assert _terminal_lifecycle(events).lifecycle == "failed"
    assert completes[-1].stop_reason == "context_error:context_summary_read_failed"
    assert errors[-1]["category"] == "context_summary_read_failed"
    assert errors[-1]["stage"] == "context"


class _NoCallClient:
    def plan(self, task: str, context: object | None = None):
        del task, context
        raise AssertionError("model should not be called")

    def continue_from_run_tools(
        self, task: str, run_tools_result: object, context: object | None = None
    ):
        del task, run_tools_result, context
        raise AssertionError("model should not be continued")


class _PlanClient:
    def __init__(self, plans: list[LLMPlan]) -> None:
        self._plans = list(plans)
        self._index = 0
        self.contexts: list[dict[str, object]] = []

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        del task
        self.contexts.append(dict(context) if isinstance(context, dict) else {})
        return self._next()

    def continue_from_run_tools(
        self, task: str, run_tools_result: object, context: object | None = None
    ) -> LLMPlan:
        del task, run_tools_result
        self.contexts.append(dict(context) if isinstance(context, dict) else {})
        return self._next()

    def _next(self) -> LLMPlan:
        if self._index >= len(self._plans):
            raise AssertionError("unexpected model call")
        plan = self._plans[self._index]
        self._index += 1
        return plan


def test_cache_usage_reported_zero_writes_fact(tmp_path: Path) -> None:
    """厂商上报 cache=0 应写入 fact，值为 0（不是 None）。"""

    class CacheZeroClient:
        def plan(self, task: str, context: object | None = None) -> LLMPlan:
            del task, context
            return LLMPlan(
                final_output="reported zero",
                observation=ModelObservation(
                    stage="plan",
                    provider="stub",
                    model="stub-model",
                    started_at="2026-09-01T00:00:00+00:00",
                    elapsed_ms=1,
                    attempt_count=1,
                    success=True,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
            )

    project = _project(tmp_path)
    loop, context, _ = _build_loop(project, CacheZeroClient())

    list(loop.run_stream(context))

    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    cache_rows = [row for row in facts if row.get("event") == "llm:cache_usage"]
    assert len(cache_rows) == 1, "报了 0 也应写 fact"
    assert cache_rows[0]["cache_read_input_tokens"] == 0
    assert cache_rows[0]["cache_creation_input_tokens"] == 0


def test_cache_usage_not_reported_no_fact(tmp_path: Path) -> None:
    """厂商完全不报 cache 用量时，observation 为 None，不写 fact。"""

    class NoCache:
        def plan(self, task: str, context: object | None = None) -> LLMPlan:
            del task, context
            return LLMPlan(
                final_output="no cache reported",
                observation=ModelObservation(
                    stage="plan",
                    provider="stub",
                    model="stub-model",
                    started_at="2026-09-01T00:00:00+00:00",
                    elapsed_ms=1,
                    attempt_count=1,
                    success=True,
                    cache_read_input_tokens=None,
                    cache_creation_input_tokens=None,
                ),
            )

    project = _project(tmp_path)
    loop, context, _ = _build_loop(project, NoCache())

    list(loop.run_stream(context))

    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    cache_rows = [row for row in facts if row.get("event") == "llm:cache_usage"]
    assert len(cache_rows) == 0, "未上报不应写 fact"


def test_cache_usage_reported_positive_writes_fact(tmp_path: Path) -> None:
    """厂商上报正数，observation 与 fact 均为该数。"""

    class CachePositiveClient:
        def plan(self, task: str, context: object | None = None) -> LLMPlan:
            del task, context
            return LLMPlan(
                final_output="positive cache",
                observation=ModelObservation(
                    stage="plan",
                    provider="stub",
                    model="stub-model",
                    started_at="2026-09-01T00:00:00+00:00",
                    elapsed_ms=1,
                    attempt_count=1,
                    success=True,
                    cache_read_input_tokens=256,
                    cache_creation_input_tokens=128,
                ),
            )

    project = _project(tmp_path)
    loop, context, _ = _build_loop(project, CachePositiveClient())

    list(loop.run_stream(context))

    facts = RunFactStore(project / ".reins" / "data").read_run(context.run_id)
    cache_rows = [row for row in facts if row.get("event") == "llm:cache_usage"]
    assert len(cache_rows) == 1
    assert cache_rows[0]["cache_read_input_tokens"] == 256
    assert cache_rows[0]["cache_creation_input_tokens"] == 128
