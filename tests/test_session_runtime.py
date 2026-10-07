"""用实际请求和持久文件验证运行中纠正与交接，不用等待时间猜测先后。

作者：xxx
时间：2026-09-14 10:00:00
"""

from __future__ import annotations
from scripts.testing.llm import (
    from_test_native_tool_then_final,
    from_test_sequence,
    from_test_stub,
)

from contextlib import closing
from pathlib import Path
from threading import Event

import pytest

from llm.messages import (
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    model_visible_text,
)
from runtime.agent_loop import AgentLoop
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.session_runtime import SessionRun, SessionRuntime
from runtime.types import RunContext, Trigger
from tasks.store import TaskStore
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk


def make_runtime(root: Path, client, registry: ToolRegistry, *, after_run=None):
    """装配真实循环与持久输入写者；传参：依赖及交接屏障；返回：协调器、运行列表。"""
    messages, facts = SessionMessageStore(root), RunFactStore(root)
    with closing(TaskStore(root)) as store:
        task = store.create_task("持续工作", is_inbox=True)
    runs: list[RunContext] = []

    def execute(request: SessionRun) -> None:
        """按身份取正文，不在新运行重复保存；传参：运行请求；返回：无。"""
        inbound = next(
            item
            for item in messages.read_entries("session-live")
            if item.entry_id == request.input_id
        )
        lease = from_trigger(
            "user", task_id=task.task_id, capabilities={"fs": {"read": [str(root)]}}
        )
        context = RunContext(
            session_id="session-live",
            compatibility_task_id=task.task_id,
            trigger=Trigger.USER,
            payload={
                "message": model_visible_text(inbound.message),
                "input_message_id": request.input_id,
            },
            capability_lease=lease,
        )
        runs.append(context)
        list(
            AgentLoop(
                root,
                llm_client=client,
                tool_registry=registry,
                cancellation=request.cancellation,
            ).run_stream(context)
        )
        if after_run is not None:
            after_run()

    return SessionRuntime(
        "session-live", messages=messages, facts=facts, run=execute
    ), runs


def capture_requests(client, monkeypatch, *, before_first=None):
    """捕获发给Adapter的真实请求，必要时在首请求设置屏障；传参：客户端/替换器；返回：请求列表。"""
    adapter = client._adapter_registry.require("scripted_test")
    original, requests = adapter.stream, []

    def stream(request, **kwargs):
        """在实际网络边界留出用户纠正窗口；传参：完整请求；返回：供应商事件流。"""
        requests.append(request)
        if len(requests) == 1 and before_first is not None:
            before_first()
        yield from original(request, **kwargs)

    monkeypatch.setattr(adapter, "stream", stream)
    return requests


def test_correction_during_native_tool_batch_enters_next_request(
    tmp_path: Path, monkeypatch
) -> None:
    """工具未回填时可持久接纳，闭合后原文各进入一次；传参：临时目录/替换器；返回：无。"""
    started, release = Event(), Event()
    effects: list[str] = []

    def execute(args):
        """记录真实执行并等候测试释放；传参：工具参数；返回：结果。"""
        effects.append(args["value"])
        started.set()
        assert release.wait(5)
        return args["value"]

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "probe",
            "查询",
            {"value": {"type": "string", "required": True}},
            "agent",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=execute,
        )
    )
    client = from_test_native_tool_then_final(
        [
            ToolCallPart("invalid", "probe", {}),
            ToolCallPart("valid", "probe", {"value": "证据"}),
        ],
        "已采用新约束",
    )
    requests = capture_requests(client, monkeypatch)
    runtime, _runs = make_runtime(tmp_path, client, registry)
    runtime.submit("调查", input_id="input-first")
    assert started.wait(5)
    runtime.submit("只使用本地文件", input_id="input-correction")
    store = SessionMessageStore(tmp_path)
    current = store.materialize("session-live")
    assert current.pending_tool_calls
    assert "只使用本地文件" not in str(current.messages)
    release.set()
    assert runtime.wait_idle(10)
    final_messages = requests[-1].messages
    assert [
        item.call_id for item in final_messages if isinstance(item, ToolResultMessage)
    ] == ["invalid", "valid"]
    assert (
        sum(
            model_visible_text(item) == "只使用本地文件"
            for item in final_messages
            if isinstance(item, UserMessage)
        )
        == 1
    )
    assert effects == ["证据"]
    assert (
        len(
            [
                entry
                for entry in store.read_entries("session-live")
                if entry.type == "inbound"
            ]
        )
        == 2
    )
    runtime.submit("只使用本地文件", input_id="input-correction")
    assert runtime.wait_idle(2)
    assert len(requests) == 2


def test_correction_during_model_discards_old_final(
    tmp_path: Path, monkeypatch
) -> None:
    """用户在生成中纠正后，旧结论不结束工作；传参：临时目录/替换器；返回：无。"""
    started, release = Event(), Event()

    def before_first():
        """证明输入发生在首请求内部；传参：无；返回：无。"""
        started.set()
        assert release.wait(5)

    client = from_test_sequence(["旧答案", "修正后的答案"])
    requests = capture_requests(client, monkeypatch, before_first=before_first)
    runtime, runs = make_runtime(tmp_path, client, ToolRegistry())
    runtime.submit("原要求")
    assert started.wait(5)
    runtime.submit("改成中文回答")
    release.set()
    assert runtime.wait_idle(10)
    assert len(runs) == 1
    assert len(requests) == 2
    assert "改成中文回答" in str(requests[-1])
    messages = SessionMessageStore(tmp_path).materialize("session-live").messages
    assert not any(model_visible_text(item) == "旧答案" for item in messages)
    assert model_visible_text(messages[-1]) == "修正后的答案"


def test_input_at_run_handoff_is_not_lost_or_replayed(tmp_path: Path) -> None:
    """最终结果已提交而宿主尚未释放时，新输入由下一运行接手；传参：临时目录；返回：无。"""
    ending, release = Event(), Event()

    def after_run():
        """停在提交与释放之间；传参：无；返回：无。"""
        ending.set()
        assert release.wait(5)

    client = from_test_sequence(["第一个结果", "第二个结果"])
    runtime, runs = make_runtime(tmp_path, client, ToolRegistry(), after_run=after_run)
    runtime.submit("第一项", input_id="first")
    assert ending.wait(5)
    runtime.submit("第二项", input_id="second")
    release.set()
    assert runtime.wait_idle(10)
    assert len(runs) == 2
    runtime.submit("第二项", input_id="second")
    assert runtime.wait_idle(2)
    assert len(runs) == 2


def test_delivered_input_survives_failure_before_request(tmp_path: Path) -> None:
    """交付标记不能掩盖未发出的请求，重启时用原输入身份恢复；传参：临时目录；返回：无。"""
    messages, facts = SessionMessageStore(tmp_path), RunFactStore(tmp_path)

    def crash(_request):
        """模拟投影落盘后的进程中断；传参：运行；返回：不返回。"""
        messages.deliver_inputs("session-live", run_id="interrupted", task_id=None)
        raise OSError("interrupted before provider request")

    runtime = SessionRuntime("session-live", messages=messages, facts=facts, run=crash)
    runtime.submit("必须继续处理", input_id="durable")
    with pytest.raises(RuntimeError, match="recoverable"):
        runtime.wait_idle(5)
    restarted, runs = make_runtime(tmp_path, from_test_stub("接续完成"), ToolRegistry())
    restarted.resume()
    assert restarted.wait_idle(10)
    assert len(runs) == 1
    entries = messages.read_entries("session-live")
    assert sum(item.type == "inbound" for item in entries) == 1
    assert sum(item.type == "delivery" for item in entries) == 1
