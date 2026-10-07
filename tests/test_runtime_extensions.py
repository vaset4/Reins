"""扩展建议与真实执行、结果存储和观察通知的责任边界。

作者：xxx
时间：2026-09-14 13:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final, from_test_stub

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, Thread

import pytest

from llm.messages import ToolCallPart, ToolResultMessage
from runtime.agent_loop import AgentLoop, State
from runtime.extensions import (
    ActionRequest,
    ResultView,
    RuntimeExtensions,
    ToolProposal,
)
from runtime.run_facts import RunFactStore
from runtime.checkpoint import load_latest_checkpoint_for_run
from runtime.session_messages import materialize_messages
from runtime.types import new_run_id
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk
from tests.test_tool_batch_execution import make_run

_WAIT_SECONDS = 5
_DISPATCH_CHECK_SECONDS = 0.2


def setup_loop(root, extensions, execute):
    """装配统一执行入口，工具调用通过真实Schema和权限检查；传参：依赖；返回：循环/运行。"""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "write",
            "写一个文件",
            {"path": {"type": "string", "required": True}},
            "file",
            ToolRisk.SAFE,
            False,
            "path",
            "builtin",
            target_scope_parameters=("path",),
            idempotent=Idempotent.NO,
            executor=execute,
        )
    )
    loop, context, _client = make_run(root, registry, [])
    loop.llm_client = from_test_native_tool_then_final(
        [ToolCallPart("original", "write", {"path": str(root / "a.txt")})],
        "完成当前答复",
    )
    loop.extensions = extensions
    return loop, context


@pytest.mark.parametrize("change", ["schema", "path", "approval", "identity"])
def test_awaitable_pre_hook_rechecks_changed_arguments(tmp_path, monkeypatch, change):
    """异步建议不能绕开参数/路径/授权边界；传参：临时根/改写类型；返回：无。"""
    effects, approvals = [], []

    async def before(proposal, _token):
        """等待后提出改参建议；传参：原建议/信号；返回：待校验建议。"""
        await asyncio.sleep(0)
        if change == "identity":
            return ToolProposal(proposal.tool, dict(proposal.arguments), "op-another")
        value = (
            7
            if change == "schema"
            else str(
                (tmp_path if change == "approval" else tmp_path.parent) / "outside.txt"
            )
        )
        return ToolProposal(proposal.tool, {"path": value}, proposal.operation_id)

    loop, context = setup_loop(
        tmp_path,
        RuntimeExtensions(before_tool=(before,)),
        lambda args: effects.append(args) or "written",
    )
    if change == "path":
        context.capability_lease.capabilities["fs"]["deny_write"] = [
            str(tmp_path.parent / "outside.txt")
        ]
    if change == "approval":
        context.capability_lease.capabilities["fs"]["write"] = []
    from approval import ApprovalDecision

    monkeypatch.setattr(
        "approval._backend",
        lambda request: approvals.append(request) or ApprovalDecision.DENY,
    )
    list(loop.run_stream(context))
    assert effects == []
    results = [
        item
        for item in materialize_messages(tmp_path, context.session_id)
        if isinstance(item, ToolResultMessage)
    ]
    assert results[0].status != "success"
    assert (
        json.loads(results[0].content[0].text)["meta"]["execution_state"]
        == "not_started"
    )
    assert bool(approvals) is (change == "approval")


def test_observer_failure_and_result_view_do_not_change_execution_fact(tmp_path):
    """派生视图不覆写原文或成功状态，观察者失败独立记录；传参：临时根；返回：无。"""
    effects = []

    def observer(_observation):
        """模拟普通观察者损坏；传参：事实快照；返回：不返回。"""
        raise RuntimeError("observer unavailable")

    def view(_observation):
        """仅生成模型展示；传参：原结果；返回：派生视图。"""
        return ResultView("用于模型的摘要", ("原文保留在操作记录",))

    loop, context = setup_loop(
        tmp_path,
        RuntimeExtensions(observers=(observer,), result_views=(view,)),
        lambda _args: effects.append(True) or "真实业务原文",
    )
    list(loop.run_stream(context))
    record = loop.operations.for_session(context.session_id)[0]
    assert record["result"]["status"] == "ok"
    assert record["result"]["output"] == "真实业务原文"
    assert record["result"]["model_view"] == "用于模型的摘要"
    result = next(
        item
        for item in materialize_messages(tmp_path, context.session_id)
        if isinstance(item, ToolResultMessage)
    )
    payload = json.loads(result.content[0].text)
    assert payload["output"] == "用于模型的摘要"
    assert result.status == "success"
    assert effects == [True]
    assert any(
        row["event"] == "extension:error"
        for row in RunFactStore(tmp_path).read_run(context.run_id)
    )


def test_post_run_action_is_deduplicated_and_uses_normal_permission_gate(
    tmp_path, monkeypatch
):
    """后处理请求先持久化，同ID只执行一次且越界仍需授权；传参：临时根；返回：无。"""
    action = ActionRequest(
        "followup", "write", {"path": str(tmp_path / "followup.txt")}
    )

    def after(_observation):
        """提出动作，不直接执行；传参：结束事实；返回：去重请求。"""
        return (action, action)

    from approval import ApprovalDecision

    approvals, effects = [], []
    monkeypatch.setattr(
        "approval._backend",
        lambda request: approvals.append(request) or ApprovalDecision.DENY,
    )
    loop, context = setup_loop(
        tmp_path,
        RuntimeExtensions(after_run=(after,)),
        lambda _args: effects.append(True) or "written",
    )
    context.capability_lease.capabilities["fs"]["write"] = []
    loop.llm_client = from_test_stub("主工作答复")
    list(loop.run_stream(context))
    assert effects == []
    assert len(approvals) == 1
    assert len(loop.operations.for_session(context.session_id)) == 1
    loop.submit_action(context, action)
    assert len(loop.operations.for_session(context.session_id)) == 1
    assert (
        load_latest_checkpoint_for_run(context.run_id, data_root=tmp_path).state.lower()
        == "done"
    )


@pytest.mark.parametrize("stage", ["result_views", "context_sources", "after_run"])
def test_stopping_during_extension_wait_releases_run(tmp_path, stage):
    """同步扩展阻塞时停止能释放宿主，已执行结果仍保留；传参：目录/扩展类型；返回：无。"""
    entered, release = Event(), Event()
    failures = []

    def hook(_observation):
        """模拟不能及时返回的扩展；传参：观察事实；返回：相应合法返回值。"""
        entered.set()
        assert release.wait(5)
        return {
            "result_views": ResultView("视图"),
            "context_sources": "材料",
            "after_run": (),
        }[stage]

    loop, context = setup_loop(
        tmp_path, RuntimeExtensions(**{stage: (hook,)}), lambda _args: "已发生的效果"
    )

    def execute():
        """捕获线程错误以检查停止结果；传参：无；返回：无。"""
        try:
            list(loop.run_stream(context))
        except BaseException as exc:
            failures.append(exc)

    worker = Thread(target=execute, daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        loop.cancellation.cancel()
        worker.join(1)
        assert not worker.is_alive()
    finally:
        release.set()
        worker.join(5)
    assert failures == []
    if stage == "result_views":
        record = loop.operations.for_session(context.session_id)[0]
        assert record["result"]["output"] == "已发生的效果"
        assert record["result"]["status"] == "ok"


def test_observer_cannot_recursively_execute_runtime(tmp_path):
    """观察者递归执行被明确拒绝且原结果不受影响；传参：临时根；返回：无。"""
    effects = []

    def observer(observation):
        """故意在观察回调直接运行；传参：事实；返回：无。"""
        if observation.event == "tool_committed":
            loop.run(context)

    loop, context = setup_loop(
        tmp_path,
        RuntimeExtensions(observers=(observer,)),
        lambda _args: effects.append(True) or "实际结果",
    )
    list(loop.run_stream(context))
    assert effects == [True]
    facts = RunFactStore(tmp_path).read_run(context.run_id)
    assert any(
        row.get("event") == "extension:error" and "ActionRequest" in row["error"]
        for row in facts
    )


def test_result_view_cannot_turn_real_failure_into_success(tmp_path):
    """后置摘要不能修改实际错误状态；传参：临时根；返回：无。"""
    from tools.types import ToolError, ToolErrorCategory

    extensions = RuntimeExtensions(
        result_views=(lambda _event: ResultView("文字声称成功"),)
    )
    loop, context = setup_loop(
        tmp_path,
        extensions,
        lambda _args: ToolError(
            ToolErrorCategory.INVALID_INPUT, "现实失败", retryable=False
        ),
    )
    list(loop.run_stream(context))
    record = loop.operations.for_session(context.session_id)[0]
    assert "现实失败" in record["result"]["error"]
    assert record["result"]["status"] != "ok"
    result = next(
        item
        for item in materialize_messages(tmp_path, context.session_id)
        if isinstance(item, ToolResultMessage)
    )
    assert result.status == "error"


def test_valid_hook_change_executes_final_arguments(tmp_path):
    """合法改参直接生效，不再依赖剩余步数；传参：临时目录；返回：无。"""
    target = tmp_path / "changed.txt"

    def before(proposal, _token):
        """将动作改到另一个已授权文件；传参：原动作及停止信号；返回：候选请求。"""
        return ToolProposal(proposal.tool, {"path": str(target)}, proposal.operation_id)

    def execute(args):
        """写入最终批准的目标；传参：参数；返回：业务结果。"""
        from pathlib import Path

        Path(args["path"]).write_text("真实修改", encoding="utf-8")
        return "已写入"

    loop, context = setup_loop(
        tmp_path, RuntimeExtensions(before_tool=(before,)), execute
    )
    # 步数上限压到 1 也不再影响改参生效
    context.capability_lease = replace(context.capability_lease, max_steps=1)
    assert loop.run(context) is State.DONE
    assert not (tmp_path / "a.txt").exists()
    assert target.read_text(encoding="utf-8") == "真实修改"


def test_final_hook_definition_serializes_conflicting_execution(tmp_path):
    """只读调用被改成写入后与前后读取串行；传参：临时目录；返回：无。"""
    release, second_prepared, writer_started = Event(), Event(), Event()
    order = []

    def read(args):
        """首项占用共享资源直到测试释放；传参：调用标签；返回：读取结果。"""
        label = args["label"]
        order.append(f"{label}:start")
        if label == "first":
            assert release.wait(_WAIT_SECONDS)
        order.append(f"{label}:end")
        return str(label)

    def write(_args):
        """记录写入开始和完成，供核对资源顺序；传参：参数；返回：写入结果。"""
        writer_started.set()
        order.extend(("write:start", "write:end"))
        return "写入完成"

    def before(proposal, _token):
        """只把第二项改成写入；传参：原动作和停止信号；返回：最终候选。"""
        if proposal.arguments["label"] == "second":
            second_prepared.set()
            return ToolProposal(
                "write_effect", dict(proposal.arguments), proposal.operation_id
            )
        return proposal

    registry = ToolRegistry()
    schema = {"label": {"type": "string", "required": True}}
    registry.register(
        ToolDefinition(
            "read_probe",
            "读取",
            schema,
            "agent",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=read,
            parallel_safe=True,
        )
    )
    registry.register(
        ToolDefinition(
            "write_effect",
            "写入",
            schema,
            "agent",
            ToolRisk.SAFE,
            False,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.NO,
            executor=write,
        )
    )
    loop, context, _client = make_run(tmp_path, registry, [])
    loop.extensions = RuntimeExtensions(before_tool=(before,))
    loop.llm_client = from_test_native_tool_then_final(
        [
            ToolCallPart(label, "read_probe", {"label": label})
            for label in ("first", "second", "third")
        ],
        "使用三项结果",
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(loop.run, context)
        try:
            assert second_prepared.wait(_WAIT_SECONDS)
            assert not writer_started.wait(_DISPATCH_CHECK_SECONDS)
        finally:
            release.set()
        assert future.result(timeout=_WAIT_SECONDS) is State.DONE
    assert order == [
        "first:start",
        "first:end",
        "write:start",
        "write:end",
        "third:start",
        "third:end",
    ]


def test_same_post_run_request_id_is_independent_between_runs(tmp_path):
    """同会话不同运行的同名后处理各执行一次；传参：临时目录；返回：无。"""
    effects = []
    action = ActionRequest(
        "same-name", "write", {"path": str(tmp_path / "followup.txt")}
    )
    extensions = RuntimeExtensions(after_run=(lambda _event: (action, action),))
    loop, context = setup_loop(
        tmp_path, extensions, lambda _args: effects.append("effect") or "完成"
    )
    loop.llm_client = from_test_stub("首次答复")
    assert loop.run(context) is State.DONE
    second = replace(
        context, run_id=new_run_id(), segment_id="", payload={"message": "另一次工作"}
    )
    second_loop = AgentLoop(
        tmp_path,
        llm_client=from_test_stub("第二次答复"),
        tool_registry=loop.tool_registry,
        extensions=extensions,
    )
    assert second_loop.run(second) is State.DONE
    assert effects == ["effect", "effect"]
    records = loop.operations.for_session(context.session_id)
    assert len({record["call"]["call_id"] for record in records}) == 2


def test_required_action_submission_failure_is_visible(tmp_path, monkeypatch):
    """必要后处理意图落盘失败不得当作观察错误吞掉；传参：目录及故障注入；返回：无。"""
    action = ActionRequest("persist", "write", {"path": str(tmp_path / "followup.txt")})
    loop, context = setup_loop(
        tmp_path,
        RuntimeExtensions(after_run=(lambda _event: (action,),)),
        lambda _args: "未调用",
    )
    loop.llm_client = from_test_stub("已保存主答复")

    def fail(*_args, **_kwargs):
        """拒绝必要提交；传参：操作记录；返回：抛出IO错误。"""
        raise OSError("action commit unavailable")

    monkeypatch.setattr(loop.operations, "create", fail)
    with pytest.raises(OSError, match="action commit unavailable"):
        loop.run(context)
    assert (
        load_latest_checkpoint_for_run(context.run_id, data_root=tmp_path).state.lower()
        == "done"
    )
