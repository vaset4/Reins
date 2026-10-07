"""验证协作等待、停止、共享决定及跨运行归属，不模拟底层执行成功。

作者：xxx
时间：2026-09-14 16:10:00
"""

from __future__ import annotations

import json
from dataclasses import replace
from threading import Event
from uuid import uuid4

import pytest

from llm.messages import AssistantMessage, TextPart
from runtime.cancellation import CancellationToken
from runtime.collaboration import ChildOutcome, CollaborationRuntime
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.shared_budget import BudgetOwner, SharedRunBudget
from runtime.tool_operations import ToolOperation, ToolOperationStore
from runtime.types import RunContext, RunToolsRequest, Trigger, new_run_id
from runtime.workspaces import WorkspaceStore


def make_host(root, runner, *, source=None):
    """组装真实持久层和会话协调器，runner只替换工作内容；传参：数据根/执行者；返回：宿主和父运行。"""
    source = source or RunContext(
        trigger=Trigger.USER,
        session_id="session-owner",
        payload={"message": "parent", "input_message_id": "parent-input"},
        capability_lease=from_trigger("user"),
    )
    messages, facts = SessionMessageStore(root), RunFactStore(root)
    WorkspaceStore(root).bind_session(source.session_id, root)
    messages.accept_input(
        source.session_id,
        str(source.payload["message"]),
        input_id=str(source.payload["input_message_id"]),
    )
    budget = SharedRunBudget(
        BudgetOwner(source.session_id, source.run_id, source.capability_lease), facts
    )
    host = CollaborationRuntime(
        source,
        messages=messages,
        facts=facts,
        operations=ToolOperationStore(root),
        budget=budget,
        cancellation=CancellationToken(),
        run_child=runner,
    )
    return host, source


def action(tool_name, **args):
    """构造已通过工具边界的协调操作；传参：工具名/字段；返回：独立操作身份。"""
    identity = f"op-{uuid4().hex}"
    return ToolOperation(
        RunToolsRequest(action=tool_name, tool_name=tool_name, arguments=args),
        identity,
        tool_name,
        args,
        operation_id=identity,
    )


def complete(root, execution, *, output="kept evidence", status="done"):
    """为受控工作记录真实消息与处理凭据；传参：数据根/执行及结果；返回：该工作真实边界。"""
    context = execution.context
    messages = SessionMessageStore(root)
    inputs = messages.deliver_inputs(
        context.session_id, run_id=context.run_id, task_id=context.material_task_id
    )
    messages.append_message(
        context.session_id,
        AssistantMessage(f"answer-{uuid4().hex}", (TextPart(output),)),
        run_id=context.run_id,
    )
    RunFactStore(root).append(
        {
            "event": "input:handled",
            "session_id": context.session_id,
            "run_id": context.run_id,
            "input_ids": list(inputs),
        }
    )
    return ChildOutcome(status, output)


def test_wait_for_parent_is_woken_by_message_and_not_lost_at_registration(tmp_path):
    """纯收信等待不会因没有子成员立即返回；传参：隔离根；返回：无。"""
    ready, finished = Event(), Event()
    outcomes = []

    def runner(execution):
        """等待父消息后确认此次输入；传参：子运行；返回：实际收信结果。"""
        host.messages.deliver_inputs(
            execution.context.session_id, run_id=execution.context.run_id, task_id=None
        )
        ready.set()
        result = host.execute(
            execution.context,
            action("agent_wait", targets=["parent"], timeout_seconds=3),
        )
        outcomes.append(json.loads(result.output)["wake_reason"])
        finished.set()
        return complete(tmp_path, execution)

    host, source = make_host(tmp_path, runner)
    try:
        host.execute(
            source, action("delegate", name="listener", task="wait for new source")
        )
        assert ready.wait(2)
        assert not finished.wait(0.1)
        sent = host.execute(
            source,
            action("agent_send", target="listener", message="new sourced discovery"),
        )
        assert sent.status == "ok"
        assert finished.wait(2)
        assert outcomes == ["message"]
    finally:
        host.close()


def test_child_cancel_wakes_wait_and_keeps_sibling_result(tmp_path):
    """取消一个等待者不改变兄弟已完成成果；传参：隔离根；返回：无。"""
    ready, stopped, sibling = Event(), Event(), Event()

    def runner(execution):
        """提供一个完成者和一个真实等待者；传参：独立运行；返回：对应结果。"""
        if execution.member["name"] == "good":
            result = complete(tmp_path, execution, output="verified sibling evidence")
            sibling.set()
            return result
        host.messages.deliver_inputs(
            execution.context.session_id, run_id=execution.context.run_id, task_id=None
        )
        ready.set()
        result = host.execute(
            execution.context,
            action("agent_wait", targets=["parent"], timeout_seconds=5),
        )
        assert json.loads(result.output)["wake_reason"] == "cancelled"
        stopped.set()
        return complete(
            tmp_path,
            execution,
            status="paused",
            output="cancelled before another action",
        )

    host, source = make_host(tmp_path, runner)
    try:
        host.execute(source, action("delegate", name="good", task="produce evidence"))
        host.execute(source, action("delegate", name="waiting", task="wait for work"))
        assert ready.wait(2) and sibling.wait(2)
        result = host.execute(source, action("agent_cancel", target="waiting"))
        assert result.status == "ok" and stopped.wait(2)
        view = json.loads(
            host.execute(source, action("agent_status", target="good")).output
        )["members"][0]
        assert view["output"] == "verified sibling evidence"
        assert not view["cancel_requested"]
    finally:
        host.close()


def test_shared_decision_requires_integrator_and_current_version(tmp_path):
    """共同决定不能被成员建议或过时写入覆盖；传参：目录；返回：无。"""
    host, source = make_host(tmp_path, lambda execution: complete(tmp_path, execution))
    try:
        child = replace(
            source,
            session_id="session-member",
            run_id=new_run_id(),
            parent_run_id=source.run_id,
        )
        denied = host.execute(
            child,
            action("agent_decision", key="budget", text="200", expected_version=0),
        )
        assert denied.status == "error"
        accepted = host.execute(
            source,
            action("agent_decision", key="budget", text="80", expected_version=0),
        )
        assert json.loads(accepted.output)["version"] == 1
        stale = host.execute(
            source,
            action("agent_decision", key="budget", text="200", expected_version=0),
        )
        assert stale.status == "error"
        assert host.store.read()["decisions"]["budget"][0]["text"] == "80"
    finally:
        host.close()


def test_new_parent_run_does_not_restart_previous_members_without_explicit_resume(
    tmp_path,
):
    """新事项的输入不会唤醒上一运行的执行者；传参：目录；返回：无。"""
    completed, executions = Event(), []

    def runner(execution):
        """记录每次真实接续身份；传参：子运行；返回：完成结果。"""
        executions.append(execution.context.run_id)
        result = complete(tmp_path, execution)
        completed.set()
        return result

    host, source = make_host(tmp_path, runner)
    host.execute(source, action("delegate", name="previous", task="old assignment"))
    assert completed.wait(2)
    host.close()
    new_source = replace(
        source,
        run_id=new_run_id(),
        payload={"message": "a different request", "input_message_id": "new-input"},
    )
    next_host, next_source = make_host(tmp_path, runner, source=new_source)
    try:
        next_host.sync_inputs()
        assert len(executions) == 1
        refused = next_host.execute(
            next_source, action("agent_send", target="previous", message="continue")
        )
        assert refused.status == "error"
        completed.clear()
        resumed = next_host.execute(
            next_source,
            action(
                "agent_send",
                target="previous",
                message="continue with new evidence",
                resume=True,
            ),
        )
        assert resumed.status == "ok" and completed.wait(2)
        assert len(executions) == len(set(executions)) == 2
    finally:
        next_host.close()


def test_user_update_retry_repairs_partial_delivery_without_repeating_input(
    tmp_path, monkeypatch
):
    """成员中途投递失败后补齐同一版本，已接纳成员不重复输入；参数：目录及故障替换；返回：无。"""
    release = Event()

    def runner(execution):
        """等待用户更正后再处理真实输入；参数：独立子执行；返回：持久完成结果。"""
        assert release.wait(10)
        return complete(tmp_path, execution)

    host, source = make_host(tmp_path, runner)
    try:
        for name in ("first", "second"):
            result = host.execute(
                source, action("delegate", name=name, task="wait for correction")
            )
            assert result.status == "ok"
        members = list(host.store.read()["members"].values())
        interrupted_recipient = members[1]["session_id"]
        original_send = host.store.send

        def interrupt_second(sender, recipient, **envelope):
            """在第二个成员接纳更正前暴露实际传递失败；参数：来源、成员及信件；返回：原凭据或异常。"""
            if (
                recipient == interrupted_recipient
                and envelope.get("reference") == "correction:v2"
            ):
                raise OSError("second recipient interrupted")
            return original_send(sender, recipient, **envelope)

        host.messages.accept_input(
            source.session_id, "budget=80", input_id="correction"
        )
        monkeypatch.setattr(host.store, "send", interrupt_second)
        with pytest.raises(OSError, match="second recipient interrupted"):
            host.sync_inputs()
        partial = host.store.read()
        assert partial["user_inputs"] == ["parent-input", "correction"]
        assert (
            len(
                [
                    item
                    for item in partial["messages"].values()
                    if item["reference"] == "correction:v2"
                ]
            )
            == 1
        )
        monkeypatch.setattr(host.store, "send", original_send)
        host.sync_inputs()
        host.sync_inputs()
        repaired = host.store.read()
        assert repaired["user_inputs"] == partial["user_inputs"]
        for member in members:
            identity = f"collab-user-correction-{member['agent_id']}"
            assert repaired["messages"][identity]["accepted"] is True
            assert repaired["messages"][identity]["reference"] == "correction:v2"
            assert (
                len(
                    [
                        entry
                        for entry in host.messages.read_entries(member["session_id"])
                        if entry.entry_id == identity
                    ]
                )
                == 1
            )
    finally:
        release.set()
        host.close()
