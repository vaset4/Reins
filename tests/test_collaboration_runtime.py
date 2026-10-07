"""以真实会话、执行器和文件验证中途协作，不把接纳标签当作理解。

作者：xxx
时间：2026-09-15 01:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_stub

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from uuid import uuid4

import pytest

from scripts.testing.llm import _ScriptedTurn, _scripted_events
from llm.messages import (
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    model_visible_text,
    validate_message_sequence,
)
from runtime.agent_loop import AgentLoop, State
from runtime.collaboration_store import CollaborationStore
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import Idempotent, ToolDefinition, ToolRisk


def tool(tool_name, **arguments):
    """按公开模型协议生成测试调用；传参：工具与字段；返回：供应商事件脚本。"""
    return _ScriptedTurn(
        calls=(ToolCallPart(f"call-{uuid4().hex}", tool_name, arguments),)
    )


def successful_results(request):
    """按真实工具回执识别已完成动作；传参：供应商请求；返回：成功回执。"""
    return [
        item
        for item in request.messages
        if isinstance(item, ToolResultMessage) and item.status == "success"
    ]


class CollaborationScenario:
    """仅替换供应商决策，协作消息、授权、预算和文件操作均使用生产实现。"""

    def __init__(self, root):
        """保存材料根和后续请求证据；传参：隔离目录；返回：无。"""
        self.root, self.requests = root, []

    def stream(
        self, request, *, model, connection, cancellation=None, prepared_body=None
    ):
        """依据执行者收到的新证据生成下一动作；传参：真实请求与供应商依赖；返回：事件流。"""
        first = next(item for item in request.messages if isinstance(item, UserMessage))
        assignment = model_visible_text(first)
        role = (
            "parent"
            if first.message_id == "root-input"
            else json.loads(assignment)["text"]
        )
        self.requests.append((role, request))
        handlers = {
            "parent": self.parent,
            "price": self.price,
            "technical": self.technical,
        }
        return iter(_scripted_events(handlers[role](request), model))

    def parent(self, request):
        """根据真实受理和产物选择委派、等待或整合；传参：请求；返回：模型动作。"""
        results = successful_results(request)
        spawned = " ".join(
            model_visible_text(item) for item in results if item.tool_name == "delegate"
        )
        for name in ("price", "technical"):
            if f'"name": "{name}"' not in spawned:
                return tool("delegate", name=name, task=name)
        if not (self.root / "decision.json").exists():
            return tool("agent_wait", targets=["price"], timeout_seconds=3)
        if not any(item.tool_name == "file_read" for item in results):
            return tool("file_read", path="decision.json")
        return _ScriptedTurn(
            text="已核对产物：高级套餐110元，超过用户更正后的80元预算，因此不购买"
        )

    def price(self, request):
        """新发现和用户纠正共同影响实际写入；传参：后续请求；返回：价格调查动作。"""
        results = successful_results(request)
        if not any(item.tool_name == "slow_price" for item in results):
            return tool("slow_price")
        if not any(item.tool_name == "file_write" for item in results):
            evidence = " ".join(
                model_visible_text(item)
                for item in request.messages
                if isinstance(item, UserMessage)
            )
            decision = {
                "requires_pro": "must use PRO" in evidence,
                "budget": 80 if "budget=80" in evidence else 200,
            }
            decision["buy"] = not decision["requires_pro"] or decision["budget"] >= 110
            return tool(
                "file_write", path="decision.json", content=json.dumps(decision)
            )
        return _ScriptedTurn(text="价格产物已保存，请父执行者读取核对")

    def technical(self, request):
        """调查后直接向同伴发送带来源发现；传参：请求；返回：模型动作。"""
        results = successful_results(request)
        if not any(item.tool_name == "file_read" for item in results):
            return tool("file_read", path="tiers.txt")
        if not any(item.tool_name == "agent_send" for item in results):
            return tool(
                "agent_send", target="price", message="must use PRO; PRO costs 110"
            )
        return _ScriptedTurn(text="技术调查完成，发现已通知价格执行者")


def approve_scoped(request):
    """仅允许当前测试要求的委派和产物写入；传参：真实审批请求；返回：本次决定。"""
    from approval import ApprovalDecision

    assert request.tool in {"delegate", "file_write"}
    return ApprovalDecision.ONCE


def test_peer_discovery_and_user_update_change_child_artifact(tmp_path, monkeypatch):
    """两执行者运行中通信，用户纠正改变后续真实文件结果；传参：目录/替换器；返回：无。"""
    started, release = Event(), Event()
    (tmp_path / "tiers.txt").write_text(
        "Required feature needs PRO; price=110", encoding="utf-8"
    )
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    monkeypatch.chdir(other_workspace)
    registry = build_tool_registry(repo_root=tmp_path, data_root=tmp_path / "data")

    def slow_price(_args):
        """保留真实可观察的工具执行窗口；传参：工具参数；返回：基础报价。"""
        started.set()
        assert release.wait(10)
        return "BASIC=50; PRO=110"

    registry.register(
        ToolDefinition(
            "slow_price",
            "read pricing",
            {},
            "agent",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=slow_price,
        )
    )
    client = from_test_stub("unused")
    scenario = CollaborationScenario(tmp_path)
    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", scenario.stream
    )
    monkeypatch.setattr("approval._backend", approve_scoped)
    messages = SessionMessageStore(tmp_path / "data")
    context = RunContext(
        trigger=Trigger.USER,
        session_id="session-parent",
        payload={
            "message": "Compare options",
            "input_message_id": "root-input",
            "toolset_policy": {"enabled_toolsets": ["full"]},
        },
        capability_lease=from_trigger(
            "user",
            max_steps=50,
            max_tokens=400000,
            capabilities={
                "fs": {
                    "project_root": str(tmp_path),
                    "read": [str(tmp_path)],
                    "write": [str(tmp_path)],
                }
            },
        ),
    )
    WorkspaceStore(tmp_path / "data").bind_session(context.session_id, tmp_path)
    messages.accept_input(context.session_id, "Compare options", input_id="root-input")
    loop = AgentLoop(tmp_path / "data", llm_client=client, tool_registry=registry)
    with ThreadPoolExecutor(1) as pool:
        running = pool.submit(loop.run, context)
        try:
            if not started.wait(10):
                running.result(timeout=1)
                pytest.fail("price tool did not start")
            deadline = time.monotonic() + 10
            while not any(
                item.get("kind") == "peer"
                for item in CollaborationStore(messages, context.session_id)
                .read()["messages"]
                .values()
            ):
                assert time.monotonic() < deadline
                time.sleep(0.02)
            messages.accept_input(
                context.session_id, "budget=80", input_id="budget-correction"
            )
            release.set()
            assert running.result(timeout=15) is State.DONE
        finally:
            release.set()
            if not running.done():
                loop.cancellation.cancel("test execution interrupted")
    assert json.loads((tmp_path / "decision.json").read_text()) == {
        "requires_pro": True,
        "budget": 80,
        "buy": False,
    }
    assert not (other_workspace / "decision.json").exists()
    verify_collaboration_evidence(
        messages, context, scenario, data_root=tmp_path / "data"
    )


def verify_collaboration_evidence(messages, context, scenario, *, data_root):
    """核对输入去重、调用配对和父子账目归属；传参：真实记录与请求；返回：无。"""
    group = CollaborationStore(messages, context.session_id).read()
    assert len(group["members"]) == 2
    assert group["user_inputs"] == ["root-input", "budget-correction"]
    for member in group["members"].values():
        assert member["session_id"] != context.session_id
        assert WorkspaceStore(data_root).for_session(
            member["session_id"]
        ) == WorkspaceStore(data_root).for_session(context.session_id)
        assert member["run_id"] != context.run_id
        current = messages.materialize(member["session_id"])
        validate_message_sequence(current.messages)
        corrections = [
            item
            for item in current.messages
            if isinstance(item, UserMessage)
            and "budget-correction:v2" in model_visible_text(item)
        ]
        assert len(corrections) == 1
    prices = [request for role, request in scenario.requests if role == "price"]
    assert any(
        "must use PRO" in str(request.messages) and "budget=80" in str(request.messages)
        for request in prices
    )
    rows = RunFactStore(data_root).read_run(context.run_id)
    settlements = [row for row in rows if row.get("event") == "budget:settled"]
    assert len({row["owner_run_id"] for row in settlements}) >= 3
    assert len({row["attempt_id"] for row in settlements}) == len(settlements)


def test_mailbox_failed_transaction_retries_without_duplicate_input(
    tmp_path, monkeypatch
):
    """同库提交失败不留下半封消息，显式重投后只保留一次输入；参数：目录/故障注入；返回：无。"""
    messages = SessionMessageStore(tmp_path)
    store = CollaborationStore(messages, "session-parent")
    source = RunContext(
        trigger=Trigger.USER,
        session_id="session-parent",
        payload={"message": "source"},
        capability_lease=from_trigger("user"),
    )
    original = store._save

    def fail_ack(value):
        """在同一事务准备发布接收凭据时注入写入失败；参数：邮箱；返回：无。"""
        if any(row.get("accepted") for row in value["messages"].values()):
            raise OSError("mailbox ack interrupted")
        original(value)

    monkeypatch.setattr(store, "_save", fail_ack)
    with pytest.raises(OSError, match="ack interrupted"):
        store.send(
            source, "session-peer", message_id="collab-once", text="source discovery"
        )
    assert not messages.exists("session-peer")
    monkeypatch.setattr(store, "_save", original)
    store.send(
        source, "session-peer", message_id="collab-once", text="source discovery"
    )
    store.deliver_pending()
    messages.deliver_inputs("session-peer", run_id="run-peer", task_id=None)
    assert len(messages.materialize("session-peer").messages) == 1
    assert store.read()["messages"]["collab-once"]["accepted"] is True


@pytest.mark.parametrize(
    "change",
    [
        {"text": "changed discovery"},
        {"recipient": "session-other"},
        {"kind": "user_update"},
        {"reference": "changed-reference"},
    ],
)
def test_accepted_mailbox_retry_retains_sources_and_rejects_identity_changes(
    tmp_path, change
):
    """已接纳重投保留最初来源，身份冲突不能借只读确认被接受；参数：目录及冲突字段；返回：无。"""
    messages = SessionMessageStore(tmp_path)
    store = CollaborationStore(messages, "session-parent")
    source = RunContext(
        trigger=Trigger.USER,
        session_id="session-parent",
        payload={},
        capability_lease=from_trigger("user"),
    )
    arguments = {
        "recipient": "session-peer",
        "message_id": "collab-once",
        "text": "source discovery",
    }
    accepted = store.send(source, **arguments)
    resumed = replace(source, run_id="run-resumed")
    assert store.send(resumed, **arguments) == accepted
    assert accepted["sender_run_id"] == source.run_id
    with pytest.raises(ValueError, match="identity has different content or recipient"):
        store.send(resumed, **{**arguments, **change})
    assert store.read()["messages"] == {"collab-once": accepted}
    assert not messages.exists("session-other")
    messages.deliver_inputs("session-peer", run_id="run-peer", task_id=None)
    assert len(messages.materialize("session-peer").messages) == 1
