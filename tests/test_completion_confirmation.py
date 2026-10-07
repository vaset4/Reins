"""验证真实提问、用户选择、版本核验和可恢复提交。

作者：xxx
时间：2026-09-24 18:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from app.completion_ui import CompletionMenu
from app.repl.session_host import SessionHost, SessionHostConfig
from app.repl.slash_commands import ReplState
from llm.client import RealLLMClient
from llm.messages import AssistantMessage, TextPart, ToolCallPart, UserMessage
from runtime.agent_loop import AgentLoop
from runtime.completion_confirmation import CompletionConfirmations
from runtime.goal_manager import GoalManager
from runtime.ledger import LedgerStore
from runtime.lease import from_trigger
from runtime.session_message_store import SessionMessageStore
from runtime.tool_operations import ToolOperationStore
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tools.native_actions import register_native_actions
from tools.tool_registry import ToolRegistry
from tests.test_harness_feedback_requests import _run


@dataclass
class ConfirmationEnvironment:
    """保存同一真实提问的测试依赖。"""

    root: Path
    confirmations: CompletionConfirmations
    question: dict[str, Any]
    client: RealLLMClient
    registry: ToolRegistry


@pytest.fixture
def confirmation_env(tmp_path: Path):
    """经真实 AgentLoop 产生待确认提案；传参：临时根；返回：可关闭的场景依赖。"""
    with closing(TaskStore(tmp_path)) as tasks:
        tasks.create_task("核对资料", task_id="goal")
        messages = SessionMessageStore(tmp_path)
        WorkspaceStore(tmp_path).bind_session("session", tmp_path)
        messages.append_message(
            "session",
            UserMessage("request", (TextPart("结果需要我认可"),)),
            task_id="goal",
        )
        messages.append_message(
            "session",
            AssistantMessage(
                "report", (TextPart("逐项核对后，两份资料的日期与数值一致"),)
            ),
            task_id="goal",
            run_id="report-run",
        )
        registry = ToolRegistry()
        register_native_actions(registry)
        client = from_test_native_tool_then_final(
            [
                ToolCallPart(
                    "confirm-call",
                    "ask_user",
                    {
                        "question": "是否认可这份核对结果？",
                        "completion": {
                            "goal_id": "goal",
                            "expected_revision": tasks.require_task("goal").revision,
                            "evidence": [
                                {
                                    "kind": "answer",
                                    "reference": "report",
                                    "reason": "已交付逐项核对结果",
                                }
                            ],
                        },
                    },
                )
            ],
            "已收到你的选择",
        )
        context = RunContext(
            task_id="goal",
            session_id="session",
            trigger=Trigger.USER,
            payload={"message": "请确认核对结果"},
            capability_lease=from_trigger("user", task_id="goal"),
        )
        list(
            AgentLoop(tmp_path, llm_client=client, tool_registry=registry).run_stream(
                context
            )
        )
        confirmations = CompletionConfirmations(
            tasks, messages, ToolOperationStore(tmp_path), LedgerStore(tmp_path)
        )
        pending = confirmations.pending("session")
        assert len(pending) == 1 and pending[0]["valid"]
        yield ConfirmationEnvironment(
            tmp_path, confirmations, pending[0], client, registry
        )


def _complete(env: ConfirmationEnvironment, reference: str) -> None:
    """用事件作为实际目标完成依据；传参：场景和引用；返回：无。"""
    service = env.confirmations
    GoalManager(
        service.tasks,
        messages=service.messages,
        operations=service.operations,
        ledger=service.ledger,
    ).complete_goal(
        "goal",
        expected_revision=env.question["expected_revision"],
        summary="用户已认可核对结果",
        evidence=[
            {
                "kind": "user_confirmation",
                "reference": reference,
                "reason": "明确确认当前版本",
            }
        ],
        session_id="session",
        run_id="complete-run",
        operation_id="complete-operation",
    )


@pytest.mark.parametrize(("choice", "accepted"), [("1", True), ("2", False)])
def test_actual_local_frontend_choice_creates_typed_confirmation(
    confirmation_env, choice, accepted
):
    """实际前台选择记录唯一用户动作，拒绝不关闭目标；传参：场景、选项和期望；返回：无。"""
    env = confirmation_env
    state = ReplState(session_id="session", current_task_id="goal")
    host = SessionHost(
        SessionHostConfig(state, env.registry, env.client, env.root, env.root)
    )
    try:
        host.submit(choice)
        host.wait_idle()
        events = [
            row
            for row in env.confirmations.ledger.read_session_events("session")
            if row.event == "goal.confirmation_decided"
        ]
        assert len(events) == 1 and events[0].payload["accepted"] is accepted
        source = events[0].payload["source_input_id"]
        assert (
            sum(
                entry.entry_id == source
                for entry in env.confirmations.messages.read_entries("session")
            )
            == 1
        )
        if accepted:
            _complete(env, events[0].event_id)
            before = env.confirmations.tasks.require_task("goal").revision
            _complete(env, events[0].event_id)
            assert env.confirmations.tasks.require_task("goal").revision == before
            completion = env.confirmations.tasks.require_task("goal").completion
            assert completion["evidence"][0]["message_id"] == source
            assert len(completion["evidence"][0]["content_sha256"]) == 64
            env.confirmations.tasks.append_grant(
                "goal", {"tool": "read", "scope": "once"}
            )
            env.confirmations.tasks.update_task_refs(
                "goal", skill_refs=["compare-method"]
            )
            env.confirmations.tasks.update_sediment_status("goal", done=True)
            assert env.confirmations.tasks.require_task("goal").completion == completion
        else:
            with pytest.raises(ValueError, match="rejected"):
                _complete(env, events[0].event_id)
            assert env.confirmations.tasks.require_task("goal").status == "active"
    finally:
        host.close()


@pytest.mark.parametrize("failure", ["revision", "branch", "pending"])
def test_old_proposal_cannot_confirm_changed_work(confirmation_env, failure):
    """版本、分支或未处理新要求使旧确认失效；传参：场景及变化；返回：无。"""
    env = confirmation_env
    service = env.confirmations
    if failure == "revision":
        service.tasks.update_task_refs("goal", spec_refs=["new-requirement"])
    elif failure == "branch":
        first = service.messages.materialize("session").entries[0].entry_id
        service.messages.branch("session", first)
    else:
        service.messages.accept_input(
            "session", "还有第三份资料", input_id="new-requirement"
        )
    with pytest.raises(ValueError):
        service.decide(
            "session", env.question["question_id"], action_id="stale", accepted=True
        )
    assert not any(
        row.event == "goal.confirmation_decided"
        for row in service.ledger.read_session_events("session")
    )
    assert service.tasks.require_task("goal").status == "active"


@pytest.mark.parametrize("boundary", ["event", "operation"])
def test_confirmation_retry_repairs_partial_commit_without_duplicate_input(
    confirmation_env, monkeypatch, boundary
):
    """输入、事件、回执间失败可用同一动作补齐；传参：场景、替换器和边界；返回：无。"""
    env = confirmation_env
    service = env.confirmations
    owner, method = (
        (service.ledger, "append_once")
        if boundary == "event"
        else (service.operations, "write")
    )
    original = getattr(owner, method)

    def unavailable(*_args, **_kwargs):
        """模拟真实持久化故障；传参：写入参数；返回：抛出错误。"""
        raise OSError("synthetic commit failure")

    monkeypatch.setattr(owner, method, unavailable)
    with pytest.raises(OSError, match="synthetic commit failure"):
        service.decide(
            "session", env.question["question_id"], action_id="retry", accepted=True
        )
    monkeypatch.setattr(owner, method, original)
    event = service.decide(
        "session", env.question["question_id"], action_id="retry", accepted=True
    )
    assert (
        service.decide(
            "session", env.question["question_id"], action_id="retry", accepted=True
        )
        == event
    )
    with pytest.raises(ValueError, match="different decision"):
        service.decide(
            "session", env.question["question_id"], action_id="retry", accepted=False
        )
    assert (
        sum(
            row.event == "goal.confirmation_decided"
            for row in service.ledger.read_session_events("session")
        )
        == 1
    )
    assert (
        sum(
            row.entry_id == event.payload["source_input_id"]
            for row in service.messages.read_entries("session")
        )
        == 1
    )
    assert (
        next(
            row
            for row in service.operations.for_session("session")
            if row["operation_id"] == env.question["question_id"]
        )["confirmation_event_id"]
        == event.event_id
    )


def test_confirmation_menu_preserves_identity_and_does_not_guess_normal_text(
    confirmation_env,
):
    """重连展示相同提案，普通回复没有确认语义；传参：场景；返回：无。"""
    env = confirmation_env
    first, second = CompletionMenu(), CompletionMenu()
    view = env.confirmations.pending("session")
    rendered = first.update(view)
    assert rendered == second.update(view)
    assert "两份资料的日期与数值一致" in rendered
    choice = first.choose("1")
    assert (
        choice == first.choose("1")
        and choice["question_id"] == env.question["question_id"]
    )
    assert all(
        first.choose(text) is None for text in ("好", "继续", "确认完成", "1 新要求")
    )


@pytest.mark.parametrize("body", ["逐项核对结果：两份资料的日期和数值一致", ""])
def test_current_answer_is_published_before_native_completion(tmp_path, body):
    """同轮真实正文先发布再完成，空正文不能自证成果；传参：临时根与正文；返回：无。"""
    from scripts.testing.llm import _ScriptedTurn

    registry = ToolRegistry()
    register_native_actions(registry)
    _, context, _adapter = _run(
        tmp_path,
        (
            _ScriptedTurn(
                text=body,
                calls=(
                    ToolCallPart(
                        "complete-report",
                        "goal",
                        {
                            "action": "complete",
                            "goal_body": "已交付逐项核对报告",
                            "expected_revision": 1,
                            "evidence": [
                                {
                                    "kind": "answer",
                                    "reference": "current_answer",
                                    "reason": "本轮正文包含完整比较结果",
                                }
                            ],
                        },
                    ),
                ),
            ),
            _ScriptedTurn(text="处理结束"),
        ),
        definition=registry.get("goal"),
    )
    with closing(TaskStore(tmp_path)) as tasks:
        task = tasks.require_task(context.task_id)
        assert task.status == ("done" if body else "active")
        if body:
            reference = task.completion["evidence"][0]
            source = next(
                entry
                for entry in SessionMessageStore(tmp_path)
                .materialize(context.session_id)
                .entries
                if entry.entry_id == reference["entry_id"]
            )
            assert source.message.content[0].text == body


def test_ask_user_success_cannot_be_used_as_delivered_work(confirmation_env):
    """问题调用成功不等于已有成果；传参：真实提问场景；返回：无。"""
    env = confirmation_env
    service = env.confirmations
    with pytest.raises(ValueError, match="asking a question"):
        GoalManager(service.tasks, messages=service.messages).complete_goal(
            "goal",
            expected_revision=env.question["expected_revision"],
            summary="不能用提问冒充成果",
            evidence=[
                {
                    "kind": "tool_result",
                    "reference": "confirm-call",
                    "reason": "问题调用返回成功",
                }
            ],
            session_id="session",
            run_id="bad-completion",
        )


def test_background_reconnection_keeps_original_confirmation(
    confirmation_env, monkeypatch
):
    """后台重连展示原提案，前端选择经真实RPC分派接回运行；传参：场景、替换器；返回：无。"""
    from app.background.frontend import BackgroundSessionHost
    from app.background.server import BackgroundServer
    from app.background.service import BackgroundService
    from app.background.sessions import (
        BackgroundSession,
        SessionRecord,
        SessionServices,
    )

    env = confirmation_env
    services = SessionServices(
        env.root, env.root, lambda _: env.client, lambda: env.registry
    )
    background = BackgroundSession(
        SessionRecord(
            "session", status="paused", intent={"run_id": env.question["run_id"]}
        ),
        services,
    )
    service = BackgroundService(services)
    monkeypatch.setattr(service, "attach", lambda _, **_options: background)
    server = BackgroundServer(service, "synthetic-local-token")

    class Connection:
        """保留生产RPC分派，只把本地传输替换为可重复调用。"""

        def call(self, method, **params):
            """调用真实服务入口；传参：动作和参数；返回：服务回执。"""
            return server.dispatch(method, params)

    monkeypatch.setattr(BackgroundSessionHost, "_connect", lambda _: Connection())
    config = SessionHostConfig(
        ReplState(session_id="session", current_task_id="goal"),
        env.registry,
        env.client,
        env.root,
        env.root,
    )
    first = BackgroundSessionHost(config)
    first.close()
    host = BackgroundSessionHost(config)
    try:
        assert (
            background.snapshot(history=True)["completion_requests"][0]["question_id"]
            == env.question["question_id"]
        )
        source = host.submit("1")
        assert background.runtime.wait_idle(10)
        events = [
            row
            for row in env.confirmations.ledger.read_session_events("session")
            if row.event == "goal.confirmation_decided"
        ]
        assert len(events) == 1 and events[0].payload["source_input_id"] == source
        _complete(env, events[0].event_id)
        assert env.confirmations.tasks.require_task("goal").status == "done"
    finally:
        host.close()
        background.close()
        server.server_close()


def test_fullscreen_tui_choice_keeps_one_canonical_user_input(confirmation_env):
    """全屏TUI的明确选择沿用已保存输入；传参：确认场景；返回：无。"""
    from frontends.tui.session import FullscreenTui

    env = confirmation_env
    tui = FullscreenTui(
        project_root=env.root,
        data_root=env.root,
        llm_client=env.client,
        tool_registry=env.registry,
    )
    tui.state = ReplState(session_id="session", current_task_id="goal")
    try:
        tui.submit("/help")
        assert any("确认完成" in item.body for item in tui.transcript.snapshot())
        tui.submit("1")
        tui._worker.join(timeout=10)
        assert not tui._worker.is_alive()
        items = tui.transcript.snapshot()
        assert any(item.title == "run finished" for item in items)
        assert all(item.role != "error" for item in items)
        events = [
            row
            for row in env.confirmations.ledger.read_session_events("session")
            if row.event == "goal.confirmation_decided"
        ]
        assert len(events) == 1
        users = [
            entry
            for entry in env.confirmations.messages.read_entries("session")
            if entry.type == "inbound"
        ]
        assert (
            len(users) == 1
            and users[0].entry_id == events[0].payload["source_input_id"]
        )
    finally:
        tui.close()
