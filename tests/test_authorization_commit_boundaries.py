"""【审批】【提交边界】验证并发核对、新输入失效与单条授权原件复核。

作者：xxx
时间：2026-10-06 20:15:36
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from approval import ApprovalDecision, ApprovalRequest, ApprovalUnavailable
from approval.batch import BatchAuthorizer, intent_digest
from approval.batch_types import ApprovalChoice, BatchDecision
from approval.session import ApprovalSession
from runtime.cancellation import CancellationToken
from runtime.ledger import LedgerStore, new_ledger_event
from runtime.lease import from_trigger
from runtime.session_message_store import SessionMessageStore
from tasks.store import TaskStore

SOURCE_READ_TIMEOUT_SECONDS = 5


@pytest.fixture
def authorization_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """装配有父子来源的真实授权服务；参数：隔离目录和替换器；返回：服务及已交付输入对应的请求。"""
    monkeypatch.setattr("approval._config_path", lambda: tmp_path / "config.yaml")
    with closing(TaskStore(tmp_path)) as tasks:
        task = tasks.create_task("经批准发布结果")
    messages = SessionMessageStore(tmp_path)
    for session in ("parent", "child"):
        messages.accept_input(session, "核对后执行", input_id=f"initial-{session}")
        messages.deliver_inputs(session, run_id=f"run-{session}", task_id=None)
    request = ApprovalRequest(
        "publish_result",
        {},
        "confirm",
        from_trigger("user", task_id=task.task_id),
        tmp_path,
        "发布结果",
        cancellation=CancellationToken(),
        superseded=lambda: bool(messages.pending_inputs("child")),
        session_id="child",
        run_id="run-child",
        owner_session_id="parent",
        operation_id="operation-1",
        batch_id="batch-1",
        definition_version="contract-1",
    )
    request = replace(request, intent_digest=intent_digest(request))
    decision = BatchDecision(
        (ApprovalChoice(request.operation_id, ApprovalDecision.ONCE),)
    )
    authorizer = BatchAuthorizer(
        ApprovalSession(), messages, LedgerStore(tmp_path), prompt=lambda _: decision
    )
    return authorizer, request


def test_choice_callback_allows_another_thread_to_read_the_source(
    authorization_boundary,
) -> None:
    """新输入回调等待另一线程读取原件时仍可完成授权；参数：实际授权环境；返回：无。"""
    authorizer, request = authorization_boundary
    with ThreadPoolExecutor(max_workers=1) as pool:

        def inspect_inputs() -> bool:
            """由独立线程读取输入，暴露持事实锁回调造成的互等；参数：无；返回：是否有新要求。"""
            pending = pool.submit(
                authorizer.messages.pending_inputs, request.session_id
            )
            return bool(pending.result(timeout=SOURCE_READ_TIMEOUT_SECONDS))

        current = replace(request, superseded=inspect_inputs)
        authorization = authorizer.authorize((current,))[current.operation_id]
    assert authorization.decision is ApprovalDecision.ONCE
    assert authorizer.ledger.read_event(authorization.event_id) is not None


@pytest.mark.parametrize(
    "change", ["child_input", "parent_input", "cancel", "approval_input"]
)
def test_choice_commit_rechecks_inputs_and_cancellation(
    authorization_boundary,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    """回调检查后到提交前的新要求和取消仍阻止批准；参数：环境、替换器及变化；返回：无。"""
    authorizer, request = authorization_boundary
    write_lock = authorizer.messages.write_lock
    other_writer = SessionMessageStore(request.data_root)
    injected = False

    @contextmanager
    def admit_change_before_commit(session_id: str):
        """在真实提交锁入口插入另一写者的接纳；参数：会话编号；返回：原事实事务。"""
        nonlocal injected
        if not injected:
            injected = True
            if change == "cancel":
                request.cancellation.cancel("用户停止")
            else:
                recipient = "parent" if change == "parent_input" else "child"
                kind = "approval" if change == "approval_input" else None
                other_writer.accept_input(
                    recipient,
                    "新的实际输入",
                    input_id="concurrent-input",
                    input_kind=kind,
                )
        with write_lock(session_id) as batch:
            yield batch

    monkeypatch.setattr(authorizer.messages, "write_lock", admit_change_before_commit)
    authorization = authorizer.authorize((request,))[request.operation_id]
    decisions = [
        event
        for event in authorizer.ledger.read_events()
        if event.event == "approval.decided"
    ]
    if change == "approval_input":
        assert authorization.decision is ApprovalDecision.ONCE
        assert len(decisions) == 1
    else:
        assert authorization.decision is ApprovalDecision.CANCELLED
        assert decisions == []
        assert not any(
            entry.input_kind == "approval"
            for entry in authorizer.messages.read_entries("child")
        )


def test_validation_reads_only_the_referenced_authorization_event(
    authorization_boundary,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """复核当前批准不展开其他账目，重新打开后仍取原件；参数：环境和替换器；返回：无。"""
    authorizer, request = authorization_boundary
    authorization = authorizer.authorize((request,))[request.operation_id]
    authorizer.ledger.append(
        new_ledger_event(
            "tool.observed", "unrelated", "runtime", {"content": "其他运行的记录"}
        )
    )

    def reject_full_read():
        """禁止当前单条凭据复核读取整本账目；参数：无；返回：无，调用即失败。"""
        pytest.fail("one authorization must not materialize the full ledger")

    monkeypatch.setattr(authorizer.ledger, "read_events", reject_full_read)
    authorizer.validate(request, authorization)
    reopened = LedgerStore(request.data_root)
    assert (
        reopened.read_event(authorization.event_id).payload["intent_digest"]
        == request.intent_digest
    )
    assert reopened.read_event("missing") is None


@pytest.mark.parametrize(
    "mutation", ["event", "source", "decision", "missing_contract", "resource"]
)
def test_existing_event_with_a_different_contract_cannot_authorize(
    authorization_boundary,
    mutation: str,
) -> None:
    """存在编号但不属于原决定的事件不能放行动作；参数：环境及不符字段；返回：无。"""
    authorizer, request = authorization_boundary
    authorization = authorizer.authorize((request,))[request.operation_id]
    event = authorizer.ledger.read_event(authorization.event_id)
    assert event is not None
    payload = dict(event.payload)
    if mutation == "decision":
        payload["decision"] = "deny"
    elif mutation == "missing_contract":
        payload.pop("definition_version")
    elif mutation == "resource":
        payload["granted_resource"] = {"kind": "file", "target": "another.txt"}
    other = replace(
        event,
        event_id="other-event",
        payload=payload,
        event="tool.observed" if mutation == "event" else event.event,
        source="policy" if mutation == "source" else event.source,
    )
    authorizer.ledger.append(other)
    with pytest.raises(ApprovalUnavailable, match="does not match approved decision"):
        authorizer.validate(request, replace(authorization, event_id=other.event_id))


def test_removed_authorization_original_cannot_be_reused(
    authorization_boundary,
) -> None:
    """原件删除后不能凭内存中的旧批准继续派发；参数：实际授权环境；返回：无。"""
    authorizer, request = authorization_boundary
    authorization = authorizer.authorize((request,))[request.operation_id]
    with authorizer.ledger.database.transaction() as batch:
        batch.delete("ledger", authorization.event_id)
    with pytest.raises(ApprovalUnavailable, match="event is missing"):
        authorizer.validate(request, authorization)


@pytest.mark.parametrize("unrelated_grant", [False, True])
def test_unmatched_grants_do_not_read_unrelated_ledger_events(
    authorization_boundary,
    monkeypatch: pytest.MonkeyPatch,
    unrelated_grant: bool,
) -> None:
    """没有可复用候选时直接询问，不展开无关账目；参数：环境、替换器及无关授权；返回：无。"""
    authorizer, request = authorization_boundary
    if unrelated_grant:
        authorizer.session.remember(
            {
                "grant_id": "other-grant",
                "scope": "session",
                "tool": "another_tool",
                "args": {},
            }
        )

    def reject_ledger_scan(*_args):
        """无匹配权限时禁止加载审批之外的事件集合；参数：查询参数；返回：无，调用即失败。"""
        pytest.fail("unmatched grants must not load ledger event collections")

    monkeypatch.setattr(authorizer.ledger, "read_events", reject_ledger_scan)
    monkeypatch.setattr(authorizer.ledger, "read_events_of_type", reject_ledger_scan)
    authorization = authorizer.authorize((request,))[request.operation_id]
    assert authorization.source == "user_action"
    assert authorization.decision is ApprovalDecision.ONCE


@pytest.mark.parametrize("revoked", [False, True])
def test_matching_grants_read_targeted_decisions_and_durable_revocations(
    authorization_boundary,
    monkeypatch: pytest.MonkeyPatch,
    revoked: bool,
) -> None:
    """精确匹配仍核对真实撤销与原决定，不能恢复已撤销权限；参数：环境、替换器及撤销状态；返回：无。"""
    authorizer, request = authorization_boundary
    authorizer.prompt = lambda _: BatchDecision(
        (ApprovalChoice(request.operation_id, ApprovalDecision.SESSION),)
    )
    initial = authorizer.authorize((request,))[request.operation_id]
    if revoked:
        authorizer.ledger.append(
            new_ledger_event(
                "approval.revoked",
                "revoke-in-other-host",
                "user_action",
                {"grant_id": initial.grant["grant_id"]},
                session_id=request.session_id,
                run_id=request.run_id,
            )
        )
    authorizer.ledger.append(
        new_ledger_event(
            "tool.observed", "other-run", "runtime", {"content": "无关历史"}
        )
    )
    following = replace(request, operation_id="operation-2", batch_id="batch-2")
    prompts = []

    def deny_new_choice(batch):
        """只在不能复用原授权时拒绝新请求；参数：实际展示批次；返回：逐项拒绝。"""
        prompts.append(batch)
        return BatchDecision(
            (ApprovalChoice(following.operation_id, ApprovalDecision.DENY),)
        )

    def reject_full_read():
        """匹配查询不得展开所有其他 Ledger 正文；参数：无；返回：无，调用即失败。"""
        pytest.fail("grant lookup must query revocations and referenced decisions only")

    authorizer.prompt = deny_new_choice
    monkeypatch.setattr(authorizer.ledger, "read_events", reject_full_read)
    authorization = authorizer.authorize((following,))[following.operation_id]
    assert authorization.decision is (
        ApprovalDecision.DENY if revoked else ApprovalDecision.ONCE
    )
    assert authorization.source == ("user_action" if revoked else "grant")
    assert len(prompts) == int(revoked)
