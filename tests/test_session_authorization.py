"""授权范围、重启、撤销和持久化失败的真实执行边界。

作者：xxx
时间：2026-09-24 12:00:00
"""

from contextlib import closing
from dataclasses import replace

import pytest

from app.approval_ui import approval_control
from approval import ApprovalDecision, ApprovalUnavailable
from approval.batch import BatchAuthorizer
from approval.batch_types import ApprovalChoice, BatchDecision
from approval.session import ApprovalMode, ApprovalSession
from runtime.ledger import LedgerStore
from runtime.lease import from_trigger
from runtime.session_message_store import SessionMessageStore
from runtime.shared_budget import BudgetOwner
from runtime.watchdog import Watchdog
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import PreparedToolExecution
from tools.types import ToolError


@pytest.fixture
def authorization_env(tmp_path, monkeypatch):
    """装配实际文件工具与隔离权限载体；传参：目录和替换器；返回：运行依赖。"""
    project_root, data_root = tmp_path / "project", tmp_path / "data"
    project_root.mkdir()
    monkeypatch.setattr("approval._config_path", lambda: data_root / "config.yaml")
    with closing(TaskStore(data_root)) as tasks:
        task = tasks.create_task("修改指定文件")
    lease = from_trigger(
        "user",
        task_id=task.task_id,
        capabilities={
            "fs": {
                "project_root": str(project_root),
                "read": [str(project_root)],
                "write": [],
            }
        },
    )
    WorkspaceStore(data_root).bind_session("session", project_root)
    messages = SessionMessageStore(data_root)
    messages.accept_input("session", "请修改指定文件", input_id="initial")
    messages.deliver_inputs("session", run_id="run", task_id=task.task_id)
    registry = build_tool_registry(repo_root=project_root, data_root=data_root)
    watchdog = Watchdog(
        lease, data_root=data_root, budget_owner=BudgetOwner("session", "run", lease)
    )
    yield project_root, task.task_id, registry, watchdog, messages
    registry.close()


def _prepared(env, name, *, operation, content="one"):
    """只准备实际文件写，不产生文件或授权副作用；传参：环境、文件、身份及内容；返回：准备结果。"""
    _, _, registry, watchdog, _ = env
    result = registry.prepare_tool_execution(
        "file_write",
        {"path": name, "content": content},
        watchdog.lease,
        watchdog=watchdog,
        operation_id=operation,
        approval_batch=f"batch-{operation}",
        cancellation=watchdog.cancellation,
        superseded=lambda: bool(env[4].pending_inputs("session")),
    )
    assert isinstance(result, PreparedToolExecution)
    return result


def _authorizer(env, session, choices):
    """为真实授权边界提供明确的逐次用户选择；传参：环境、宿主、选择列表；返回：服务和展示记录。"""
    _, _, _, _, messages = env
    shown = []

    def prompt(batch):
        """对显示的项目逐项选择作用域；传参：批次；返回：完整回执。"""
        shown.append(batch)
        decision, resource_scope = choices.pop(0)
        return BatchDecision(
            tuple(
                ApprovalChoice(request.operation_id, decision, resource_scope)
                for request in batch.requests
            )
        )

    return BatchAuthorizer(
        session, messages, LedgerStore(messages.database.data_root), prompt=prompt
    ), shown


def _authorize(authorizer, prepared):
    """把已提交的授权绑定到原准备对象；传参：服务和准备结果；返回：可派发请求。"""
    result = authorizer.authorize((prepared.approval_request,))[prepared.operation_id]
    return replace(prepared, authorization=result, authorizer=authorizer)


def test_directory_session_grant_reconnects_but_does_not_survive_new_host(
    authorization_env,
):
    """相同宿主复用目录范围，前缀碰撞及新宿主重新询问；传参：实际环境；返回：无。"""
    env = authorization_env
    root, _, registry, _, _ = env
    (root / "a").mkdir()
    (root / "ab").mkdir()
    session = ApprovalSession()
    authorizer, shown = _authorizer(
        env, session, [(ApprovalDecision.SESSION, "directory")]
    )
    assert not isinstance(
        registry.execute_prepared_tool(
            _authorize(authorizer, _prepared(env, "a/first.txt", operation="one"))
        ),
        ToolError,
    )
    reconnected, next_shown = _authorizer(
        env, session, [(ApprovalDecision.DENY, "exact")]
    )
    assert not isinstance(
        registry.execute_prepared_tool(
            _authorize(reconnected, _prepared(env, "a/second.txt", operation="two"))
        ),
        ToolError,
    )
    assert isinstance(
        registry.execute_prepared_tool(
            _authorize(reconnected, _prepared(env, "ab/other.txt", operation="three"))
        ),
        ToolError,
    )
    restarted, restart_shown = _authorizer(
        env, ApprovalSession(), [(ApprovalDecision.DENY, "exact")]
    )
    assert isinstance(
        registry.execute_prepared_tool(
            _authorize(restarted, _prepared(env, "a/third.txt", operation="four"))
        ),
        ToolError,
    )
    assert len(shown) == len(next_shown) == len(restart_shown) == 1
    with closing(TaskStore(env[4].database.data_root)) as tasks:
        assert not tasks.require_task(env[1]).grants
    assert not (env[4].database.data_root / "config.yaml").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "path",
        "content",
        "operation",
        "definition",
        "revoke",
        "mode",
        "deny",
        "cron",
        "goal",
    ],
)
def test_queued_approval_is_revalidated_before_effect(authorization_env, mutation):
    """排队期间意图、合同、撤销或模式改变不能沿用旧批准；传参：环境及改变；返回：无。"""
    env = authorization_env
    root, _, registry, _, _ = env
    session = ApprovalSession()
    authorizer, _ = _authorizer(env, session, [(ApprovalDecision.ONCE, "exact")])
    prepared = _authorize(authorizer, _prepared(env, "result.txt", operation="one"))
    if mutation in {"path", "content"}:
        prepared = replace(
            prepared, arguments={**prepared.arguments, mutation: "changed.txt"}
        )
    elif mutation == "operation":
        prepared = replace(prepared, operation_id="another-operation")
    elif mutation == "definition":
        registry.replace(replace(registry.get("file_write"), readonly=True))
    elif mutation == "revoke":
        session.revoke(str(prepared.authorization.grant["grant_id"]))
    elif mutation == "deny":
        prepared = replace(
            prepared,
            lease=replace(
                prepared.lease,
                capabilities={
                    **prepared.lease.capabilities,
                    "fs": {
                        **prepared.lease.capabilities["fs"],
                        "deny_write": [str(root / "result.txt")],
                    },
                },
            ),
        )
    elif mutation == "cron":
        prepared = replace(prepared, lease=replace(prepared.lease, trigger="cron"))
    elif mutation == "goal":
        prepared = replace(
            prepared, lease=replace(prepared.lease, task_id="another-goal")
        )
    else:
        session.set_mode(ApprovalMode.READ_ONLY)
    assert isinstance(registry.execute_prepared_tool(prepared), ToolError)
    assert not (root / "result.txt").exists() and not (root / "changed.txt").exists()


@pytest.mark.parametrize("scope", [ApprovalDecision.TASK, ApprovalDecision.PERMANENT])
def test_persistent_grants_require_event_and_can_be_revoked(authorization_env, scope):
    """持久授权绑定事件，用户撤销清除实际载体并拒绝排队项；传参：环境和范围；返回：无。"""
    env = authorization_env
    root, task, registry, _, _ = env
    session = ApprovalSession()
    authorizer, _ = _authorizer(env, session, [(scope, "exact")])
    prepared = _authorize(authorizer, _prepared(env, "result.txt", operation="one"))
    identity = str(prepared.authorization.grant["grant_id"])
    text = approval_control(
        f"/revoke {identity}",
        session,
        data_root=env[4].database.data_root,
        session_id="session",
        task_id=task,
        action_id="revoke",
    )
    assert "已撤回" in text
    assert isinstance(registry.execute_prepared_tool(prepared), ToolError)
    assert not (root / "result.txt").exists()
    assert (
        sum(
            row.event == "approval.revoked"
            for row in LedgerStore(env[4].database.data_root).read_events()
        )
        == 1
    )


def test_grant_save_failure_has_no_authorized_effect_and_retry_is_idempotent(
    authorization_env, monkeypatch
):
    """事件成功而权限发布失败时不能派发，原动作重试只补缺失载体；传参：环境和替换器；返回：无。"""
    env = authorization_env
    _, task, registry, _, messages = env
    authorizer, shown = _authorizer(
        env, ApprovalSession(), [(ApprovalDecision.TASK, "exact")]
    )
    prepared = _prepared(env, "result.txt", operation="one")
    with monkeypatch.context() as failure:
        failure.setattr(
            TaskStore,
            "append_grant",
            lambda *_: (_ for _ in ()).throw(OSError("grant publication failed")),
        )
        with pytest.raises(OSError, match="grant publication"):
            _authorize(authorizer, prepared)
    assert isinstance(registry.execute_prepared_tool(replace(prepared)), ToolError)
    approved = _authorize(authorizer, prepared)
    assert not isinstance(registry.execute_prepared_tool(approved), ToolError)
    with closing(TaskStore(messages.database.data_root)) as tasks:
        assert len(tasks.require_task(task).grants) == 1
    assert len(shown) == 1
    assert (
        len(
            [
                entry
                for entry in messages.read_entries("session")
                if entry.input_kind == "approval"
            ]
        )
        == 1
    )


def test_approval_controls_do_not_become_new_user_requirements(authorization_env):
    """授权正文可审计但不替代真正问题、不进入待处理输入；传参：环境；返回：无。"""
    env = authorization_env
    authorizer, _ = _authorizer(
        env, ApprovalSession(), [(ApprovalDecision.DENY, "exact")]
    )
    _authorize(authorizer, _prepared(env, "result.txt", operation="one"))
    messages = env[4]
    assert not messages.pending_inputs("session")
    assert (
        messages.materialize("session").messages[-1].content[0].text == "请修改指定文件"
    )
    assert any(
        entry.input_kind == "approval" for entry in messages.read_entries("session")
    )


def test_missing_grant_event_is_not_treated_as_legacy(authorization_env):
    """新授权丢失事件原件时明确失败，不能降为旧授权放行；传参：环境；返回：无。"""
    env = authorization_env
    authorizer, _ = _authorizer(
        env, ApprovalSession(), [(ApprovalDecision.TASK, "exact")]
    )
    prepared = _authorize(authorizer, _prepared(env, "result.txt", operation="one"))
    with authorizer.ledger.database.transaction() as batch:
        batch.delete("ledger", prepared.authorization.event_id)
    with pytest.raises(ApprovalUnavailable, match="missing"):
        authorizer.authorize(
            (_prepared(env, "result.txt", operation="two").approval_request,)
        )


def test_edited_grant_cannot_expand_recorded_user_decision(
    authorization_env, monkeypatch
):
    """权限载体被扩大但事件未变时拒绝执行；传参：环境和替换器；返回：无。"""
    import approval

    env = authorization_env
    authorizer, _ = _authorizer(
        env, ApprovalSession(), [(ApprovalDecision.TASK, "exact")]
    )
    prepared = _authorize(authorizer, _prepared(env, "result.txt", operation="one"))
    expanded = {
        **prepared.authorization.grant,
        "resource": {
            "kind": "directory",
            "action": "file_write",
            "target": str(env[0]).casefold(),
        },
    }
    monkeypatch.setattr(approval, "_task_grants", lambda _: [expanded])
    with pytest.raises(ApprovalUnavailable, match="recorded decision"):
        authorizer.authorize(
            (_prepared(env, "other.txt", operation="two").approval_request,)
        )


@pytest.mark.parametrize("mode", [ApprovalMode.WORKSPACE, ApprovalMode.AUTO])
def test_cron_cannot_use_session_grant_or_user_mode(authorization_env, mode):
    """无人值守运行不沿用用户会话模式或临时授权；传参：环境与用户模式；返回：无。"""
    env = authorization_env
    session = ApprovalSession()
    authorizer, shown = _authorizer(env, session, [(ApprovalDecision.SESSION, "exact")])
    _authorize(authorizer, _prepared(env, "result.txt", operation="one"))
    session.set_mode(mode)
    prepared = _prepared(env, "result.txt", operation="two")
    prepared = replace(
        prepared,
        lease=replace(prepared.lease, trigger="cron"),
        approval_request=replace(
            prepared.approval_request, lease=replace(prepared.lease, trigger="cron")
        ),
    )
    assert isinstance(
        env[2].execute_prepared_tool(_authorize(authorizer, prepared)), ToolError
    )
    assert len(shown) == 1 and not (env[0] / "result.txt").exists()


def test_revoke_failure_does_not_restore_actual_permission(
    authorization_env, monkeypatch
):
    """撤销审计失败也保持真实载体已撤回；传参：环境和替换器；返回：无。"""
    from runtime.ledger_writer import LedgerWriter

    env = authorization_env
    _, task, registry, _, _ = env
    session = ApprovalSession()
    authorizer, _ = _authorizer(env, session, [(ApprovalDecision.TASK, "exact")])
    prepared = _authorize(authorizer, _prepared(env, "result.txt", operation="one"))
    identity = str(prepared.authorization.grant["grant_id"])
    with monkeypatch.context() as failure:
        failure.setattr(
            LedgerWriter,
            "record_once",
            lambda *_: (_ for _ in ()).throw(OSError("audit unavailable")),
        )
        with pytest.raises(OSError, match="audit unavailable"):
            approval_control(
                f"/revoke {identity}",
                session,
                data_root=env[4].database.data_root,
                session_id="session",
                task_id=task,
                action_id="revoke",
            )
    assert isinstance(registry.execute_prepared_tool(prepared), ToolError)
    with closing(TaskStore(env[4].database.data_root)) as tasks:
        assert not tasks.require_task(task).grants
    approval_control(
        f"/revoke {identity}",
        session,
        data_root=env[4].database.data_root,
        session_id="session",
        task_id=task,
        action_id="revoke",
    )
    assert (
        sum(
            row.event == "approval.revoked"
            for row in LedgerStore(env[4].database.data_root).read_events()
        )
        == 1
    )


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "unknown"])
def test_incomplete_batch_never_saves_a_partial_grant(authorization_env, invalid):
    """缺项、重复和陌生编号都不能产生部分批准；传参：环境和无效回执；返回：无。"""
    env = authorization_env
    first = _prepared(env, "a.txt", operation="a")
    second = _prepared(env, "b.txt", operation="b")
    requests = tuple(
        replace(item.approval_request, batch_id="group") for item in (first, second)
    )
    identities = {
        "missing": ["a"],
        "duplicate": ["a", "a"],
        "unknown": ["a", "unknown"],
    }[invalid]
    decision = BatchDecision(
        tuple(
            ApprovalChoice(identity, ApprovalDecision.TASK) for identity in identities
        )
    )
    authorizer = BatchAuthorizer(
        ApprovalSession(),
        env[4],
        LedgerStore(env[4].database.data_root),
        prompt=lambda _: decision,
    )
    with pytest.raises(ApprovalUnavailable):
        authorizer.authorize(requests)
    assert not any(
        row.event == "approval.decided" for row in authorizer.ledger.read_events()
    )
    with closing(TaskStore(env[4].database.data_root)) as tasks:
        assert not tasks.require_task(env[1]).grants
    assert isinstance(env[2].execute_prepared_tool(first), ToolError)
    assert not (env[0] / "a.txt").exists()


@pytest.mark.parametrize("interrupt", ["cancel", "input"])
def test_interrupting_user_choice_prevents_the_entire_operation(
    authorization_env, interrupt
):
    """用户选择期间的停止或新要求使旧动作不开始；传参：环境和中断类型；返回：无。"""
    env = authorization_env
    prepared = _prepared(env, "result.txt", operation="one")

    def prompt(batch):
        """在真实决定提交窗口插入中断；传参：批次；返回：已失效的批准。"""
        if interrupt == "cancel":
            env[3].cancellation.cancel()
        else:
            env[4].accept_input("session", "改为先核对来源")
        return BatchDecision(
            (ApprovalChoice(batch.requests[0].operation_id, ApprovalDecision.ONCE),)
        )

    authorizer = BatchAuthorizer(
        ApprovalSession(), env[4], LedgerStore(env[4].database.data_root), prompt=prompt
    )
    result = env[2].execute_prepared_tool(_authorize(authorizer, prepared))
    assert (
        isinstance(result, ToolError)
        and result.details["approval_state"] == "cancelled"
    )
    assert not (env[0] / "result.txt").exists()


@pytest.mark.parametrize("scope", [ApprovalDecision.TASK, ApprovalDecision.PERMANENT])
def test_persistent_grant_reloads_without_inventing_a_new_user_choice(
    authorization_env, scope
):
    """同合同的持久权限可在新宿主复用，来源仍指原决定；传参：环境和范围；返回：无。"""
    env = authorization_env
    initial, _ = _authorizer(env, ApprovalSession(), [(scope, "exact")])
    _authorize(initial, _prepared(env, "result.txt", operation="one"))
    restarted, shown = _authorizer(env, ApprovalSession(), [])
    assert not isinstance(
        env[2].execute_prepared_tool(
            _authorize(restarted, _prepared(env, "result.txt", operation="two"))
        ),
        ToolError,
    )
    assert shown == []
    decisions = [
        row for row in restarted.ledger.read_events() if row.event == "approval.decided"
    ]
    assert (
        decisions[-1].source == "grant"
        and decisions[-1].payload["source_input_id"] is None
    )
    assert decisions[-1].payload["inherited_event_id"] == decisions[0].event_id


def test_legacy_task_grant_keeps_exact_argument_scope(authorization_env):
    """旧记录可读但不升级为目录授权，自动复用标注legacy来源；传参：环境；返回：无。"""
    env = authorization_env
    with closing(TaskStore(env[4].database.data_root)) as tasks:
        tasks.append_grant(
            env[1],
            {
                "tool": "file_write",
                "args": {"path": "result.txt", "content": "one"},
                "scope": "task",
            },
        )
    authorizer, shown = _authorizer(
        env, ApprovalSession(), [(ApprovalDecision.DENY, "exact")]
    )
    prepared = _authorize(authorizer, _prepared(env, "result.txt", operation="one"))
    assert prepared.authorization.source == "legacy"
    assert not isinstance(env[2].execute_prepared_tool(prepared), ToolError)
    assert isinstance(
        env[2].execute_prepared_tool(
            _authorize(authorizer, _prepared(env, "other.txt", operation="two"))
        ),
        ToolError,
    )
    assert len(shown) == 1


@pytest.mark.parametrize("protected", [False, True])
def test_auto_policy_does_not_impersonate_user_or_skip_protected_choice(
    authorization_env, protected
):
    """自动策略不伪造用户来源，受保护替换标记始终取得新选择；传参：环境和受保护标记；返回：无。"""
    from approval.batch import intent_digest

    env = authorization_env
    session = ApprovalSession()
    session.set_mode(ApprovalMode.AUTO)
    authorizer, shown = _authorizer(env, session, [(ApprovalDecision.ONCE, "exact")])
    prepared = _prepared(env, "result.txt", operation="one")
    request = replace(prepared.approval_request, force_confirmation=protected)
    prepared = replace(
        prepared,
        approval_request=replace(request, intent_digest=intent_digest(request)),
    )
    approved = _authorize(authorizer, prepared)
    assert not isinstance(env[2].execute_prepared_tool(approved), ToolError)
    assert len(shown) == int(protected)
    event = next(
        row
        for row in authorizer.ledger.read_events()
        if row.event_id == approved.authorization.event_id
    )
    assert event.source == ("user_action" if protected else "policy")
    assert bool(event.payload["source_input_id"]) is protected


def test_revoke_audit_retry_cannot_target_another_goal(authorization_env):
    """补撤销审计也不能越过当前可管理目标；传参：授权环境；返回：无。"""
    env = authorization_env
    session = ApprovalSession()
    authorizer, _ = _authorizer(env, session, [(ApprovalDecision.TASK, "exact")])
    prepared = _authorize(authorizer, _prepared(env, "result.txt", operation="one"))
    identity = prepared.authorization.grant["grant_id"]
    with closing(TaskStore(env[4].database.data_root)) as tasks:
        other = tasks.create_task("另一项目")
    with pytest.raises(ValueError, match="可管理"):
        approval_control(
            f"/revoke {identity}",
            session,
            data_root=env[4].database.data_root,
            session_id="session",
            task_id=other.task_id,
        )
    assert not any(
        row.event == "approval.revoked"
        for row in LedgerStore(env[4].database.data_root).read_events()
    )
    assert not session.is_revoked(identity)


def test_approval_input_cannot_be_replayed_as_an_ordinary_request(authorization_env):
    """授权选择只能作为授权来源，不能伪装当前新要求；传参：授权环境；返回：无。"""
    from triggers.user import make_run_context

    env = authorization_env
    env[4].accept_input("session", "一次批准", input_id="choice", input_kind="approval")
    with pytest.raises(ValueError, match="identity"):
        make_run_context(
            "一次批准",
            data_root=env[4].database.data_root,
            task_id=env[1],
            session_id="session",
            input_id="choice",
        )
