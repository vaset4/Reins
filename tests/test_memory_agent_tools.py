"""【记忆工具】【统一入口】保持搜索、归档、恢复和便签真实行为。

作者：xxx
时间：2026-10-02 16:20:00
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from uuid import uuid4

import pytest

from memory.store import MemoryStore
from runtime.memory_actions import MemoryActions
from runtime.native_actions import NativeActionContext
from runtime.session_message_store import SessionMessageStore
from runtime.tool_operations import ToolOperation
from runtime.types import RunToolsRequest
from tasks.store import TaskStore
from tests.test_tool_batch_execution import make_run
from tools.builtin_tools import build_tool_registry
from tools.types import ToolError


@pytest.fixture
def memory_runtime(tmp_path):
    """为真实存储和原生动作注入当前用户出处；参数：根；返回：目录及执行服务。"""
    with (
        closing(
            build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
        ) as registry,
        closing(TaskStore(tmp_path)) as tasks,
    ):
        loop, run, _ = make_run(tmp_path, registry, [])
        services = NativeActionContext(
            run,
            tasks,
            SessionMessageStore(tmp_path),
            loop.session_states,
            loop.operations,
            loop.run_facts,
            registry,
        )
        yield registry, MemoryActions(services, data_root=tmp_path)


def invoke_memory(runtime, name, arguments):
    """校验当前schema后执行真实知识动作；参数：运行、工具、参数；返回：实际结果。"""
    registry, actions = runtime
    validated = registry.validate_model_request(tool_name=name, arguments=arguments)
    if isinstance(validated, ToolError):
        return validated
    identity = uuid4().hex
    return actions.execute(
        ToolOperation(
            RunToolsRequest(name, arguments=arguments),
            identity,
            name,
            arguments,
            operation_id=identity,
        )
    )


def test_registry_exposes_only_canonical_memory_actions(memory_runtime):
    """新目录只有正式查询、修改和便签，旧入口退出；参数：运行；返回：无。"""
    registry, _ = memory_runtime
    assert registry.get("memory_search") is registry.get("memory_archive") is None
    assert registry.get("memory_query").readonly
    assert not registry.get("memory_manage").readonly
    assert not registry.get("memory_note").readonly


def test_search_archive_restore_preserves_version_sources_and_recall(
    tmp_path, memory_runtime
):
    """搜到真实身份后归档再恢复，归档可查但不自动召回，历史及出处保留；参数：根、运行；返回：无。"""
    from context.memory_recall import recall_memories

    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory(
            "fact", "remember archive boundary rsync", ["memory"]
        )
        original = store.load_memory(identity)
    search = invoke_memory(
        memory_runtime,
        "memory_query",
        {"action": "search", "query": "archive boundary"},
    )
    assert search.status == "ok"
    found = next(row for row in search.meta["records"] if row["memory_id"] == identity)
    assert found["version"] == original.version
    archived = invoke_memory(
        memory_runtime,
        "memory_manage",
        {
            "action": "archive",
            "memory_id": identity,
            "expected_version": found["version"],
            "reason": "用户要求收起旧方法",
        },
    )
    assert archived.status == "ok" and archived.meta["record"]["state"] == "archived"
    assert identity not in {
        row.memory.memory_id
        for row in recall_memories(
            tmp_path, task_summary="archive boundary", task_tags=[]
        )
    }
    searched = invoke_memory(
        memory_runtime,
        "memory_query",
        {"action": "search", "query": "archive boundary"},
    )
    found = next(
        row for row in searched.meta["records"] if row["memory_id"] == identity
    )
    assert found["state"] == "archived"
    restored = invoke_memory(
        memory_runtime,
        "memory_manage",
        {
            "action": "restore",
            "memory_id": identity,
            "expected_version": found["version"],
            "reason": "用户要求恢复方法",
        },
    )
    assert restored.status == "ok" and restored.meta["record"]["state"] == "active"
    with closing(MemoryStore(tmp_path)) as store:
        current = store.load_memory(identity)
        assert (
            current.details.sources and current.details.sources[-1].kind == "user_input"
        )
        assert (
            store.load_memory(identity, version=original.version).content
            == original.content
        )


@pytest.mark.parametrize(
    ("state", "action", "error"),
    [
        ("draft", "archive", "draft->archived"),
        ("draft", "restore", "only an archived"),
        ("active", "restore", "only an archived"),
    ],
)
def test_invalid_state_changes_preserve_original(
    tmp_path, memory_runtime, state, action, error
):
    """草稿不能获批或归档，启用记录不能假恢复；参数：根、动作及原状态；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory(
            "fact", "state boundary", ["memory"], state=state
        )
        version = store.load_memory(identity).version
    result = invoke_memory(
        memory_runtime,
        "memory_manage",
        {
            "action": action,
            "memory_id": identity,
            "expected_version": version,
            "reason": "状态核对",
        },
    )
    assert result.status == "error" and error in result.error
    with closing(MemoryStore(tmp_path)) as store:
        assert store.load_memory(identity).state == state


def test_archive_requires_explicit_current_version_and_reports_unknown_id(
    tmp_path, memory_runtime
):
    """缺版本、旧版本和未知身份均明确拒绝，不能静默改最新记录；参数：根、运行；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory("fact", "version boundary", ["memory"])
        version = store.load_memory(identity).version
    args = {"action": "archive", "memory_id": identity, "reason": "归档"}
    assert isinstance(invoke_memory(memory_runtime, "memory_manage", args), ToolError)
    result = invoke_memory(
        memory_runtime, "memory_manage", {**args, "expected_version": "old-version"}
    )
    assert result.status == "error"
    unknown = invoke_memory(
        memory_runtime,
        "memory_manage",
        {**args, "memory_id": "no-such-memory", "expected_version": version},
    )
    assert unknown.status == "error" and "no-such-memory" in unknown.error
    with closing(MemoryStore(tmp_path)) as store:
        assert store.load_memory(identity).version == version


def test_agent_tools_refuse_without_an_active_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证已获授权的协作请求仍须真实运行上下文；参数：隔离目录与替换器；返回：无。"""
    from approval import ApprovalDecision
    from runtime.lease import Lease
    from tests.support.approval import install_approval
    from tools.tool_registry import PreparedToolExecution
    from tools.types import ToolErrorCategory

    # 1. 明确授权此次调用，保证拒绝原因来自缺少运行而非审批或参数错误
    install_approval(monkeypatch, lambda _request: ApprovalDecision.ONCE)
    with closing(TaskStore(tmp_path)) as tasks:
        tasks.create_task("没有运行的协作请求", task_id="no-active-run")
    lease = Lease(task_id="no-active-run")
    with closing(
        build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    ) as registry:
        for name, arguments in (
            ("ask_user", {"question": "Which file should I update?"}),
            ("delegate", {"name": "inspector", "task": "inspect repository state"}),
        ):
            prepared = registry.prepare_tool_execution(name, arguments, lease)
            assert isinstance(prepared, PreparedToolExecution), name
            result = registry.execute_prepared_tool(prepared)
            assert isinstance(result, ToolError), name
            assert result.category is ToolErrorCategory.INVALID_INPUT, name
            assert "requires an active session runtime" in result.message, name
            assert result.retryable is False, name


def test_memory_note_second_identical_write_is_skipped_as_duplicate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # AC7：memory_note 连写两遍相同 note，第二遍经 write_memory 撞冲突被拒，
    # 回执 committed=false 并指向已有那条（端到端过真实运行边界，
    # 证明冲突收口在 write_memory 后上游受益）
    note = "记住 D:/repo 的边界约束"
    arguments = {"note": note, "scope": "memory"}
    # memory_note 是 CONFIRM 工具，这次测的是写入口去重而不是审批闸门，先给出授权
    import approval

    from tests.support.approval import install_approval

    install_approval(monkeypatch, lambda _request: approval.ApprovalDecision.ONCE)

    first = _native_result(
        tmp_path,
        monkeypatch,
        session="note-dup",
        tool_name="memory_note",
        arguments=arguments,
    )
    second = _native_result(
        tmp_path,
        monkeypatch,
        session="note-dup",
        tool_name="memory_note",
        arguments=arguments,
    )

    first_payload = json.loads(str(first["content"]))
    assert first["status"] == "ok"
    assert first_payload["committed"] is True
    second_payload = json.loads(str(second["content"]))
    assert second["status"] == "ok"
    assert second_payload["committed"] is False
    assert second_payload["duplicate_of"] == first_payload["record"]["memory_id"]


def _native_result(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    session: str,
    tool_name: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    """让记忆工具经真实运行边界执行一次；传参：数据根/替换器/会话与调用；返回：结果字典。"""
    from runtime.tool_operations import ToolOperationStore
    from tests.test_memory_native_actions import run_action
    from tools.builtin_tools import build_tool_registry

    # run_action 交回的是整个 session 的操作，同一 session 连调两次会拿到上一轮那条，
    # 所以先记下已有身份，只认这次新增的那一条
    before = {
        row["operation_id"] for row in ToolOperationStore(root).for_session(session)
    }
    _, operations = run_action(
        root,
        monkeypatch,
        session=session,
        message="记忆维护",
        registry=build_tool_registry(repo_root=root, data_root=root),
        tool_name=tool_name,
        arguments=arguments,
    )
    fresh = [
        item
        for item in operations
        if item["operation_id"] not in before and item["call"]["tool_name"] == tool_name
    ]
    assert len(fresh) == 1, fresh
    return fresh[0]["result"]
