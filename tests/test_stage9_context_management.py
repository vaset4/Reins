"""【上下文】【后台查看验证】控制、分页和记忆差异沿真实原件验证。

作者：xxx
时间：2026-10-01 12:40:00
"""

from __future__ import annotations

import json
from contextlib import closing
from threading import Thread
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest

from app.background.context_management import ContextManagement
from app.background.server import BackgroundServer
from memory.records import MemoryDetails, MemorySource
from memory.store import MemoryStore
from runtime.knowledge_maintenance import KnowledgeMaintenance
from runtime.lease import from_trigger
from runtime.session_messages import append_user_message
from runtime.tool_operations import ToolOperationStore
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from scripts.testing.llm import from_test_turns


def context_work(root, *, session_id="context-session"):
    """以真实新输入接纳隔离维护工作；参数：数据根和会话；返回：服务、归属和工作。"""
    WorkspaceStore(root).bind_session(session_id, root)
    identity = append_user_message(root, session_id, "请记住项目只在本地运行")
    context = RunContext(
        trigger=Trigger.USER,
        session_id=session_id,
        run_id="source-run",
        payload={"input_message_id": identity},
        capability_lease=from_trigger(
            "user",
            capabilities={
                "background_run": {"enabled": True},
                "fs": {"read": [str(root)], "project_root": str(root)},
            },
        ),
    )
    manager = ContextManagement(root)
    row = manager.knowledge.observe(context, client=from_test_turns([]))
    assert row is not None
    owner = {"session_id": session_id, "data_space_id": manager.database.data_space_id}
    return manager, owner, row


def test_settings_reads_do_not_enable_history_backfill_and_controls_persist(tmp_path):
    """查看无启用副作用，两个开关独立持久；参数：隔离空间；返回：无。"""
    WorkspaceStore(tmp_path).bind_session("view", tmp_path)
    manager = ContextManagement(tmp_path)
    owner = {"session_id": "view", "data_space_id": manager.database.data_space_id}
    with manager.database.snapshot() as snapshot:
        before = snapshot.sequence
    view = manager.query({**owner, "action": "overview"})
    assert view["history"]["enabled"] and view["knowledge"]["enabled"]
    assert manager.knowledge.status()["boundary"] is None
    with manager.database.snapshot() as snapshot:
        assert snapshot.sequence == before
    manager.query(
        {**owner, "action": "configure", "domain": "history", "enabled": False}
    )
    manager.query(
        {**owner, "action": "configure", "domain": "knowledge", "enabled": False}
    )
    restarted = ContextManagement(tmp_path).query({**owner, "action": "overview"})
    assert not restarted["history"]["enabled"] and not restarted["knowledge"]["enabled"]
    with pytest.raises(ValueError, match="数据空间"):
        manager.query(
            {
                **owner,
                "data_space_id": "old-space",
                "action": "configure",
                "domain": "knowledge",
                "enabled": True,
            }
        )


def test_work_control_checks_original_session_and_frozen_detail(tmp_path):
    """旧详情保持原快照，跨会话不能取消；参数：隔离空间；返回：无。"""
    manager, owner, row = context_work(tmp_path)
    page = manager.query({**owner, "action": "list", "limit": 1})
    assert page["total"] == 1 and page["items"][0]["work_id"] == row["work_id"]
    selection = {
        **owner,
        "domain": "knowledge",
        "work_id": row["work_id"],
        "action": "detail",
    }
    old = manager.query(selection)
    assert "等待执行" in old["text"]
    manager.query({**selection, "action": "cancel"})
    frozen = manager.query({**selection, "commit": old["commit"]})
    assert frozen["text"] == old["text"]
    assert "已取消" in manager.query(selection)["text"]
    WorkspaceStore(tmp_path).bind_session("another", tmp_path)
    with pytest.raises(ValueError, match="selected session"):
        manager.query({**selection, "session_id": "another", "action": "cancel"})


def test_memory_detail_reads_committed_revision_after_later_user_edit(tmp_path):
    """查看固定已提交修改前后版本，不用后续内容替代；参数：隔离空间；返回：无。"""
    manager, owner, row = context_work(tmp_path)
    workspace = WorkspaceStore(tmp_path).for_session(owner["session_id"])
    source = MemorySource(
        "user_input", reference="original", session_id=owner["session_id"]
    )
    with closing(MemoryStore(tmp_path)) as memories:
        identity = memories.create_memory(
            "rule",
            "端口使用8000",
            [],
            details=MemoryDetails(
                scope=f"project:{workspace.workspace_id}", sources=(source,)
            ),
        )
        old = memories.load_memory(identity)
        memories.revise_memory(
            identity,
            "端口改为9000",
            expected_version=old.version,
            reason="用户明确更正端口",
            sources=(source,),
        )
        revised = memories.load_memory(identity)
        manager.knowledge.update(
            row["work_id"],
            state="completed",
            commits=[
                {
                    "memory_id": identity,
                    "version": revised.version,
                    "operation_id": "saved-change",
                }
            ],
        )
        memories.revise_memory(
            identity,
            "端口再次改为9100",
            expected_version=revised.version,
            reason="之后的用户更正",
            sources=(source,),
        )
    detail = manager.query(
        {**owner, "action": "detail", "domain": "knowledge", "work_id": row["work_id"]}
    )
    assert "-端口使用8000" in detail["text"] and "+端口改为9000" in detail["text"]
    assert "9100" not in detail["text"] and "用户明确更正端口" in detail["text"]


def test_authenticated_rpc_reaches_persistent_maintenance_controls(tmp_path):
    """真实HTTP入口查询与控制同一领域状态；参数：隔离空间；返回：无。"""
    manager, owner, _ = context_work(tmp_path)
    service = SimpleNamespace(
        services=SimpleNamespace(data_root=tmp_path),
        workspaces=WorkspaceStore(tmp_path),
    )
    server = BackgroundServer(service, "fixture-context-token")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = {
            "instance": server.instance,
            "data_space_id": server.data_space_id,
            "method": "context_management",
            "params": {
                "payload": {
                    **owner,
                    "action": "configure",
                    "domain": "knowledge",
                    "enabled": False,
                }
            },
        }
        request = Request(
            f"http://127.0.0.1:{server.server_port}/rpc",
            json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer fixture-context-token",
                "Content-Type": "application/json",
            },
        )
        with urlopen(request, timeout=5) as response:
            result = json.load(response)
        assert not result["knowledge"]["enabled"]
        assert not KnowledgeMaintenance(tmp_path).status()["boundary"]["enabled"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_partial_knowledge_commit_is_visible_without_claiming_work_completed(
    tmp_path, state
):
    """失败或取消前实际保存的版本仍可查看，固定详情不混入后续写入；参数：隔离空间/状态；返回：无。"""
    manager, owner, row = context_work(tmp_path)
    WorkspaceStore(tmp_path).bind_session("worker", tmp_path)
    with closing(MemoryStore(tmp_path)) as memories:
        identity = memories.create_memory(
            "fact",
            "已保存的部分知识",
            [],
            details=MemoryDetails(scope=f"session:{owner['session_id']}"),
        )
        saved = memories.load_memory(identity)
    manager.knowledge.update(
        row["work_id"],
        state=state,
        worker_session_id="worker",
        error="unfinished review",
    )
    operations = ToolOperationStore(tmp_path)
    operations.write(
        {
            "session_id": "worker",
            "run_id": "worker-run",
            "operation_id": "partial-save",
        },
        {
            "state": "completed",
            "call": {"tool_name": "memory_manage", "args": {"memory_id": identity}},
            "result": {
                "status": "ok",
                "meta": {
                    "committed": True,
                    "record": {"memory_id": identity, "version": saved.version},
                },
            },
        },
    )
    selection = {
        **owner,
        "action": "detail",
        "domain": "knowledge",
        "work_id": row["work_id"],
    }
    detail = manager.query(selection)
    assert detail["work"]["state"] == state
    assert "已保存的部分知识" in detail["text"] and saved.version in detail["text"]
    with closing(MemoryStore(tmp_path)) as memories:
        later = memories.revise_memory(
            identity,
            "后续新知识",
            expected_version=saved.version,
            reason="后续更正",
            sources=(MemorySource("user_input", "later"),),
        )
    operations.write(
        {"session_id": "worker", "run_id": "worker-run", "operation_id": "later-save"},
        {
            "state": "completed",
            "call": {"tool_name": "memory_manage", "args": {"memory_id": identity}},
            "result": {
                "status": "ok",
                "meta": {
                    "committed": True,
                    "record": {"memory_id": identity, "version": later.version},
                },
            },
        },
    )
    assert (
        manager.query({**selection, "commit": detail["commit"]})["text"]
        == detail["text"]
    )
