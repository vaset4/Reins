"""【项目记忆】【工作区隔离】同名目录仍按持久身份决定适用范围。

作者：xxx
时间：2026-09-30 20:00:00
"""

import pytest

from context.engine import recall_context_body
from context.memory_recall import recall_memories
from memory.records import MemoryDetails, MemorySource
from memory.store import MemoryStore
from runtime.workspaces import WorkspaceStore
from tests.test_memory_native_actions import run_action


def test_same_name_workspaces_keep_project_memory_in_original_requests(
    tmp_path, monkeypatch
):
    """验证项目保存、跨会话召回与其他同名目录隔离；参数：临时根及替换器；返回：无。"""
    data, first, second = (
        tmp_path / "data",
        tmp_path / "one/project",
        tmp_path / "two/project",
    )
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    _, operations = run_action(
        data,
        monkeypatch,
        session="first",
        project=first,
        message="请记住这个项目的代号是蓝鲸，不适用于其他项目",
        arguments={
            "action": "create",
            "kind": "fact",
            "content": "这个项目的代号是蓝鲸",
            "memory_scope": "project",
            "subject": "项目",
            "fact_key": "代号",
            "tags": ["项目", "代号"],
        },
    )
    record = operations[0]["result"]["meta"]["record"]
    original = WorkspaceStore(data).for_session("first")
    assert record["details"]["scope"] == f"project:{original.workspace_id}"
    same_requests, _ = run_action(
        data, monkeypatch, session="same", project=first, message="这个项目的代号是什么"
    )
    other_requests, _ = run_action(
        data,
        monkeypatch,
        session="other",
        project=second,
        message="这个项目的代号是什么",
    )
    assert "蓝鲸" in str(same_requests[0])
    assert "蓝鲸" not in str(other_requests[0])
    assert (
        WorkspaceStore(data).for_session("other").workspace_id != original.workspace_id
    )
    # 【项目记忆】【范围查询】1. 查询当前项目简称复用同一持久身份
    _, queries = run_action(
        data,
        monkeypatch,
        session="same",
        project=first,
        message="查询本项目记忆",
        tool_name="memory_query",
        arguments={"action": "list", "memory_scope": "project"},
    )
    assert (
        queries[-1]["result"]["meta"]["records"][0]["memory_id"] == record["memory_id"]
    )
    # 【项目记忆】【标签隔离】2. 任务标签不能将其他目录的事实引入当前请求
    body = recall_context_body(
        data,
        task_summary="项目代号",
        task_tags=[record["details"]["scope"]],
        skill_refs=[],
        session_id="other",
    )
    assert "蓝鲸" not in body


def test_project_memory_write_rejects_another_workspace_identity(tmp_path, monkeypatch):
    """模型指定他处工作区不会改写知识适用范围；参数：目录及替换器；返回：无。"""
    data = tmp_path / "data"
    original = WorkspaceStore(data).bind_session("elsewhere", tmp_path / "elsewhere")
    _, operations = run_action(
        data,
        monkeypatch,
        session="writer",
        project=tmp_path,
        message="保存当前项目代号",
        arguments={
            "action": "create",
            "kind": "fact",
            "content": "项目代号是蓝鲸",
            "memory_scope": f"project:{original.workspace_id}",
            "subject": "项目",
            "fact_key": "代号",
            "tags": [],
        },
    )
    assert operations[0]["result"]["status"] == "error"
    assert "this session workspace" in operations[0]["result"]["error"]
    assert MemoryStore(data).list_memories() == []


@pytest.mark.parametrize("bind_first", [False, True])
def test_session_memory_keeps_published_location_when_workspace_is_bound(
    tmp_path, bind_first
):
    """先记忆或先绑定均保留原稿位置、版本和会话适用范围；参数：目录及创建顺序；返回：无。"""
    data, project = tmp_path / "data", tmp_path / "project"
    project.mkdir()
    workspaces = WorkspaceStore(data)
    if bind_first:
        workspaces.bind_session("chat", project)
    store = MemoryStore(data)
    identity = store.create_memory(
        "fact", "服务端口8000", ["端口"], details=MemoryDetails(scope="session:chat")
    )
    original, path = store.load_memory(identity), store.current_path(identity)
    workspaces.bind_session("chat", project)

    # 1. 【会话记忆】【首次绑定】工作区确定后，已发布原稿保持原身份和存放位置
    reopened = MemoryStore(data)
    assert reopened.load_memory(identity) == original
    assert reopened.current_path(identity) == path
    assert (
        recall_memories(
            data, task_summary="端口", task_tags=[], scopes=["session:chat"]
        )[0].memory.version
        == original.version
    )
    assert (
        recall_memories(
            data, task_summary="端口", task_tags=[], scopes=["session:other"]
        )
        == []
    )

    # 2. 【会话记忆】【后续修订】相同会话的修订留在原目录，历史与作用域保持可追溯
    updated = reopened.revise_memory(
        identity,
        "服务端口9000",
        expected_version=original.version,
        reason="用户更正端口",
        sources=(MemorySource("user_input", "input-2"),),
    )
    assert reopened.current_path(identity) == path
    assert updated.details.scope == "session:chat"
    assert (
        reopened.load_memory(identity, version=original.version).content
        == original.content
    )
    assert reopened.load_memory(identity).content == "服务端口9000"
