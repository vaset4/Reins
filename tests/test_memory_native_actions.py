"""从生产模型请求验证跨会话更正、作用域与出处边界。

作者：xxx
时间：2026-09-15 05:40:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_native_tool_then_final, from_test_stub

from contextlib import closing
from uuid import uuid4

from context.engine import recall_context_body
from llm.messages import ToolCallPart
from memory.records import MemoryDetails
from memory.store import MemoryStore
from runtime.agent_loop import AgentLoop
from runtime.lease import from_trigger
from runtime.persistence import RuntimeStore
from runtime.session_messages import append_user_message
from runtime.tool_operations import ToolOperationStore
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tests.test_session_runtime import capture_requests
from tools.knowledge_tools import register_knowledge_tools
from tools.tool_registry import ToolRegistry


def run_action(
    root,
    monkeypatch,
    *,
    session,
    message,
    arguments=None,
    tool_name="memory_manage",
    registry=None,
    project=None,
):
    """让原生动作经过完整模型/执行边界；传参：存储、替换器及会话输入；返回：实际请求与操作。"""
    registry = registry or ToolRegistry()
    register_knowledge_tools(registry)
    if arguments is not None:
        definition = registry.get(tool_name)
        definition.deferred = False
        registry.publish([definition], replace_names=(tool_name,))
    client = (
        from_test_native_tool_then_final(
            [ToolCallPart(f"knowledge-call-{uuid4().hex}", tool_name, arguments)],
            "已处理",
        )
        if arguments is not None
        else from_test_stub('{"type":"final","final_output":"查询完成"}')
    )
    requests = capture_requests(client, monkeypatch)
    with closing(TaskStore(root)) as store:
        task = store.create_task(message, is_inbox=True)
    workspace = WorkspaceStore(root).bind_session(session, project or root)
    context = RunContext(
        session_id=session,
        compatibility_task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": message},
        capability_lease=from_trigger(
            "user",
            task_id=task.task_id,
            capabilities={
                "fs": {
                    "project_root": str(workspace.project_root),
                    "read": [str(workspace.project_root)],
                    "write": [str(workspace.project_root)],
                }
            },
        ),
    )
    append_user_message(root, session, message)
    list(AgentLoop(root, llm_client=client, tool_registry=registry).run_stream(context))
    return requests, ToolOperationStore(root).for_session(session)


def test_correction_is_used_in_new_session_actual_request(tmp_path, monkeypatch):
    """第三会话没有更正对话，实际请求仍采用新版并可追到第二会话；传参：目录/替换器；返回：无。"""
    _, first_ops = run_action(
        tmp_path,
        monkeypatch,
        session="first",
        message="请记住客户甲的预算是800元",
        arguments={
            "action": "create",
            "kind": "fact",
            "content": "客户甲的预算是800元",
            "memory_scope": "global",
            "subject": "客户甲",
            "fact_key": "预算",
            "tags": ["预算"],
        },
    )
    first = first_ops[0]["result"]["meta"]["record"]
    _, correction_ops = run_action(
        tmp_path,
        monkeypatch,
        session="correction",
        message="客户甲的预算改为600元，请更正",
        arguments={
            "action": "revise",
            "memory_id": first["memory_id"],
            "expected_version": first["version"],
            "content": "客户甲的预算是600元",
            "reason": "用户更正预算",
        },
    )
    corrected = correction_ops[0]["result"]["meta"]["record"]
    requests, _ = run_action(
        tmp_path, monkeypatch, session="new-session", message="客户甲的预算是多少"
    )
    assert "客户甲的预算是600元" in str(requests[0])
    assert "客户甲的预算是800元" not in str(requests[0])
    assert corrected["version"] in str(requests[0])
    assert corrected["details"]["sources"][0]["session_id"] == "correction"
    assert corrected["details"]["sources"][0]["kind"] == "user_input"
    with closing(MemoryStore(tmp_path)) as store:
        assert (
            store.load_memory(first["memory_id"], version=first["version"]).content
            == "客户甲的预算是800元"
        )


def test_source_ids_cannot_fabricate_user_confirmation(tmp_path, monkeypatch):
    """模型编造或引用分支外输入不能成为用户来源；传参：目录与替换器；返回：无。"""
    requests, operations = run_action(
        tmp_path,
        monkeypatch,
        session="sources",
        message="调查客户甲预算",
        arguments={
            "action": "create",
            "kind": "fact",
            "content": "用户已确认预算900元",
            "memory_scope": "global",
            "subject": "客户甲",
            "fact_key": "预算",
            "source_mode": "user_inputs",
            "source_input_ids": ["fabricated-input"],
        },
    )
    assert operations[0]["result"]["status"] == "error"
    assert "delivered user input" in str(requests[-1])
    with closing(MemoryStore(tmp_path)) as store:
        assert store.list_memories() == []


def test_scope_and_explicit_expiry_control_automatic_recall(tmp_path):
    """同名不同项目不混用，过期仅来自明确时间，便签不泄漏到其他会话；传参：目录；返回：无。"""
    workspace = WorkspaceStore(tmp_path).bind_session("current", tmp_path / "甲")
    other = WorkspaceStore(tmp_path).bind_session("other", tmp_path / "乙")
    with closing(MemoryStore(tmp_path)) as store:
        store.create_memory(
            "fact",
            "甲项目王先生预算500元",
            ["预算"],
            details=MemoryDetails(
                scope=f"project:{workspace.workspace_id}",
                subject="王先生",
                fact_key="预算",
            ),
        )
        store.create_memory(
            "fact",
            "乙项目王先生预算900元",
            ["预算"],
            details=MemoryDetails(
                scope=f"project:{other.workspace_id}", subject="王先生", fact_key="预算"
            ),
        )
        store.create_memory(
            "fact",
            "限时活动折扣",
            ["预算"],
            details=MemoryDetails(expires_at="2020-01-01T00:00:00+00:00"),
        )
        store.create_memory(
            "fact",
            "当前会话临时便签",
            ["预算"],
            details=MemoryDetails(kind="note", scope="session:prior"),
        )
    body = recall_context_body(
        tmp_path,
        task_summary="王先生预算",
        task_tags=["预算", "project:甲"],
        skill_refs=[],
        session_id="current",
    )
    assert "甲项目王先生预算500元" in body
    assert "乙项目王先生预算900元" not in body
    assert "限时活动折扣" not in body and "当前会话临时便签" not in body


def test_missing_index_rows_rebuild_before_actual_model_request(tmp_path, monkeypatch):
    """派生缺行在请求前由原件重建，原文继续可用；传参：目录/替换器；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        store.create_memory("fact", "原件有据可查", [])
        with RuntimeStore(tmp_path).index_connection() as connection:
            connection.execute("DELETE FROM memories")
    requests, _ = run_action(
        tmp_path, monkeypatch, session="index-fault", message="查询原件"
    )
    assert "原件有据可查" in str(requests[0])
    assert MemoryStore(tmp_path).index_status().state == "current"


def test_correcting_applicability_and_exact_fields_changes_future_queries(
    tmp_path, monkeypatch
):
    """误写的全局事实可修订范围与精确字段，其他项目不再召回它；传参：目录/替换器；返回：无。"""
    with closing(MemoryStore(tmp_path)) as store:
        identity = store.create_memory(
            "fact",
            "林先生预算800元",
            ["预算"],
            details=MemoryDetails(
                subject="林先生", fact_key="预算", exact_fields={"预算": "800"}
            ),
        )
        old = store.load_memory(identity)
    _, operations = run_action(
        tmp_path,
        monkeypatch,
        session="scope-correction",
        message="更正：620元预算只适用于星河项目",
        arguments={
            "action": "revise",
            "memory_id": identity,
            "expected_version": old.version,
            "content": "林先生预算620元",
            "reason": "用户更正预算和适用范围",
            "memory_scope": "project",
            "exact_fields": {"预算": "620"},
            "tags": ["星河"],
        },
    )
    assert operations[0]["result"]["status"] == "ok"
    with closing(MemoryStore(tmp_path)) as store:
        scope = f"project:{WorkspaceStore(tmp_path).for_session('scope-correction').workspace_id}"
        assert (
            store.list_memories(exact_fields={"预算": "620"})[0].details.scope == scope
        )
        assert store.list_memories(exact_fields={"预算": "800"}) == []
        assert (
            store.load_memory(identity, version=old.version).details.scope == "global"
        )
    other = recall_context_body(
        tmp_path, task_summary="林先生预算", task_tags=["project:蓝岸"], skill_refs=[]
    )
    assert "林先生预算" not in other


def test_archival_original_is_readable_from_a_different_session(tmp_path, monkeypatch):
    """档案索引跨会话可补读真实操作原件，而非只有一个无法兑现的引用；传参：目录/替换器；返回：无。"""
    from tools.builtin_tools import build_tool_registry

    data = tmp_path / "data"
    (tmp_path / "contract.txt").write_text(
        "合同编号A-2026-018，原始条款：交付后30日付款", encoding="utf-8"
    )
    registry = build_tool_registry(repo_root=tmp_path, data_root=data)
    _, read_ops = run_action(
        data,
        monkeypatch,
        project=tmp_path,
        session="archive-origin",
        message="读取合同",
        registry=registry,
        tool_name="file_read",
        arguments={"path": "contract.txt"},
    )
    source_id = read_ops[0]["operation_id"]
    _, operations = run_action(
        data,
        monkeypatch,
        project=tmp_path,
        session="archive-origin",
        message="保留刚才合同的原件引用",
        arguments={
            "action": "create",
            "kind": "archive",
            "content": "合同A-2026-018的付款条款",
            "memory_scope": "project",
            "subject": "合同A-2026-018",
            "fact_key": "付款条款",
            "exact_fields": {"合同编号": "A-2026-018"},
            "source_mode": "tool_results",
            "source_operation_ids": [source_id],
        },
    )
    archive = next(
        row["result"]["meta"]["record"]
        for row in operations
        if row["call"]["tool_name"] == "memory_manage"
    )
    requests, new_ops = run_action(
        data,
        monkeypatch,
        project=tmp_path,
        session="archive-reader",
        message="查看这条档案的原件",
        tool_name="memory_query",
        arguments={"action": "sources", "memory_id": archive["memory_id"]},
    )
    assert new_ops[0]["result"]["status"] == "ok"
    assert "交付后30日付款" in str(requests[-1])
    assert archive["details"]["archive_ref"] == f"operation:{source_id}"
