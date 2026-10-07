"""验证实际知识查询的来源有效性与自动维护范围。

作者：xxx
时间：2026-10-01 18:00:00
"""

from contextlib import closing
import hashlib

import pytest

from context.memory_recall import recall_memories_with_outcome
from memory.records import MemoryDetails, MemoryReplacement, MemorySource
from memory.store import MemoryStore
from runtime.lease import from_trigger
from runtime.memory_actions import MemoryActions
from runtime.native_actions import NativeActionContext
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.session_state import SessionStateStore
from runtime.tool_operations import ToolOperation, ToolOperationStore
from runtime.types import RunContext, RunToolsRequest, Trigger
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry


@pytest.fixture
def query_context(tmp_path):
    """装配真实查询依赖且不启动模型；参数：隔离目录；返回：动作上下文。"""
    WorkspaceStore(tmp_path).bind_session("source", tmp_path)
    run = RunContext(
        session_id="source",
        trigger=Trigger.USER,
        payload={},
        capability_lease=from_trigger("user", task_id="query"),
    )
    with closing(TaskStore(tmp_path)) as tasks, closing(ToolRegistry()) as registry:
        yield NativeActionContext(
            run,
            tasks,
            SessionMessageStore(tmp_path),
            SessionStateStore(tmp_path),
            ToolOperationStore(tmp_path),
            RunFactStore(tmp_path),
            registry,
        )


def referenced_memory(root, *, memory_type="fact"):
    """保存可核对的文件观察与记忆；参数：目录/知识类型；返回：原记忆和真实文件。"""
    path = root / "port.txt"
    path.write_text("port=8000", encoding="utf-8")
    source = MemorySource(
        "tool_result", "read-port", session_id="source", run_id="source-run"
    )
    ToolOperationStore(root).write(
        {"session_id": "source", "run_id": "source-run", "operation_id": "read-port"},
        {
            "state": "completed",
            "call": {
                "tool_name": "file_read",
                "call_id": "read-port",
                "args": {"path": str(path)},
            },
            "result": {
                "status": "ok",
                "meta": {
                    "resolved_path": str(path),
                    "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
            },
        },
    )
    sources = (
        (source,)
        if memory_type == "fact"
        else (MemorySource("user_input", "port-rule"), source)
    )
    scope = f"project:{WorkspaceStore(root).for_session('source').workspace_id}"
    with closing(MemoryStore(root)) as store:
        identity = store.create_memory(
            memory_type,
            "服务端口使用8000",
            ["端口"],
            details=MemoryDetails(scope=scope, sources=sources),
        )
        return store.load_memory(identity), path


def query_action(context, root, arguments):
    """从原生查询边界取得完整回执；参数：上下文/目录/查询参数；返回：结果。"""
    call = ToolOperation(
        RunToolsRequest("memory_query"), "query", "memory_query", arguments
    )
    result = MemoryActions(context, data_root=root).execute(call)
    assert result.status == "ok", result.output
    return result.meta


@pytest.mark.parametrize("action", ["read", "list"])
def test_current_query_marks_changed_source_without_rewriting_original(
    tmp_path, query_context, action
):
    """精确读取和目录查询也检查当前相关来源；参数：隔离依赖/入口；返回：无。"""
    memory, path = referenced_memory(tmp_path)
    path.write_text("port=9000", encoding="utf-8")
    args = (
        {"action": action, "memory_id": memory.memory_id, "version": memory.version}
        if action == "read"
        else {"action": action}
    )
    row = query_action(query_context, tmp_path, args)["records"][0]
    assert row["source_validity"]["state"] == "needs_verification"
    assert row["source_validity"]["changes"][0]["path"] == str(path.resolve())
    assert row["content"] == memory.content
    with closing(MemoryStore(tmp_path)) as store:
        assert store.load_memory(memory.memory_id).version == memory.version


def test_file_observation_does_not_cancel_original_user_requirement(
    tmp_path, query_context
):
    """当前文件与要求不一致只产生核验线索，用户约束继续有效；参数：隔离依赖；返回：无。"""
    memory, path = referenced_memory(tmp_path, memory_type="rule")
    path.write_text("port=9000", encoding="utf-8")
    outcome = recall_memories_with_outcome(
        tmp_path,
        task_summary="服务端口",
        task_tags=[],
        scopes=["session:source", memory.details.scope],
    )
    assert outcome.selected[0].required
    assert "用户要求仍有效" in outcome.selected[0].memory.content
    assert "不能作为已确认的当前事实" not in outcome.selected[0].memory.content


def test_historical_read_retains_frozen_content_without_new_source_work(
    tmp_path, query_context
):
    """查看旧版不是重新采用当前知识，不为旧观察创建维护工作；参数：隔离依赖；返回：无。"""
    memory, path = referenced_memory(tmp_path)
    with closing(MemoryStore(tmp_path)) as store:
        store.revise_memory(
            memory.memory_id,
            "服务端口使用9000",
            expected_version=memory.version,
            reason="用户更正",
            sources=(MemorySource("user_input", "correction"),),
        )
    path.unlink()
    row = query_action(
        query_context,
        tmp_path,
        {"action": "read", "memory_id": memory.memory_id, "version": memory.version},
    )["records"][0]
    assert row["historical_version"] and row["content"] == memory.content
    assert "source_validity" not in row
    with query_context.messages.database.snapshot() as source:
        assert not source.list("knowledge_maintenance")


@pytest.mark.parametrize("state", ["archived", "superseded"])
def test_inactive_requirement_is_not_reactivated_by_source_change(
    tmp_path, query_context, state
):
    """归档或已替代规则的来源变化不能恢复旧要求或新建核验工作；参数：隔离依赖/状态；返回：无。"""
    memory, path = referenced_memory(tmp_path, memory_type="rule")
    with closing(MemoryStore(tmp_path)) as store:
        if state == "archived":
            store.archive_memory(memory.memory_id)
        else:
            store.create_memory(
                "rule",
                "服务端口使用9000",
                [],
                details=MemoryDetails(
                    scope=memory.details.scope,
                    sources=(MemorySource("user_input", "correction"),),
                    supersedes=(
                        MemoryReplacement(memory.memory_id, memory.version, "用户更正"),
                    ),
                ),
            )
    path.write_text("port=9100", encoding="utf-8")
    row = query_action(
        query_context, tmp_path, {"action": "read", "memory_id": memory.memory_id}
    )["records"][0]
    assert row["effective_state"] == state
    assert "source_validity" not in row
    with query_context.messages.database.snapshot() as source:
        assert not source.list("knowledge_maintenance")


def test_worker_search_uses_original_session_and_never_checks_global_sources(
    tmp_path, query_context, monkeypatch
):
    """自动维护先按授权范围筛选再读来源；参数：隔离依赖/观察器；返回：无。"""
    import runtime.knowledge_validity as validity

    workspace = WorkspaceStore(tmp_path).bind_session("worker", tmp_path)
    query_context.run.session_id = "worker"
    query_context.run.payload["knowledge_origin"] = {
        "work_kind": "knowledge_maintenance",
        "source_session_id": "source",
        "workspace_id": workspace.workspace_id,
    }
    with closing(MemoryStore(tmp_path)) as store:
        source_id = store.create_memory(
            "fact",
            "端口原会话知识",
            ["端口"],
            details=MemoryDetails(scope="session:source"),
        )
        global_id = store.create_memory(
            "fact", "端口全局知识", ["端口"], details=MemoryDetails(scope="global")
        )
    checked = []
    original = validity.changed_memory_sources

    def observe_sources(data_root, memory, *, session_id):
        """记录真正检查过的来源范围；参数：当前记忆；返回：真实检查结果。"""
        checked.append((memory.memory_id, session_id))
        return original(data_root, memory, session_id=session_id)

    monkeypatch.setattr(validity, "changed_memory_sources", observe_sources)
    result = query_action(
        query_context, tmp_path, {"action": "search", "query": "端口"}
    )
    assert [row["memory_id"] for row in result["records"]] == [source_id]
    assert (source_id, "source") in checked
    assert global_id not in {identity for identity, _ in checked}


def test_worker_context_uses_accepted_scope_without_skill_material(
    tmp_path, query_context
):
    """生产上下文沿原会话召回，不向后台提供全局或技能内容；参数：隔离依赖；返回：无。"""
    from context.production_builder import HistorySelection, ProductionContextBuilder
    from skills.store import SkillStore, build_skill_markdown

    workspace = WorkspaceStore(tmp_path).bind_session("worker", tmp_path)
    task = query_context.tasks.create_task("核对端口知识", is_inbox=True)
    query_context.run.session_id = "worker"
    query_context.run.compatibility_task_id = task.task_id
    query_context.run.payload["knowledge_origin"] = {
        "work_kind": "knowledge_maintenance",
        "source_session_id": "source",
        "workspace_id": workspace.workspace_id,
    }
    with closing(MemoryStore(tmp_path)) as store:
        source_id = store.create_memory(
            "fact",
            "端口原会话知识",
            ["端口"],
            details=MemoryDetails(scope="session:source"),
        )
        store.create_memory(
            "fact", "端口全局知识", ["端口"], details=MemoryDetails(scope="global")
        )
    SkillStore(tmp_path).create_skill(
        "port-method", build_skill_markdown(name="端口知识", body="敏感方法正文")
    )
    builder = ProductionContextBuilder(tmp_path, system_prompt_provider=lambda: "test")
    _, materials = builder.recall_for_context(
        query_context.run,
        store=query_context.tasks,
        history=HistorySelection((), False),
    )
    assert len(materials) == 1
    assert materials[0].identity.startswith(f"memory:{source_id}@")
    assert materials[0].scope == "session:source"
