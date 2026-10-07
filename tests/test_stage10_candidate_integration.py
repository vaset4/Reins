"""【知识维护】【持久候选】真实记忆提交与多次完成检查之间保留明确处置。

作者：xxx
时间：2026-10-02 12:02:00
"""

from contextlib import closing
from dataclasses import asdict

from memory.store import MemoryStore
from runtime.knowledge_jobs import KnowledgeJobs
from runtime.knowledge_maintenance import KnowledgeMaintenance
from runtime.lease import from_trigger
from runtime.memory_actions import MemoryActions
from runtime.native_actions import NativeActionContext
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.session_state import SessionStateStore
from runtime.tool_operations import ToolOperation, ToolOperationStore
from runtime.types import RunContext, RunToolsRequest, Trigger
from runtime.workspaces import WorkspaceStore
from scripts.testing.llm import from_test_stub
from tasks.store import TaskStore
from tests.test_automatic_knowledge import start_source
from tools.tool_registry import ToolRegistry


def execute_action(context, root, tool, arguments, identity):
    """记录实际领域动作的开始和回执；参数：依赖、目录、工具、参数及身份；返回：真实工具结果。"""
    operation = ToolOperation(
        RunToolsRequest(tool), identity, tool, arguments, operation_id=identity
    )
    owner = {
        "session_id": context.run.session_id,
        "run_id": context.run.run_id,
        "operation_id": identity,
    }
    context.operations.write(
        owner,
        {
            "state": "started",
            "call": {"call_id": identity, "tool_name": tool, "args": arguments},
        },
    )
    actions = (
        MemoryActions(context, data_root=root)
        if tool == "memory_manage"
        else KnowledgeJobs(context, data_root=root, client=from_test_stub("unused"))
    )
    result = actions.execute(operation)
    context.operations.write(owner, {"state": "completed", "result": asdict(result)})
    return result


def test_resolved_failure_survives_another_unresolved_candidate_and_restart(tmp_path):
    """七次真实提交后分两次处置失败，首次完成被拒也不撤回已保存替代；参数：隔离根；返回：无。"""
    _, work = start_source(tmp_path)
    WorkspaceStore(tmp_path).bind_session("worker", tmp_path)
    manager = KnowledgeMaintenance(tmp_path)
    manager.update(
        work["work_id"],
        state="running",
        worker_session_id="worker",
        run_id="worker-run",
    )
    run = RunContext(
        session_id="worker",
        run_id="worker-run",
        trigger=Trigger.CRON,
        payload={
            "knowledge_origin": work,
            "source_session_id": work["source_session_id"],
            "source_run_id": work["source_run_id"],
        },
        capability_lease=from_trigger("cron"),
    )
    with closing(TaskStore(tmp_path)) as tasks, closing(ToolRegistry()) as registry:
        context = NativeActionContext(
            run,
            tasks,
            SessionMessageStore(tmp_path),
            SessionStateStore(tmp_path),
            ToolOperationStore(tmp_path),
            RunFactStore(tmp_path),
            registry,
        )
        arguments = {
            "action": "create",
            "type": "fact",
            "kind": "fact",
            "memory_scope": "project",
            "subject": "金额",
            "content": "金额保留两位小数",
            "source_mode": "origin_inputs",
        }
        for name in ("article_and_commit_conventions", "unverified-other"):
            assert (
                execute_action(
                    context,
                    tmp_path,
                    "memory_manage",
                    {
                        **arguments,
                        "fact_key": name,
                        "source_input_ids": ["wrong-source"],
                    },
                    name,
                ).status
                == "error"
            )
        for index in range(7):
            assert (
                execute_action(
                    context,
                    tmp_path,
                    "memory_manage",
                    {
                        **arguments,
                        "fact_key": f"amount-{index}",
                        "content": f"金额字段{index}保留两位小数",
                    },
                    f"saved-{index}",
                ).status
                == "ok"
            )
        assert (
            execute_action(
                context, tmp_path, "knowledge_read", {"action": "messages"}, "read"
            ).status
            == "ok"
        )
        first = execute_action(
            context,
            tmp_path,
            "knowledge_finish",
            {
                "outcome": "completed",
                "reason": "部分候选已处理",
                "resolutions": [
                    {
                        "operation_id": "article_and_commit_conventions",
                        "disposition": "superseded",
                        "replacement_operation_id": "saved-0",
                        "reason": "已根据正确原文换名保存",
                    }
                ],
            },
            "finish-first",
        )
        assert first.status == "error" and first.meta["candidate_resolutions"]
        saved = KnowledgeMaintenance(tmp_path).load(work["work_id"])
        assert len(saved["candidate_resolutions"]) == 1 and saved["outcome"] is None
        second = execute_action(
            context,
            tmp_path,
            "knowledge_finish",
            {
                "outcome": "completed",
                "reason": "所有候选已明确处置",
                "resolutions": [
                    {
                        "operation_id": "unverified-other",
                        "disposition": "abandoned",
                        "reason": "缺少证据，不保存该候选",
                    }
                ],
            },
            "finish-last",
        )
        assert second.status == "ok" and len(second.meta["commits"]) == 7
        assert (
            context.operations.load(
                {
                    "session_id": "worker",
                    "run_id": "worker-run",
                    "operation_id": "article_and_commit_conventions",
                }
            )["result"]["status"]
            == "error"
        )
    assert (
        KnowledgeMaintenance(tmp_path).status()["works"][0]["failed_operation_ids"]
        == []
    )
    with closing(MemoryStore(tmp_path)) as memories:
        assert len(memories.list_memories()) == 7
