"""恢复场景只验证原件、模型自主结束与权限边界，作者：xxx。"""

from __future__ import annotations

from contextlib import closing
from collections.abc import Sequence
from dataclasses import asdict
from functools import partial
import hashlib
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

from approval import ApprovalDecision
from approval.batch_types import ApprovalBatch, ApprovalChoice, BatchDecision
from approval.session import ApprovalSession
from llm.messages import AssistantMessage, ToolCallPart
from runtime.agent_loop import AgentLoop
from runtime.checkpoint import (
    Checkpoint,
    load_checkpoint,
    load_latest_checkpoint,
    save_pre_tool_checkpoint,
)
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_user_message
from runtime.tool_operations import ToolOperation, ToolOperationStore, operation_payload
from runtime.types import RunContext, RunToolsRequest
from runtime.workspaces import WorkspaceStore
from scripts.testing.llm import from_test_sequence
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import ToolRegistry
from triggers.resume import make_run_context

if TYPE_CHECKING:
    from scripts.stable_ux_scenarios import (
        ObservedEvidence,
        _RunCapture,
        _ScenarioEnvironment,
    )


def run_resume_scenario(environment: _ScenarioEnvironment) -> ObservedEvidence:
    """真实恢复先查询后结束，再显式重试并验证审批拒绝；参数：隔离环境；返回：可核对原件证据。"""
    from scripts.stable_ux_scenarios import (
        _common_observed,
        _facts_path,
        _final,
        _relative,
        _tool,
    )

    checkpoint, identity, before_bytes = _prepare_source(environment)
    operations = ToolOperationStore(environment.data_root)
    original = operations.load(identity)
    approval_batches: list[dict[str, object]] = []
    query_message = "先查询原操作，不执行历史写入"
    retry_message = "明确重试原操作，但用户拒绝本次写入授权"
    with closing(
        build_tool_registry(
            repo_root=environment.project_root, data_root=environment.data_root
        )
    ) as registry:
        for name in ("operation_status", "resume_operation"):
            definition = registry.get(name)
            assert definition is not None
            definition.deferred = False
        query_context = make_run_context(
            checkpoint.task_id, data_root=environment.data_root, checkpoint=checkpoint
        )
        query_context.payload.update(message=query_message, resume_action="replay")
        query = _execute(
            environment,
            query_context,
            registry,
            (
                _tool("operation_status", {"operation_id": identity["operation_id"]}),
                _final("原操作仍待处理，先保留证据。"),
            ),
        )
        after_query = load_latest_checkpoint(
            checkpoint.task_id, data_root=environment.data_root
        )
        query_operations = [
            row
            for row in operations.for_session(checkpoint.session_id)
            if row["run_id"] == query.run_id
        ]
        query_bytes = (environment.project_root / "pending.txt").read_bytes()
        retry_context = make_run_context(
            checkpoint.task_id, data_root=environment.data_root, checkpoint=checkpoint
        )
        retry_context.payload["message"] = retry_message
        retry = _execute(
            environment,
            retry_context,
            registry,
            (
                _tool(
                    "resume_operation",
                    {"operation_id": identity["operation_id"], "action": "retry"},
                ),
                _final("写入被当前权限拒绝，原文件未变。"),
            ),
            approval_batches=approval_batches,
        )
    after_retry = load_latest_checkpoint(
        checkpoint.task_id, data_root=environment.data_root
    )
    source = load_checkpoint(
        checkpoint.task_id, checkpoint.checkpoint_id, data_root=environment.data_root
    )
    assert source is not None and after_query is not None and after_retry is not None
    return _common_observed(environment, retry, [query_message, retry_message]) | {
        "run_ids": [query.run_id, retry.run_id],
        "source_checkpoint": asdict(checkpoint),
        "source_checkpoint_after": asdict(source),
        "checkpoint_after_query": asdict(after_query),
        "checkpoint_after_retry": asdict(after_retry),
        "source_operation": original,
        "source_operation_after": operations.load(identity),
        "query_operations": query_operations,
        "retry_operations": [
            row
            for row in operations.for_session(checkpoint.session_id)
            if row["run_id"] == retry.run_id
        ],
        "approval_batches": approval_batches,
        "model_requests": list(query.model_requests),
        "retry_model_requests": list(retry.model_requests),
        "query_facts": list(query.facts),
        "retry_facts": list(retry.facts),
        "facts": [*query.facts, *retry.facts],
        "before_sha256": hashlib.sha256(before_bytes).hexdigest(),
        "after_query_sha256": hashlib.sha256(query_bytes).hexdigest(),
        "after_retry_sha256": hashlib.sha256(
            (environment.project_root / "pending.txt").read_bytes()
        ).hexdigest(),
        "fact_files": list(
            dict.fromkeys(
                _relative(environment.root, _facts_path(environment, run))
                for run in (query, retry)
            )
        ),
        "artifact_files": ["project/pending.txt"],
    }


def _prepare_source(
    environment: _ScenarioEnvironment,
) -> tuple[Checkpoint, dict[str, str], bytes]:
    """保存真实未启动操作及恢复点并预置哨兵文件；参数：隔离环境；返回：恢复点、操作身份、原字节。"""
    from scripts.stable_ux_scenarios import (
        _create_task,
        _lease,
        _persist_source_checkpoint,
    )

    task_id = _create_task(environment, "恢复待决文件写入")
    session_id, run_id = (
        "session-acceptance-resume-source",
        "run-acceptance-resume-source",
    )
    WorkspaceStore(environment.data_root).bind_session(
        session_id, environment.project_root
    )
    before = b"original content must survive\n"
    (environment.project_root / "pending.txt").write_bytes(before)
    arguments: dict[str, str] = {
        "path": "pending.txt",
        "content": "must not run",
        "expected_sha256": hashlib.sha256(before).hexdigest(),
    }
    append_user_message(
        environment.data_root, session_id, "保留文件，先检查未决写入", task_id=task_id
    )
    SessionMessageStore(environment.data_root).append_message(
        session_id,
        AssistantMessage(
            "source-message", (ToolCallPart("call-pending", "file_write", arguments),)
        ),
        task_id=task_id,
        run_id=run_id,
    )
    operation = ToolOperation(
        RunToolsRequest("file_write", arguments=dict(arguments)),
        "call-pending",
        "file_write",
        dict(arguments),
        task_id=task_id,
        operation_id="operation-pending",
    )
    identity = {
        "session_id": session_id,
        "run_id": run_id,
        "operation_id": operation.operation_id,
    }
    ToolOperationStore(environment.data_root).write(
        identity, operation_payload(operation, state="not_started")
    )
    checkpoint = save_pre_tool_checkpoint(
        "source-segment",
        "file_write",
        arguments,
        "call-pending",
        {},
        task_id=task_id,
        session_id=session_id,
        run_id=run_id,
        focus_task_id=task_id,
        lease_snapshot=asdict(_lease(environment, task_id)),
    )
    _persist_source_checkpoint(environment, checkpoint)
    return checkpoint, identity, before


def _execute(
    environment: _ScenarioEnvironment,
    context: RunContext,
    registry: ToolRegistry,
    responses: Sequence[str],
    *,
    approval_batches: list[dict[str, object]] | None = None,
) -> _RunCapture:
    """使用真实模型请求、权限和执行循环；参数：环境、恢复上下文、目录、响应及审批记录；返回：落盘证据。"""
    from scripts.stable_ux_scenarios import _RunCapture, _model_request_payloads

    with closing(ApprovalSession()) as approval_session:
        loop = AgentLoop(
            environment.data_root,
            llm_client=from_test_sequence(responses, protocol_mode="text_json"),
            tool_registry=registry,
            approval_session=approval_session,
        )
        with patch.dict(os.environ, {"REINS_TRACE_LEVEL": "debug"}):
            if approval_batches is None:
                list(loop.run_stream(context))
            else:
                with patch(
                    "approval.batch._batch_backend",
                    partial(_deny_batch, approval_batches),
                ):
                    list(loop.run_stream(context))
    return _RunCapture(
        context.storage_task_id,
        context.session_id,
        context.run_id,
        loop.last_output,
        tuple(RunFactStore(environment.data_root).read_run(context.run_id)),
        _model_request_payloads(environment, context),
    )


def _deny_batch(
    records: list[dict[str, object]], batch: ApprovalBatch
) -> BatchDecision:
    """记录实际展示的请求并给出明确拒绝；参数：回执容器和真实审批批次；返回：逐项拒绝。"""
    records.append(
        {
            "batch_id": batch.batch_id,
            "requests": [
                {
                    "tool": item.tool,
                    "args": dict(item.args),
                    "operation_id": item.operation_id,
                    "risk": item.risk,
                }
                for item in batch.requests
            ],
        }
    )
    return BatchDecision(
        tuple(
            ApprovalChoice(item.operation_id, ApprovalDecision.DENY)
            for item in batch.requests
        )
    )
