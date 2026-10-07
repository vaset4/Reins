"""【工具体系】【历史恢复】检验领域加载和旧调用的真实持久边界。

作者：xxx
时间：2026-10-02 16:00:00
"""

from contextlib import closing
from dataclasses import replace

import pytest

from context.production_builder import ProductionContextBuilder
from llm.messages import AssistantMessage, ToolCallPart
from llm.tool_selection import select_tools
from runtime.native_actions import NativeActionContext, NativeActions
from runtime.session_compaction import SessionCompactionStore, SummarySource
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_user_message
from runtime.tool_operations import ToolOperation, operation_payload
from runtime.types import RunToolsRequest
from scripts.testing.llm import from_test_native_tool_then_final
from tasks.store import TaskStore
from tests.test_tool_batch_execution import make_run
from tools.builtin_tools import build_tool_registry


def test_group_loading_survives_compaction_restart_and_respects_branch(tmp_path):
    """从原操作重建领域，压缩和重建owner不丢失，切回加载前不泄入工具；参数：根；返回：无。"""
    with closing(
        build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
    ) as registry:
        loop, run, _ = make_run(tmp_path, registry, [])
        loop.llm_client = from_test_native_tool_then_final(
            [
                ToolCallPart(
                    "load-group",
                    "capabilities",
                    {"action": "load", "domain": "collaboration"},
                )
            ],
            "已加载协作",
        )
        loop.run(run)
        owner = SessionMessageStore(tmp_path)
        before_load = owner.materialize(run.session_id).entries[0].entry_id
        append_user_message(tmp_path, run.session_id, "继续核对")
        view = owner.materialize(run.session_id)
        SessionCompactionStore(owner).publish(
            SummarySource(view, len(view.messages) - 1),
            "已查看可用协作能力",
            request_ids=("test-summary-request",),
        )
        builder = ProductionContextBuilder(tmp_path, system_prompt_provider=lambda: "")
        bundle = builder.build(
            task="继续核对", context=run, tool_registry=registry, toolset_policy={}
        )
        loaded = bundle.model_context["loaded_tools"]
        assert {
            "delegate",
            "agent_send",
            "agent_status",
            "agent_wait",
            "agent_cancel",
            "agent_decision",
            "read_history",
        } <= loaded
        schemas = select_tools(registry, loaded_tools=loaded).selected_definitions
        assert [row.format_for_openai_tool() for row in schemas] == [
            row.format_for_openai_tool()
            for row in select_tools(
                registry.snapshot(), loaded_tools=loaded
            ).selected_definitions
        ]
        SessionMessageStore(tmp_path).branch(run.session_id, before_load)
        history = ProductionContextBuilder(
            tmp_path, system_prompt_provider=lambda: ""
        ).read_conversation_history(run.session_id)
        assert not history.loaded_tools and history.summary is None


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("find_file", {"path": ".", "query": "资料"}),
        ("file_read", {"path": "资料.txt", "offset": 10}),
    ],
)
def test_incompatible_saved_call_remains_inspectable_but_is_not_reinterpreted(
    tmp_path, name, arguments
):
    """历史名字和参数原样保留，不把字符位移猜成行号或重放已变更合同；参数：根、旧请求；返回：无。"""
    with (
        closing(
            build_tool_registry(repo_root=tmp_path, data_root=tmp_path)
        ) as registry,
        closing(TaskStore(tmp_path)) as tasks,
    ):
        loop, run, _ = make_run(tmp_path, registry, [])
        messages = SessionMessageStore(tmp_path)
        messages.append_message(
            run.session_id,
            AssistantMessage(
                message_id="old-message",
                content=(ToolCallPart("old-call", name, arguments),),
            ),
        )
        old = ToolOperation(
            RunToolsRequest(name, arguments=arguments),
            "old-call",
            name,
            arguments,
            operation_id="op-old",
        )
        identity = {
            "session_id": run.session_id,
            "run_id": run.run_id,
            "operation_id": old.operation_id,
        }
        loop.operations.write(identity, operation_payload(old, state="not_started"))
        assert {"operation_status", "resume_operation"} <= ProductionContextBuilder(
            tmp_path, system_prompt_provider=lambda: ""
        ).read_conversation_history(run.session_id).loaded_tools
        native = NativeActions(
            NativeActionContext(
                run,
                tasks,
                messages,
                loop.session_states,
                loop.operations,
                loop.run_facts,
                registry,
            )
        )
        retry = replace(
            old,
            tool_name="resume_operation",
            args={"operation_id": old.operation_id, "action": "retry"},
        )
        result = native.execute(retry)
        assert result.status == "error" and ("no longer" in result.error)
        record = loop.operations.load(identity)
        assert (
            record["call"]["tool_name"] == name and record["call"]["args"] == arguments
        )
        assert record["state"] == "not_started" and "retry_operation_id" not in record
