"""【上下文】【请求边界】核对撤销权限与并发保护的实际效果。

作者：xxx
时间：2026-10-01 21:00:00
"""

from dataclasses import replace
import json

import pytest

from context.selection_store import MaterialSelectionStore, selection_scope
from llm.messages import AssistantMessage, TextPart, ToolCallPart, ToolResultMessage
from llm.providers.openai_chat import OpenAIChatAdapter
from runtime.tool_operations import ToolOperationStore
from tests.test_stage9_requests import action, send, setup_request


@pytest.mark.parametrize(
    "tool_name,status", [("file_read", "error"), ("file_write", "success")]
)
def test_revocation_removes_receipt_content_and_error_without_a_successful_read(
    tmp_path, tool_name, status
):
    """撤权同时移除成功/失败回执的正文与错误细节；参数：工具与状态；返回：无。"""
    client, views, context, actions = setup_request(tmp_path)
    path = str(tmp_path / "private.txt")
    call = AssistantMessage(
        "call", (ToolCallPart("operation", tool_name, {"path": path}),)
    )
    original = ToolResultMessage(
        "result",
        "operation",
        tool_name,
        (TextPart("撤销前私有正文"),),
        status,
        error="撤销前私有错误" if status == "error" else None,
    )
    actions.context.messages.append_message("s", call)
    actions.context.messages.append_message("s", original)
    ToolOperationStore(tmp_path).write(
        {"session_id": "s", "run_id": "r", "operation_id": "resource-op"},
        {
            "state": "completed",
            "call": {
                "call_id": "operation",
                "tool_name": tool_name,
                "resource": {"known": True, "path": path},
            },
            "result": {"status": "error" if status == "error" else "ok", "meta": {}},
        },
    )
    context = {
        **context,
        "conversation_history": actions.context.messages.materialize("s").messages,
    }
    send(client, context, views.prepare("继续", context))
    lease = context["capability_lease"]
    narrowed = {
        **context,
        "capability_lease": replace(
            lease,
            capabilities={
                **lease.capabilities,
                "fs": {**lease.capabilities["fs"], "deny_read": [path]},
            },
        ),
    }
    prepared = views.prepare("继续", narrowed)
    payload = json.dumps(
        OpenAIChatAdapter().build_request(prepared.request, model_id="fixture"),
        ensure_ascii=False,
    )
    assert "撤销前私有正文" not in payload and "撤销前私有错误" not in payload
    assert prepared.request.messages[1] == call
    assert prepared.request.messages[2].status == status
    assert actions.context.messages.materialize("s").messages[2] == original


def test_release_rechecks_current_protection_after_concurrent_adoption(tmp_path):
    """原件核对期间新请求保护同版材料，旧释放不能覆盖；参数：临时目录；返回：无。"""
    client, views, context, _ = setup_request(tmp_path)
    send(client, context, views.prepare("继续", context))
    store, scope = MaterialSelectionStore(tmp_path), selection_scope(context)
    old_row = store.read(scope)["catalog"][0]
    protected = {
        **context,
        "context_materials": (
            replace(context["context_materials"][0], protected=True),
        ),
    }
    send(client, protected, views.prepare("继续", protected))
    with pytest.raises(ValueError, match="protected"):
        store.release(
            scope,
            {**old_row, "representation": "reference"},
            operation_id="stale-release",
        )
    assert store.read(scope)["catalog"][0]["protected"]


def test_release_rejects_missing_retained_original(tmp_path):
    """原件已损坏时拒绝释放并继续保留正文；参数：隔离空间；返回：无。"""
    client, views, context, actions = setup_request(tmp_path)
    material = replace(
        context["context_materials"][0],
        reference=json.dumps(
            {
                "tool": "read_artifact",
                "arguments": {"artifact_id": "art-missing", "mode": "full"},
            }
        ),
    )
    context = {**context, "context_materials": (material,)}
    send(client, context, views.prepare("继续", context))
    released = action(
        actions,
        "context_release",
        {"identity": material.identity, "version": material.version},
    )
    assert released.meta["status"] == "rejected"
    assert material.text in views.prepare("继续", context).render_text_to_model


@pytest.mark.parametrize("change", ["permission", "requirements", "branch"])
def test_prepared_request_cannot_bypass_changed_context_boundary(tmp_path, change):
    """已准备请求不能覆盖派发时新范围与更正；参数：真实边界变化；返回：无。"""
    client, views, context, _ = setup_request(tmp_path)
    prepared = views.prepare("继续", context)
    if change == "permission":
        lease = context["capability_lease"]
        changed = {
            **context,
            "capability_lease": replace(
                lease,
                capabilities={
                    **lease.capabilities,
                    "fs": {
                        **lease.capabilities["fs"],
                        "deny_read": [str(tmp_path / "private.txt")],
                    },
                },
            ),
        }
    elif change == "requirements":
        changed = {**context, "effective_requirements": "新更正：不得联网"}
    else:
        changed = {**context, "context_branch_id": "changed-branch"}
    with pytest.raises(ValueError, match="context boundary changed"):
        client.plan("继续", {**changed, "prepared_request": prepared})
    assert not MaterialSelectionStore(tmp_path).read(selection_scope(context))


def test_direct_markdown_edit_replaces_old_memory_in_next_sent_request(tmp_path):
    """用户直接编辑记忆后，下次召回与发送采用新正文；参数：隔离空间；返回：无。"""
    from context.engine import recall_context_materials
    from memory.store import MemoryStore

    client, views, context, _ = setup_request(tmp_path)
    store = MemoryStore(tmp_path)
    identity = store.create_memory("fact", "品牌颜色采用深蓝", ["品牌颜色"])
    old = store.load_memory(identity)
    _, materials = recall_context_materials(
        tmp_path, task_summary="品牌颜色", task_tags=["品牌颜色"], skill_refs=[]
    )
    first_context = {**context, "context_materials": materials}
    first = views.prepare("继续", first_context)
    send(client, first_context, first)
    path = store.current_path(identity)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "品牌颜色采用深蓝", "品牌颜色改为浅绿"
        ),
        encoding="utf-8",
    )
    _, revised = recall_context_materials(
        tmp_path, task_summary="品牌颜色", task_tags=["品牌颜色"], skill_refs=[]
    )
    current = {**context, "context_materials": revised}
    prepared = views.prepare("继续", current)
    plan = send(client, current, prepared)
    payload = json.dumps(
        plan.model_attempts[-1].request, ensure_ascii=False, default=dict
    )
    assert "品牌颜色改为浅绿" in payload and "品牌颜色采用深蓝" not in payload
    assert (
        prepared.context_baseline["baseline_id"]
        != first.context_baseline["baseline_id"]
    )
    assert store.load_memory(identity, version=old.version).content == old.content
