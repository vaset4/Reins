"""【上下文】【阶段九请求】实际发送采用、稳定前缀与同源释放。

作者：xxx
时间：2026-10-01 14:00:00
"""

from dataclasses import replace
from types import SimpleNamespace
import json

import pytest

from context.materials import ContextMaterial
from context.selection_store import MaterialSelectionStore, selection_scope
from llm.messages import (
    AssistantMessage,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
    thaw_json_value,
)
from llm.model_request import (
    Capability,
    model_request_from_mapping,
    model_request_to_mapping,
)
from llm.client import RealLLMClient
from llm.model_registry import ModelRegistry
from llm.provider_adapter import AdapterRegistry
from llm.providers.anthropic_messages import AnthropicMessagesAdapter
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.providers.openai_responses import OpenAIResponsesAdapter
from runtime.context_actions import ContextActions
from runtime.lease import from_trigger
from runtime.persistence import RuntimeStore
from runtime.session_message_store import SessionMessageStore
from runtime.tool_operations import ToolOperation
from runtime.tool_result_views import ToolResultViews
from runtime.types import RunContext, RunToolsRequest
from scripts.testing.llm import (
    from_test_turns,
    _test_config,
    _test_connection,
    _test_model_registry,
)
from tests.test_image_attachments import ADAPTERS, fixture_sse
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry


def setup_request(tmp_path):
    """建立真实隔离消息与请求组装；参数：临时目录；返回：客户端、组装器、上下文及动作。"""
    tasks = TaskStore(tmp_path)
    task = tasks.create_task("继续核对")
    messages = SessionMessageStore(tmp_path)
    messages.create_session("s")
    entry = messages.append_message(
        "s", UserMessage("user", (TextPart("不得上传；端口8000"),))
    )
    lease = from_trigger(
        "user",
        capabilities={"fs": {"project_root": str(tmp_path), "read": [str(tmp_path)]}},
    )
    run = RunContext(
        trigger="user",
        payload={},
        capability_lease=lease,
        session_id="s",
        run_id="r",
        task_id=task.task_id,
    )
    native = SimpleNamespace(run=run, messages=messages)
    client = from_test_turns(["继续"] * 5)
    views = ToolResultViews(
        tmp_path, task_id=task.task_id, prepare=client.prepare_request
    )
    context = {
        "data_root": str(tmp_path),
        "session_id": "s",
        "context_branch_id": entry.entry_id,
        "tool_working_directory": str(tmp_path),
        "capability_lease": lease,
        "tool_registry": ToolRegistry(),
        "input_message_id": "user",
        "conversation_history": messages.materialize("s").messages,
        "context_materials": (
            ContextMaterial(
                identity="one",
                source="history",
                version="v1",
                scope="s",
                text="稳定历史正文",
            ),
        ),
        "effective_requirements": "不得上传；端口8000",
    }
    return client, views, context, ContextActions(native, data_root=tmp_path)


def send(client, context, bundle):
    """经过真实客户端派发采用请求；参数：客户端、上下文、预览；返回：实际计划。"""
    plan = client.plan("继续", {**context, "prepared_request": bundle})
    assert plan.model_error is None
    return plan


def action(actions, name, args):
    """通过公开原生动作契约调用；参数：执行器、工具名、参数；返回：结果。"""
    request = RunToolsRequest(
        action=name, tool_name=name, arguments=args, call_id="action"
    )
    return actions.execute(
        ToolOperation(
            request=request,
            call_id="action",
            tool_name=name,
            args=args,
            operation_id="op",
        )
    )


@pytest.mark.parametrize(
    "adapter",
    [OpenAIChatAdapter(), OpenAIResponsesAdapter(), AnthropicMessagesAdapter()],
)
def test_baseline_delta_stable_in_provider_bodies_only_send_adopts(tmp_path, adapter):
    """预算/通知变化保持前方正文，新增只进增量；参数：协议；返回：无。"""
    client, views, context, _ = setup_request(tmp_path)
    first = views.prepare("继续", context)
    with RuntimeStore(tmp_path).snapshot() as source:
        assert not source.list_raw("context_baseline")
        assert not source.list_raw("context_material_selection")
        assert not source.list_raw("prompt_snapshot")
    send(client, context, first)
    second = views.prepare(
        "继续",
        {
            **context,
            "budget_evidence": {"steps_used": 7, "steps_limit": 20},
            "recoverable_error_notice": "临时网络错误",
        },
    )
    assert (
        first.context_baseline["baseline_id"] == second.context_baseline["baseline_id"]
    )
    assert second.context_baseline["local_reused"] is True
    left = adapter.build_request(first.request, model_id="fixture")
    right = adapter.build_request(second.request, model_id="fixture")
    if adapter.api_family == "openai_responses":
        assert right["instructions"].startswith(
            left["instructions"].split("Runtime context")[0]
        )
    elif adapter.api_family == "openai_chat":
        assert right["messages"][0]["content"].startswith(
            left["messages"][0]["content"].split("Runtime context")[0]
        )
    else:
        assert left["system"][:2] == right["system"][:2]
    added = replace(context["context_materials"][0], identity="two", text="新增历史")
    third_context = {
        **context,
        "context_materials": (*context["context_materials"], added),
    }
    third = views.prepare("继续", third_context)
    assert (
        third.context_baseline["baseline_id"] == first.context_baseline["baseline_id"]
    )
    assert len(third.context_baseline["delta"]) == 1
    send(client, third_context, third)
    fourth = views.prepare("继续", third_context)
    assert third.context_baseline["delta_id"] == fourth.context_baseline["delta_id"]
    assert third.request.instructions == fourth.request.instructions


def test_material_release_survives_requests_and_version_changes_invalidate(tmp_path):
    """释放当前已采用材料，重启选档并在新版恢复正文；参数：临时根；返回：无。"""
    client, views, context, actions = setup_request(tmp_path)
    send(client, context, views.prepare("继续", context))
    released = action(actions, "context_release", {"identity": "one", "version": "v1"})
    assert released.status == "ok" and released.meta["status"] == "applied"
    for _ in range(2):
        bundle = views.prepare("继续", context)
        assert "稳定历史正文" not in bundle.render_text_to_model
        assert "read_artifact" in bundle.render_text_to_model
        send(client, context, bundle)
    changed = {
        **context,
        "context_materials": (
            replace(context["context_materials"][0], version="v2", text="新版本正文"),
        ),
    }
    assert "新版本正文" in views.prepare("继续", changed).render_text_to_model
    assert MaterialSelectionStore(tmp_path).read(selection_scope(context))["decisions"]


def test_tool_release_retains_original_and_call_arguments(tmp_path):
    """原工具正文可读且真实参数不改；参数：隔离消息；返回：无。"""
    client, views, context, actions = setup_request(tmp_path)
    messages = actions.context.messages
    call = AssistantMessage(
        "call",
        (ToolCallPart("read", "file_read", {"path": "original.txt", "offset": 17}),),
    )
    result = ToolResultMessage(
        "result", "read", "file_read", (TextPart("工具原始字节"),), "success"
    )
    messages.append_message("s", call)
    messages.append_message("s", result)
    context = {**context, "conversation_history": messages.materialize("s").messages}
    send(client, context, views.prepare("继续", context))
    row = next(
        row
        for row in action(actions, "context_inspect", {}).meta["materials"]
        if row["source"] == "tool_result"
    )
    released = action(
        actions,
        "context_release",
        {"identity": row["identity"], "version": row["version"]},
    )
    assert released.meta["status"] == "applied"
    bundle = views.prepare("继续", context)
    assert bundle.request.messages[1] == call
    assert "工具原始字节" not in bundle.render_text_to_model
    assert messages.materialize("s").messages[-1] == result
    from artifacts.store import ArtifactStore

    reference = json.loads(released.meta["read_reference"])
    assert "工具原始字节" in ArtifactStore(tmp_path).read_path(
        reference["arguments"]["artifact_id"]
    ).read_text(encoding="utf-8")


def test_protected_requirements_branch_and_permission_boundaries(tmp_path):
    """保护要求，分支及权限变化不能复用旧选档；参数：隔离根；返回：无。"""
    client, views, context, actions = setup_request(tmp_path)
    protected = replace(context["context_materials"][0], protected=True)
    context = {**context, "context_materials": (protected,)}
    send(client, context, views.prepare("继续", context))
    assert (
        action(actions, "context_release", {"identity": "one", "version": "v1"}).meta[
            "status"
        ]
        == "rejected"
    )
    before = views.prepare("继续", context)
    corrected = views.prepare(
        "继续", {**context, "effective_requirements": "禁止上传；端口9000"}
    )
    assert (
        corrected.context_baseline["baseline_id"]
        != before.context_baseline["baseline_id"]
    )
    assert "端口9000" in corrected.render_text_to_model
    assert not MaterialSelectionStore(tmp_path).read(
        {**selection_scope(context), "branch_id": "other"}
    )
    assert not MaterialSelectionStore(tmp_path).read(
        {**selection_scope(context), "authority": "narrowed"}
    )


def test_neutral_layers_roundtrip_and_verified_anthropic_target_only(tmp_path):
    """层边界进入canonical，只有已知目标发显式提示；参数：隔离请求；返回：无。"""
    _, views, context, _ = setup_request(tmp_path)
    request = views.prepare("继续", context).request
    assert model_request_from_mapping(model_request_to_mapping(request)) == request
    adapter = AnthropicMessagesAdapter()
    model = _test_model_registry().require("production")
    model = replace(
        model,
        api_family="anthropic_messages",
        capabilities={**model.capabilities, Capability.PROMPT_CACHE: True},
    )
    connection = replace(_test_connection(), base_url="https://api.anthropic.com")
    body = adapter.build_request(request, model_id=model.model_id)
    marked = adapter.apply_cache_hints(
        body, request=request, model=model, connection=connection
    )
    assert marked["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in marked["system"][-1]
    gateway = replace(connection, base_url="https://compatible.invalid")
    assert (
        adapter.apply_cache_hints(
            body, request=request, model=model, connection=gateway
        )
        == body
    )
    unknown = replace(
        model, capabilities={**model.capabilities, Capability.PROMPT_CACHE: False}
    )
    assert (
        adapter.apply_cache_hints(
            body, request=request, model=unknown, connection=connection
        )
        == body
    )


@pytest.mark.parametrize("family,adapter_type,http", ADAPTERS)
def test_sdk_actual_send_and_sources_keep_stable_prefix(
    tmp_path, family, adapter_type, http
):
    """通过真实SDK序列化并记录实际采用来源；参数：协议/SDK；返回：无，未联网。"""
    _, original_views, context, actions = setup_request(tmp_path)
    captured, attempts = [], []

    def transport(request):
        """捕获HTTP正文并返回合成SSE；参数：真实SDK请求；返回：本地响应。"""
        captured.append(json.loads(request.content))
        return http.Response(
            200, headers={"content-type": "text/event-stream"}, text=fixture_sse(family)
        )

    descriptor = _test_model_registry().require("production")
    descriptor = replace(
        descriptor,
        api_family=family,
        capabilities={**descriptor.capabilities, Capability.PROMPT_CACHE: True},
    )
    connection = _test_connection()
    if family == "anthropic_messages":
        connection = replace(connection, base_url="https://api.anthropic.com")
    with http.Client(transport=http.MockTransport(transport)) as network:
        client = RealLLMClient(
            _test_config(),
            adapter_registry=AdapterRegistry([adapter_type(http_client=network)]),
            model_registry=ModelRegistry([descriptor]),
            connection=connection,
        )
        views = ToolResultViews(
            tmp_path, task_id=original_views.task_id, prepare=client.prepare_request
        )
        for index in range(3):
            current = {
                **context,
                "model_request_id": f"sdk-{index}",
                "model_attempt_recorder": attempts.append,
                "budget_evidence": {"steps_used": index, "steps_limit": 20},
            }
            if index:
                current["context_materials"] = (
                    *context["context_materials"],
                    replace(
                        context["context_materials"][0],
                        identity="sdk-delta",
                        text="新增且稳定的增量正文",
                    ),
                )
            send(client, current, views.prepare("继续", current))
    assert len(captured) == 3
    starts = [item for item in attempts if item.phase == "started"]
    from llm.messages import thaw_json_value

    assert [thaw_json_value(item.request) for item in starts] == captured
    versions = [
        item.sources["composition"]["context_baseline"]["baseline_id"]
        for item in starts
    ]
    assert versions[0] == versions[1] == versions[2]
    assert action(actions, "context_inspect", {}).meta["adopted_request_id"] == "sdk-2"
    if family == "anthropic_messages":
        # 1. 缓存标记会从基线末尾移动到增量末尾，但已发送基线正文保持完全一致
        baseline_texts = [
            [block["text"] for block in body["system"][:2]] for body in captured
        ]
        assert baseline_texts[0] == baseline_texts[1] == baseline_texts[2]
        assert captured[1]["system"][:3] == captured[2]["system"][:3]
        assert captured[0]["system"][1]["cache_control"] == {"type": "ephemeral"}
        assert captured[1]["system"][2]["cache_control"] == {"type": "ephemeral"}
    else:
        assert "cache_control" not in json.dumps(captured)
        texts = [
            body["instructions"]
            if family == "openai_responses"
            else body["messages"][0]["content"]
            for body in captured
        ]
        stable_prefixes = [text.split("Runtime context")[0] for text in texts]
        assert stable_prefixes[1].startswith(stable_prefixes[0])
        assert stable_prefixes[1] == stable_prefixes[2]
        assert "新增且稳定的增量正文" in stable_prefixes[2]


def test_history_downgrade_replays_and_needed_reread_is_protected(tmp_path):
    """同源降档不调用模型，补读正文在当前用途下不再释放；参数：隔离根；返回：无。"""
    from scripts.testing.llm import ScriptedTurnOptions

    _, _, context, actions = setup_request(tmp_path)
    client = from_test_turns(["继续"], options=ScriptedTurnOptions(context_window=3500))
    views = ToolResultViews(
        tmp_path,
        task_id=actions.context.run.storage_task_id,
        prepare=client.prepare_request,
    )
    history = ContextMaterial(
        identity="segment",
        source="history",
        version="summary/P2",
        scope="s",
        text="详细历史" * 5000,
        reference='{"tool":"read_history","arguments":{"summary_id":"summary"}}',
    )
    current = {
        **context,
        "context_materials": (),
        "history_materials": (history,),
        "history_material_levels": {
            "segment": {
                "P1": history.text,
                "P2": history.text,
                "P3": "紧凑结论",
                "P4": "定位",
            }
        },
    }
    bundle = views.prepare("继续", current)
    assert bundle.trim_delta["materials"][0]["representation"] == "P3"
    send(client, current, bundle)
    assert (
        views.prepare("继续", current).context_baseline["baseline_id"]
        == bundle.context_baseline["baseline_id"]
    )
    messages = actions.context.messages
    messages.append_message(
        "s", AssistantMessage("call", (ToolCallPart("read", "read_history", {}),))
    )
    messages.append_message(
        "s",
        ToolResultMessage(
            "result", "read", "read_history", (TextPart("当前需要的原文"),), "success"
        ),
    )
    current = {**context, "conversation_history": messages.materialize("s").messages}
    reread = views.prepare("继续", current)
    assert "当前需要的原文" in reread.render_text_to_model
    assert next(
        row
        for row in reread.material_selection["catalog"]
        if row["source"] == "tool_result"
    )["protected"]


def test_pre_dispatch_cancel_and_evidence_failure_do_not_adopt(tmp_path):
    """取消或证据写失败没有采用记录；参数：临时空间；返回：无。"""
    from runtime.cancellation import CancellationToken

    client, views, context, _ = setup_request(tmp_path)
    bundle = views.prepare("继续", context)

    def fail_record(event):
        """模拟真实派发前证据失败；参数：尝试事件；返回：抛出存储错误。"""
        raise OSError("evidence unavailable")

    with pytest.raises(OSError, match="evidence unavailable"):
        client.plan(
            "继续",
            {
                **context,
                "prepared_request": bundle,
                "model_attempt_recorder": fail_record,
            },
        )
    cancellation = CancellationToken()
    cancellation.cancel("test")
    plan = client.plan(
        "继续", {**context, "prepared_request": bundle, "cancellation": cancellation}
    )
    assert plan.model_error.category == "cancelled"
    with RuntimeStore(tmp_path).snapshot() as source:
        assert not source.list_raw("context_baseline")
        assert not source.list_raw("context_material_selection")


def test_cancel_after_attempt_started_closes_attempt_without_adopting(tmp_path):
    """派发前最后取消仍闭合尝试，未发送内容不记采用；参数：隔离根；返回：无。"""
    from runtime.cancellation import CancellationToken

    client, views, context, _ = setup_request(tmp_path)
    cancellation, events = CancellationToken(), []

    def record(event):
        """在已记录开始后取消；参数：尝试事件；返回：无。"""
        events.append(event)
        if event.phase == "started":
            cancellation.cancel("cancel before send")

    plan = client.plan(
        "继续",
        {
            **context,
            "prepared_request": views.prepare("继续", context),
            "model_attempt_recorder": record,
            "cancellation": cancellation,
        },
    )
    assert plan.model_error.category == "cancelled"
    assert [event.phase for event in events] == ["started", "finished"]
    assert events[-1].error.category == "cancelled"
    assert events[-1].usage.cache_read_input_tokens.value is None
    with RuntimeStore(tmp_path).snapshot() as source:
        assert not source.list_raw("context_baseline")


def test_unreported_cache_remains_unknown_after_actual_dispatch(tmp_path):
    """本地采用不制造供应商缓存命中数字；参数：隔离根；返回：无。"""
    client, views, context, _ = setup_request(tmp_path)
    plan = send(client, context, views.prepare("继续", context))
    usage = plan.model_attempts[-1].usage
    assert usage.cache_read_input_tokens.status.value == "unknown"
    assert usage.cache_write_input_tokens.status.value == "unknown"
    assert usage.cache_read_input_tokens.value is None
    evidence = plan.request_bundle_evidence["context_baseline"]
    assert evidence["adoption_boundary"] == "local_dispatch"
    assert evidence["provider_delivery"] == "unknown_until_response"


def test_actual_model_switch_rebuilds_and_rejects_old_prepared_request(tmp_path):
    """相同窗口更换模型也改变请求合同；参数：临时根；返回：无。"""
    from llm.context_baseline import adopt_request_context

    client, views, context, _ = setup_request(tmp_path)
    old = views.prepare("继续", context)
    adopt_request_context(old, request_id="old")
    descriptor = replace(
        _test_model_registry().require("production"), model_id="different-model"
    )
    changed = RealLLMClient(
        _test_config(),
        model_registry=ModelRegistry([descriptor]),
        connection=_test_connection(),
    )
    fresh = changed.prepare_request("继续", context)
    assert fresh.context_baseline["baseline_id"] != old.context_baseline["baseline_id"]
    with pytest.raises(ValueError, match="target changed"):
        changed.plan("继续", {**context, "prepared_request": old})


def test_real_agent_loop_inspects_releases_and_keeps_tool_original(tmp_path):
    """正式入口运行读取→查看→释放，下一请求实际去掉旧正文；参数：隔离目录；返回：无。"""
    from app.run_task import run_task
    from runtime.session_messages import materialize_messages
    from tools.builtin_tools import build_tool_registry

    marker = "完整原件在释放后依然保留"
    (tmp_path / "source.txt").write_text(marker, encoding="utf-8")
    bodies = []

    def transport(body, connection):
        """按实际查看结果选择待释放版本；参数：发送体/连接；返回：受控原生响应。"""
        bodies.append(thaw_json_value(body))
        index = len(bodies)
        if index == 4:
            return [
                {
                    "id": "response-4",
                    "choices": [
                        {
                            "delta": {"content": "已释放并保留原件"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ]
        if index == 1:
            name, args = "file_read", {"path": str(tmp_path / "source.txt")}
        elif index == 2:
            name, args = "context_inspect", {}
        else:
            receipt = next(
                row for row in reversed(body["messages"]) if row["role"] == "tool"
            )
            envelope = json.loads(receipt["content"])
            inspected = json.loads(envelope["output"])
            row = next(
                row for row in inspected["materials"] if row["source"] == "tool_result"
            )
            name, args = (
                "context_release",
                {"identity": row["identity"], "version": row["version"]},
            )
        return [
            {
                "id": f"response-{index}",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": f"call-{index}",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(args),
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        ]

    descriptor = replace(
        _test_model_registry().require("production"), api_family="openai_chat"
    )
    client = RealLLMClient(
        _test_config(),
        adapter_registry=AdapterRegistry([OpenAIChatAdapter(stream_factory=transport)]),
        model_registry=ModelRegistry([descriptor]),
        connection=_test_connection(),
    )
    registry = build_tool_registry(repo_root=tmp_path, data_root=tmp_path / "data")
    try:
        result = run_task(
            "读取source.txt、查看上下文并释放旧工具正文",
            tmp_path,
            data_root=tmp_path / "data",
            session_id="session-" + "a" * 32,
            llm_client=client,
            tool_registry=registry,
        )
    finally:
        registry.close()
    assert result.status == "done" and len(bodies) == 4
    assert marker in json.dumps(bodies[2], ensure_ascii=False)
    assert marker not in json.dumps(bodies[3], ensure_ascii=False)
    originals = materialize_messages(tmp_path / "data", "session-" + "a" * 32)
    source = next(
        item
        for item in originals
        if isinstance(item, ToolResultMessage) and item.tool_name == "file_read"
    )
    assert marker in str(source)
    assert (tmp_path / "source.txt").read_text(encoding="utf-8") == marker


def test_permission_revocation_removes_old_tool_body_from_next_request(tmp_path):
    """当前文件权限收缩后实际去掉旧正文，仍保留真实参数；参数：隔离根；返回：无。"""
    from runtime.tool_operations import ToolOperationStore

    client, views, context, actions = setup_request(tmp_path)
    path = str(tmp_path / "private.txt")
    messages = actions.context.messages
    call = AssistantMessage(
        "call", (ToolCallPart("read", "file_read", {"path": path}),)
    )
    messages.append_message("s", call)
    messages.append_message(
        "s",
        ToolResultMessage(
            "result", "read", "file_read", (TextPart("撤销前可见正文"),), "success"
        ),
    )
    ToolOperationStore(tmp_path).write(
        {"session_id": "s", "run_id": "r", "operation_id": "read-op"},
        {
            "state": "completed",
            "call": {
                "call_id": "read",
                "tool_name": "file_read",
                "resource": {"known": True, "path": path},
            },
            "result": {"status": "ok", "meta": {}},
        },
    )
    context = {**context, "conversation_history": messages.materialize("s").messages}
    send(client, context, views.prepare("继续", context))
    old_lease = context["capability_lease"]
    capabilities = {
        **old_lease.capabilities,
        "fs": {**old_lease.capabilities["fs"], "deny_read": [path]},
    }
    narrowed = {
        **context,
        "capability_lease": replace(old_lease, capabilities=capabilities),
    }
    bundle = views.prepare("继续", narrowed)
    assert "撤销前可见正文" not in bundle.render_text_to_model
    assert bundle.request.messages[1] == call
    assert bundle.trim_delta["file_reads"][0]["reason"] == "permission_removed"
