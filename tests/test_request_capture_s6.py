"""【阶段六】【发送捕获】三类真实协议适配只构造一次并保存真正交给发送层的内容。

作者：xxx
时间：2026-09-30 14:00:00
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import partial
from threading import Event

import pytest

from llm.client import RealLLMClient
from llm.messages import thaw_json_value
from llm.messages import UserMessage
from llm.model_registry import ModelRegistry
from llm.model_request import Capability
from llm.production_target import PRODUCTION_MODEL_KEY
from llm.provider_adapter import AdapterRegistry
from llm.providers.anthropic_messages import AnthropicMessagesAdapter
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.providers.openai_responses import OpenAIResponsesAdapter
from runtime.run_evidence import RunEvidenceStore
from runtime.request_export import freeze_export, write_export
from runtime.secret_redaction import redact_value
from scripts.testing.llm import _test_config, _test_connection, _test_model_registry
from tests.test_request_inspection_export import _detail, _store
from tests.test_image_attachments import ADAPTERS, fixture_sse, image_bytes


def _raw(family):
    """提供协议原生响应，供应商解析仍执行生产实现；参数：协议；返回：事件序列。"""
    if family == "openai_chat":
        return [
            {
                "id": "response-1",
                "choices": [{"delta": {"content": "你好"}, "finish_reason": "stop"}],
            }
        ]
    if family == "openai_responses":
        return [
            {"type": "response.created", "response": {"id": "response-1"}},
            {
                "type": "response.output_text.delta",
                "item_id": "message-1",
                "content_index": 0,
                "delta": "你好",
            },
            {
                "type": "response.output_text.done",
                "item_id": "message-1",
                "content_index": 0,
                "text": "你好",
            },
            {
                "type": "response.completed",
                "response": {"id": "response-1", "status": "completed"},
            },
        ]
    return [
        {
            "type": "message_start",
            "message": {
                "id": "response-1",
                "usage": {"input_tokens": 2, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "你好"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]


@pytest.mark.parametrize(
    "adapter_type",
    [OpenAIChatAdapter, OpenAIResponsesAdapter, AnthropicMessagesAdapter],
)
@pytest.mark.parametrize("level", ["off", "basic", "debug"])
def test_actual_body_is_built_once_sent_and_read_at_all_levels(
    tmp_path, monkeypatch, adapter_type, level
):
    """一次构造对象同时进入捕获和真实发送层；参数：协议及记录级别；返回：无。"""
    monkeypatch.setenv("REINS_TRACE_LEVEL", level)
    inspection, writer, context = _store(tmp_path)
    built, sent, recorded = [], [], []
    family = adapter_type.api_family

    def transport(body, connection):
        """记录真实发送层收到的对象；参数：发送体和连接；返回：原协议响应。"""
        sent.append(body)
        return _raw(family)

    adapter = adapter_type(stream_factory=transport)
    original = adapter.build_request

    def build(request, *, model_id):
        """检查重复构造会改变内容的边界；参数：请求；返回：本次唯一适配结果。"""
        body = original(request, model_id=model_id)
        built.append(body)
        return body

    def capture(attempt):
        """保存发送前及收尾证据；参数：真实尝试；返回：无。"""
        recorded.append(attempt)
        writer.record_attempt(context, attempt)

    monkeypatch.setattr(adapter, "build_request", build)
    descriptor = replace(
        _test_model_registry().require(PRODUCTION_MODEL_KEY), api_family=family
    )
    client = RealLLMClient(
        _test_config(),
        adapter_registry=AdapterRegistry([adapter]),
        model_registry=ModelRegistry([descriptor]),
        connection=_test_connection(),
    )
    plan = client.plan(
        "核对实际发送内容",
        {"model_request_id": "request-1", "model_attempt_recorder": capture},
    )
    assert plan.model_error is None
    assert len(built) == len(sent) == 1
    assert sent[0] is recorded[0].request
    assert thaw_json_value(sent[0]) == built[0]
    with pytest.raises(TypeError):
        sent[0]["model"] = "changed"
    assert json.loads(
        _detail(inspection, request_id="request-1", attempt_id=recorded[0].attempt_id)
    ) == redact_value(sent[0])
    assert (
        json.loads(
            _detail(
                inspection,
                section="response",
                request_id="request-1",
                attempt_id=recorded[0].attempt_id,
            )
        )["text"]
        == "你好"
    )
    assert recorded[0].api_family == family


def test_empty_stream_retry_keeps_each_actual_body(tmp_path, monkeypatch):
    """第一次空流失败与重试成功各有证据；参数：临时根；返回：无。"""
    inspection, writer, context = _store(tmp_path)
    bodies = []

    def transport(body, connection):
        """第一次断流后第二次返回；参数：发送体；返回：实际协议事件。"""
        bodies.append(body)
        return [] if len(bodies) == 1 else _raw("openai_chat")

    monkeypatch.setattr("llm.client.sleep", lambda _: None)
    descriptor = replace(
        _test_model_registry().require(PRODUCTION_MODEL_KEY), api_family="openai_chat"
    )
    client = RealLLMClient(
        _test_config(),
        adapter_registry=AdapterRegistry([OpenAIChatAdapter(stream_factory=transport)]),
        model_registry=ModelRegistry([descriptor]),
        connection=_test_connection(),
    )
    plan = client.plan(
        "测试重试",
        {
            "model_request_id": "request-1",
            "model_attempt_recorder": partial(writer.record_attempt, context),
        },
    )
    assert plan.model_error is None
    attempts = inspection.query(
        {
            "action": "attempts",
            "session_id": "s",
            "run_id": "r",
            "request_id": "request-1",
        }
    )["items"]
    assert [row["status"] for row in attempts] == ["transport_error", "completed"]
    store = RunEvidenceStore(tmp_path)
    records = store.list_records(session_id="s", run_id="r", kind="attempt_request")
    assert [row["payload"]["request"] for row in records] == [
        redact_value(body) for body in bodies
    ]


def _frozen_attachments(workspace, data_root):
    """读取真实图片、文字与PDF后删除源文件；参数：工作区和数据根；返回：接纳时冻结的消息内容。"""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    from app.background.attachments import compose_attachment_input
    from app.session_assembly import user_lease
    from tools.redacted_files import RedactedFiles

    workspace.mkdir()
    (workspace / "text.txt").write_text("文本附件冻结标记", encoding="utf-8")
    (workspace / "picture.png").write_bytes(image_bytes())
    writer = PdfWriter()
    page_size = 200
    page = writer.add_blank_page(width=page_size, height=page_size)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    contents = DecodedStreamObject()
    contents.set_data(b"BT /F1 12 Tf 20 150 Td (Frozen PDF evidence) Tj ET")
    page.replace_contents(contents)
    writer.write(workspace / "document.pdf")
    writer.close()
    prepared = compose_attachment_input(
        "核对全部附件",
        attachment_paths=("text.txt", "document.pdf", "picture.png"),
        reference_paths=(),
        project_root=workspace,
        lease=user_lease(task_id="task", project_root=workspace, data_root=data_root),
        redacted_files=RedactedFiles(),
        session_id="s",
    )
    for name in ("text.txt", "document.pdf", "picture.png"):
        (workspace / name).unlink()
    return prepared


@pytest.mark.parametrize("family,adapter_type,http", ADAPTERS)
def test_sdk_wire_view_export_reconstruct_frozen_attachments(
    tmp_path, family, adapter_type, http
):
    """实际SDK发送正文与查看及JSON导出逐字段一致；参数：三种协议；返回：无。"""
    inspection, writer, context = _store(tmp_path / "data")
    attachments = _frozen_attachments(tmp_path / "workspace", inspection.db.data_root)
    captured = []

    def transport(request):
        """捕获SDK实际HTTP正文；参数：请求；返回：合成供应商SSE，生产编码解析照常执行。"""
        captured.append(json.loads(request.content))
        return http.Response(
            200, headers={"content-type": "text/event-stream"}, text=fixture_sse(family)
        )

    descriptor = _test_model_registry().require(PRODUCTION_MODEL_KEY)
    descriptor = replace(
        descriptor,
        api_family=family,
        capabilities={**descriptor.capabilities, Capability.IMAGE_INPUT: True},
    )
    with http.Client(transport=http.MockTransport(transport)) as network:
        client = RealLLMClient(
            _test_config(),
            adapter_registry=AdapterRegistry([adapter_type(http_client=network)]),
            model_registry=ModelRegistry([descriptor]),
            connection=_test_connection(),
        )
        plan = client.plan(
            "核对全部附件",
            {
                "model_request_id": "request-1",
                "model_attempt_recorder": partial(writer.record_attempt, context),
                "conversation_history": (
                    UserMessage("attachment-input", attachments.content),
                ),
            },
        )
    assert plan.model_error is None
    assert len(captured) == 1
    attempt = plan.model_attempts[0]
    viewed = json.loads(
        _detail(inspection, request_id="request-1", attempt_id=attempt.attempt_id)
    )
    assert captured[0] == thaw_json_value(attempt.request)
    assert viewed == redact_value(captured[0])
    body = json.dumps(viewed, ensure_ascii=False)
    assert "文本附件冻结标记" in body and "Frozen PDF evidence" in body
    target = tmp_path / "export"
    target.mkdir()
    write_export(
        inspection,
        freeze_export(inspection, {"session_id": "s", "run_id": "r"}),
        target,
        Event(),
    )
    exported = json.loads((target / "requests.json").read_text(encoding="utf-8"))
    assert exported["requests"][0]["attempts"][0]["request"]["request"] == redact_value(
        captured[0]
    )
    images = [
        item for item in exported["attachments"] if item.get("mime_type") == "image/png"
    ]
    assert len(images) == 1
    assert (target / images[0]["package_path"]).read_bytes() == image_bytes()
