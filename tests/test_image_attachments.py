"""合成图片经持久化、真实SDK请求和恢复后的保真验证。

作者：xxx
时间：2026-09-30 11:00:00
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import httpx
import httpx2
import pytest
from PIL import Image

from app.background.image_attachments import read_image_attachment
from app.background.sessions import BackgroundSession
from app.session_assembly import user_lease
from context.token_estimate import estimate_agent_messages_tokens
from llm.image_input import image_from_bytes
from llm.messages import ImagePart, TextPart, UserMessage
from llm.model_registry import ModelRegistry
from llm.model_request import Capability, CapabilityRequirement, ModelRequest
from llm.provider_adapter import ProviderAdapterError
from llm.provider_stream import StreamAssembler
from llm.providers.anthropic_messages import AnthropicMessagesAdapter
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.providers.openai_responses import OpenAIResponsesAdapter
from scripts.testing.llm import from_test_sequence
from tests.test_background_sessions import session_for
from tests.test_provider_sdk_transport import _connection, _model
from tests.test_session_runtime import capture_requests
from tools.tool_registry import ToolRegistry

ADAPTERS = (
    ("openai_chat", OpenAIChatAdapter, httpx2),
    ("openai_responses", OpenAIResponsesAdapter, httpx2),
    ("anthropic_messages", AnthropicMessagesAdapter, httpx),
)


def image_bytes(color="red", *, size=(24, 18), compress_level=6):
    """生成不含真实内容的PNG；参数：色彩、尺寸和压缩；返回：图片字节。"""
    output = BytesIO()
    Image.new("RGB", size, color).save(
        output, format="PNG", compress_level=compress_level
    )
    return output.getvalue()


def image_request(part):
    """构造文字图片穿插的规范请求；参数：冻结图片；返回：请求。"""
    return ModelRequest(
        instructions=(),
        messages=(
            UserMessage(
                "input-picture",
                (
                    TextPart("第一张"),
                    part,
                    TextPart("第二张"),
                    part,
                    TextPart("比较图片"),
                ),
            ),
        ),
        tools=(),
        stream=True,
        required_capabilities=frozenset(
            {
                CapabilityRequirement(Capability.IMAGE_INPUT),
                CapabilityRequirement(Capability.STREAMING),
            }
        ),
    )


def fixture_sse(family):
    """把已固定的供应商合成事件交给真实SDK；参数：协议；返回：SSE正文。"""
    path = (
        Path(__file__).parent
        / "fixtures"
        / "llm"
        / "providers"
        / family
        / "stream_text.jsonl"
    )
    blocks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        prefix = f"event: {row['type']}\n" if "type" in row else ""
        blocks.append(f"{prefix}data: {line}\n\n")
    if family == "openai_chat":
        blocks.append("data: [DONE]\n\n")
    return "".join(blocks)


@pytest.mark.parametrize("family,adapter_type,http", ADAPTERS)
def test_real_sdk_wire_contains_image_bytes_in_order(family, adapter_type, http):
    """三个真实SDK的最终HTTP正文包含同一图片字节，顺序不变；参数：协议组合；返回：无。"""
    raw = image_bytes()
    part = image_from_bytes(raw)
    captured = []

    def handler(request):
        """捕获SDK编码后的请求；参数：HTTP请求；返回：合成成功SSE。"""
        captured.append(json.loads(request.content))
        return http.Response(
            200, headers={"content-type": "text/event-stream"}, text=fixture_sse(family)
        )

    with http.Client(transport=http.MockTransport(handler)) as client:
        adapter = adapter_type(http_client=client)
        model = _model(family, "test-provider", "gpt-fixture")
        model = replace(
            model, capabilities={**model.capabilities, Capability.IMAGE_INPUT: True}
        )
        result = StreamAssembler().assemble(
            list(
                adapter.stream(
                    image_request(part),
                    model=model,
                    connection=_connection(family, "https://image.invalid/v1"),
                )
            )
        )
    assert result.error is None and result.message is not None
    assert len(captured) == 1
    blocks = captured[0]["input" if family == "openai_responses" else "messages"][0][
        "content"
    ]
    assert [block.get("text") for block in blocks[::2]] == [
        "第一张",
        "第二张",
        "比较图片",
    ]
    for block in blocks[1::2]:
        if family == "anthropic_messages":
            assert block["source"]["media_type"] == "image/png"
            encoded = block["source"]["data"]
        else:
            url = (
                block["image_url"]["url"]
                if family == "openai_chat"
                else block["image_url"]
            )
            encoded = url.split(",", 1)[1]
        assert base64.b64decode(encoded) == raw


@pytest.mark.parametrize("family,adapter_type,http", ADAPTERS)
def test_model_without_image_capability_fails_before_transport(
    family, adapter_type, http
):
    """模型目录明确不支持图片时不得发送或静默删图；参数：协议组合；返回：无。"""
    adapter = adapter_type()
    with pytest.raises(
        ProviderAdapterError, match="unsupported_capability:image_input"
    ):
        list(
            adapter.stream(
                image_request(image_from_bytes(image_bytes())),
                model=_model(family, "test-provider", "text-only"),
                connection=_connection(family, "https://must-not-send.invalid"),
            )
        )


def test_background_restart_and_branch_use_frozen_image(tmp_path, monkeypatch):
    """接纳后原文件变化/删除，重启和分支仍发送原图；参数：临时目录与捕获器；返回：无。"""
    client = from_test_sequence(["已读取图片", "仍使用原图片"])
    descriptor = client._require_model_registry().require("production")
    client._model_registry = ModelRegistry(
        [
            replace(
                descriptor,
                capabilities={**descriptor.capabilities, Capability.IMAGE_INPUT: True},
            )
        ]
    )
    requests = capture_requests(client, monkeypatch)
    original = session_for(tmp_path, client, registry=ToolRegistry())
    path = tmp_path / "picture.png"
    raw = image_bytes()
    path.write_bytes(raw)
    with monkeypatch.context() as pending:
        pending.setattr(original.runtime, "_start_locked", lambda: None)
        original.submit(
            "检查图片",
            input_id="input-image",
            model_config={},
            attachment_paths=("picture.png",),
        )
    original.runtime.close()
    path.write_bytes(image_bytes("blue"))
    path.unlink()
    restarted = BackgroundSession(original.record, original.services)
    try:
        restarted.recover()
        assert restarted.runtime.wait_idle(10)
        assert len(requests) == 1
        delivery = next(
            row
            for row in restarted.messages.read_entries(restarted.record.session_id)
            if row.type == "delivery"
        )
        restarted.branch(delivery.entry_id)
        restarted.submit("再看原图", input_id="input-image-followup", model_config={})
        assert restarted.runtime.wait_idle(10)
        assert len(requests) == 2
        for request in requests:
            images = [
                part
                for message in request.messages
                for part in message.content
                if isinstance(part, ImagePart)
            ]
            assert len(images) == 1
            assert base64.b64decode(images[0].source_ref.split(",", 1)[1]) == raw
            assert (
                CapabilityRequirement(Capability.IMAGE_INPUT)
                in request.required_capabilities
            )
    finally:
        restarted.runtime.close()


def test_picture_mime_comes_from_content_and_permissions_are_kept(tmp_path):
    """扩展名不决定MIME，越界/显式拒绝/损坏图片均不能发送；参数：临时目录；返回：无。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path = workspace / "actually-png.jpg"
    path.write_bytes(image_bytes())
    lease = user_lease(
        task_id="picture", project_root=workspace, data_root=tmp_path / "data"
    )
    part = read_image_attachment(path.name, project_root=workspace, lease=lease)
    assert part.mime_type == "image/png"
    with pytest.raises(ValueError, match="REJECTED_PATH"):
        read_image_attachment("../outside.png", project_root=workspace, lease=lease)
    (workspace / "broken.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="损坏"):
        read_image_attachment("broken.png", project_root=workspace, lease=lease)
    # 【图片附件】【秘密保护】敏感文件即便内部是图片，也交给原脱敏读取器
    (workspace / ".env").write_bytes(image_bytes())
    assert read_image_attachment(".env", project_root=workspace, lease=lease) is None
    denied = replace(
        lease,
        capabilities={
            **lease.capabilities,
            "fs": {**lease.capabilities["fs"], "deny_read": ["*.jpg"]},
        },
    )
    with pytest.raises(ValueError, match="PERMISSION_DENIED"):
        read_image_attachment(path.name, project_root=workspace, lease=denied)


def test_image_estimate_is_independent_of_base64_compression():
    """同像素图片不同编码体积不会误算为文本token；参数：无；返回：无。"""
    first = image_from_bytes(image_bytes(size=(1024, 1024), compress_level=0))
    second = image_from_bytes(image_bytes(size=(1024, 1024), compress_level=9))
    assert len(first.source_ref) > len(second.source_ref) * 10
    estimates = [
        estimate_agent_messages_tokens((UserMessage("picture", (part,)),))
        for part in (first, second)
    ]
    assert estimates[0] == estimates[1]
    assert 1024 <= estimates[0] < 1200


@pytest.mark.parametrize("family,adapter_type,http", ADAPTERS)
def test_provider_rejects_unread_or_mislabeled_image(family, adapter_type, http):
    """路径和伪造MIME不允许在provider里静默漏图；参数：协议组合；返回：无。"""
    adapter = adapter_type()
    with pytest.raises(ProviderAdapterError, match="invalid_image_input"):
        adapter.build_request(
            image_request(ImagePart("C:/unread.png", "image/png")), model_id="picture"
        )
    real = image_from_bytes(image_bytes())
    forged = replace(
        real,
        mime_type="image/jpeg",
        source_ref=real.source_ref.replace("image/png", "image/jpeg"),
    )
    with pytest.raises(ProviderAdapterError, match="MIME"):
        adapter.build_request(image_request(forged), model_id="picture")


@pytest.mark.parametrize("family,adapter_type,http", ADAPTERS)
@pytest.mark.parametrize(
    "status,reason",
    [(400, "model does not support image input"), (413, "image request too large")],
)
def test_remote_image_rejection_preserves_error_without_retry(
    family, adapter_type, http, status, reason
):
    """未知端点拒绝图片或体积超限时保留真实错误且不切模型；参数：协议和远端错误；返回：无。"""
    calls = []

    def handler(request):
        """模拟供应商真实HTTP拒绝；参数：完整请求；返回：明确错误。"""
        calls.append(json.loads(request.content))
        return http.Response(
            status, json={"error": {"type": "invalid_request_error", "message": reason}}
        )

    with http.Client(transport=http.MockTransport(handler)) as client:
        model = _model(family, "test-provider", "image-rejected")
        model = replace(
            model, capabilities={**model.capabilities, Capability.IMAGE_INPUT: True}
        )
        result = StreamAssembler().assemble(
            list(
                adapter_type(http_client=client).stream(
                    image_request(image_from_bytes(image_bytes())),
                    model=model,
                    connection=_connection(family, "https://image.invalid/v1"),
                )
            )
        )
    assert result.message is None and result.error is not None
    assert result.error.http_status == status and reason in result.error.summary
    assert len(calls) == 1


def test_anthropic_image_dimensions_and_encoded_size_are_enforced(monkeypatch):
    """官方尺寸及编码字节边界在发送前拒绝；参数：局部测试上限替换；返回：无。"""
    tall = image_from_bytes(image_bytes(size=(1, 8001)))
    with pytest.raises(ProviderAdapterError, match="image_dimensions_exceeded"):
        AnthropicMessagesAdapter().build_request(
            image_request(replace(tall, width=None, height=None)), model_id="picture"
        )
    part = image_from_bytes(image_bytes())
    monkeypatch.setattr(
        "llm.providers.anthropic_messages.IMAGE_BASE64_MAX_BYTES",
        len(part.source_ref.split(",", 1)[1]) - 1,
    )
    with pytest.raises(ProviderAdapterError, match="image_too_large"):
        AnthropicMessagesAdapter().build_request(
            image_request(part), model_id="picture"
        )


@pytest.mark.parametrize("family,adapter_type,http", ADAPTERS)
def test_protocol_request_size_limit_is_checked_before_transport(
    family, adapter_type, http, monkeypatch
):
    """实际编码请求超协议体积上限明确失败；参数：协议与测试上限；返回：无。"""
    name = (
        "IMAGE_REQUEST_MAX_BYTES"
        if family == "anthropic_messages"
        else "OPENAI_IMAGE_REQUEST_BYTES"
    )
    monkeypatch.setattr(f"llm.providers.{family}.{name}", 1)
    with pytest.raises(ProviderAdapterError, match="image_request_too_large"):
        adapter_type().build_request(
            image_request(image_from_bytes(image_bytes())), model_id="picture"
        )
