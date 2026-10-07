"""模型选择必须贯通配置、冻结快照和实际发送参数。

作者：xxx
时间：2026-09-30 00:00:00
"""

from dataclasses import replace
import json

import pytest

from llm.profiles import load_model_profiles, switch_active_model_profile
from llm.reasoning import reasoning_options
from llm.resolved_target import resolve_model_target
from llm.client import RealLLMClient
from llm.config import LLMProviderConfig
from llm.public_config import public_model_config
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.providers.openai_responses import OpenAIResponsesAdapter
from llm.providers.anthropic_messages import AnthropicMessagesAdapter
from llm.model_request import ModelPreference, PreferenceKind
from tests.provider_test_support import text_request


def test_selection_persists_once_and_preserves_frozen_request(tmp_path):
    """传参：临时目录；返回：无，验证切换后旧请求不会读到新强度。"""
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "active_provider": "proxy",
                "active_model": "a",
                "providers": {
                    "proxy": {
                        "model_provider": "custom",
                        "base_url": "https://proxy.test/v1",
                        "credential": "proxy-key",
                        "models": {
                            "a": {"model": "grok-4.7"},
                            "b": {"model": "glm-5.3-flash"},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    switch_active_model_profile("proxy:b", path, reasoning_effort="low")
    profile = load_model_profiles(path).active_profile
    assert profile is not None and profile.reasoning_effort == "low"
    target = replace(
        resolve_model_target(cli_overrides=profile.as_config()),
        profile_name=profile.name,
    )
    client = RealLLMClient(
        config=LLMProviderConfig(target.base_url, target.model, None, 30),
        resolved_target=target,
    )
    frozen = public_model_config(client)
    request = client.prepare_request("你好", {}).request
    assert (
        OpenAIChatAdapter().build_request(request, model_id=profile.model)[
            "reasoning_effort"
        ]
        == "low"
    )
    switch_active_model_profile("proxy:b", path, reasoning_effort="max")
    assert frozen["reasoning_effort"] == "low"
    switch_active_model_profile("proxy:b", path, reasoning_effort="default")
    assert load_model_profiles(path).active_profile.reasoning_effort is None
    assert (
        "reasoning_effort"
        not in json.loads(path.read_text())["providers"]["proxy"]["models"]["b"]
    )


@pytest.mark.parametrize(
    ("model", "values"),
    [
        ("glm-5.2", ("high", "max")),
        ("glm-5.3-flash", ("low", "high", "max")),
        ("grok-4.7", ("low", "medium", "high", "xhigh")),
        ("deepseek-v4.1-flash", ("none", "low", "high", "max")),
        ("unknown", ()),
    ],
)
def test_only_documented_levels_are_selectable(model, values):
    """传参：模型及真实档位；返回：无，不为代理供应商臆造通用强度。"""
    assert reasoning_options(model, "chat_completions") == values


def test_invalid_selection_does_not_change_file(tmp_path):
    """传参：临时目录；返回：无，错误强度不先切换活动模型。"""
    path = tmp_path / "models.json"
    text = json.dumps(
        {
            "active_provider": "proxy",
            "active_model": "a",
            "providers": {
                "proxy": {
                    "model_provider": "custom",
                    "base_url": "https://example.test",
                    "models": {"a": {"model": "glm-5.2"}},
                }
            },
        }
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="reasoning_effort"):
        switch_active_model_profile("proxy:a", path, reasoning_effort="low")
    assert path.read_text(encoding="utf-8") == text


def test_responses_wire_and_default_snapshot():
    """传参：无；返回：无，验证Responses独立结构和显式默认快照。"""
    target = resolve_model_target(
        cli_overrides={
            "model": "o3",
            "base_url": "https://example.test",
            "api_mode": "responses",
            "reasoning_effort": "high",
        }
    )
    client = RealLLMClient(
        config=LLMProviderConfig(target.base_url, target.model, None, 30),
        resolved_target=target,
    )
    body = OpenAIResponsesAdapter().build_request(
        client.prepare_request("你好", {}).request, model_id="o3"
    )
    assert body["reasoning"] == {"effort": "high"}
    default = RealLLMClient(
        config=LLMProviderConfig(target.base_url, target.model, None, 30),
        resolved_target=replace(target, reasoning_effort=None),
    )
    assert public_model_config(default)["reasoning_effort"] == "default"


@pytest.mark.parametrize(
    ("mode", "adapter", "expected"),
    [
        (
            "chat_completions",
            OpenAIChatAdapter(),
            {
                "reasoning_effort": "max",
                "extra_body": {"thinking": {"type": "enabled"}},
            },
        ),
        ("responses", OpenAIResponsesAdapter(), {"reasoning": {"effort": "max"}}),
        (
            "anthropic_messages",
            AnthropicMessagesAdapter(),
            {"thinking": {"type": "enabled"}, "output_config": {"effort": "max"}},
        ),
    ],
)
def test_deepseek_wire_keeps_protocol_specific_parameters(mode, adapter, expected):
    """传参：协议及预期参数；返回：无，三个API不会误用另一家的参数。"""
    request = replace(
        text_request(),
        optional_preferences=(ModelPreference(PreferenceKind.REASONING_LEVEL, "max"),),
    )
    body = adapter.build_request(request, model_id="deepseek-v4.1-flash")
    assert {key: body[key] for key in expected} == expected


def test_background_rebuild_honors_frozen_default(monkeypatch):
    """传参：隔离替身；返回：无，后台重建不从已更改profile继承思考强度。"""
    import app.cli as cli
    from llm.profiles import ModelProfile
    from types import SimpleNamespace

    profile = ModelProfile(
        "proxy:b",
        "proxy",
        "http://localhost/v1",
        "glm-5.3-flash",
        reasoning_effort="max",
    )
    monkeypatch.setattr(cli, "_load_named_model_profile", lambda name: profile)
    monkeypatch.setattr(
        cli, "SecretsVault", lambda: SimpleNamespace(get=lambda name: None)
    )
    frozen = {
        **profile.as_config(),
        "profile_name": profile.name,
        "reasoning_effort": "default",
    }
    rebuilt = cli.build_llm_client(frozen)
    assert rebuilt.resolved_target.reasoning_effort is None
    assert not rebuilt.prepare_request("你好", {}).request.optional_preferences


def test_unsupported_model_explicit_effort_is_never_silently_dropped():
    """传参：无；返回：无，绕过界面的不支持档位仍被Adapter明确拒绝。"""
    request = replace(
        text_request(),
        optional_preferences=(ModelPreference(PreferenceKind.REASONING_LEVEL, "max"),),
    )
    with pytest.raises(ValueError, match="unsupported reasoning_effort"):
        OpenAIChatAdapter().build_request(request, model_id="unknown")


def test_real_sdk_serializes_deepseek_extension_into_http_body():
    """传参：无；返回：无，离线HTTP捕获证明SDK真正发送扩展字段。"""
    import httpx
    from llm.provider_connection import ResolvedConnection

    captured = []

    def respond(request):
        """传参：SDK请求；返回：真实SSE形状响应，同时保存已编码正文。"""
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text='data: {"id":"a","choices":[{"delta":{"content":"好"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
        )

    request = replace(
        text_request(),
        optional_preferences=(ModelPreference(PreferenceKind.REASONING_LEVEL, "low"),),
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        adapter = OpenAIChatAdapter(http_client=transport)
        body = adapter.build_request(request, model_id="deepseek-v4.1-flash")
        list(
            adapter._raw_stream(
                body, ResolvedConnection("https://example.test/v1", 5, "test-only", {})
            )
        )
    assert captured[0]["reasoning_effort"] == "low"
    assert captured[0]["thinking"] == {"type": "enabled"}
    assert "extra_body" not in captured[0]


def test_reasoning_preamble_merges_only_with_matching_request():
    """传参：无；返回：无，多工具共用本轮来源，邻接其他轮思考不能冒充调用前言。"""
    from llm.messages import (
        AssistantMessage,
        ThinkingPart,
        ToolCallPart,
        ToolResultMessage,
        TextPart,
    )
    from llm.model_request import Capability, CapabilityRequirement
    from tests.provider_test_support import tool_roundtrip_request

    original = tool_roundtrip_request()
    source = AssistantMessage("request-a", (ThinkingPart("本轮思考", "visible"),))
    calls = AssistantMessage(
        "request-a:tool-calls",
        (ToolCallPart("c1", "read", {}), ToolCallPart("c2", "read", {})),
    )
    result2 = ToolResultMessage("m4", "c2", "read", (TextPart("结果二"),), "success")
    request = replace(
        original,
        messages=(original.messages[0], source, calls, original.messages[2], result2),
        required_capabilities=original.required_capabilities
        | {CapabilityRequirement(Capability.REASONING)},
        optional_preferences=(ModelPreference(PreferenceKind.REASONING_LEVEL, "high"),),
    )
    body = OpenAIChatAdapter().build_request(request, model_id="deepseek-v4.1-flash")
    row = next(row for row in body["messages"] if row.get("tool_calls"))
    assert len(row["tool_calls"]) == 2 and row["reasoning_content"] == "本轮思考"
    mismatch = replace(
        request,
        messages=(
            request.messages[0],
            replace(source, message_id="other"),
            *request.messages[2:],
        ),
    )
    body = OpenAIChatAdapter().build_request(mismatch, model_id="deepseek-v4.1-flash")
    assert "reasoning_content" not in next(
        row for row in body["messages"] if row.get("tool_calls")
    )


def test_failed_atomic_replace_preserves_original_selection(tmp_path, monkeypatch):
    """传参：临时目录和替换器；返回：无，文件替换失败不破坏原选择且清理临时文件。"""
    import llm.model_catalog as catalog

    path = tmp_path / "models.json"
    original = json.dumps(
        {
            "active_provider": "proxy",
            "active_model": "a",
            "providers": {
                "proxy": {
                    "model_provider": "custom",
                    "base_url": "https://example.test",
                    "models": {"a": {"model": "glm-5.2"}},
                }
            },
        }
    )
    path.write_text(original, encoding="utf-8")

    def fail_replace(source, target):
        """传参：文件路径；返回：无，模拟Windows占用目标文件。"""
        raise PermissionError("locked")

    monkeypatch.setattr(catalog.os, "replace", fail_replace)
    with pytest.raises(PermissionError, match="locked"):
        switch_active_model_profile("proxy:a", path, reasoning_effort="max")
    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [path]
