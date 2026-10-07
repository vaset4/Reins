from __future__ import annotations
from scripts.testing.llm import (
    from_test_error,
    from_test_native_tool_calls,
    from_test_stub,
    from_test_text_json_stub,
    from_test_turns,
)

import pytest

from llm.client import RealLLMClient
from scripts.testing.llm import ScriptedTurnOptions, scripted_provider_error
from llm.messages import (
    AssistantMessage,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
)
from tools.tool_registry import (
    IDEMPOTENT_YES,
    TARGET_SCOPE_LOGICAL,
    TOOL_RISK_SAFE,
    TOOL_SOURCE_BUILTIN,
    ToolDefinition,
    ToolRegistry,
)


def test_real_llm_client_returns_final_output_from_provider_stub() -> None:
    client = from_test_stub(
        provider_text='{"type":"final","content":"hello real model"}'
    )

    plan = client.plan("say hello")

    assert plan.final_output == "hello real model"
    assert plan.run_tools_request is None
    assert plan.render_text_to_model.startswith("[system]\n")
    assert "tool_history_recent" not in plan.prompt_context


def test_real_llm_client_native_prompt_omits_text_parameter_catalog() -> None:
    client = from_test_stub(
        provider_text='{"type":"final","content":"hello real model"}'
    )

    plan = client.plan("summarize tools")

    assert "native tool schemas" in plan.render_text_to_model
    assert "Tool catalog:" not in plan.render_text_to_model
    assert "parameters=" not in plan.render_text_to_model
    assert '"tool":"list","arguments":{"path":"tools"}' not in (
        plan.render_text_to_model
    )
    assert "ask_user tool call" in plan.render_text_to_model
    assert "Allowed model actions this turn:" in plan.render_text_to_model
    assert plan.request_bundle_evidence["tool_selection"]["selected"]


def test_real_llm_client_project_artifact_prompt_requires_grounding() -> None:
    """独立论文工作区不被当成Reins源码，不强制读取不存在的文档；参数：无；返回：无。"""
    client = from_test_stub(provider_text='{"type":"final","content":"ok"}')

    plan = client.plan("讲解当前文件夹中的论文，并告诉我你是谁")

    prompt = plan.render_text_to_model
    system_prompt = prompt.split("[user]", 1)[0]
    assert (
        "relevant materials actually available in the current workspace"
        in system_prompt
    )
    assert "Describe your identity from your system instructions" in system_prompt
    assert "read README.md and docs/Architecture.md" not in system_prompt
    assert "not the Reins source repository" in system_prompt
    assert "artifact_output_dir is a managed scratch workspace" in prompt
    assert "prefer that directory" not in prompt


def test_real_llm_client_returns_run_tools_request_from_provider_stub() -> None:
    client = from_test_text_json_stub(
        provider_text='{"type":"run_tools","tool":"list","arguments":{"path":"runtime"}}'
    )

    plan = client.plan("inspect runtime")

    assert plan.run_tools_request is not None
    assert plan.run_tools_request.action == "list"
    assert plan.run_tools_request.arguments == {"path": "runtime"}
    assert plan.render_text_to_model.startswith("[system]\n")


def test_real_llm_client_returns_native_tool_call_from_provider_stub() -> None:
    client = from_test_native_tool_calls(
        [ToolCallPart("call-native-1", "list", {"path": "tools"})]
    )

    plan = client.plan("inspect tools")

    assert plan.run_tools_request is not None
    assert plan.run_tools_request.tool_name == "list"
    assert plan.run_tools_request.arguments == {"path": "tools"}
    assert plan.run_tools_request.call_id == "call-native-1"
    assert plan.raw_model_response["tool_calls"][0]["id"] == "call-native-1"


def test_real_llm_client_uses_runtime_context_registry_for_tool_schemas() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="custom_tool",
            description="Custom context-bound tool.",
            parameters={},
            toolset="custom",
            risk_level=TOOL_RISK_SAFE,
            readonly=True,
            target_scope_rule=TARGET_SCOPE_LOGICAL,
            source=TOOL_SOURCE_BUILTIN,
            idempotent=IDEMPOTENT_YES,
            executor=lambda _args: {"ok": True},
        )
    )
    client = from_test_stub(provider_text='{"type":"final","content":"ok"}')

    plan = client.plan(
        "use custom",
        context={
            "tool_registry": registry,
            "toolset_policy": {
                "enabled_toolsets": ["full"],
                "source": "payload",
            },
        },
    )

    tools = plan.raw_model_request["tools"]
    assert isinstance(tools, tuple)
    assert [tool["function"]["name"] for tool in tools] == ["custom_tool"]


def test_real_llm_client_plain_text_is_final_answer_in_native_mode() -> None:
    client = from_test_stub(provider_text="plain final answer")

    plan = client.plan("say hello")

    assert plan.final_output == "plain final answer"
    assert plan.model_error is None


def test_real_llm_client_returns_provider_error_as_stable_output() -> None:
    client = from_test_error("provider unavailable")

    plan = client.plan("say hello")

    assert plan.final_output == "MODEL_PROVIDER_ERROR: provider unavailable"
    assert plan.model_error is not None
    assert plan.model_error.category == "provider_error"
    assert plan.model_error.retryable is False
    assert plan.observation is not None
    assert plan.observation.attempt_count == 1
    assert plan.observation.was_retried is False


def test_real_llm_client_does_not_retry_non_retryable_error() -> None:
    # 脚本只排一轮错误轮：真发生第二次调用会用尽脚本并改写摘要，attempt_count 也会变
    client = from_test_turns(
        [scripted_provider_error("provider_error", "provider unavailable")]
    )

    plan = client.plan("say hello")

    assert plan.final_output == "MODEL_PROVIDER_ERROR: provider unavailable"
    assert plan.observation is not None
    assert plan.observation.attempt_count == 1
    assert plan.observation.was_retried is False


def test_real_llm_client_retries_once_for_retryable_error() -> None:
    # 首轮可重试超时、次轮成功；脚本只有两轮，多重试一次就会用尽脚本拿不到这段终稿
    client = from_test_turns(
        [
            scripted_provider_error("timeout", "timed out", retryable=True),
            '{"type":"final","content":"hello after retry"}',
        ]
    )

    plan = client.plan("say hello")

    assert plan.final_output == "hello after retry"
    assert plan.observation is not None
    assert plan.observation.attempt_count == 2
    assert plan.observation.was_retried is True
    assert plan.observation.success is True


def test_real_llm_client_returns_stable_provider_error_after_retry_failure() -> None:
    # 每次调用都撞同一个可重试超时，重试预算烧完后错误分类与摘要仍要原样交出
    client = from_test_error("timed out", category="timeout", retryable=True)

    plan = client.plan("say hello")

    assert plan.final_output == "MODEL_PROVIDER_ERROR: timed out"
    assert plan.model_error is not None
    assert plan.model_error.category == "timeout"
    assert plan.observation is not None
    assert plan.observation.attempt_count == 3
    assert plan.observation.was_retried is True
    assert plan.observation.success is False


def test_real_llm_client_does_not_retry_context_overflow_error() -> None:
    # 本轮消息本就很短，裁剪压不出更小的 token，溢出错误就地收口不再发第二次
    client = from_test_turns(
        [scripted_provider_error("context_overflow", "maximum context length exceeded")]
    )

    plan = client.plan("say hello")

    assert plan.final_output == "MODEL_PROVIDER_ERROR: maximum context length exceeded"
    assert plan.model_error is not None
    assert plan.model_error.category == "context_overflow"
    assert plan.observation is not None
    assert plan.observation.attempt_count == 1
    assert plan.observation.was_retried is False


def test_real_llm_client_exposes_overflow_without_discarding_tool_evidence() -> None:
    """没有Session摘要边界的独立调用不能删掉工具证据冒充成功；传参：无；返回：无。"""
    client = from_test_turns(
        ['{"type":"final","content":"ok"}'],
        options=ScriptedTurnOptions(context_window=200),
    )

    plan = client.plan(
        "continue",
        context={
            "conversation_history": (
                AssistantMessage(
                    "a1",
                    (ToolCallPart("call-1", "file_read", {"path": "big.txt"}),),
                ),
                ToolResultMessage(
                    "t1",
                    "call-1",
                    "file_read",
                    (TextPart(" ".join(f"word{i}" for i in range(1000))),),
                    "success",
                ),
            ),
        },
    )

    assert plan.model_error.category == "context_overflow"
    assert plan.observation.attempt_count == 0
    assert plan.trim_delta is None
    assert "word999" in plan.render_text_to_model


def test_cli_can_build_real_llm_client_from_config(monkeypatch) -> None:
    from app.cli import build_llm_client

    monkeypatch.setenv("XIANGMU_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("XIANGMU_LLM_MODEL", "local-model")

    client = build_llm_client(cli_overrides={})

    assert isinstance(client, RealLLMClient)


def test_cli_vault_initialization_failure_is_visible(monkeypatch) -> None:
    import app.cli as cli

    monkeypatch.setenv("XIANGMU_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("XIANGMU_LLM_MODEL", "local-model")
    monkeypatch.setattr(cli, "load_project_llm_defaults", lambda _root: {})
    monkeypatch.setattr(cli, "load_saved_config", lambda: {})
    monkeypatch.setattr(cli, "_load_active_model_profile", lambda: None)
    monkeypatch.setattr(
        cli,
        "SecretsVault",
        lambda: (_ for _ in ()).throw(RuntimeError("vault unavailable")),
    )

    with pytest.raises(RuntimeError, match="secrets vault unavailable"):
        cli.build_llm_client(cli_overrides={})


def test_cli_builds_client_from_active_models_json_profile(monkeypatch) -> None:
    import app.cli as cli
    from llm.profiles import ModelProfile, ModelProfilesConfig

    profile = ModelProfile(
        name="elysiver:flash",
        provider="elysiver",
        base_url="https://elysiver.example/v1",
        model="deepseek-v4-flash",
        credential="elysiver_api_key",
        context_window=300000,
        timeout_seconds=60,
    )
    profiles = ModelProfilesConfig(
        path=cli.Path("models.json"),
        active=profile.name,
        profiles={profile.name: profile},
        source="models_json",
    )

    monkeypatch.setattr(cli, "load_model_profiles", lambda: profiles)
    monkeypatch.setattr(cli, "load_project_llm_defaults", lambda _root: {})
    monkeypatch.setattr(cli, "load_saved_config", lambda: {})

    client = cli.build_llm_client(cli_overrides={})

    assert isinstance(client, RealLLMClient)
    assert client.resolved_target is not None
    assert client.resolved_target.provider == "elysiver"
    assert client.resolved_target.model == "deepseek-v4-flash"
    assert client.resolved_target.context_window == 300000
    assert client.resolved_target.profile_name == "elysiver:flash"
    assert client.resolved_target.credential_name == "elysiver_api_key"

    from app.repl import _describe_model, _describe_provider

    assert _describe_provider(client) == "elysiver"
    assert _describe_model(client) == "deepseek-v4-flash"
