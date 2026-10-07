"""思考续工具的来源和相邻边界。

作者：xxx
"""

from dataclasses import replace

from llm.messages import AssistantMessage, TextPart, ThinkingPart, UserMessage
from llm.model_request import (
    Capability,
    CapabilityRequirement,
    ModelPreference,
    PreferenceKind,
)
from llm.providers.openai_chat import OpenAIChatAdapter
from tests.provider_test_support import tool_roundtrip_request
from schedules.store import ScheduleStore
from runtime.workspaces import WorkspaceStore
from llm.profiles import ModelProfile
from llm.public_config import public_model_config
from llm.providers.openai_responses import OpenAIResponsesAdapter
from types import SimpleNamespace
import pytest


def test_nonadjacent_matching_source_is_not_merged_into_tool_call():
    """来源编号相同但中间已有用户消息时不得跨消息合并；参数：无；返回：无。"""
    original = tool_roundtrip_request()
    thought = AssistantMessage(
        "request-source", (ThinkingPart("原来的思考", "visible"),)
    )
    intervention = UserMessage("intervention", (TextPart("后来的用户补充"),))
    calls = replace(original.messages[1], message_id="request-source:tool-calls")
    request = replace(
        original,
        messages=(
            original.messages[0],
            thought,
            intervention,
            calls,
            original.messages[2],
        ),
        required_capabilities=original.required_capabilities
        | {CapabilityRequirement(Capability.REASONING)},
        optional_preferences=(ModelPreference(PreferenceKind.REASONING_LEVEL, "high"),),
    )
    body = OpenAIChatAdapter().build_request(request, model_id="deepseek-v4.1-flash")
    rows = body["messages"]
    call_row = next(row for row in rows if row.get("tool_calls"))
    assert "reasoning_content" not in call_row
    assert any(row.get("content") == "后来的用户补充" for row in rows)
    assert any(row.get("reasoning_content") == "原来的思考" for row in rows)


@pytest.mark.parametrize("effort", ["default", "high"])
def test_schedule_roundtrip_keeps_reasoning_selection_after_profile_change(
    tmp_path, monkeypatch, effort
):
    """调度重读按冻结强度重建请求，不继承后来全局档位；参数：目录、替换器、档位；返回：无。"""
    import app.cli as cli

    profile = ModelProfile(
        "configured",
        "proxy",
        "https://example.test/v1",
        "o3",
        api_mode="responses",
        reasoning_effort="low",
    )
    monkeypatch.setattr(cli, "_load_named_model_profile", lambda name: profile)
    monkeypatch.setattr(cli, "load_project_llm_defaults", lambda root: {})
    monkeypatch.setattr(cli, "load_saved_config", lambda: {})
    monkeypatch.setattr(
        cli, "SecretsVault", lambda: SimpleNamespace(get=lambda name: None)
    )
    selected = cli.build_llm_client(
        {
            **profile.as_config(),
            "profile_name": profile.name,
            "reasoning_effort": effort,
        }
    )
    frozen = public_model_config(selected)
    ScheduleStore(tmp_path).create_schedule(
        "scheduled",
        "interval:60",
        prompt="核对资料",
        model_config=frozen,
        workspace_id=WorkspaceStore(tmp_path).register(tmp_path).workspace_id,
    )
    loaded = ScheduleStore(tmp_path).load_schedule("scheduled")
    assert loaded is not None and loaded.model_config == frozen
    rebuilt = cli.build_llm_client(loaded.model_config)
    wire = OpenAIResponsesAdapter().build_request(
        rebuilt.prepare_request("核对资料", {}).request, model_id="o3"
    )
    assert wire.get("reasoning") == (
        None if effort == "default" else {"effort": "high"}
    )
