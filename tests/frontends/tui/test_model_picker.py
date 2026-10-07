"""模型选择使用真实控件与配置，不发送外部模型请求。

作者：xxx
时间：2026-09-30 00:00:00
"""

import asyncio
import json

import pytest
from textual.widgets import Select

from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui
from frontends.tui.settings import SettingsScreen
from llm.profiles import load_model_profiles
from llm.public_config import public_model_config
from tests.frontends.tui.test_interactive import DisplayBridge
from tests.frontends.tui.test_settings import configured_bridge


def picker_data():
    """提供两个供应商、三模型的公开事实；参数：无；返回：设置快照。"""
    return {
        "session_id": "",
        "input_model": {"profile_name": "甲:first"},
        "approval_mode": "workspace",
        "profiles": [
            {
                "name": "甲:first",
                "provider": "proxy",
                "provider_group": "甲",
                "model": "custom",
            },
            {
                "name": "乙:second",
                "provider": "proxy",
                "provider_group": "乙",
                "model": "glm-5.2",
                "reasoning_options": ["high", "max"],
            },
            {
                "name": "乙:third",
                "provider": "proxy",
                "provider_group": "乙",
                "model": "custom",
            },
        ],
    }


def test_model_command_opens_keyboard_picker_and_f6_keeps_draft():
    """无参数命令与F6共用选择，键盘可选供应商及强度；参数：无；返回：无。"""

    async def scenario():
        bridge = DisplayBridge()
        bridge.settings = picker_data
        bridge.release.set()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(110, 50)) as pilot:
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("/model")
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            assert bridge.submitted == []
            provider = app.screen.query_one("#provider-choice", Select)
            provider.focus()
            await pilot.press("enter", "home", "enter")
            # 1. 按供应商排序通过键盘切到乙，仅显示乙的模型配置
            assert provider.value == "乙"
            await pilot.pause()
            assert app.screen.query_one("#model-choice", Select).value == "乙:second"
            effort = app.screen.query_one("#reasoning-choice", Select)
            effort.focus()
            await pilot.press("enter", "end", "enter")
            assert effort.value == "max"
            await pilot.click("#apply-model")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert bridge.submitted == [
                "/model profile use 乙:second reasoning_effort=max"
            ]
            composer.load_text("草稿保留")
            await pilot.press("f6")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, SettingsScreen)
            await pilot.press("escape")
            assert composer.text == "草稿保留"

    asyncio.run(scenario())


def test_profile_effort_reaches_new_input_and_preserves_previous(tmp_path, monkeypatch):
    """真实工厂选择强度后公开请求冻结新值；参数：隔离配置；返回：无。"""
    bridge, host, _ = configured_bridge(tmp_path, monkeypatch)
    config = load_model_profiles()
    raw = json.loads(config.path.read_text(encoding="utf-8"))
    raw["providers"]["proxy"]["models"]["new"]["model"] = "glm-5.2"
    config.path.write_text(json.dumps(raw), encoding="utf-8")
    previous = host.config.llm_client
    bridge.submit("/model profile use proxy:new reasoning_effort=max")
    assert public_model_config(host.config.llm_client)["reasoning_effort"] == "max"
    assert public_model_config(previous)["reasoning_effort"] == "default"
    assert load_model_profiles().active_profile.reasoning_effort == "max"
    bridge.submit("/model profile use proxy:new reasoning_effort=default")
    assert public_model_config(host.config.llm_client)["reasoning_effort"] == "default"
    assert load_model_profiles().active_profile.reasoning_effort is None


def test_invalid_effort_does_not_change_saved_profile(tmp_path, monkeypatch):
    """能力未知时拒绝强度，不持久化也不替换客户端；参数：隔离配置；返回：无。"""
    bridge, host, _ = configured_bridge(tmp_path, monkeypatch)
    previous = host.config.llm_client
    with pytest.raises(ValueError, match="unsupported reasoning_effort"):
        bridge.submit("/model profile use proxy:new reasoning_effort=high")
    assert load_model_profiles().active == "proxy:old"
    assert host.config.llm_client is previous


def test_json_provider_alias_groups_models_without_exposing_credentials(
    tmp_path, monkeypatch
):
    """真实JSON目录按冒号供应商分组，凭据不进入UI；参数：隔离配置；返回：无。"""
    bridge, host, _ = configured_bridge(tmp_path, monkeypatch)
    catalog = tmp_path / "models.json"
    provider = {
        "model_provider": "openai_compatible",
        "base_url": "https://example.invalid/v1",
        "credential": "private-reference",
        "models": {"one": {"model": "glm-5.2"}, "two": {"model": "custom"}},
    }
    catalog.write_text(
        json.dumps(
            {
                "active_provider": "proxy-a",
                "active_model": "one",
                "providers": {"proxy-a": provider, "proxy-b": provider},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("llm.profiles.MODELS_JSON_CONFIG_PATH", catalog)
    host.browse = lambda *args, **kwargs: {"session_id": host.session_id}
    data = bridge.settings()
    assert len(data["profiles"]) == 4
    assert {row["provider_group"] for row in data["profiles"]} == {"proxy-a", "proxy-b"}
    assert "private-reference" not in str(data)
    screen = SettingsScreen(data)
    assert [name for _, name in screen.model_options("proxy-a")] == [
        "proxy-a:one",
        "proxy-a:two",
    ]
