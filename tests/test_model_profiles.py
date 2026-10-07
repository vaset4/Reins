"""models.json 目录读取、校验及模型选择回归。"""

from __future__ import annotations

import json

import pytest

from llm.profiles import load_model_profiles, switch_active_model_profile


def test_missing_json_does_not_load_old_yaml(tmp_path, monkeypatch):
    """传参：隔离目录；返回：无，旧文件不能成为默认模型来源。"""
    path = tmp_path / "models.json"
    (tmp_path / "models.yaml").write_text("active: old\nprofiles: {}", encoding="utf-8")
    monkeypatch.setattr("llm.profiles.MODELS_JSON_CONFIG_PATH", path)
    config = load_model_profiles()
    assert config.path == path
    assert config.source == "missing"
    assert config.active_profile is None
    assert config.profiles == {}


def test_loads_models_json_catalog(tmp_path):
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "active_provider": "elysiver",
                "active_model": "flash",
                "providers": {
                    "elysiver": {
                        "model_provider": "custom",
                        "provider_name": "elysiver",
                        "base_url": "https://elysiver.example/v1",
                        "credential": "elysiver_api_key",
                        "api_mode": "chat_completions",
                        "timeout_seconds": 60,
                        "models": {
                            "flash": {
                                "model": "deepseek-v4-flash",
                                "context_window": 300000,
                            },
                            "reasoning": {
                                "model": "deepseek-r1",
                                "context_window": 128000,
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_model_profiles(config_path)
    profile = config.active_profile

    assert config.source == "models_json"
    assert config.active == "elysiver:flash"
    assert set(config.profiles) == {"elysiver:flash", "elysiver:reasoning"}
    assert profile is not None
    assert profile.provider == "elysiver"
    assert profile.base_url == "https://elysiver.example/v1"
    assert profile.model == "deepseek-v4-flash"
    assert profile.credential == "elysiver_api_key"
    assert profile.context_window == 300000
    assert profile.timeout_seconds == 60


def test_models_json_switches_active_provider_model(tmp_path):
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "active_provider": "elysiver",
                "active_model": "flash",
                "providers": {
                    "elysiver": {
                        "model_provider": "custom",
                        "base_url": "https://elysiver.example/v1",
                        "models": {
                            "flash": {"model": "deepseek-v4-flash"},
                            "reasoning": {"model": "deepseek-r1"},
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    switch_active_model_profile("elysiver:reasoning", config_path)
    config = load_model_profiles(config_path)

    assert config.active == "elysiver:reasoning"
    assert config.active_profile is not None
    assert config.active_profile.model == "deepseek-r1"


def test_models_json_accepts_trailing_commas(tmp_path):
    config_path = tmp_path / "models.json"
    config_path.write_text(
        """
        {
          "active_provider": "nvidia",
          "active_model": "kimi",
          "providers": {
            "nvidia": {
              "model_provider": "custom",
              "provider_name": "nvidia",
              "base_url": "https://integrate.api.nvidia.com/v1",
              "credential": "nvidia_api_key",
              "models": {
                "kimi": {
                  "model": "moonshotai/kimi-k2.6",
                  "context_window": 300000,
                },
              },
            },
          }
        }
        """,
        encoding="utf-8",
    )

    config = load_model_profiles(config_path)

    assert config.active == "nvidia:kimi"
    assert config.active_profile is not None
    assert config.active_profile.model == "moonshotai/kimi-k2.6"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("api_key", "synthetic-secret", "credential fields"),
        ("api_mode", "unknown", "unsupported api_mode"),
        ("context_window", "invalid", "context_window"),
        ("base_url", "", "base_url"),
    ],
)
def test_rejects_invalid_json_provider(tmp_path, field, value, message):
    """传参：非法配置字段；返回：无，JSON 同样校验凭据与模型参数。"""
    provider = {
        "model_provider": "custom",
        "base_url": "https://example.test/v1",
        "models": {"main": {"model": "test-model"}},
        field: value,
    }
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "active_provider": "proxy",
                "active_model": "main",
                "providers": {"proxy": provider},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_model_profiles(path)


def test_explicit_yaml_is_not_parsed(tmp_path):
    """传参：旧格式文件；返回：无，显式旧配置不能继续被解析。"""
    path = tmp_path / "models.yaml"
    path.write_text("active: old\nprofiles: {}", encoding="utf-8")
    with pytest.raises(ValueError, match="failed to parse"):
        load_model_profiles(path)


@pytest.mark.parametrize("entry", ["cli", "probe"])
def test_entries_ignore_old_model_files(tmp_path, monkeypatch, entry):
    """传参：入口和隔离配置；返回：无，旧模型文件不能覆盖 JSON 选择。"""
    from app import cli
    from scripts import probe_llm_api

    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "active_provider": "proxy",
                "active_model": "main",
                "providers": {
                    "proxy": {
                        "model_provider": "custom",
                        "base_url": "http://localhost/v1",
                        "credential": "fixture-key",
                        "models": {"main": {"model": "current-model"}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "llm.json").write_text('{"model": "old-project"}', encoding="utf-8")
    (tmp_path / "config.yaml").write_text("llm:\n  model: old-user\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("llm.profiles.MODELS_JSON_CONFIG_PATH", path)
    monkeypatch.setattr(cli, "SecretsVault", lambda: {"fixture-key": "synthetic-token"})
    monkeypatch.setattr(
        "reins_secrets.store.SecretsVault", lambda: {"fixture-key": "synthetic-token"}
    )
    monkeypatch.delenv("XIANGMU_LLM_MODEL", raising=False)
    monkeypatch.delenv("XIANGMU_LLM_BASE_URL", raising=False)
    if entry == "cli":
        target = cli.build_llm_client({}, project_root=tmp_path).resolved_target
    else:
        target = probe_llm_api._resolve_target(probe_llm_api._parse_args([]))
    assert target.model == "current-model"
    assert target.profile_name == "proxy:main"
    assert target.api_key == "synthetic-token"
