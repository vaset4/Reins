"""Tests for ~/.reins/models.yaml provider profiles."""

from __future__ import annotations

import json

import pytest

from llm.profiles import (
    load_model_profiles,
    save_model_profile,
    switch_active_model_profile,
)


def test_missing_models_yaml_returns_empty_config(tmp_path):
    config = load_model_profiles(tmp_path / "models.yaml")

    assert config.active == ""
    assert config.active_profile is None
    assert config.profiles == {}


def test_loads_active_profile(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        "\n".join(
            [
                "active: glm-main",
                "profiles:",
                "  glm-main:",
                "    provider: openai_compatible",
                "    base_url: https://provider.example/v1",
                "    model: glm-5.1",
                "    credential: openai_compatible_api_key",
                "    api_mode: chat_completions",
                "    context_window: 30000",
            ]
        ),
        encoding="utf-8",
    )

    config = load_model_profiles(config_path)
    profile = config.active_profile

    assert profile is not None
    assert profile.name == "glm-main"
    assert profile.provider == "openai_compatible"
    assert profile.base_url == "https://provider.example/v1"
    assert profile.model == "glm-5.1"
    assert profile.credential == "openai_compatible_api_key"
    assert profile.as_config()["context_window"] == 30000


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
    ("content", "message"),
    [
        ("[]\n", "must contain a YAML object"),
        (
            "active: missing\nprofiles: {}\n",
            "profiles must contain at least one profile",
        ),
        (
            "active: missing\nprofiles:\n  glm:\n    provider: openai_compatible\n    base_url: https://x/v1\n    model: glm\n",
            "active profile not found",
        ),
        (
            "active: glm\nprofiles:\n  glm:\n    provider: openai_compatible\n    model: glm\n",
            "base_url",
        ),
        (
            "active: glm\nprofiles:\n  glm:\n    provider: openai_compatible\n    base_url: https://x/v1\n    model: glm\n    api_mode: unknown\n",
            "unsupported api_mode",
        ),
        (
            "active: glm\nprofiles:\n  glm:\n    provider: openai_compatible\n    base_url: https://x/v1\n    model: glm\n    context_window: no\n",
            "context_window",
        ),
        (
            "active: glm\nprofiles:\n  glm:\n    provider: openai_compatible\n    base_url: https://x/v1\n    model: glm\n    api_key: sk-test\n",
            "credential fields",
        ),
    ],
)
def test_rejects_invalid_models_yaml(tmp_path, content, message):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_model_profiles(config_path)


def test_save_profile_writes_non_secret_fields(tmp_path):
    config_path = tmp_path / "models.yaml"

    save_model_profile(
        "glm-main",
        {
            "provider": "openai_compatible",
            "base_url": "https://provider.example/v1",
            "model": "glm-5.1",
            "credential": "openai_compatible_api_key",
        },
        config_path,
    )

    text = config_path.read_text(encoding="utf-8")
    config = load_model_profiles(config_path)

    assert config.active == "glm-main"
    assert "openai_compatible_api_key" in text
    assert "api_key:" not in text
    assert "sk-" not in text


def test_save_profile_rejects_raw_secret_fields(tmp_path):
    with pytest.raises(ValueError, match="credential fields"):
        save_model_profile(
            "glm-main",
            {
                "provider": "openai_compatible",
                "base_url": "https://provider.example/v1",
                "model": "glm-5.1",
                "api_key": "sk-test",
            },
            tmp_path / "models.yaml",
        )


def test_switch_active_profile_preserves_profiles(tmp_path):
    config_path = tmp_path / "models.yaml"
    save_model_profile(
        "first",
        {
            "provider": "openai_compatible",
            "base_url": "https://first.example/v1",
            "model": "first-model",
        },
        config_path,
    )
    save_model_profile(
        "second",
        {
            "provider": "openai_compatible",
            "base_url": "https://second.example/v1",
            "model": "second-model",
        },
        config_path,
    )

    switch_active_model_profile("second", config_path)
    config = load_model_profiles(config_path)

    assert config.active == "second"
    assert set(config.profiles) == {"first", "second"}
