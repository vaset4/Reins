"""Tests for llm/resolved_target.py — provider runtime config rules."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm.resolved_target import (
    CredentialResolutionError,
    infer_provider,
    resolve_model_target,
    sanitize_base_url,
)


class TestResolveModelTarget:
    def test_cli_overrides_all_layers(self, monkeypatch):
        monkeypatch.setenv("XIANGMU_LLM_MODEL", "env-model")
        target = resolve_model_target(
            cli_overrides={"model": "cli-model", "base_url": "http://cli.local/v1"},
            saved_config={"model": "saved-model"},
            file_defaults={"model": "file-model", "base_url": "http://file.local/v1"},
        )
        assert target.model == "cli-model"
        assert target.base_url == "http://cli.local/v1"
        assert target.config_source == "cli"

    def test_saved_config_overrides_env(self, monkeypatch):
        monkeypatch.setenv("XIANGMU_LLM_MODEL", "env-model")
        target = resolve_model_target(
            cli_overrides={},
            saved_config={"model": "saved-model", "base_url": "http://saved.local/v1"},
            file_defaults={"model": "file-model"},
        )
        assert target.model == "saved-model"
        assert target.config_source == "saved_config"

    def test_env_overrides_file_defaults(self, monkeypatch):
        monkeypatch.setenv("XIANGMU_LLM_MODEL", "env-model")
        monkeypatch.setenv("XIANGMU_LLM_BASE_URL", "http://env.local/v1")
        target = resolve_model_target(
            cli_overrides={},
            saved_config={},
            file_defaults={"model": "file-model", "base_url": "http://file.local/v1"},
        )
        assert target.model == "env-model"
        assert target.config_source == "env"

    def test_file_defaults_used_when_nothing_else(self, monkeypatch):
        monkeypatch.delenv("XIANGMU_LLM_MODEL", raising=False)
        monkeypatch.delenv("XIANGMU_LLM_BASE_URL", raising=False)
        target = resolve_model_target(
            cli_overrides={},
            saved_config={},
            file_defaults={"model": "file-model", "base_url": "http://file.local/v1"},
        )
        assert target.model == "file-model"
        assert target.config_source == "file_default"
        assert target.context_window == 30000
        assert target.context_window_source == "builtin_default"
        assert target.context_window_defaulted is True

    def test_api_mode_is_chat_completions(self, monkeypatch):
        monkeypatch.delenv("XIANGMU_LLM_MODEL", raising=False)
        monkeypatch.delenv("XIANGMU_LLM_BASE_URL", raising=False)
        target = resolve_model_target(
            cli_overrides={"model": "m", "base_url": "http://x/v1"},
        )
        assert target.api_mode == "chat_completions"

    def test_context_window_source_is_observable(self, monkeypatch):
        monkeypatch.setenv("XIANGMU_LLM_CONTEXT_WINDOW", "128000")
        target = resolve_model_target(
            cli_overrides={"model": "m", "base_url": "http://x/v1"},
        )
        assert target.context_window == 128000
        assert target.context_window_source == "env"
        assert target.context_window_defaulted is False

    def test_invalid_context_window_is_not_defaulted(self):
        with pytest.raises(ValueError, match="context_window from saved_config"):
            resolve_model_target(
                cli_overrides={"model": "m", "base_url": "http://x/v1"},
                saved_config={"context_window": "no"},
            )


class TestInferProvider:
    def test_explicit_provider_wins(self):
        assert infer_provider("https://api.openai.com/v1", "anthropic") == "anthropic"

    def test_openai_host_inferred(self):
        assert infer_provider("https://api.openai.com/v1", None) == "openai"

    def test_anthropic_host_inferred(self):
        assert infer_provider("https://api.anthropic.com/v1", None) == "anthropic"

    def test_unknown_host_falls_back_to_openai_compatible(self):
        assert infer_provider("http://localhost:11434/v1", None) == "openai_compatible"

    def test_empty_explicit_provider_treated_as_none(self):
        assert infer_provider("https://api.openai.com/v1", "") == "openai"


class TestSanitizeBaseUrl:
    def test_strips_path_keeps_scheme_and_host(self):
        assert (
            sanitize_base_url("https://api.openai.com/v1/chat")
            == "https://api.openai.com"
        )

    def test_handles_localhost_with_port(self):
        assert sanitize_base_url("http://localhost:11434/v1") == "http://localhost"

    def test_returns_input_when_no_hostname(self):
        assert sanitize_base_url("not-a-url") == "not-a-url"


class TestCredentialResolution:
    def test_cli_api_key_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("XIANGMU_LLM_API_KEY", "env-key")
        target = resolve_model_target(
            cli_overrides={
                "api_key": "cli-key",
                "model": "m",
                "base_url": "http://x/v1",
            },
        )
        assert target.api_key_present is True
        assert target.credential_source == "cli"

    def test_vault_provider_prefix_used_when_available(self, monkeypatch):
        monkeypatch.delenv("XIANGMU_LLM_API_KEY", raising=False)
        vault = MagicMock()
        vault.get = MagicMock(
            side_effect=lambda key: "scoped-key" if key == "openai_api_key" else None
        )
        target = resolve_model_target(
            cli_overrides={"model": "m", "base_url": "https://api.openai.com/v1"},
            secrets_vault=vault,
        )
        assert target.api_key_present is True
        assert target.credential_source == "secrets_vault"

    def test_vault_generic_key_when_provider_prefix_absent(self, monkeypatch):
        monkeypatch.delenv("XIANGMU_LLM_API_KEY", raising=False)
        vault = MagicMock()
        vault.get = MagicMock(
            side_effect=lambda key: "generic-key" if key == "llm_api_key" else None
        )
        target = resolve_model_target(
            cli_overrides={"model": "m", "base_url": "http://x/v1"},
            secrets_vault=vault,
        )
        assert target.api_key_present is True
        assert target.credential_source == "secrets_vault"

    def test_env_used_when_vault_empty(self, monkeypatch):
        monkeypatch.setenv("XIANGMU_LLM_API_KEY", "env-key")
        target = resolve_model_target(
            cli_overrides={"model": "m", "base_url": "http://x/v1"},
        )
        assert target.api_key_present is True
        assert target.credential_source == "env"

    def test_credential_source_none_when_no_key(self, monkeypatch):
        monkeypatch.delenv("XIANGMU_LLM_API_KEY", raising=False)
        target = resolve_model_target(
            cli_overrides={"model": "m", "base_url": "http://x/v1"},
        )
        assert target.api_key_present is False
        assert target.credential_source == "none"

    def test_vault_exception_is_visible_and_does_not_fallback(self, monkeypatch):
        monkeypatch.setenv("XIANGMU_LLM_API_KEY", "env-key")
        vault = MagicMock()
        vault.get = MagicMock(side_effect=RuntimeError("vault unavailable"))

        with pytest.raises(CredentialResolutionError, match="vault unavailable"):
            resolve_model_target(
                cli_overrides={"model": "m", "base_url": "http://x/v1"},
                secrets_vault=vault,
            )

        assert vault.get.call_count == 1
