from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlparse

from context.window import output_reserve
from llm.reasoning import validate_reasoning_effort
from llm.api_modes import DEFAULT_API_MODE, require_api_mode
from llm.config import DEFAULT_CONTEXT_WINDOW, parse_positive_int


@dataclass(frozen=True, slots=True)
class ResolvedModelTarget:
    provider: str
    model: str
    base_url: str
    api_mode: str
    timeout_seconds: float
    config_source: str
    credential_source: str
    api_key_present: bool
    api_key: str | None = None
    unsupported_reason: str = ""
    context_window: int = 30000
    context_window_source: str = "builtin_default"
    context_window_defaulted: bool = True
    profile_name: str = ""
    credential_name: str = ""
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None

    @property
    def output_token_limit(self) -> int:
        """取本次实际发送的输出预算；传参：无；返回：配置值或窗口内默认额度，不代表供应商最大能力。"""
        return (
            self.max_output_tokens
            if self.max_output_tokens is not None
            else output_reserve(self.context_window)
        )


class CredentialResolutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _ConfigSources:
    """按统一优先级保存配置来源，只读取本次解析输入。"""

    cli: Mapping[str, object]
    saved: Mapping[str, object]
    file: Mapping[str, object]
    env_prefix: str

    def resolve(self, key: str, default: object) -> tuple[object, str]:
        """解析一个字段并保留来源；传参：字段和缺省值；返回：值与来源，额度空串交给校验报错。"""
        candidates = (
            (self.cli, "cli"),
            (self.saved, "saved_config"),
            ({key: os.getenv(f"{self.env_prefix}{key.upper()}")}, "env"),
            (self.file, "file_default"),
        )
        for values, source in candidates:
            value = values.get(key)
            if value is not None and (
                value != "" or key in ("context_window", "max_output_tokens")
            ):
                return value, source
        return default, "builtin_default"


def resolve_model_target(
    *,
    cli_overrides: dict[str, object] | None = None,
    saved_config: dict[str, object] | None = None,
    env_prefix: str = "XIANGMU_LLM_",
    file_defaults: dict[str, object] | None = None,
    secrets_vault: object | None = None,
) -> ResolvedModelTarget:
    """合并模型、协议、额度和凭据来源；传参：各配置层；返回：经校验且可追溯的发送目标。"""
    cli = cli_overrides or {}
    saved = saved_config or {}
    file = file_defaults or {}
    fields = _ConfigSources(cli=cli, saved=saved, file=file, env_prefix=env_prefix)
    base_url, _ = fields.resolve("base_url", "")
    model, model_src = fields.resolve("model", "")
    timeout_raw, _ = fields.resolve("timeout_seconds", 30)
    explicit_provider, _ = fields.resolve("provider", None)
    context_window_raw, context_window_src = fields.resolve(
        "context_window", DEFAULT_CONTEXT_WINDOW
    )
    context_window = parse_positive_int(
        context_window_raw, f"context_window from {context_window_src}"
    )
    api_mode, _ = fields.resolve("api_mode", DEFAULT_API_MODE)
    effort_raw, _ = fields.resolve("reasoning_effort", None)
    effort = validate_reasoning_effort(effort_raw, str(model), str(api_mode))
    output_raw, output_source = fields.resolve("max_output_tokens", None)
    output = (
        parse_positive_int(output_raw, f"max_output_tokens from {output_source}")
        if output_raw is not None
        else None
    )
    if output is not None and output > context_window:
        raise ValueError("max_output_tokens cannot exceed context_window")

    provider = infer_provider(str(base_url), explicit_provider)
    config_source = model_src

    api_key, credential_source = _resolve_credential(
        cli=cli,
        saved=saved,
        env_key=f"{env_prefix}API_KEY",
        file_defaults=file,
        secrets_vault=secrets_vault,
        provider=provider,
    )

    return ResolvedModelTarget(
        provider=provider,
        model=str(model).strip(),
        base_url=str(base_url).strip(),
        api_mode=require_api_mode(api_mode),
        timeout_seconds=_as_float(timeout_raw),
        config_source=config_source,
        credential_source=credential_source,
        api_key_present=api_key is not None and str(api_key).strip() != "",
        api_key=str(api_key) if api_key not in (None, "") else None,
        context_window=context_window,
        context_window_source=context_window_src,
        context_window_defaulted=context_window_src == "builtin_default",
        max_output_tokens=output,
        reasoning_effort=effort,
    )


def infer_provider(base_url: str, explicit_provider: object | None) -> str:
    if explicit_provider and str(explicit_provider).strip():
        return str(explicit_provider).strip()
    host = urlparse(base_url).hostname or ""
    if "api.openai.com" in host:
        return "openai"
    if "api.anthropic.com" in host:
        return "anthropic"
    return "openai_compatible"


def sanitize_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.hostname:
        return f"{parsed.scheme}://{parsed.hostname}"
    return base_url


def requires_api_key(base_url: str) -> bool:
    parsed = urlparse(base_url)
    if parsed.scheme.lower() != "https":
        return False
    host = (parsed.hostname or "").lower()
    return host not in {"localhost", "127.0.0.1", "::1"}


def _resolve_credential(
    *,
    cli: dict[str, object],
    saved: dict[str, object],
    env_key: str,
    file_defaults: dict[str, object],
    secrets_vault: object | None,
    provider: str,
) -> tuple[str | None, str]:
    cli_key = cli.get("api_key")
    if cli_key not in (None, ""):
        return str(cli_key), "cli"
    saved_key = saved.get("api_key")
    if saved_key not in (None, ""):
        return str(saved_key), "saved_config"
    if secrets_vault is not None:
        getter = getattr(secrets_vault, "get", None)
        if callable(getter):
            credential_name = file_defaults.get("credential")
            if credential_name not in (None, ""):
                try:
                    named_key = getter(str(credential_name))
                except Exception as exc:
                    raise CredentialResolutionError(
                        f"credential lookup failed for {credential_name}: {exc}"
                    ) from exc
                if named_key not in (None, ""):
                    return str(named_key), "secrets_vault"
            try:
                provider_key = getter(f"{provider}_api_key")
            except Exception as exc:
                raise CredentialResolutionError(
                    f"credential lookup failed for {provider}_api_key: {exc}"
                ) from exc
            if provider_key not in (None, ""):
                return str(provider_key), "secrets_vault"
            try:
                generic_key = getter("llm_api_key")
            except Exception as exc:
                raise CredentialResolutionError(
                    f"credential lookup failed for llm_api_key: {exc}"
                ) from exc
            if generic_key not in (None, ""):
                return str(generic_key), "secrets_vault"
    env_value = os.getenv(env_key)
    if env_value not in (None, ""):
        return env_value, "env"
    file_key = file_defaults.get("api_key")
    if file_key not in (None, ""):
        return str(file_key), "file_default"
    return None, "none"


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    return 30.0


__all__ = [
    "CredentialResolutionError",
    "ResolvedModelTarget",
    "infer_provider",
    "requires_api_key",
    "resolve_model_target",
    "sanitize_base_url",
]
