from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


from llm.api_modes import DEFAULT_API_MODE, require_api_mode
from llm.config import parse_positive_int
from llm.reasoning import validate_reasoning_effort
from llm.model_catalog import (
    MODELS_JSON_CONFIG_PATH,
    load_model_catalog,
    switch_model_catalog_profile,
)

DEFAULT_CONTEXT_WINDOW = 30000
DEFAULT_TIMEOUT_SECONDS = 30.0
ALLOWED_PROFILE_KEYS = frozenset(
    {
        "provider",
        "base_url",
        "model",
        "credential",
        "api_mode",
        "context_window",
        "timeout_seconds",
        "max_output_tokens",
        "reasoning_effort",
    }
)
REQUIRED_PROFILE_KEYS = frozenset({"provider", "base_url", "model"})
FORBIDDEN_PROFILE_KEYS = frozenset(
    {"api_key", "api_token", "authorization", "bearer_token", "token", "password"}
)


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    provider: str
    base_url: str
    model: str
    credential: str = ""
    api_mode: str = DEFAULT_API_MODE
    context_window: int = DEFAULT_CONTEXT_WINDOW
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None

    def as_config(self) -> dict[str, object]:
        """导出非机密的模型设置；传参：无；返回：协议、模型和已配置窗口/输出上限。"""
        result: dict[str, object] = {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "credential": self.credential,
            "api_mode": self.api_mode,
            "context_window": self.context_window,
            "timeout_seconds": self.timeout_seconds,
        }
        if self.max_output_tokens is not None:
            result["max_output_tokens"] = self.max_output_tokens
        result["reasoning_effort"] = self.reasoning_effort or "default"
        return result


@dataclass(frozen=True, slots=True)
class ModelProfilesConfig:
    path: Path
    active: str = ""
    profiles: Mapping[str, ModelProfile] = field(default_factory=dict)
    source: str = "missing"

    @property
    def active_profile(self) -> ModelProfile | None:
        return self.profiles.get(self.active)


def load_model_profiles(path: Path | None = None) -> ModelProfilesConfig:
    """读取模型目录；传参：可选 JSON 路径；返回：活动模型及可选配置。"""
    config_path = path or MODELS_JSON_CONFIG_PATH
    catalog = load_model_catalog(config_path)
    profiles = {
        name: _parse_profile(name, values) for name, values in catalog.profiles.items()
    }
    return ModelProfilesConfig(
        path=catalog.path,
        active=catalog.active,
        profiles=profiles,
        source="models_json" if profiles else "missing",
    )


def switch_active_model_profile(
    name: str, path: Path | None = None, *, reasoning_effort: str | None = None
) -> None:
    """保存模型及可选强度；传参：provider:model、JSON 路径及档位；返回：无。"""
    switch_model_catalog_profile(
        name, path or MODELS_JSON_CONFIG_PATH, reasoning_effort=reasoning_effort
    )


def _parse_profile(name: str, raw: object) -> ModelProfile:
    if not isinstance(raw, dict):
        raise ValueError(f"profile must contain a JSON object: {name}")
    _reject_forbidden_keys(raw)
    _reject_unknown_profile_keys(raw)
    _require_profile_fields(raw, name)
    api_mode = require_api_mode(_profile_text(raw, "api_mode", DEFAULT_API_MODE))
    window_raw = raw.get("context_window")
    context_window = parse_positive_int(
        DEFAULT_CONTEXT_WINDOW if window_raw is None else window_raw, "context_window"
    )
    output = (
        parse_positive_int(raw["max_output_tokens"], "max_output_tokens")
        if raw.get("max_output_tokens") is not None
        else None
    )
    if output is not None and output > context_window:
        raise ValueError("max_output_tokens cannot exceed context_window")
    return ModelProfile(
        name=name,
        provider=_profile_text(raw, "provider"),
        base_url=_profile_text(raw, "base_url"),
        model=_profile_text(raw, "model"),
        credential=_profile_text(raw, "credential", ""),
        api_mode=api_mode,
        context_window=context_window,
        timeout_seconds=_positive_float(raw.get("timeout_seconds"), "timeout_seconds"),
        max_output_tokens=output,
        reasoning_effort=validate_reasoning_effort(
            raw.get("reasoning_effort"), _profile_text(raw, "model"), api_mode
        ),
    )


def _require_profile_fields(raw: dict[object, object], name: str) -> None:
    missing = [
        field
        for field in sorted(REQUIRED_PROFILE_KEYS)
        if not _optional_text(raw.get(field))
    ]
    if missing:
        raise ValueError(
            f"profile {name} missing required fields: {', '.join(missing)}"
        )


def _reject_forbidden_keys(value: object) -> None:
    found: list[str] = []
    _collect_forbidden_keys(value, found)
    if found:
        raise ValueError(
            "models.json must not contain credential fields: "
            f"{', '.join(sorted(set(found)))}"
        )


def _collect_forbidden_keys(value: object, found: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_PROFILE_KEYS:
                found.append(str(key))
            _collect_forbidden_keys(child, found)
    elif isinstance(value, list):
        for child in value:
            _collect_forbidden_keys(child, found)


def _reject_unknown_profile_keys(raw: Mapping[Any, object]) -> None:
    unknown = sorted(str(key) for key in raw if str(key) not in ALLOWED_PROFILE_KEYS)
    if unknown:
        raise ValueError(f"unknown profile fields: {', '.join(unknown)}")


def _required_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _profile_text(
    raw: dict[object, object], field_name: str, default: str | None = None
) -> str:
    value = raw.get(field_name, default)
    return (
        _required_text(value, field_name) if default is None else _optional_text(value)
    )


def _optional_text(value: object) -> str:
    if value in (None, ""):
        return ""
    return str(value).strip()


def _positive_float(value: object, field_name: str) -> float:
    if value in (None, ""):
        return DEFAULT_TIMEOUT_SECONDS
    try:
        parsed = float(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a positive number") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return parsed


__all__ = [
    "MODELS_JSON_CONFIG_PATH",
    "ModelProfile",
    "ModelProfilesConfig",
    "load_model_profiles",
    "switch_active_model_profile",
]
