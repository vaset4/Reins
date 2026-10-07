from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
from typing import Any, Mapping

import yaml

from llm.api_modes import DEFAULT_API_MODE, require_api_mode
from llm.config import parse_positive_int
from llm.reasoning import validate_reasoning_effort
from llm.model_catalog import (
    MODELS_JSON_CONFIG_PATH,
    load_model_catalog,
    switch_model_catalog_profile,
)

MODELS_CONFIG_PATH = Path.home() / ".reins" / "models.yaml"
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
    if path is None and MODELS_JSON_CONFIG_PATH.exists():
        return _load_json_profiles(MODELS_JSON_CONFIG_PATH)
    config_path = _resolve_path(path)
    if config_path.suffix.lower() == ".json":
        return _load_json_profiles(config_path)
    if not config_path.exists():
        return ModelProfilesConfig(path=config_path)
    raw = _read_yaml_object(config_path)
    active = _required_text(raw.get("active"), "active")
    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ValueError("profiles must contain at least one profile")
    profiles = _parse_profiles(profiles_raw)
    if active not in profiles:
        raise ValueError(f"active profile not found: {active}")
    return ModelProfilesConfig(
        path=config_path, active=active, profiles=profiles, source="models_yaml"
    )


def save_model_profile(
    name: str, updates: dict[str, object], path: Path | None = None
) -> None:
    if path is None and MODELS_JSON_CONFIG_PATH.exists():
        raise ValueError("models.json profiles must be edited in models.json")
    profile_name = _clean_profile_name(name)
    _reject_forbidden_keys(updates)
    _reject_unknown_profile_keys(updates)
    config_path = _resolve_path(path)
    if config_path.suffix.lower() == ".json":
        raise ValueError("models.json profiles must be edited in models.json")
    raw = _read_yaml_object(config_path) if config_path.exists() else {}
    profiles = _profiles_for_write(raw)
    existing = profiles.get(profile_name, {})
    if not isinstance(existing, dict):
        raise ValueError(f"profile must contain a YAML object: {profile_name}")
    merged = dict(existing)
    merged.update(updates)
    profile = _parse_profile(profile_name, merged)
    profiles[profile_name] = _profile_to_yaml(profile)
    if not _optional_text(raw.get("active")):
        raw["active"] = profile_name
    _validate_raw_for_write(raw)
    _write_yaml(config_path, raw)


def switch_active_model_profile(
    name: str, path: Path | None = None, *, reasoning_effort: str | None = None
) -> None:
    """一次保存模型及可选强度；传参：配置名、路径及档位；返回：无，校验失败不写入。"""
    profile_name = _clean_profile_name(name)
    if path is None and MODELS_JSON_CONFIG_PATH.exists():
        switch_model_catalog_profile(
            profile_name, MODELS_JSON_CONFIG_PATH, reasoning_effort=reasoning_effort
        )
        return
    config_path = _resolve_path(path)
    if config_path.suffix.lower() == ".json":
        switch_model_catalog_profile(
            profile_name, config_path, reasoning_effort=reasoning_effort
        )
        return
    config = load_model_profiles(config_path)
    if profile_name not in config.profiles:
        raise ValueError(f"profile not found: {profile_name}")
    raw = _read_yaml_object(config_path)
    if reasoning_effort is not None:
        profile = config.profiles[profile_name]
        effort = validate_reasoning_effort(
            reasoning_effort, profile.model, profile.api_mode
        )
        if effort is None:
            raw["profiles"][profile_name].pop("reasoning_effort", None)
        else:
            raw["profiles"][profile_name]["reasoning_effort"] = effort
    raw["active"] = profile_name
    _write_yaml(config_path, raw)


def _resolve_path(path: Path | None) -> Path:
    return path or MODELS_CONFIG_PATH


def _load_json_profiles(path: Path) -> ModelProfilesConfig:
    catalog = load_model_catalog(path)
    profiles = {
        name: _parse_profile(name, values) for name, values in catalog.profiles.items()
    }
    return ModelProfilesConfig(
        path=catalog.path,
        active=catalog.active,
        profiles=profiles,
        source="models_json",
    )


def _read_yaml_object(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("models.yaml must contain a YAML object")
    _reject_forbidden_keys(loaded)
    return dict(loaded)


def _parse_profiles(raw: dict[object, object]) -> dict[str, ModelProfile]:
    profiles: dict[str, ModelProfile] = {}
    for name, value in raw.items():
        profile_name = _clean_profile_name(name)
        profiles[profile_name] = _parse_profile(profile_name, value)
    return profiles


def _parse_profile(name: str, raw: object) -> ModelProfile:
    if not isinstance(raw, dict):
        raise ValueError(f"profile must contain a YAML object: {name}")
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


def _profiles_for_write(raw: dict[str, Any]) -> dict[str, Any]:
    profiles = raw.setdefault("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("profiles must contain a YAML object")
    return profiles


def _validate_raw_for_write(raw: dict[str, Any]) -> None:
    active = _optional_text(raw.get("active"))
    if active and active not in _profiles_for_write(raw):
        raise ValueError(f"active profile not found: {active}")
    _reject_forbidden_keys(raw)


def _profile_to_yaml(profile: ModelProfile) -> dict[str, object]:
    values = profile.as_config()
    return {key: value for key, value in values.items() if value not in ("", None)}


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
            "models.yaml must not contain credential fields: "
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


def _clean_profile_name(value: object) -> str:
    text = _optional_text(value)
    if not text:
        raise ValueError("profile name must be non-empty")
    return text


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


def _write_yaml(path: Path, raw: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 【模型配置】【保存选择】临时文件完整写入后替换，读者不会看到半套模型和强度
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        pending = Path(stream.name)
        try:
            stream.write(yaml.safe_dump(raw, allow_unicode=True, sort_keys=True))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            pending.unlink()
            raise
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


__all__ = [
    "ALLOWED_PROFILE_KEYS",
    "MODELS_JSON_CONFIG_PATH",
    "MODELS_CONFIG_PATH",
    "ModelProfile",
    "ModelProfilesConfig",
    "load_model_profiles",
    "save_model_profile",
    "switch_active_model_profile",
]
