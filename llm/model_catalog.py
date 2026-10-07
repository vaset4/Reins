from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
from typing import Any, Mapping

from llm.reasoning import validate_reasoning_effort
from llm.api_modes import DEFAULT_API_MODE, require_api_mode

MODELS_JSON_CONFIG_PATH = Path.home() / ".reins" / "models.json"
DEFAULT_CONTEXT_WINDOW = 30000
DEFAULT_TIMEOUT_SECONDS = 30.0
ALLOWED_TOP_LEVEL_KEYS = frozenset({"active_provider", "active_model", "providers"})
ALLOWED_PROVIDER_KEYS = frozenset(
    {
        "model_provider",
        "provider_name",
        "base_url",
        "credential",
        "api_mode",
        "timeout_seconds",
        "context_window",
        "max_output_tokens",
        "models",
    }
)
ALLOWED_MODEL_KEYS = frozenset(
    {
        "model",
        "context_window",
        "timeout_seconds",
        "max_output_tokens",
        "reasoning_effort",
    }
)
FORBIDDEN_SECRET_KEYS = frozenset(
    {"api_key", "api_token", "authorization", "bearer_token", "token", "password"}
)


@dataclass(frozen=True, slots=True)
class ModelCatalogConfig:
    path: Path
    active: str = ""
    profiles: Mapping[str, dict[str, object]] = field(default_factory=dict)


def load_model_catalog(path: Path | None = None) -> ModelCatalogConfig:
    config_path = path or MODELS_JSON_CONFIG_PATH
    if not config_path.exists():
        return ModelCatalogConfig(path=config_path)
    raw = _read_json_object(config_path)
    _reject_unknown_keys(raw, ALLOWED_TOP_LEVEL_KEYS, "models.json")
    active_provider = _required_text(raw.get("active_provider"), "active_provider")
    active_model = _required_text(raw.get("active_model"), "active_model")
    providers = _required_mapping(raw.get("providers"), "providers")
    profiles = _flatten_profiles(providers)
    active = _profile_name(active_provider, active_model)
    if active not in profiles:
        raise ValueError(f"active provider/model not found: {active}")
    return ModelCatalogConfig(path=config_path, active=active, profiles=profiles)


def switch_model_catalog_profile(
    name: str, path: Path | None = None, *, reasoning_effort: str | None = None
) -> None:
    """一次保存目录选择及模型强度；传参：配置名、路径及可选档位；返回：无。"""
    provider_name, model_name = _split_profile_name(name)
    config_path = path or MODELS_JSON_CONFIG_PATH
    raw = _read_json_object(config_path)
    profiles = load_model_catalog(config_path).profiles
    if _profile_name(provider_name, model_name) not in profiles:
        raise ValueError(f"profile not found: {name}")
    if reasoning_effort is not None:
        profile = profiles[name]
        effort = validate_reasoning_effort(
            reasoning_effort, str(profile["model"]), str(profile["api_mode"])
        )
        selected = raw["providers"][provider_name]["models"][model_name]
        if effort is None:
            selected.pop("reasoning_effort", None)
        else:
            selected["reasoning_effort"] = effort
    raw["active_provider"] = provider_name
    raw["active_model"] = model_name
    _write_json(config_path, raw)


def _flatten_profiles(providers: Mapping[Any, object]) -> dict[str, dict[str, object]]:
    profiles: dict[str, dict[str, object]] = {}
    for provider_key, provider_value in providers.items():
        key = _required_text(provider_key, "provider name")
        provider = _provider_config(key, provider_value)
        models = _required_mapping(provider["models"], f"providers.{key}.models")
        for model_key, model_value in models.items():
            alias = _required_text(model_key, "model alias")
            profile_name = _profile_name(key, alias)
            profiles[profile_name] = _profile_values(provider, alias, model_value)
    return profiles


def _provider_config(name: str, value: object) -> dict[str, object]:
    raw = _required_mapping(value, f"providers.{name}")
    _reject_unknown_keys(raw, ALLOWED_PROVIDER_KEYS, f"providers.{name}")
    return {
        "provider": _optional_text(raw.get("provider_name")) or name,
        "model_provider": _required_text(raw.get("model_provider"), "model_provider"),
        "base_url": _required_text(raw.get("base_url"), "base_url"),
        "credential": _optional_text(raw.get("credential")),
        "api_mode": require_api_mode(raw.get("api_mode", DEFAULT_API_MODE)),
        "timeout_seconds": raw.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
        "context_window": raw.get("context_window", DEFAULT_CONTEXT_WINDOW),
        "max_output_tokens": raw.get("max_output_tokens"),
        "models": raw.get("models"),
    }


def _profile_values(
    provider: Mapping[str, object], alias: str, value: object
) -> dict[str, object]:
    raw = _required_mapping(value, f"models.{alias}")
    _reject_unknown_keys(raw, ALLOWED_MODEL_KEYS, f"models.{alias}")
    return {
        "provider": provider["provider"],
        "base_url": provider["base_url"],
        "model": _required_text(raw.get("model"), "model"),
        "reasoning_effort": raw.get("reasoning_effort"),
        "credential": provider["credential"],
        "api_mode": provider["api_mode"],
        "context_window": raw.get("context_window", provider["context_window"]),
        "timeout_seconds": raw.get("timeout_seconds", provider["timeout_seconds"]),
        "max_output_tokens": raw.get(
            "max_output_tokens", provider["max_output_tokens"]
        ),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        stripped = _strip_trailing_commas(text)
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            raise ValueError(f"failed to parse {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("models.json must contain a JSON object")
    _reject_forbidden_keys(loaded)
    return dict(loaded)


def _strip_trailing_commas(text: str) -> str:
    result: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            result.append(char)
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            result.append(char)
            continue
        if char == "," and _next_non_ws_is_closer(text, index + 1):
            continue
        result.append(char)
    return "".join(result)


def _next_non_ws_is_closer(text: str, start: int) -> bool:
    for char in text[start:]:
        if char.isspace():
            continue
        return char in ("}", "]")
    return False


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
            if lowered in FORBIDDEN_SECRET_KEYS:
                found.append(str(key))
            _collect_forbidden_keys(child, found)
    elif isinstance(value, list):
        for child in value:
            _collect_forbidden_keys(child, found)


def _required_mapping(value: object, field_name: str) -> Mapping[Any, object]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{field_name} must contain a JSON object")
    return value


def _reject_unknown_keys(
    raw: Mapping[Any, object], allowed: frozenset[str], location: str
) -> None:
    unknown = sorted(str(key) for key in raw if str(key) not in allowed)
    if unknown:
        raise ValueError(f"unknown {location} fields: {', '.join(unknown)}")


def _split_profile_name(name: str) -> tuple[str, str]:
    provider, sep, model = name.partition(":")
    if not sep or not provider.strip() or not model.strip():
        raise ValueError("profile name must use provider:model")
    return provider.strip(), model.strip()


def _profile_name(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def _required_text(value: object, field_name: str) -> str:
    text = _optional_text(value)
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _optional_text(value: object) -> str:
    if value in (None, ""):
        return ""
    return str(value).strip()


def _write_json(path: Path, raw: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
    # 【模型配置】【保存选择】临时文件完整写入后替换，读者不会看到半套模型和强度
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        pending = Path(stream.name)
        try:
            stream.write(text)
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
    "MODELS_JSON_CONFIG_PATH",
    "ModelCatalogConfig",
    "load_model_catalog",
    "switch_model_catalog_profile",
]
