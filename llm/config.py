from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from llm.api_modes import require_api_mode

SAVED_CONFIG_PATH = Path.home() / ".reins" / "config.yaml"
ALLOWED_SAVED_KEYS = frozenset(
    {"provider", "model", "base_url", "context_window", "api_mode", "max_output_tokens"}
)
DEFAULT_CONTEXT_WINDOW = 30000


@dataclass(slots=True)
class LLMProviderConfig:
    base_url: str
    model: str
    api_key: str | None
    timeout_seconds: float

    @property
    def is_configured(self) -> bool:
        return not self.missing_required_fields

    @property
    def missing_required_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.base_url.strip():
            missing.append("base_url")
        if not self.model.strip():
            missing.append("model")
        return missing


def load_project_llm_defaults(project_root: Path) -> dict[str, object]:
    config_path = project_root / "llm.json"
    if not config_path.exists():
        return {}

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("llm.json must contain a JSON object")
    return raw


def load_saved_config(path: Path | None = None) -> dict[str, object]:
    config_path = path or SAVED_CONFIG_PATH
    if not config_path.exists():
        return {}
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        print(
            f"warning: failed to parse {config_path}: {exc}",
            file=sys.stderr,
        )
        return {}
    if not isinstance(raw, dict):
        return {}
    llm_section = raw.get("llm")
    result: dict[str, object] = {}
    if isinstance(llm_section, dict):
        for key, value in llm_section.items():
            result[str(key)] = value
    auxiliary_models = raw.get("auxiliary_models")
    if isinstance(auxiliary_models, dict):
        result["auxiliary_models"] = auxiliary_models
    return result


def save_user_config(
    updates: dict[str, object],
    path: Path | None = None,
) -> None:
    rejected = [key for key in updates if key not in ALLOWED_SAVED_KEYS]
    if rejected:
        raise ValueError(
            f"saved config does not allow keys: {', '.join(sorted(rejected))}; "
            f"allowed keys are {sorted(ALLOWED_SAVED_KEYS)}"
        )
    if "api_mode" in updates:
        require_api_mode(updates["api_mode"])
    config_path = path or SAVED_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            loaded = None
        if isinstance(loaded, dict):
            existing = dict(loaded)

    llm_section = existing.get("llm")
    if not isinstance(llm_section, dict):
        llm_section = {}
    for key, value in updates.items():
        llm_section[key] = value
    for key in ("context_window", "max_output_tokens"):
        if llm_section.get(key) is not None:
            llm_section[key] = parse_positive_int(llm_section[key], key)
    if llm_section.get("max_output_tokens") is not None:
        window = llm_section.get("context_window") or DEFAULT_CONTEXT_WINDOW
        if llm_section["max_output_tokens"] > window:
            raise ValueError("max_output_tokens cannot exceed context_window")
    existing["llm"] = llm_section

    config_path.write_text(
        yaml.safe_dump(existing, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def parse_positive_int(value: object, field_name: str) -> int:
    """严格解析窗口和输出额度；传参：配置值与报错字段；返回：正整数，拒绝截断、布尔及非有限数。"""
    error = f"{field_name} must be a positive integer"
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(error)
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(error) from exc
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise ValueError(error)
    return int(number)
