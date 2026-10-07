from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

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
