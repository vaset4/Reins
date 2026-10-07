"""生产模型配置与Adapter协议名称的唯一映射。

作者：xxx
时间：2026-09-14 20:00:00
"""

from __future__ import annotations

from types import MappingProxyType

DEFAULT_API_MODE = "chat_completions"
API_MODE_FAMILIES = MappingProxyType(
    {
        "chat_completions": "openai_chat",
        "responses": "openai_responses",
        "anthropic_messages": "anthropic_messages",
    }
)


def require_api_mode(value: object) -> str:
    """校验明确选择的协议，不根据URL替用户切换；传参：配置值；返回：合法模式。"""
    if not isinstance(value, str) or value.strip() not in API_MODE_FAMILIES:
        raise ValueError(f"unsupported api_mode: {value}")
    return value.strip()
