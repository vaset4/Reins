from __future__ import annotations

from llm.provider_adapter import AdapterRegistry
from llm.providers.anthropic_messages import AnthropicMessagesAdapter
from llm.providers.openai_chat import OpenAIChatAdapter
from llm.providers.openai_responses import OpenAIResponsesAdapter


def build_builtin_adapter_registry() -> AdapterRegistry:
    """显式组合三套内建 API-family Adapter。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：无
    返回：不依赖 import-time 全局状态的 AdapterRegistry
    """
    return AdapterRegistry(
        [OpenAIChatAdapter(), OpenAIResponsesAdapter(), AnthropicMessagesAdapter()]
    )


__all__ = ["build_builtin_adapter_registry"]
