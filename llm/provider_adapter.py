from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Protocol

from llm.model_registry import ModelDescriptor
from llm.model_request import ModelRequest
from llm.provider_connection import ResolvedConnection
from llm.provider_stream import ModelStreamEvent
from runtime.cancellation import CancellationToken


class ProviderAdapterError(ValueError):
    """表示 Adapter 注册或模型能力合同失败。"""


class ProviderAdapter(Protocol):
    """定义 Provider Core 唯一的 API-family 调用 port。"""

    @property
    def api_family(self) -> str:
        """返回 Adapter 唯一 API-family key。"""
        ...

    def stream(
        self,
        request: ModelRequest,
        *,
        model: ModelDescriptor,
        connection: ResolvedConnection,
        cancellation: CancellationToken | None = None,
        prepared_body: Mapping[str, object] | None = None,
    ) -> Iterator[ModelStreamEvent]:
        """将 canonical 请求翻译为统一事件迭代器。"""
        ...

    def build_request(
        self, request: ModelRequest, *, model_id: str
    ) -> dict[str, object]:
        """构造一次实际发送内容；参数：冻结请求和模型；返回：不含认证头的发送体。"""
        ...


class AdapterRegistry:
    """保存显式构造且无全局副作用的 API-family Adapter 快照。"""

    def __init__(self, adapters: Iterable[ProviderAdapter]) -> None:
        """构造注册表并拒绝空或重复 API family。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：adapters 为本次调用可见的 Adapter
        返回：无
        """
        values: dict[str, ProviderAdapter] = {}
        for adapter in adapters:
            family = adapter.api_family
            if not isinstance(family, str) or not family.strip():
                raise ProviderAdapterError("blank_api_family")
            if family in values:
                raise ProviderAdapterError(f"duplicate_api_family:{family}")
            values[family] = adapter
        self._values = MappingProxyType(values)

    def require(self, api_family: str) -> ProviderAdapter:
        """取得指定 API-family 的唯一 Adapter。

        作者：xxx
        时间：2026-08-19 00:00:00
        传参：api_family 为 ModelDescriptor 声明的 family
        返回：唯一 ProviderAdapter；未知时抛 ProviderAdapterError
        """
        try:
            return self._values[api_family]
        except KeyError as exc:
            raise ProviderAdapterError(f"unknown_api_family:{api_family}") from exc


def validate_adapter_target(adapter: ProviderAdapter, model: ModelDescriptor) -> None:
    """在发送前再次验证 descriptor 与 Adapter family 一致。

    作者：xxx
    时间：2026-08-19 00:00:00
    传参：adapter 为已解析 port；model 为选中模型事实
    返回：无；不一致时抛 ProviderAdapterError
    """
    if adapter.api_family != model.api_family:
        raise ProviderAdapterError(
            f"adapter_family_mismatch:{adapter.api_family}:{model.api_family}"
        )


__all__ = [
    "AdapterRegistry",
    "ProviderAdapter",
    "ProviderAdapterError",
    "validate_adapter_target",
]
