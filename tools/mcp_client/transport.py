"""由MCP官方SDK承接标准传输，配置名不自动猜测或降级。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from anyio.abc import ObjectSendStream
    from anyio.streams.memory import MemoryObjectReceiveStream
    from mcp.shared.message import SessionMessage

SUPPORTED_TRANSPORTS = frozenset({"stdio", "streamable_http", "sse"})
DEFAULT_MCP_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class ConnectionSettings:
    """保存单个连接的配置事实，机密只在内存中参与传输。"""

    name: str
    transport: str = "stdio"
    command: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    url: str = ""
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    timeout_seconds: float = DEFAULT_MCP_TIMEOUT_SECONDS
    protocol_version: str | None = None
    cwd: str | None = None

    def __post_init__(self) -> None:
        """验证真实传输所需字段；传参：构造配置；返回：无，未知传输或无效端点直接报错。"""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("MCP server name must be a non-empty string")
        if (
            not isinstance(self.transport, str)
            or self.transport not in SUPPORTED_TRANSPORTS
        ):
            raise ValueError(f"unsupported MCP transport: {self.transport}")
        if self.transport == "stdio" and (
            not self.command
            or any(
                not isinstance(item, str) or not item.strip() for item in self.command
            )
        ):
            raise ValueError("MCP stdio command requires non-empty arguments")
        if not isinstance(self.url, str):
            raise ValueError("MCP URL must be a string")
        parsed = urlparse(self.url)
        if self.transport != "stdio" and (
            parsed.scheme not in {"http", "https"} or not parsed.hostname
        ):
            raise ValueError("MCP HTTP transport requires an http(s) URL with a host")
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise ValueError("MCP timeout_seconds must be a positive number")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("MCP timeout_seconds must be positive and finite")
        for field_name, value in (
            ("protocol_version", self.protocol_version),
            ("cwd", self.cwd),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"MCP {field_name} must be a non-empty string when provided"
                )


@asynccontextmanager
async def open_transport(
    settings: ConnectionSettings,
) -> AsyncIterator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        ObjectSendStream[SessionMessage],
    ]
]:
    """按显式配置打开标准MCP通道；传参：连接设置；返回：SDK接收/发送流，退出时由SDK关闭会话和进程。"""
    from mcp import StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    if settings.transport == "stdio":
        parameters = StdioServerParameters(
            command=settings.command[0],
            args=list(settings.command[1:]),
            env=dict(settings.env),
            cwd=settings.cwd,
        )
        async with stdio_client(parameters) as streams:
            yield streams
        return
    if settings.transport == "sse":
        async with sse_client(
            settings.url,
            headers=dict(settings.headers),
            timeout=settings.timeout_seconds,
            sse_read_timeout=settings.timeout_seconds,
        ) as streams:
            yield streams
        return
    import httpx

    async with httpx.AsyncClient(
        headers=dict(settings.headers), timeout=settings.timeout_seconds
    ) as client:
        async with streamable_http_client(
            settings.url, http_client=client, terminate_on_close=True
        ) as streams:
            yield streams[0], streams[1]
