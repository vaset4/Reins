"""在官方ClientSession旁关联操作、通知与取消，不自行分配JSON-RPC请求身份。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from http import HTTPStatus
from typing import Any, cast
from uuid import uuid4

from anyio.abc import ObjectSendStream
from anyio.streams.memory import MemoryObjectSendStream
from mcp import ClientSession, McpError, types
from mcp.shared.message import SessionMessage

from runtime.cancellation import CancellationToken, ExecutionCancelled
from tools.mcp_client.transport import ConnectionSettings, open_transport
from tools.types import ToolError, ToolErrorCategory

_LOG = logging.getLogger(__name__)
OPERATION_META_KEY = "reins/operation_id"


@dataclass(slots=True)
class _CallState:
    """保留SDK实际分配的请求编号及调用停止信号。"""

    token: CancellationToken
    request_id: str | int | None = None
    cancel_sent: bool = False


class _ObservedSendStream(ObjectSendStream[SessionMessage]):
    """旁观官方SDK发出的请求身份，传输与关联回复仍归官方SDK。"""

    def __init__(
        self,
        stream: ObjectSendStream[SessionMessage],
        observe: Callable[[SessionMessage], None],
    ) -> None:
        """注入SDK发送流和证据回调；传参：发送流与观察者；返回：无。"""
        self.stream, self.observe = stream, observe

    async def send(self, item: SessionMessage) -> None:
        """发送前记录真实请求身份；传参：SDK消息；返回：无，取消发生在发送前则不派发。"""
        self.observe(item)
        await self.stream.send(item)

    async def aclose(self) -> None:
        """沿SDK生命周期关闭发送流；传参：无；返回：无。"""
        await self.stream.aclose()


class ManagedSession:
    """连接内维护通知代数与在途操作，不持有另一个协议实现。"""

    def __init__(self, settings: ConnectionSettings) -> None:
        """记录连接设置；传参：配置；返回：无。"""
        self.settings = settings
        self.connection_id = f"mcp-connection-{uuid4().hex}"
        self.client: ClientSession | None = None
        self.protocol_version = ""
        self.server_info: dict[str, Any] = {}
        self.catalog_generation = 0
        self.closed = False
        self.calls: dict[str, _CallState] = {}

    def observe_request(self, message: SessionMessage) -> None:
        """关联SDK实际请求ID与操作元数据；传参：即将发送的消息；返回：无。"""
        request = message.message.root
        if (
            not isinstance(request, types.JSONRPCRequest)
            or request.method != "tools/call"
        ):
            return
        meta = (request.params or {}).get("_meta", {})
        operation_id = (
            str(meta.get(OPERATION_META_KEY, "")) if isinstance(meta, dict) else ""
        )
        state = self.calls.get(operation_id)
        if state is None:
            raise ValueError("MCP tool request is missing its operation identity")
        if state.token.cancelled:
            state.token.report_backend(execution_state="not_started")
            raise ExecutionCancelled("MCP tool cancelled before dispatch")
        state.request_id = request.id
        state.token.report_backend(
            mcp_request_id=request.id, mcp_server=self.settings.name
        )

    async def on_message(self, message: object) -> None:
        """接收标准目录变更通知；传参：SDK分派的消息；返回：无，后续请求重新发布目录。"""
        if isinstance(message, types.ServerNotification) and isinstance(
            message.root, types.ToolListChangedNotification
        ):
            self.catalog_generation += 1
        if isinstance(message, Exception):
            _LOG.error(
                "【MCP】【协议消息】服务%s返回无效消息: %s", self.settings.name, message
            )

    async def on_log(self, params: types.LoggingMessageNotificationParams) -> None:
        """分离服务器诊断与工具正文；传参：标准日志通知；返回：无。"""
        _LOG.info("【MCP】【服务诊断】%s: %s", self.settings.name, params.data)

    async def list_tools(self) -> list[dict[str, Any]]:
        """读取所有标准目录页，拒绝循环游标和重复工具；传参：无；返回：完整工具声明。"""
        assert self.client is not None
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            page = await self.client.list_tools(cursor=cursor)
            tools.extend(
                item.model_dump(mode="json", by_alias=True, exclude_none=True)
                for item in page.tools
            )
            cursor = page.nextCursor
            if cursor is None:
                break
            if cursor in seen:
                raise ValueError("MCP tools/list returned a repeated cursor")
            seen.add(cursor)
        if len({item["name"] for item in tools}) != len(tools):
            raise ValueError("MCP tools/list returned duplicate tool names")
        return tools

    async def cancel(self, operation_id: str, reason: str) -> None:
        """向实际在途ID发送标准取消通知；传参：操作身份及原因；返回：无，不把发送当停止证明。"""
        state = self.calls.get(operation_id)
        if state is None or state.request_id is None or state.cancel_sent:
            return
        assert self.client is not None
        await self.client.send_notification(
            types.ClientNotification(
                types.CancelledNotification(
                    params=types.CancelledNotificationParams(
                        requestId=state.request_id, reason=reason
                    )
                )
            )
        )
        state.cancel_sent = True
        state.token.report_backend(
            mcp_cancel_sent=True, mcp_request_id=state.request_id
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        operation_id: str,
        token: CancellationToken,
        timeout_seconds: float,
    ) -> dict[str, object] | ToolError:
        """执行标准工具调用并保留真实关联和错误；传参：工具、参数与操作边界；返回：完整结果或明确错误。"""
        assert self.client is not None
        state = _CallState(token)
        if operation_id in self.calls:
            raise ValueError("MCP operation is already in flight")
        self.calls[operation_id] = state
        try:
            result = await self.client.call_tool(
                name,
                arguments,
                read_timeout_seconds=timedelta(seconds=timeout_seconds),
                meta={OPERATION_META_KEY: operation_id},
            )
            return {
                **result.model_dump(mode="json", by_alias=True, exclude_none=True),
                "mcp": self._evidence(state, operation_id),
            }
        except McpError as exc:
            if exc.error.code == types.CONNECTION_CLOSED:
                self.closed = True
            if exc.error.code == HTTPStatus.REQUEST_TIMEOUT:
                await self.cancel(operation_id, "tool timeout")
            category = (
                ToolErrorCategory.CANCELLED
                if token.cancelled
                else ToolErrorCategory.TRANSPORT
            )
            if exc.error.code == HTTPStatus.REQUEST_TIMEOUT:
                category = ToolErrorCategory.TIMEOUT
            return ToolError(
                category,
                exc.error.message,
                retryable=False,
                details={
                    "mcp_error": exc.error.model_dump(mode="json", by_alias=True),
                    "mcp": self._evidence(state, operation_id),
                    "execution_state": "unknown"
                    if state.request_id is not None
                    else "not_started",
                },
            )
        except ExecutionCancelled as exc:
            return ToolError(
                ToolErrorCategory.CANCELLED,
                str(exc),
                retryable=False,
                details={"execution_state": "not_started"},
            )
        finally:
            self.calls.pop(operation_id, None)

    def _evidence(self, state: _CallState, operation_id: str) -> dict[str, object]:
        """投影连接、协议和请求关联；传参：在途事实和操作身份；返回：不含凭据的证据。"""
        return {
            "server": self.settings.name,
            "protocol_version": self.protocol_version,
            "connection_id": self.connection_id,
            "request_id": state.request_id,
            "operation_id": operation_id,
            "cancel_sent": state.cancel_sent,
        }


@asynccontextmanager
async def open_session(settings: ConnectionSettings) -> AsyncIterator[ManagedSession]:
    """在同一异步任务内完成初始化及关闭；传参：连接配置；返回：已握手会话。"""
    managed = ManagedSession(settings)
    async with open_transport(settings) as (receive, send):
        observed = _ObservedSendStream(send, managed.observe_request)
        # 【MCP】【请求关联】SDK只使用ObjectSendStream接口；其类型声明限定Memory流，旁观层不改变实际发送行为
        async with ClientSession(
            receive,
            cast(MemoryObjectSendStream[SessionMessage], observed),
            read_timeout_seconds=timedelta(seconds=settings.timeout_seconds),
            message_handler=managed.on_message,
            logging_callback=managed.on_log,
        ) as client:
            managed.client = client
            result = await client.initialize()
            managed.protocol_version = str(result.protocolVersion)
            managed.server_info = result.serverInfo.model_dump(mode="json")
            if (
                settings.protocol_version is not None
                and settings.protocol_version != result.protocolVersion
            ):
                raise ValueError(
                    f"MCP negotiated {result.protocolVersion}, expected {settings.protocol_version}"
                )
            try:
                yield managed
            finally:
                managed.closed = True
