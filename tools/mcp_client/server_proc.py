"""为同步执行器持有一个可复用的标准MCP会话和明确的关闭责任。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from builtins import BaseExceptionGroup, ExceptionGroup
from contextlib import ExitStack
from functools import partial
from threading import RLock
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from runtime.cancellation import CancellationToken
from tools.mcp_client.transport import ConnectionSettings
from tools.types import ToolError, ToolErrorCategory

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal
    from tools.mcp_client.session import ManagedSession


class ServerUnavailableError(RuntimeError):
    """表示连接未建立或已失效，调用方不得据此重放未知副作用。"""


class ServerProc:
    """单服务器连接所有者；并发请求共用SDK关联表，普通调用不自动重连。"""

    def __init__(self, settings: ConnectionSettings) -> None:
        """保存惰性启动配置；传参：已校验连接设置；返回：无。"""
        self.settings = settings
        self.name = settings.name
        self.status = "unstarted"
        self.error = ""
        self._lock = RLock()
        self._stack: ExitStack | None = None
        self._portal: BlockingPortal | None = None
        self._session: ManagedSession | None = None
        self._catalog_seen = -1

    def start(self) -> None:
        """首次使用完成标准初始化；传参：无；返回：无，失败保留原因并等待显式刷新。"""
        with self._lock:
            if self.status == "running":
                return
            if self.status != "unstarted":
                raise ServerUnavailableError(
                    f"MCP server {self.name}: {self.error or self.status}"
                )
            stack = ExitStack()
            try:
                from anyio.from_thread import start_blocking_portal
                from tools.mcp_client.session import open_session

                portal = stack.enter_context(start_blocking_portal())
                session = stack.enter_context(
                    portal.wrap_async_context_manager(open_session(self.settings))
                )
            except Exception as exc:
                try:
                    stack.close()
                except Exception as cleanup:
                    exc = ExceptionGroup(
                        "MCP initialization and cleanup failed", [exc, cleanup]
                    )
                raise self._failure("initialization", exc) from exc
            self._stack, self._portal, self._session = stack, portal, session
            self.status = "running"

    def list_tools(self) -> list[dict[str, Any]]:
        """枚举实际SDK目录并记录已观察代数；传参：无；返回：完整标准定义。"""
        self.start()
        assert self._portal is not None and self._session is not None
        generation = self._session.catalog_generation
        try:
            tools = self._portal.call(self._session.list_tools)
        except Exception as exc:
            raise self._failure("discovery", exc) from exc
        self._catalog_seen = generation
        return cast(list[dict[str, Any]], tools)

    @property
    def needs_refresh(self) -> bool:
        """查询SDK是否收到尚未发布的目录通知；传参：无；返回：是否需读取新目录。"""
        return (
            self._session is not None
            and self._session.catalog_generation != self._catalog_seen
        )

    def check_available(self) -> tuple[bool, str | None]:
        """返回已知连接状态，首次使用仍可启动；传参：无；返回：可调用性及失败原因。"""
        return self.status in {"unstarted", "running"}, self.error or None

    def call_tool(
        self,
        name: str,
        arguments: dict[str, object],
        *,
        cancellation: CancellationToken | None = None,
        operation_id: str = "",
        timeout_seconds: float | None = None,
    ) -> dict[str, object] | ToolError:
        """在同一会话派发并等待关联结果；传参：工具及执行边界；返回：完整结果，关闭不冒充远端已停止。"""
        self.start()
        assert self._portal is not None and self._session is not None
        token = cancellation or CancellationToken()
        identity = operation_id or f"mcp-operation-{uuid4().hex}"
        if token.cancelled:
            return ToolError(
                ToolErrorCategory.CANCELLED,
                "MCP call cancelled before dispatch",
                retryable=False,
                details={"execution_state": "not_started"},
            )
        token.register_closer(partial(self.cancel, identity, token))
        token.report_backend(supports_stop=True)
        operation = partial(
            self._session.call_tool,
            name,
            arguments,
            operation_id=identity,
            token=token,
            timeout_seconds=timeout_seconds
            if timeout_seconds is not None
            else self.settings.timeout_seconds,
        )
        try:
            result = self._portal.start_task_soon(operation).result()
        except CancelledError:
            return ToolError(
                ToolErrorCategory.CANCELLED,
                "MCP connection closed while tool effects remain unknown",
                retryable=False,
                details={"execution_state": "unknown"},
            )
        except Exception as exc:
            raise self._failure("call", exc) from exc
        with self._lock:
            if self._session.closed and self.status == "running":
                self.status, self.error = (
                    "unavailable",
                    "MCP connection closed; reconnect explicitly",
                )
        return result

    def _failure(self, action: str, error: Exception) -> ServerUnavailableError:
        """保存SDK任务组中的具体根因；传参：失败动作及异常；返回：可见连接错误，不吞掉嵌套失败。"""
        pending: list[BaseException] = [error]
        reasons = []
        while pending:
            current = pending.pop()
            if isinstance(current, BaseExceptionGroup):
                pending.extend(reversed(current.exceptions))
            else:
                reasons.append(f"{type(current).__name__}: {current}")
        self.status, self.error = "unavailable", "; ".join(reasons)
        return ServerUnavailableError(
            f"MCP server {self.name} {action} failed: {self.error}"
        )

    def cancel(self, operation_id: str, token: CancellationToken) -> None:
        """把停止通知送到原请求的SDK会话；传参：操作身份和停止信号；返回：无。"""
        if (
            self._portal is not None
            and self._session is not None
            and not self._session.closed
        ):
            self._portal.call(
                self._session.cancel,
                operation_id,
                token.reason or "operation cancelled",
            )

    def describe(self) -> dict[str, object]:
        """返回不含凭据的支持与连接事实；传参：无；返回：状态快照。"""
        return {
            "server": self.name,
            "transport": self.settings.transport,
            "status": self.status,
            "error": self.error,
            "connection_id": self._session.connection_id if self._session else None,
            "protocol_version": self._session.protocol_version
            if self._session
            else None,
            "server_info": self._session.server_info if self._session else None,
        }

    def close(self) -> None:
        """由宿主关闭SDK会话、HTTP连接或stdio进程树；传参：无；返回：无，清理失败向上暴露。"""
        with self._lock:
            stack, self._stack = self._stack, None
            self.status = "closed"
        if stack is not None:
            stack.close()
