"""将标准MCP连接与真实定义发布到同一工具目录。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from builtins import ExceptionGroup
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import cast

import yaml

from reins_secrets.resolver import resolve_secret_refs
from runtime.cancellation import CancellationToken
from runtime.lease import Lease
from tools.argument_schema import normalize_tool_schema
from tools.mcp_client.server_proc import ServerProc, ServerUnavailableError
from tools.mcp_client import config as mcp_config
from tools.mcp_client.transport import ConnectionSettings, DEFAULT_MCP_TIMEOUT_SECONDS
from tools.tool_registry import (
    Idempotent,
    MCPToolRiskError,
    ToolDefinition,
    ToolRegistry,
    ToolRisk,
)
from tools.types import ToolError, ToolErrorCategory

_LOG = logging.getLogger(__name__)
_PROVIDER_TOOL_NAME_LIMIT = 64
_NAME_DIGEST_LENGTH = 12


@dataclass(frozen=True, slots=True)
class MCPServerConfig(ConnectionSettings):
    """扩展连接配置中的工具风险覆盖与旧版离线声明。"""

    tool_risk_overrides: Mapping[str, str] = field(default_factory=dict)
    tools: tuple[dict[str, object], ...] = ()


class MCPRegistry:
    """由ToolRegistry持有连接，目录刷新只影响后续请求的快照。"""

    def __init__(
        self,
        lease: Lease,
        tool_registry: ToolRegistry,
        *,
        config_path: Path | str | None = None,
        vault: object | None = None,
        server_factory: Callable[[ConnectionSettings], ServerProc] = ServerProc,
    ) -> None:
        """接收目录、租约及连接工厂；传参：配置定位与凭据引用解析器；返回：无，不启动未授权服务器。"""
        self.lease, self.tool_registry = lease, tool_registry
        self.config_path = Path(config_path) if config_path is not None else None
        self.vault, self.server_factory = vault, server_factory
        self.servers: dict[str, ServerProc] = {}
        self.configs: dict[str, MCPServerConfig] = {}
        self.failures: dict[str, str] = {}
        self._config_digest = ""
        self._schemas: dict[str, tuple[dict[str, object], ...]] = {}
        self._published_servers: dict[str, ServerProc] = {}
        self._discovered: set[str] = set()
        self._lock = RLock()
        self._reload_configs(lease)
        tool_registry.attach_source("mcp", self)

    def register_allowed_tools(self, *, discover: bool = False) -> None:
        """接通配置声明或实际发现；传参：是否显式刷新真实目录；返回：无，各服务器失败分别可查。"""
        self.refresh(self.lease, force=discover)

    def refresh(self, lease: Lease, *, force: bool = False) -> None:
        """响应配置和标准list_changed通知，普通失败不盲目重连；传参：当前授权及显式刷新；返回：无。"""
        with self._lock:
            self.lease = lease
            self._reload_configs(lease, force=force)
            allowed = _allowed_servers(lease)
            for name in tuple(self.servers):
                if (
                    name not in allowed
                    or name not in self.configs
                    or self.servers[name].settings != self.configs[name]
                ):
                    self._retire_server(name)
            for name in allowed:
                config = self.configs.get(name)
                if config is None:
                    self.failures.setdefault(
                        name, "mcp_not_configured: no valid server configuration"
                    )
                    continue
                if (
                    force
                    and name in self.servers
                    and self.servers[name].status != "running"
                ):
                    self._retire_server(name)
                if name not in self.servers:
                    self.servers[name] = self.server_factory(config)
                server = self.servers[name]
                if server.status == "unavailable" and not force:
                    continue
                try:
                    if (
                        force
                        or server.needs_refresh
                        or (name not in self._schemas and not config.tools)
                    ):
                        self._publish(server, config, server.list_tools())
                        self._discovered.add(name)
                    elif name not in self._schemas:
                        self._publish(server, config, config.tools)
                    self.failures.pop(name, None)
                except MCPToolRiskError:
                    # 【MCP】【风险锁】安全级别锁定是配置写错，不是这个服务器暂时连不上；
                    # 按服务器失败吞掉会让 MCPToolRiskError(ValueError) 静默变成 failures 条目
                    raise
                except (ServerUnavailableError, ValueError, OSError) as exc:
                    self.failures[name] = str(exc)
                    _LOG.error("【MCP】【能力目录】%s不可用: %s", name, exc)

    def ensure_current(self, server: ServerProc) -> None:
        """旧配置声明首次执行前与真实服务对账；传参：准备执行的原连接；返回：无，目录变更统一发布。"""
        with self._lock:
            if self.servers.get(server.name) is not server:
                raise ServerUnavailableError(
                    "MCP connection configuration changed; reload the current tool"
                )
            if server.name not in self._discovered or server.needs_refresh:
                self._publish(server, self.configs[server.name], server.list_tools())
                self._discovered.add(server.name)

    def _publish(
        self,
        server: ServerProc,
        config: MCPServerConfig,
        schemas: Sequence[dict[str, object]],
    ) -> None:
        """完整校验后一次发布某服务器的全部定义；传参：连接、配置和标准目录；返回：无，不发布半个目录。"""
        definitions = [
            _definition(self, server, config=config, schema=schema)
            for schema in schemas
        ]
        previous = self._schemas.get(config.name, ())
        old_signature = [_schema_signature(item) for item in previous]
        new_signature = [_schema_signature(item) for item in schemas]
        if (
            self._published_servers.get(config.name) is server
            and old_signature == new_signature
        ):
            return
        old_names = tuple(
            item.name
            for item in self.tool_registry.list_definitions()
            if item.mcp_server == config.name
        )
        self.tool_registry.publish(definitions, replace_names=old_names)
        self._schemas[config.name] = tuple(dict(item) for item in schemas)
        self._published_servers[config.name] = server

    def _retire_server(self, name: str) -> None:
        """配置移除或显式重连时撤下旧连接及目录；传参：服务器名；返回：无，关闭失败不冒充成功。"""
        server = self.servers.pop(name)
        names = tuple(
            item.name
            for item in self.tool_registry.list_definitions()
            if item.mcp_server == name
        )
        if names:
            self.tool_registry.publish((), replace_names=names)
        self._schemas.pop(name, None)
        self._published_servers.pop(name, None)
        self._discovered.discard(name)
        server.close()

    def _reload_configs(self, lease: Lease, *, force: bool = False) -> None:
        """按文件内容变化读取配置，保留每个服务器的独立错误；传参：租约和刷新选项；返回：无。"""
        path = self.config_path or _config_path_from_lease(lease)
        content = path.read_bytes() if path.is_file() else b""
        digest = hashlib.sha256(
            str(path.resolve()).encode("utf-8") + content
        ).hexdigest()
        if not force and digest == self._config_digest:
            return
        data = yaml.safe_load(content) if content else {"servers": {}}
        if not isinstance(data, dict) or not isinstance(data.get("servers"), dict):
            raise ValueError("MCP config must contain a servers mapping")
        configs: dict[str, MCPServerConfig] = {}
        failures: dict[str, str] = {}
        for name, value in data["servers"].items():
            try:
                configs[str(name)] = _server_config(str(name), value, self.vault)
            except ValueError as exc:
                failures[str(name)] = str(exc)
        self.configs, self.failures, self._config_digest = configs, failures, digest

    def describe(self) -> dict[str, object]:
        """给模型和界面提供连接支持事实；传参：无；返回：实际协议、连接状态和失败原因。"""
        with self._lock:
            return {
                "servers": [server.describe() for server in self.servers.values()],
                "errors": dict(self.failures),
            }

    def close(self) -> None:
        """关闭所有拥有的连接后汇总失败；传参：无；返回：无，不使用进程全局服务器表。"""
        failures: list[Exception] = []
        for server in tuple(self.servers.values()):
            try:
                server.close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise ExceptionGroup("MCP connections failed to close", failures)


def attach_mcp_registry(lease: Lease, registry: ToolRegistry) -> MCPRegistry:
    """复用当前宿主的MCP owner；传参：新租约和已有目录；返回：已按当前权限刷新过的连接集合。"""
    existing = registry.source("mcp")
    if existing is not None and not isinstance(existing, MCPRegistry):
        raise ValueError("MCP capability source has an incompatible owner")
    owner = (
        existing if isinstance(existing, MCPRegistry) else MCPRegistry(lease, registry)
    )
    owner.refresh(lease)
    return owner


def _server_config(name: str, data: object, vault: object | None) -> MCPServerConfig:
    """校验真实连接参数并在内存解析机密引用；传参：服务器名、原始配置及凭据提供者；返回：连接配置。"""
    if not isinstance(data, dict):
        raise ValueError(f"MCP server {name} config must be an object")
    command = data.get("command", [])
    if not isinstance(command, list) or any(
        not isinstance(item, str) or not item.strip() for item in command
    ):
        raise ValueError("MCP command must be a list of non-empty strings")
    env, headers = _string_dict(data.get("env")), _string_dict(data.get("headers"))
    timeout = data.get("timeout_seconds", DEFAULT_MCP_TIMEOUT_SECONDS)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float, str)):
        raise ValueError("MCP timeout_seconds must be a positive number")
    try:
        timeout_seconds = float(timeout)
    except ValueError as exc:
        raise ValueError("MCP timeout_seconds must be a positive number") from exc
    return MCPServerConfig(
        name=name,
        command=tuple(command),
        transport=cast(str, data.get("transport", "stdio")),
        env={
            key: resolve_secret_refs(value, vault=vault) for key, value in env.items()
        },
        url=cast(str, data.get("url", "")),
        headers={
            key: resolve_secret_refs(value, vault=vault)
            for key, value in headers.items()
        },
        tool_risk_overrides=_string_dict(data.get("tool_risk_overrides")),
        tools=_tools_from_config(data.get("tools")),
        timeout_seconds=timeout_seconds,
        protocol_version=cast(str | None, data.get("protocol_version")),
        cwd=cast(str | None, data.get("cwd")),
    )


def _string_dict(value: object) -> dict[str, str]:
    """校验配置中的字符串映射；传参：原始字段；返回：独立映射，不吞掉格式错误。"""
    if value is None:
        return {}
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError(
            "MCP environment, headers and risk overrides must be string mappings"
        )
    return dict(value)


def _tools_from_config(value: object) -> tuple[dict[str, object], ...]:
    """保留旧版具名/列表声明，在首次真实执行前核对；传参：tools配置；返回：离线定义集合。"""
    if value is None:
        return ()
    if isinstance(value, dict):
        if any(not isinstance(item, dict) for item in value.values()):
            raise ValueError("MCP configured tool must be an object")
        return tuple({"name": str(name), **item} for name, item in value.items())
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        return tuple(dict(item) for item in value)
    raise ValueError("MCP tools must be a mapping or list")


def _parameters_from_schema(schema: dict[str, object]) -> dict[str, object]:
    """保留标准JSON Schema，旧parameters声明仍经同一校验；传参：目录条目；返回：独立标准Schema。"""
    raw = schema.get(
        "inputSchema", schema.get("input_schema", schema.get("parameters", {}))
    )
    if not isinstance(raw, dict):
        raise ValueError("MCP tool input schema must be an object")
    return normalize_tool_schema(raw)


def _schema_signature(schema: dict[str, object]) -> str:
    """比较会影响请求含义与执行边界的完整声明；传参：标准或旧配置Schema；返回：内容签名。"""
    payload = {
        "name": schema.get("name"),
        "description": schema.get("description", "MCP tool"),
        "inputSchema": _parameters_from_schema(schema),
        "annotations": schema.get("annotations", {}),
        "outputSchema": schema.get("outputSchema"),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _definition(
    owner: MCPRegistry,
    server: ServerProc,
    *,
    config: MCPServerConfig,
    schema: dict[str, object],
) -> ToolDefinition:
    """将一个标准工具接入共同授权/执行器；传参：owner、连接、配置及Schema；返回：有版本可发布定义。"""
    name = schema.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("MCP tool requires a non-empty name")
    parameters = _parameters_from_schema(schema)
    properties = cast(dict[str, object], parameters.get("properties", {}))
    paths = tuple(
        key for key, spec in properties.items() if _is_path_parameter(key, spec)
    )
    annotations = schema.get("annotations", {})
    readonly = isinstance(annotations, dict) and annotations.get("readOnlyHint") is True
    idempotent = (
        readonly
        and isinstance(annotations, dict)
        and annotations.get("idempotentHint") is True
    )
    return ToolDefinition(
        name=_registered_name(config.name, name),
        description=str(schema.get("description", "MCP tool")),
        parameters=parameters,
        toolset="agent",
        risk_level=ToolRisk(config.tool_risk_overrides.get(name, "confirm")),
        readonly=readonly,
        target_scope_rule="path" if paths else "logical_scope",
        source="mcp_reserved",
        target_scope_parameters=paths,
        idempotent=Idempotent.YES if idempotent else Idempotent.NO,
        deferred=True,
        parallel_safe=idempotent,
        executor=_MCPExecutor(owner, server, name, _schema_signature(schema)),
        availability_check=server.check_available,
        mcp_server=config.name,
    )


def _registered_name(server: str, tool: str) -> str:
    """将外部名字映射到各模型均可发送的名字；传参：服务与工具名；返回：保留普通旧名的唯一名称。"""
    original = f"mcp_{server}_{tool}"
    if len(original) <= _PROVIDER_TOOL_NAME_LIMIT and re.fullmatch(
        r"[A-Za-z0-9_-]+", original
    ):
        return original
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:_NAME_DIGEST_LENGTH]
    prefix = re.sub(r"[^A-Za-z0-9_-]", "_", original)[
        : _PROVIDER_TOOL_NAME_LIMIT - _NAME_DIGEST_LENGTH - 1
    ]
    return f"{prefix}_{digest}"


def _is_path_parameter(name: str, spec: object) -> bool:
    """识别标准或显式标记的本地路径参数；传参：名称及Schema；返回：是否经过文件授权边界。"""
    if name in {"filePath", "targetPath"} or "path" in name.casefold():
        return True
    if not isinstance(spec, dict):
        return False
    annotations = spec.get("annotations", {})
    return bool(
        spec.get("x-reins-path")
        or spec.get("path")
        or isinstance(annotations, dict)
        and annotations.get("path")
        or str(spec.get("format", "")).casefold()
        in {"path", "file-path", "filepath", "directory-path"}
    )


@dataclass(frozen=True, slots=True)
class _MCPExecutor:
    """绑定实际连接和请求定义，不在调用时换到同名的新服务。"""

    owner: MCPRegistry
    server: ServerProc
    tool: str
    expected_schema: str

    def __call__(self, args: dict[str, object]) -> object:
        """校验当前租约及真实Schema后派发；传参：共同执行器注入参数；返回：完整内容块或业务/传输错误。"""
        lease = args.get("__lease__")
        if not isinstance(lease, Lease) or self.server.name not in _allowed_servers(
            lease
        ):
            return ToolError(
                ToolErrorCategory.PERMISSION,
                "MCP server is outside the current lease",
                retryable=False,
                details={"execution_state": "not_started"},
            )
        try:
            self.owner.ensure_current(self.server)
        except (ServerUnavailableError, ValueError) as exc:
            return ToolError(
                ToolErrorCategory.TRANSPORT,
                str(exc),
                retryable=False,
                details={"execution_state": "not_started"},
            )
        current = next(
            (
                item
                for item in self.owner._schemas[self.server.name]
                if item["name"] == self.tool
            ),
            None,
        )
        if current is None or _schema_signature(current) != self.expected_schema:
            return ToolError(
                ToolErrorCategory.INVALID_INPUT,
                "MCP definition changed; reload the current native schema",
                retryable=False,
                details={"execution_state": "not_started"},
            )
        try:
            result = self.server.call_tool(
                self.tool,
                {key: value for key, value in args.items() if not key.startswith("__")},
                cancellation=cast(
                    CancellationToken | None, args.get("__cancellation__")
                ),
                operation_id=str(args.get("__operation_id__", "")),
                timeout_seconds=cast(float | None, args.get("__timeout_seconds__")),
            )
        except ServerUnavailableError as exc:
            return ToolError(
                ToolErrorCategory.TRANSPORT,
                str(exc),
                retryable=False,
                details={"execution_state": "unknown"},
            )
        if isinstance(result, ToolError):
            return result
        if result.get("isError") is not True:
            # 【MCP】【结果保存】标准content是内容块列表；整体放入运行时正文，避免丢失structuredContent与请求证据
            return {"content": result, "meta": {"mcp": result["mcp"]}}
        blocks = cast(list[dict[str, object]], result.get("content", []))
        message = "\n".join(
            str(item["text"]) for item in blocks if item.get("type") == "text"
        )
        return ToolError(
            ToolErrorCategory.BUSINESS,
            message or "MCP tool returned isError=true",
            retryable=False,
            details={"mcp_result": result, "execution_state": "completed"},
        )


def _allowed_servers(lease: Lease) -> set[str]:
    """读取当前MCP授权，不从工具名猜服务器；传参：租约；返回：允许服务集合。"""
    config = lease.capabilities.get("mcp")
    if not isinstance(config, dict) or config.get("enabled") is False:
        return set()
    allowed = config.get("allow_servers", [])
    if not isinstance(allowed, (list, tuple)):
        raise ValueError("MCP allow_servers must be a list")
    return {str(item) for item in allowed if item}


def _config_path_from_lease(lease: Lease) -> Path:
    """解析现有用户配置定位；传参：租约；返回：显式路径或既有默认位置。"""
    mcp = lease.capabilities.get("mcp")
    value = mcp.get("config_path") if isinstance(mcp, dict) else None
    return (
        Path(value).expanduser()
        if isinstance(value, str) and value.strip()
        else mcp_config.MCP_CONFIG_PATH
    )
