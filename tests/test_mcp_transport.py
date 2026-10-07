"""标准MCP传输的配置边界；真实三种传输往返另由标准服务集成覆盖。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

import pytest
import yaml

from runtime.lease import Lease
from tools.mcp_client.config import load_mcp_access
from tools.mcp_client.registry import MCPRegistry
from tools.mcp_client.transport import ConnectionSettings
from tools.tool_registry import ToolRegistry


@pytest.mark.parametrize(
    "overrides",
    [
        {"transport": "websocket"},
        {"command": ()},
        {"transport": "sse", "url": "file:///local"},
        {"transport": "streamable_http", "url": "http:///missing-host"},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": True},
        {"protocol_version": 20250618},
        {"cwd": []},
    ],
)
def test_transport_rejects_invalid_configuration(overrides):
    """错误端点、超时及字段类型均明确失败；传参：无效设置；返回：无。"""
    with pytest.raises(ValueError):
        ConnectionSettings(**{"name": "test", "command": ("unused",), **overrides})


@pytest.mark.parametrize(
    "overrides",
    [
        {"timeout_seconds": None},
        {"timeout_seconds": []},
        {"timeout_seconds": "later"},
        {"protocol_version": 20250618},
        {"cwd": []},
        {"transport": None},
    ],
)
def test_invalid_server_config_does_not_hide_valid_server(tmp_path, overrides):
    """一个服务器字段损坏仍保留其他配置，并给出具名错误；传参：隔离根和坏字段；返回：无。"""
    path = tmp_path / "mcp.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "servers": {
                    "good": {"command": ["unused"]},
                    "bad": {"command": ["unused"], **overrides},
                }
            }
        ),
        encoding="utf-8",
    )
    owner = MCPRegistry(Lease(), ToolRegistry(), config_path=path)
    assert set(owner.configs) == {"good"}
    assert owner.failures["bad"]


def test_declaring_server_does_not_grant_production_access(tmp_path):
    """服务器声明与调用授权分别读取；传参：隔离配置；返回：无。"""
    path = tmp_path / "mcp.yaml"
    data = {"servers": {"local": {"command": ["unused"]}}}
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_mcp_access(path)["allow_servers"] == []
    path.write_text(
        yaml.safe_dump({**data, "allow_servers": ["local"]}), encoding="utf-8"
    )
    assert load_mcp_access(path)["allow_servers"] == ["local"]
