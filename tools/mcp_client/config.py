"""从真实用户配置读取MCP授权，定义服务器不自动授予调用权限。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

from pathlib import Path

import yaml

MCP_CONFIG_PATH = Path.home() / ".reins" / "mcp.yaml"


def load_mcp_access(path: Path | None = None) -> dict[str, object]:
    """生产入口读取显式服务器允许名单；传参：可选配置路径；返回：租约字段，不启动任何连接。"""
    config_path = path or MCP_CONFIG_PATH
    data = (
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if config_path.is_file()
        else {}
    )
    if not isinstance(data, dict):
        raise ValueError("MCP config must be an object")
    enabled, allowed = data.get("enabled", True), data.get("allow_servers", [])
    if not isinstance(enabled, bool):
        raise ValueError("MCP enabled must be a boolean")
    if not isinstance(allowed, list) or any(
        not isinstance(item, str) or not item.strip() for item in allowed
    ):
        raise ValueError("MCP allow_servers must be a list of non-empty server names")
    return {
        "enabled": enabled,
        "allow_servers": list(allowed),
        "config_path": str(config_path),
    }
