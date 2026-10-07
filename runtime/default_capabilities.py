from __future__ import annotations

from pathlib import Path
from tools.mcp_client.config import load_mcp_access

# 【文件权限】【默认来源】默认敏感策略独立于用户禁令，旧租约不据名单内容推测来源


def build_local_agent_capabilities(
    project_root: Path, data_root: Path
) -> dict[str, object]:
    """生成本地宿主的默认能力；传参：项目和数据目录；返回：带明确默认敏感策略的新快照。"""
    project = project_root.resolve()
    data = data_root.resolve()
    workspace = (project / ".reins" / "workspace").resolve()
    roots = [str(project), str(data), str(workspace)]
    return {
        "fs": {
            "project_root": str(project),
            "read": list(roots),
            "write": list(roots),
            "default_sensitive_policy": "redacted",
        },
        "terminal": {"enabled": True, "allow_commands": []},
        "browser": {
            "enabled": True,
            "profile": "default",
            "deny_domains": [],
            "headless": True,
        },
        "mouse_keyboard": {"enabled": False},
        "network": {"enabled": True, "deny_domains": []},
        "background_run": {"enabled": True},
        "mcp": load_mcp_access(),
    }
