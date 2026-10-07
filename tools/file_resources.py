"""文件操作的真实资源身份，供授权复核、并行调度和变化记录共用。

作者：xxx
时间：2026-09-24 23:00:00
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import path_security
from runtime.lease import Lease

if TYPE_CHECKING:
    from tools.tool_registry import ToolDefinition

FILE_WRITERS = frozenset({"file_write", "file_patch"})
FILE_READERS = frozenset({"file_read", "list", "grep", "find_path"})


def file_identity(path: Path) -> str | None:
    """读取Windows卷和文件身份，硬链接共享此编号；传参：路径；返回：身份，文件不存在为None。"""
    try:
        status = path.stat()
    except FileNotFoundError:
        return None
    return f"{status.st_dev}:{status.st_ino}"


def describe_resource(
    definition: ToolDefinition | None, arguments: Mapping[str, object], lease: Lease
) -> dict[str, object]:
    """按最终工具声明固定可证明的文件范围；传参：定义、参数和权限；返回：资源或明确未知原因。"""
    if (
        definition is None
        or definition.source != "builtin"
        or definition.name not in FILE_WRITERS | FILE_READERS
    ):
        return {"known": False, "reason": "resource_not_declared"}
    target = arguments.get("path")
    if not isinstance(target, str) or not target.strip():
        return {"known": False, "reason": "missing_file_path"}
    try:
        path = path_security.resolve_target(Path(target), lease)
        return {
            "known": True,
            "path": str(path),
            "identity": file_identity(path),
            "directory": definition.name in {"list", "find_path"}
            or (definition.name == "grep" and path.is_dir()),
            "write": definition.name in FILE_WRITERS,
        }
    except OSError as exc:
        return {"known": False, "reason": type(exc).__name__}


def resources_conflict(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """核对同文件、硬链接和目录内写入；传参：两个已知资源；返回：是否必须保持公告顺序。"""
    if not left.get("write") and not right.get("write"):
        return False
    first, second = Path(str(left["path"])), Path(str(right["path"]))
    if first == second or (
        left.get("identity") is not None and left["identity"] == right.get("identity")
    ):
        return True
    return bool(
        (left.get("directory") and second.is_relative_to(first))
        or (right.get("directory") and first.is_relative_to(second))
    )
