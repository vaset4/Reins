"""从真实文件工具入口验证脱敏读写和原有权限边界。

作者：xxx
时间：2026-09-24 22:00:00
"""

import re

import pytest

from runtime.default_capabilities import build_local_agent_capabilities
from runtime.lease import from_trigger
from tools.builtin_tools import build_tool_registry
from tools.types import ToolError


def _api(root, *, deny=False):
    """装配本地默认能力和生产注册表；传参：隔离根、显式禁令；返回：目录与租约。"""
    capabilities = build_local_agent_capabilities(root, root / "data")
    if deny:
        capabilities["fs"]["deny_read"] = [".env"]
    return build_tool_registry(repo_root=root, data_root=root / "data"), from_trigger(
        "user", task_id="edit", capabilities=capabilities
    )


@pytest.mark.parametrize("tool", ["file_write", "file_patch"])
def test_registry_edits_plain_setting_without_disclosing_secret(tmp_path, tool):
    """真实入口修改端口、保留密钥且回执无秘密；传参：目录和工具；返回：无。"""
    path = tmp_path / ".env"
    original = b"TOKEN=ORIGINAL_PRIVATE\r\nPORT=3000\r\n"
    path.write_bytes(original)
    registry, lease = _api(tmp_path)
    view = registry.execute_tool("file_read", {"path": ".env"}, lease)
    assert not isinstance(view, ToolError), view
    assert "ORIGINAL_PRIVATE" not in str(view) and "PORT=3000" in view["content"]
    args = {
        "path": ".env",
        "view_id": view["meta"]["view_id"],
        "expected_sha256": view["meta"]["content_sha256"],
    }
    if tool == "file_write":
        args["content"] = view["content"].replace("3000", "4000")
    else:
        args.update(old_text="PORT=3000", new_text="PORT=4000")
    result = registry.execute_tool(tool, args, lease)
    assert not isinstance(result, ToolError), result
    assert path.read_bytes() == original.replace(b"3000", b"4000")
    assert "ORIGINAL_PRIVATE" not in str(result)


def test_custom_deny_and_legacy_lease_still_refuse_sensitive_read(tmp_path):
    """显式禁令和来源不明的旧权限不会因脱敏被放宽；传参：目录；返回：无。"""
    (tmp_path / ".env").write_bytes(b"TOKEN=PRIVATE\n")
    registry, lease = _api(tmp_path, deny=True)
    assert isinstance(
        registry.execute_tool("file_read", {"path": ".env"}, lease), ToolError
    )
    old = from_trigger(
        "user",
        task_id="old",
        capabilities={"fs": {"project_root": str(tmp_path), "read": [str(tmp_path)]}},
    )
    assert isinstance(
        registry.execute_tool("file_read", {"path": ".env"}, old), ToolError
    )


def test_recursive_search_returns_sensitive_path_without_fragments(tmp_path):
    """递归搜索只展示敏感文件名，不展示命中片段和行号；传参：目录；返回：无。"""
    (tmp_path / ".env").write_bytes(b"TOKEN=PRIVATE_NEEDLE\n")
    registry, lease = _api(tmp_path)
    result = registry.execute_tool("grep", {"path": ".", "query": "NEEDLE"}, lease)
    assert not isinstance(result, ToolError), result
    assert ".env" in result["content"] and "内容已脱敏" in result["content"]
    assert "PRIVATE" not in str(result) and not re.search(
        r"\.env:\d", result["content"]
    )


def test_private_key_creation_is_refused_even_with_plain_filename(tmp_path):
    """私钥内容不能借普通文件名写入；传参：目录；返回：无。"""
    registry, lease = _api(tmp_path)
    result = registry.execute_tool(
        "file_write",
        {"path": "plain.txt", "content": "-----BEGIN PRIVATE KEY-----\nPRIVATE\n"},
        lease,
    )
    assert isinstance(result, ToolError)
    assert not (tmp_path / "plain.txt").exists()


def test_private_key_with_plain_filename_is_never_read_or_searched(tmp_path):
    """私钥正文识别不依赖路径名单；传参：目录；返回：无。"""
    (tmp_path / "plain.txt").write_bytes(
        b"-----BEGIN PRIVATE KEY-----\nPRIVATE_NEEDLE\n"
    )
    registry, lease = _api(tmp_path)
    view = registry.execute_tool("file_read", {"path": "plain.txt"}, lease)
    assert view["meta"]["metadata_only"] and "PRIVATE_NEEDLE" not in str(view)
    search = registry.execute_tool("grep", {"path": ".", "query": "NEEDLE"}, lease)
    assert "PRIVATE_NEEDLE" not in str(search) and "内容已脱敏" in search["content"]


def test_host_memory_survives_registry_rebuild_and_expires_on_close(tmp_path):
    """后台每轮重建目录仍能使用原视图，宿主重启后明确失效；传参：目录；返回：无。"""
    from runtime.shared_budget import BudgetOwner
    from runtime.watchdog import Watchdog
    from tools.redacted_files import RedactedFiles

    (tmp_path / ".env").write_bytes(b"TOKEN=PRIVATE\nPORT=3000\n")
    first, lease = _api(tmp_path)
    files = RedactedFiles()
    first.bind_redacted_files(files)
    watchdog = Watchdog(
        lease,
        data_root=tmp_path / "data",
        budget_owner=BudgetOwner("session", "run", lease),
    )
    view = first.execute_tool("file_read", {"path": ".env"}, lease, watchdog=watchdog)
    first.close()
    second = build_tool_registry(
        repo_root=tmp_path, data_root=tmp_path / "data", redacted_files=files
    )
    args = {
        "path": ".env",
        "view_id": view["meta"]["view_id"],
        "expected_sha256": view["meta"]["content_sha256"],
        "old_text": "PORT=3000",
        "new_text": "PORT=4000",
    }
    result = second.execute_tool("file_patch", args, lease, watchdog=watchdog)
    assert not isinstance(result, ToolError), result
    refreshed = second.execute_tool(
        "file_read", {"path": ".env"}, lease, watchdog=watchdog
    )
    files.close()
    args.update(
        view_id=refreshed["meta"]["view_id"],
        expected_sha256=refreshed["meta"]["content_sha256"],
    )
    assert isinstance(
        second.execute_tool("file_patch", args, lease, watchdog=watchdog), ToolError
    )


def test_sensitive_link_name_keeps_redaction_and_explicit_deny(tmp_path):
    """链接名与真实目标均参与敏感策略，不因解析路径丢失禁令；传参：目录；返回：无。"""
    target = tmp_path / "plain.txt"
    target.write_text("TOKEN=LINK_PRIVATE\n", encoding="utf-8")
    (tmp_path / ".env").symlink_to(target)
    registry, lease = _api(tmp_path)
    view = registry.execute_tool("file_read", {"path": ".env"}, lease)
    assert "LINK_PRIVATE" not in str(view)
    assert view["meta"]["redacted"]
    search = registry.execute_tool("grep", {"path": ".env", "query": "TOKEN"}, lease)
    assert "LINK_PRIVATE" not in str(search)
    denied, lease = _api(tmp_path, deny=True)
    assert isinstance(
        denied.execute_tool("file_read", {"path": ".env"}, lease), ToolError
    )
