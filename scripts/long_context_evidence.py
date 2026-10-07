"""固定长会话评测的源码及连接身份，供自然对照和专项共用。

作者：xxx
时间：2026-09-28 11:15:37
"""

from __future__ import annotations

import hashlib
import json
import platform
import tomllib
from pathlib import Path
from typing import Sequence

from llm.resolved_target import ResolvedModelTarget


def evaluation_source_files(project_root: Path) -> tuple[str, ...]:
    """定位可安装源码；传参：项目根目录；返回：源码及依赖声明的相对路径，不读取用户数据。"""
    configuration = tomllib.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    packaging = configuration["tool"]["setuptools"]
    paths = {project_root / "pyproject.toml"}
    paths.update(project_root / f"{name}.py" for name in packaging["py-modules"])
    # 【阶段三评测】【源码冻结】1. 跟随发布包清单收集完整运行源码，排除研究副本与会话数据
    for pattern in packaging["packages"]["find"]["include"]:
        for package in project_root.glob(pattern):
            if package.is_dir() and (package / "__init__.py").is_file():
                paths.update(package.rglob("*.py"))
    return tuple(sorted(path.relative_to(project_root).as_posix() for path in paths))


def snapshot_sources(
    project_root: Path,
    names: Sequence[str],
    destination: Path,
    *,
    resume: bool,
) -> dict[str, str]:
    """冻结或核对受测源码；传参：根目录、文件清单、快照目录和接续标记；返回：逐文件哈希。"""
    fingerprints = {}
    for name in names:
        content = (project_root / name).read_bytes()
        fingerprints[name] = hashlib.sha256(content).hexdigest()
        if not resume:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    return fingerprints


def model_manifest(target: ResolvedModelTarget) -> dict[str, object]:
    """记录本次解析目标；传参：真实模型配置；返回：公开参数与连接指纹，不包含密钥或原始地址。"""
    connection = {
        "base_url": target.base_url,
        "credential_name": target.credential_name,
    }
    encoded = json.dumps(connection, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "profile": target.profile_name,
        "provider": target.provider,
        "model": target.model,
        "api_mode": target.api_mode,
        "context_window": target.context_window,
        "output_tokens": target.output_token_limit,
        "timeout_seconds": target.timeout_seconds,
        "connection_sha256": hashlib.sha256(encoded).hexdigest(),
        "python_version": platform.python_version(),
    }
