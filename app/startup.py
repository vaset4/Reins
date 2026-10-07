from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StartupIdentity:
    """启动阶段解析后的项目根和运行数据根身份"""

    project_root: Path
    data_root: Path


def resolve_startup_identity(
    *,
    project_root: Path | str | None = None,
    data_root: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    user_home: Path | str | None = None,
) -> StartupIdentity:
    """解析入口共用的项目与数据根目录

    参数：project_root 为显式项目根；data_root 为显式数据根；environ/user_home 用于隔离路径解析
    返回：StartupIdentity，两个路径均已展开用户目录并解析为绝对路径
    """
    env = os.environ if environ is None else environ
    selected_project = project_root
    if selected_project is None:
        selected_project = (
            env.get("REINS_PROJECT_ROOT")
            or env.get("XIANGMU_PROJECT_ROOT")
            or Path.cwd()
        )
    resolved_project = _resolve_path(selected_project)
    selected_data = data_root
    if selected_data is None:
        selected_data = (
            Path(user_home) / ".reins" / "data"
            if user_home is not None
            else Path.home() / ".reins" / "data"
        )
    return StartupIdentity(
        project_root=resolved_project,
        data_root=_resolve_path(selected_data),
    )


def _resolve_path(value: Path | str) -> Path:
    """统一展开用户目录并解析启动路径

    参数：value 为待解析的路径
    返回：规范化后的绝对 Path
    """
    return Path(value).expanduser().resolve()
