"""【会话】【工作区归属】保存空间内稳定目录身份，执行时校验原目录。

作者：xxx
时间：2026-09-30 11:00:00
"""

from __future__ import annotations

import os
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from runtime.persistence import RuntimeStore
from runtime.session_message_store import SessionMessageStore

WORKSPACE_PATH_LABEL_LENGTH = 80
WORKSPACE_PATH_HASH_LENGTH = 16


@dataclass(frozen=True, slots=True)
class Workspace:
    """工作区目录的持久身份；路径不会随连接的启动目录变化。"""

    workspace_id: str
    project_root: Path
    name: str

    def require_available(self) -> None:
        """核对执行所需原目录；参数：无；返回：无，缺失或无访问权限明确失败。"""
        if not self.project_root.is_dir():
            raise FileNotFoundError(
                f"原工作区不可用，历史仍可查看：{self.project_root}"
            )
        with os.scandir(self.project_root):
            pass

    def snapshot(self) -> dict[str, Any]:
        """投影公开归属及执行可用性；参数：无；返回：只读界面字段，不创建目录或加载模型。"""
        error = None
        try:
            self.require_available()
        except OSError as exc:
            error = str(exc)
        return {
            "workspace_id": self.workspace_id,
            "project_root": str(self.project_root),
            "workspace_name": self.name,
            "workspace_available": error is None,
            "workspace_error": error,
        }


class WorkspaceStore:
    """在同一事实库维护工作区与会话归属，不复制到前端配置。"""

    def __init__(self, data_root: Path | str) -> None:
        """初始化目录身份表；参数：数据空间根；返回：无，不持有连接。"""
        self.database = RuntimeStore(data_root)
        self.messages = SessionMessageStore(data_root)

    def register(self, project_root: Path | str) -> Workspace:
        """按Windows规范绝对路径复用工作区；参数：明确目录；返回：稳定身份，不创建该目录。"""
        root = Path(project_root).expanduser().resolve()
        key = os.path.normcase(str(root))
        with self.database.transaction() as batch:
            existing = batch.list("workspace")
            for row in existing:
                if row["path_key"] == key:
                    return self.get(row["workspace_id"])
            directory = re.sub(r"[^\w.-]+", "__", str(root)).strip("._")[
                :WORKSPACE_PATH_LABEL_LENGTH
            ]
            directory += (
                "--"
                + hashlib.sha256(key.encode("utf-8")).hexdigest()[
                    :WORKSPACE_PATH_HASH_LENGTH
                ]
            )
            if any(row["directory"] == directory for row in existing):
                raise ValueError("workspace directory identity collision")
            identity = f"workspace-{uuid4().hex}"
            row = {
                "workspace_id": identity,
                "path_key": key,
                "project_root": str(root),
                "name": root.name or str(root),
                "directory": directory,
            }
            batch.put("workspace", identity, row)
            return Workspace(identity, root, row["name"])

    def get(self, workspace_id: str) -> Workspace:
        """读取已保存目录；参数：工作区编号；返回：原工作区，未知身份明确失败。"""
        with self.database.snapshot() as source:
            row = source.get("workspace", workspace_id)
            if row is None:
                raise ValueError(f"workspace identity not found: {workspace_id}")
            return Workspace(
                row["workspace_id"], Path(row["project_root"]), row["name"]
            )

    def for_session(self, session_id: str) -> Workspace:
        """查询会话原工作区；参数：会话编号；返回：持久归属，不使用启动目录兜底。"""
        workspace = self.find_for_session(session_id)
        if workspace is None:
            raise ValueError(f"session workspace is missing: {session_id}")
        return workspace

    def find_for_session(self, session_id: str) -> Workspace | None:
        """读取可选会话归属；参数：会话编号；返回：工作区或未绑定，执行必须用严格查询。"""
        with self.database.snapshot() as source:
            row = source.get("session_workspace", session_id)
            return None if row is None else self.get(row["workspace_id"])

    def bind_session(self, session_id: str, project_root: Path | str) -> Workspace:
        """原子创建会话并绑定工作区；参数：会话编号与明确目录；返回：归属，已有会话禁止改绑。"""
        with self.database.transaction() as batch:
            workspace = self.register(project_root)
            row = batch.get("session_workspace", session_id)
            if row is not None and row["workspace_id"] != workspace.workspace_id:
                raise ValueError("session workspace cannot be rebound")
            if row is None:
                batch.put(
                    "session_workspace",
                    session_id,
                    {"session_id": session_id, "workspace_id": workspace.workspace_id},
                    workspace_id=workspace.workspace_id,
                    session_id=session_id,
                )
            if not self.messages.exists(session_id):
                self.messages.create_session(session_id)
            return workspace

    def inherit_session(self, session_id: str, source_session_id: str) -> Workspace:
        """让子执行继承父会话原工作区；参数：子与父会话编号；返回：同一持久工作区。"""
        return self.bind_session(
            session_id, self.for_session(source_session_id).project_root
        )
