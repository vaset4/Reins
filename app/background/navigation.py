"""全屏界面的只读会话目录与树投影。

作者：xxx
时间：2026-09-29 18:00:00
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.background.sessions import session_history_page
from runtime.session_directory import directory_page, search_content, search_matches
from runtime.session_message_store import DEFAULT_HISTORY_PAGE_SIZE, SessionMessageStore


def history_page(
    data_root: Path,
    session_id: str,
    *,
    leaf_id: str | None = None,
    before: str | None = None,
    limit: int = DEFAULT_HISTORY_PAGE_SIZE,
) -> dict[str, Any]:
    """投影一页固定分支历史；传参：目录、会话及锚点；返回：正文和下一页游标。"""
    return session_history_page(
        data_root, session_id, leaf_id=leaf_id, before=before, limit=limit
    )


def list_sessions(
    data_root: Path,
    query: str = "",
    *,
    before: list[str] | None = None,
    workspace_id: str | None = None,
    purpose: str = "chat",
    owner_session_id: str | None = None,
) -> dict[str, Any]:
    """查询指定工作区或全部会话；传参：数据根、检索词、游标与工作区；返回：目录投影。"""
    return directory_page(
        data_root,
        query,
        before=before,
        workspace_id=workspace_id,
        purpose=purpose,
        owner_session_id=owner_session_id,
    )


def session_tree(
    data_root: Path,
    session_id: str,
    *,
    after: int = 0,
    view: str = "turns",
    query: str = "",
    selected_entry: str | None = None,
) -> dict[str, Any]:
    """读取一页节点摘要而不改当前分支；传参：数据根、会话及游标；返回：摘要及当前叶。"""
    messages = SessionMessageStore(data_root)
    if not messages.exists(session_id):
        return {
            "session_id": session_id,
            "entries": [],
            "leaf_id": None,
            "next_after": None,
        }
    page = messages.tree_page(
        session_id, after=after, view=view, query=query, selected_entry=selected_entry
    )
    return {
        "session_id": session_id,
        **page,
        "leaf_id": messages.current_leaf(session_id),
    }


def session_search(data_root: Path, params: dict[str, Any]) -> dict[str, Any]:
    """转交历史搜索的只读操作；参数：数据根和已认证请求；返回：命中页或原件正文页。"""
    action = params.get("action")
    if action == "matches":
        session_id, query = params.get("session_id"), params.get("query")
        if not isinstance(session_id, str) or not isinstance(query, str):
            raise ValueError("search requires session identity and text query")
        return search_matches(
            data_root, session_id, query, after=params.get("after", 0)
        )
    if action == "content":
        source = params.get("source")
        if not isinstance(source, dict):
            raise ValueError("search requires a structured source identity")
        return search_content(
            data_root,
            source,
            offset=params.get("offset", 0),
            limit=params.get("limit", 8000),
        )
    raise ValueError("unknown session search action")
