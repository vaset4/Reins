"""【会话】【目录与搜索】从可重建索引分页定位文件原件。

作者：xxx
时间：2026-09-30 16:00:00
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from runtime.persistence import RuntimeStore
from runtime.workspaces import WorkspaceStore

SESSION_PAGE_SIZE = 100
SEARCH_PAGE_SIZE = 50
SEARCH_CONTENT_PAGE_CHARS = 8000
_DIRECTORY_SQL = """
WITH matches AS (
 SELECT r.*, ROW_NUMBER() OVER (PARTITION BY r.session_id ORDER BY r.sequence DESC,r.record_id DESC) AS position
 FROM records r JOIN records_fts f ON f.kind=r.kind AND f.record_id=r.record_id
 WHERE records_fts MATCH ? AND r.kind IN ('session_entry','tool_operation')
), auxiliary AS (
 SELECT json_extract(payload,'$.worker_session_id') AS session_id,
        json_extract(payload,'$.source_session_id') AS owner_session_id
 FROM records WHERE kind='knowledge_maintenance' AND json_extract(payload,'$.record_type')='work'
 UNION
 SELECT json_extract(payload,'$.session_id'),json_extract(payload,'$.schedule_snapshot.knowledge_origin.source_session_id')
 FROM records WHERE kind='schedule_occurrence'
   AND json_extract(payload,'$.schedule_snapshot.knowledge_origin.work_kind')='knowledge_maintenance'
), directory AS (
 SELECT s.record_id AS session_id, json_extract(s.payload,'$.title') AS title,
 COALESCE(NULLIF(json_extract(t.payload,'$.updated_at'),''),json_extract(s.payload,'$.created_at')) AS updated_at,
 COALESCE(NULLIF(json_extract(t.payload,'$.last_run_status'),''),'idle') AS status,
 w.record_id AS workspace_id,json_extract(w.payload,'$.project_root') AS project_root,
 json_extract(w.payload,'$.name') AS workspace_name,
 CASE WHEN a.owner_session_id IS NULL THEN 'chat' ELSE 'maintenance' END AS purpose,a.owner_session_id,
 json_extract(owner.payload,'$.title') AS owner_title,
 (SELECT count(*) FROM auxiliary children WHERE children.owner_session_id=s.record_id AND children.session_id IS NOT NULL) AS maintenance_count,
 m.kind AS match_kind,m.record_id AS match_id,m.sequence AS match_sequence,m.source_path,m.source_offset
 FROM records s
 LEFT JOIN records t ON t.kind='session_state' AND t.record_id=s.record_id
 LEFT JOIN records sw ON sw.kind='session_workspace' AND sw.record_id=s.record_id
 LEFT JOIN records w ON w.kind='workspace' AND w.record_id=json_extract(sw.payload,'$.workspace_id')
 LEFT JOIN matches m ON m.session_id=s.record_id AND m.position=1
 LEFT JOIN auxiliary a ON a.session_id=s.record_id
 LEFT JOIN records owner ON owner.kind='session' AND owner.record_id=a.owner_session_id
 WHERE s.kind='session'
)
SELECT * FROM directory
WHERE (? IS NULL OR (julianday(updated_at),session_id)<(julianday(?),?))
 AND (? IS NULL OR workspace_id=?)
 AND (?='all' OR purpose=?)
 AND (? IS NULL OR owner_session_id=?)
 AND (?='' OR instr(lower(session_id)||' '||lower(title)||' '||lower(COALESCE(project_root,''))||' '||
                    lower(COALESCE(workspace_name,'')),?)>0 OR match_id IS NOT NULL)
ORDER BY julianday(updated_at) DESC,session_id DESC LIMIT ?
"""


def fulltext_query(query: str) -> str:
    """把用户文字转换为字面FTS词组；参数：原查询；返回：安全的中文双字及英文词表达式。"""
    words = re.findall(r"[\u3400-\u9fff]+|[^\W_]+", query.casefold())
    tokens: list[str] = []
    for word in words:
        if len(word) > 1 and all("\u3400" <= char <= "\u9fff" for char in word):
            tokens.extend(word[index : index + 2] for index in range(len(word) - 1))
        else:
            tokens.append(word)
    return (
        " AND ".join('"' + token.replace('"', '""') + '"' for token in tokens) or '""'
    )


def directory_page(
    data_root: Path,
    query: str = "",
    *,
    before: list[str] | None = None,
    workspace_id: str | None = None,
    purpose: str = "chat",
    owner_session_id: str | None = None,
) -> dict[str, Any]:
    """查询有界全局会话页；参数：根、搜索、游标、工作区；返回：目录及原件命中位置。"""
    _validate_directory_filters(data_root, before, workspace_id)
    if purpose not in {"chat", "maintenance", "all"}:
        raise ValueError("session purpose must be chat, maintenance or all")
    if owner_session_id is not None and (
        not isinstance(owner_session_id, str) or not owner_session_id
    ):
        raise ValueError("maintenance owner must be a non-empty session identity")
    store = RuntimeStore(data_root)
    cursor: tuple[str | None, str | None] = (
        (before[0], before[1]) if before else (None, None)
    )
    with store.index_connection() as connection:
        workspace_rows = [
            {"workspace_id": row["record_id"], **json.loads(row["payload"])}
            for row in connection.execute(
                "SELECT record_id,payload FROM records WHERE kind='workspace' ORDER BY lower(json_extract(payload,'$.name')),record_id"
            )
        ]
        results = connection.execute(
            _DIRECTORY_SQL,
            (
                fulltext_query(query),
                cursor[0],
                cursor[0],
                cursor[1],
                workspace_id,
                workspace_id,
                purpose,
                purpose,
                owner_session_id,
                owner_session_id,
                query.casefold(),
                query.casefold(),
                SESSION_PAGE_SIZE + 1,
            ),
        ).fetchall()
    rows = [_directory_item(record, query) for record in results[:SESSION_PAGE_SIZE]]
    next_before = (
        [rows[-1]["updated_at"], rows[-1]["session_id"]]
        if len(results) > SESSION_PAGE_SIZE
        else None
    )
    return {
        "sessions": rows,
        "has_more": next_before is not None,
        "next_before": next_before,
        "workspaces": workspace_rows,
        "workspace_id": workspace_id,
        "index_status": store.index_status,
        "purpose": purpose,
        "owner_session_id": owner_session_id,
        "data_space_id": store.data_space_id,
        "data_root": str(data_root),
    }


def _validate_directory_filters(
    root: Path, before: list[str] | None, workspace_id: str | None
) -> None:
    """核对分页身份和工作区；参数：数据根、游标、工作区；返回：无，不接受未知归属。"""
    if before is not None and (
        not isinstance(before, list)
        or len(before) != 2
        or not all(isinstance(item, str) for item in before)
    ):
        raise ValueError("session directory cursor must contain timestamp and identity")
    if workspace_id is not None:
        if not isinstance(workspace_id, str) or not workspace_id:
            raise ValueError("workspace filter must be a non-empty identity")
        WorkspaceStore(root).get(workspace_id)


def _directory_item(record: sqlite3.Row, query: str) -> dict[str, Any]:
    """区分普通目录命中与正文原件命中；参数：派生摘要和查询；返回：可直接供界面使用的目录行。"""
    row = {
        key: record[key]
        for key in (
            "session_id",
            "title",
            "updated_at",
            "status",
            "workspace_id",
            "project_root",
            "workspace_name",
            "purpose",
            "owner_session_id",
            "owner_title",
            "maintenance_count",
        )
    }
    row["title"] = row["title"] or (
        "自动知识维护" if row["purpose"] == "maintenance" else "新会话"
    )
    directory_text = " ".join(
        str(row.get(key) or "")
        for key in ("session_id", "title", "project_root", "workspace_name")
    ).casefold()
    if record["match_id"] is not None and query.casefold() not in directory_text:
        row["match"] = {
            "kind": record["match_kind"],
            "record_id": record["match_id"],
            "sequence": record["match_sequence"],
            "source_path": record["source_path"],
            "source_offset": record["source_offset"],
            "session_id": record["session_id"],
        }
    return row


def search_matches(
    data_root: Path, session_id: str, query: str, *, after: int = 0
) -> dict[str, Any]:
    """列出选中会话的每个正文命中；参数：根、会话、查询及位置；返回：轻量原件分页。"""
    if type(after) is not int or after < 0:
        raise ValueError("search cursor must be a nonnegative integer")
    store = RuntimeStore(data_root)
    with store.index_connection() as connection:
        rows = connection.execute(
            """
            SELECT r.kind,r.record_id,r.sequence,r.source_path,r.source_offset,r.payload
            FROM records r JOIN records_fts f ON f.kind=r.kind AND f.record_id=r.record_id
            WHERE records_fts MATCH ? AND r.session_id=? AND r.kind IN ('session_entry','tool_operation')
            ORDER BY r.sequence DESC,r.record_id DESC LIMIT ? OFFSET ?
        """,
            (fulltext_query(query), session_id, SEARCH_PAGE_SIZE + 1, after),
        ).fetchall()
    results = []
    for row in rows[:SEARCH_PAGE_SIZE]:
        payload = json.loads(row["payload"])
        results.append(
            {
                "kind": row["kind"],
                "record_id": row["record_id"],
                "sequence": row["sequence"],
                "source_path": row["source_path"],
                "source_offset": row["source_offset"],
                "session_id": session_id,
                "entry_id": payload.get("entry_id"),
                "run_id": payload.get("run_id"),
                "label": "对话原文" if row["kind"] == "session_entry" else "工具结果",
            }
        )
    return {
        "matches": results,
        "next_after": after + SEARCH_PAGE_SIZE
        if len(rows) > SEARCH_PAGE_SIZE
        else None,
    }


def search_content(
    data_root: Path,
    source: dict[str, Any],
    *,
    offset: int = 0,
    limit: int = SEARCH_CONTENT_PAGE_CHARS,
) -> dict[str, Any]:
    """分页读取命中的冻结原件；参数：根、来源身份及字符页；返回：原文，不改变执行叶。"""
    from runtime.evidence_content import content_page, evidence_node

    kind, record_id = source.get("kind"), source.get("record_id")
    if kind not in {"session_entry", "tool_operation"} or not isinstance(
        record_id, str
    ):
        raise ValueError("invalid searchable source identity")
    sequence = source.get("sequence")
    if type(sequence) is not int:
        raise ValueError("search source requires committed sequence")
    store = RuntimeStore(data_root)
    with store.snapshot(sequence=sequence) as snapshot:
        record = snapshot.raw(kind, record_id)
        if record is None or record.session_id != source.get("session_id"):
            raise ValueError("search source does not belong to the selected session")
    node = evidence_node(record)
    body = node.get("message") if kind == "session_entry" else node.get("result", node)
    return {
        **content_page(store, body, offset=offset, limit=limit),
        "status": "历史原件",
        "retention": "captured",
    }
