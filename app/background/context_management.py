"""【上下文】【查看与控制】将历史整理和知识维护原件接到现有后台查询。

作者：xxx
时间：2026-10-01 12:30:00
"""

from __future__ import annotations

import difflib
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping

from memory.store import MemoryStore
from runtime.context_compaction_jobs import ContextCompactionJobs
from runtime.evidence_content import DEFAULT_DETAIL_CHARS
from runtime.knowledge_maintenance import KnowledgeMaintenance
from runtime.knowledge_worker import committed_changes
from runtime.persistence import RuntimeStore, SourceSnapshot
from runtime.workspaces import WorkspaceStore

PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
KINDS = {"history": "context_compaction_job", "knowledge": "knowledge_maintenance"}
STATE_LABELS = {
    "queued": "等待执行",
    "running": "正在执行",
    "published": "历史已发布",
    "completed": "知识已更新",
    "no_op": "已核验，无需保存",
    "failed": "失败",
    "cancelled": "已取消",
    "cancelling": "正在取消",
}


class ContextManagement:
    """只投影已有领域状态，暂停或取消仍由各领域写者处理。"""

    def __init__(self, data_root: Path | str) -> None:
        """注入所选数据空间；参数：数据根；返回：无，不启动模型。"""
        self.root = Path(data_root)
        self.database = RuntimeStore(data_root)
        self.history = ContextCompactionJobs(data_root)
        self.knowledge = KnowledgeMaintenance(data_root)

    def query(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """校验界面归属后分派查看或明确控制；参数：界面请求；返回：持久事实及分页。"""
        session_id = _text(payload, "session_id")
        workspace = WorkspaceStore(self.root).for_session(session_id)
        space_id = self.database.data_space_id
        if payload.get("data_space_id") != space_id:
            raise ValueError("上下文查看的数据空间已变化，请重新连接")
        owner = {"session_id": session_id, "data_space_id": space_id}
        action = payload.get("action", "overview")
        if action == "overview":
            history = self.history.status(session_id=session_id)
            knowledge = self.knowledge.status(session_id=session_id)
            boundary = knowledge["boundary"]
            return {
                **owner,
                "history": {"enabled": history["enabled"]},
                "knowledge": {
                    "enabled": boundary["enabled"] if boundary is not None else True,
                    "enabled_at": boundary["enabled_at"]
                    if boundary is not None
                    else None,
                    "admission": next(iter(knowledge["admissions"]), None),
                },
            }
        if action == "configure":
            manager = self._manager(_text(payload, "domain"))
            enabled = payload.get("enabled")
            if type(enabled) is not bool:
                raise ValueError("自动维护开关必须是启用或暂停")
            manager.configure(enabled=enabled)
            return self.query({**owner, "action": "overview"})
        if action == "cancel":
            domain, identity = _text(payload, "domain"), _text(payload, "work_id")
            manager = self._manager(domain)
            row = manager.load(identity)
            _require_session(row, session_id)
            return {**owner, "work": manager.cancel(identity)}
        if action == "list":
            return {**owner, **self._list(payload, session_id)}
        if action != "detail":
            raise ValueError("unknown context management action")
        return {**owner, **self._detail(payload, session_id, workspace.workspace_id)}

    def _manager(self, domain: str) -> ContextCompactionJobs | KnowledgeMaintenance:
        """选择实际领域写者；参数：整理或知识领域；返回：已有管理器。"""
        if domain not in KINDS:
            raise ValueError("unknown context work domain")
        return self.history if domain == "history" else self.knowledge

    def _list(self, payload: Mapping[str, Any], session_id: str) -> dict[str, Any]:
        """按稳定创建位置分页，新增工作不会挤掉旧页；参数：游标和会话；返回：工作目录。"""
        limit = _number(
            payload.get("limit", PAGE_SIZE), minimum=1, maximum=MAX_PAGE_SIZE
        )
        before = payload.get("before")
        if before is not None and (
            not isinstance(before, list)
            or len(before) != 2
            or not all(isinstance(value, str) for value in before)
        ):
            raise ValueError("invalid context work cursor")
        with self.database.snapshot() as source:
            rows = [
                work_summary(domain, row)
                for domain, kind in KINDS.items()
                for row in source.list(kind, session_id=session_id)
                if row.get("job_id") or row.get("record_type") == "work"
            ]
        rows.sort(key=lambda row: (row["created_at"], row["work_id"]), reverse=True)
        total = len(rows)
        if before is not None:
            rows = [
                row
                for row in rows
                if (row["created_at"], row["work_id"]) < tuple(before)
            ]
        page = rows[:limit]
        next_before = (
            [page[-1]["created_at"], page[-1]["work_id"]] if len(rows) > limit else None
        )
        return {"items": page, "total": total, "next_before": next_before}

    def _detail(
        self, payload: Mapping[str, Any], session_id: str, workspace_id: str
    ) -> dict[str, Any]:
        """按提交位置冻结阅读内容，记忆差异使用已提交版本；参数：来源/分页和范围；返回：正文页。"""
        domain, identity = _text(payload, "domain"), _text(payload, "work_id")
        self._manager(domain)
        sequence = payload.get("commit")
        if sequence is not None:
            sequence = _number(sequence, minimum=0)
        offset = _number(payload.get("offset", 0), minimum=0)
        limit = _number(
            payload.get("limit", DEFAULT_DETAIL_CHARS),
            minimum=1,
            maximum=DEFAULT_DETAIL_CHARS,
        )
        with self.database.snapshot(sequence=sequence) as source:
            row = source.get(KINDS[domain], identity)
            if row is None:
                raise FileNotFoundError(identity)
            _require_session(row, session_id)
            if domain == "knowledge" and row.get("worker_session_id"):
                operations = sorted(
                    source.list("tool_operation", session_id=row["worker_session_id"]),
                    key=lambda operation: (
                        operation["updated_at"],
                        operation["operation_id"],
                    ),
                )
                commits, failures = committed_changes(operations)
                known = {
                    change["operation_id"]: change for change in row.get("commits", [])
                }
                known.update({change["operation_id"]: change for change in commits})
                row = {
                    **row,
                    "commits": list(known.values()),
                    "failed_operation_ids": failures,
                }
            summary = work_summary(domain, row)
            execution = execution_source(source, row)
            commit = source.sequence
        body = work_text(summary, row)
        for change in row.get("commits", []):
            body += "\n\n" + memory_change_text(
                self.root, change, session_id=session_id, workspace_id=workspace_id
            )
        end = min(offset + limit, len(body))
        if offset > len(body):
            raise ValueError("context detail offset exceeds captured body")
        return {
            "work": summary,
            "execution": execution,
            "commit": commit,
            "text": body[offset:end],
            "total_chars": len(body),
            "offset": offset,
            "has_more": end < len(body),
            "next_offset": end if end < len(body) else None,
        }


def work_summary(domain: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """从原件提取可比较的目录元信息；参数：领域和原记录；返回：无正文摘要。"""
    history = domain == "history"
    return {
        "domain": domain,
        "work_id": row["job_id"] if history else row["work_id"],
        "title": "历史整理" if history else "自动知识维护",
        "state": row["status"] if history else row["state"],
        "created_at": row["accepted_at"] if history else row["created_at"],
        "source_count": row["covered_count"] if history else len(row["message_ids"]),
        "cancel_requested": bool(row.get("cancel_requested")),
        "error": row.get("error"),
        "summary_id": row.get("summary_id"),
        "source_session_id": row["source_session_id"],
    }


def execution_source(
    source: SourceSnapshot, row: Mapping[str, Any]
) -> dict[str, str] | None:
    """沿已接纳的发生定位实际模型记录；参数：快照和工作；返回：可打开的请求归属或尚未执行。"""
    schedule_id = row.get("schedule_id") or f"schedule-{row['job_id']}"
    occurrences = source.list(
        "schedule_occurrence", filters={"schedule_id": schedule_id}
    )
    for occurrence in reversed(occurrences):
        if occurrence.get("session_id") and occurrence.get("run_id"):
            return {
                "session_id": occurrence["session_id"],
                "run_id": occurrence["run_id"],
            }
    return None


def work_text(summary: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """将真实执行与发布状态写成用户可读说明；参数：目录和原记录；返回：详情正文。"""
    state = summary["state"]
    lines = [
        f"{summary['title']} · {STATE_LABELS.get(state, state)}",
        f"接纳时间：{summary['created_at']}",
        f"来源范围：{summary['source_count']} 条消息",
        f"来源会话：{summary['source_session_id']}",
    ]
    if summary["cancel_requested"]:
        lines.append("已请求取消；实际停止以执行回执为准")
    if summary["summary_id"]:
        lines.extend(
            [
                f"已发布历史版本：{summary['summary_id']}",
                "发布表示后台整理已保存；具体哪次请求采用，请查看实际请求的来源记录",
            ]
        )
    if row.get("reason"):
        lines.append(f"处理原因：{row['reason']}")
    if row.get("reasons"):
        lines.append("触发来源：" + "、".join(row["reasons"]))
    if summary["error"]:
        lines.append(f"失败或中断：{summary['error']}")
    lines.extend(
        [
            "",
            "模型请求、重试与供应商用量可从“查看模型请求及用量”展开；未报告的用量保持未知",
            "本地材料复用、历史发布、实际请求采用和供应商缓存命中分别记录",
        ]
    )
    return "\n".join(lines)


def memory_change_text(
    root: Path, change: Mapping[str, Any], *, session_id: str, workspace_id: str
) -> str:
    """读取维护实际提交的旧版、新版、原因与来源；参数：提交引用和授权范围；返回：差异正文。"""
    identity, version = _text(change, "memory_id"), _text(change, "version")
    with closing(MemoryStore(root)) as memories:
        current = memories.load_memory(identity, version=version)
        if current.details.scope not in {
            f"session:{session_id}",
            f"project:{workspace_id}",
        }:
            raise ValueError("knowledge change belongs to another workspace or session")
        previous = (
            memories.load_memory(identity, version=current.previous_version)
            if current.previous_version
            else None
        )
    lines = [
        f"记忆 {identity} · 第 {current.revision} 版",
        f"修改原因：{current.reason}",
        f"适用范围：{current.details.scope}",
        f"已提交版本：{version}",
    ]
    lines.extend(
        f"来源：{item.kind} · {item.reference} · 会话 {item.session_id}"
        for item in current.details.sources
    )
    if previous is None:
        lines.extend(["新增内容：", current.content])
    else:
        lines.extend([f"上一版：{previous.version}", "修改差异："])
        lines.extend(
            difflib.unified_diff(
                previous.content.splitlines(),
                current.content.splitlines(),
                fromfile="修改前",
                tofile="修改后",
                lineterm="",
            )
        )
    return "\n".join(lines)


def _require_session(row: Mapping[str, Any], session_id: str) -> None:
    """拒绝串入其他会话的工作控制；参数：原工作与界面会话；返回：无。"""
    if row.get("source_session_id") != session_id:
        raise ValueError("context work does not belong to the selected session")


def _text(payload: Mapping[str, Any], key: str) -> str:
    """读取不可为空的界面身份；参数：请求与字段；返回：原字符串。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be non-empty text")
    return value


def _number(value: Any, *, minimum: int, maximum: int | None = None) -> int:
    """校验稳定阅读页的边界；参数：数值及允许范围；返回：整数。"""
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError("invalid context page boundary")
    return value
