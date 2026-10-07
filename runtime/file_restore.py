"""【文件恢复】【预览服务】按已保存原件生成可核验、只能确认一次的恢复计划。

作者：xxx
时间：2026-09-30 22:00:00
"""

from __future__ import annotations

import codecs
import difflib
import hashlib
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import path_security
from approval.session import ApprovalMode
from runtime.file_restore_records import (
    RestoreAuthority,
    RestoreRecords,
    operation_view,
)
from runtime.file_snapshots import SnapshotStore, missing_state
from runtime.tool_operations import ToolOperationStore, file_changes
from runtime.workspaces import WorkspaceStore
from tasks.ids import utc_now

DEFAULT_PAGE_SIZE = 30
DEFAULT_DIFF_PAGE_SIZE = 8000
CHOICES = frozenset({"keep", "restore", "copy"})


class FileRestoreService:
    """文件恢复领域入口，查询不依赖模型，作业由后台持有。"""

    def __init__(
        self,
        data_root: Path | str,
        *,
        authority: Callable[[str], RestoreAuthority],
        changed: Callable[[str, tuple[str, ...]], None] | None = None,
    ) -> None:
        """注入当前权限和文件变化通知；参数：数据根、权限、通知；返回：无。"""
        self.snapshots = SnapshotStore(data_root)
        self.records = RestoreRecords(self.snapshots.database)
        self.workspaces = WorkspaceStore(data_root)
        self.authority, self.changed = authority, changed

    def query(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """读取恢复点、计划差异或作业；参数：固定身份与分页；返回：不含敏感正文的视图。"""
        action = payload.get("action", "list")
        if action == "status":
            return self.status(payload)
        if action == "diff":
            return self.diff(payload)
        session_id = require_text(payload, "session_id")
        workspace = self.workspaces.for_session(session_id)
        if (
            payload.get("workspace_id", workspace.workspace_id)
            != workspace.workspace_id
        ):
            raise ValueError("恢复查询与会话工作区不一致")
        if action == "list":
            return self.list_points(payload, session_id)
        if action in {"detail", "preview"}:
            return self.preview(payload, session_id, persist=action == "preview")
        raise ValueError("未知文件恢复查询")

    def list_points(
        self, payload: Mapping[str, Any], session_id: str
    ) -> dict[str, Any]:
        """按轮次和具体动作列目录；参数：分页及会话；返回：覆盖范围与可下钻身份。"""
        workspace = self.workspaces.for_session(session_id)
        points = self._points(session_id)
        if any(payload.get(key) for key in ("operation_id", "run_id", "call_id")):
            operations = ToolOperationStore(self.snapshots.data_root).for_session(
                session_id
            )
            identities = {
                row["operation_id"]
                for row in operations
                if (not payload.get("run_id") or row["run_id"] == payload["run_id"])
                and (
                    not payload.get("call_id")
                    or row["call"].get("call_id") == payload["call_id"]
                )
                and (
                    not payload.get("operation_id")
                    or row["operation_id"] == payload["operation_id"]
                )
            }
            if (
                payload.get("operation_id")
                and not payload.get("run_id")
                and not payload.get("call_id")
            ):
                identities.add(require_text(payload, "operation_id"))
            points = [point for point in points if point["operation_id"] in identities]
        points = list(reversed(sorted(points, key=lambda point: point["started_at"])))
        offset = page_number(payload.get("cursor", 0) or 0)
        limit = page_number(payload.get("limit", DEFAULT_PAGE_SIZE), positive=True)
        page = points[offset : offset + limit]
        turns = {
            point["input_id"]: {
                "input_id": point["input_id"],
                "started_at": point["started_at"],
                "session_id": point["session_id"],
            }
            for point in page
            if point.get("input_id")
        }
        fields = (
            "point_id",
            "input_id",
            "operation_id",
            "session_id",
            "run_id",
            "status",
            "started_at",
            "completed_at",
            "scope",
            "attribution",
            "exclusions",
            "errors",
        )
        return {
            "workspace_root": str(workspace.project_root),
            "workspace_id": workspace.workspace_id,
            "turns": list(turns.values()),
            "points": [
                {
                    **{key: point.get(key) for key in fields},
                    "entry_count": len(self._affected(point)),
                }
                for point in page
            ],
            "next_cursor": offset + limit if offset + limit < len(points) else None,
        }

    def _points(self, session_id: str) -> list[dict[str, Any]]:
        """保留首次发布顺序，秒级时间相同也不按随机身份排先后；参数：会话；返回：来源点。"""
        workspace = self.workspaces.for_session(session_id)
        with self.snapshots.database.snapshot() as source:
            points = list(
                source.list(
                    "file_restore_point",
                    workspace_id=workspace.workspace_id,
                    session_id=session_id,
                )
            )
        return points + self._legacy_points(session_id)

    def _legacy_points(self, session_id: str) -> list[dict[str, Any]]:
        """只展示旧哈希记录，不伪造历史字节；参数：会话；返回：不可恢复的目录项。"""
        rows = ToolOperationStore(self.snapshots.data_root).for_session(session_id)
        result = []
        for row in rows:
            for change in cast(list[dict[str, Any]], file_changes([row])["changes"]):
                if change.get("restore_point_ids"):
                    continue
                path = change.get("resolved_path") or change.get("requested_path")
                if not isinstance(path, str):
                    continue
                before = {
                    **missing_state(path),
                    "kind": "legacy",
                    "restorable": False,
                    "error": "仅有变化记录，无法恢复",
                }
                result.append(
                    {
                        "point_id": "legacy-" + row["operation_id"],
                        "operation_id": row["operation_id"],
                        "session_id": session_id,
                        "run_id": row.get("run_id", ""),
                        "input_id": row.get("input_id", ""),
                        "started_at": row["updated_at"],
                        "status": "legacy",
                        "exclusions": [],
                        "errors": [],
                        "entries": [
                            {
                                "path": path,
                                "before": before,
                                "after": None,
                                "attribution": "unknown",
                            }
                        ],
                    }
                )
        return result

    def _affected(self, point: Mapping[str, Any]) -> list[dict[str, Any]]:
        """筛出可观察变化，后态缺失仍保留有效旧版本；参数：恢复点；返回：变化项。"""
        result = []
        for entry in point["entries"]:
            before, after = (
                entry.get("displaced_before", entry["before"]),
                entry.get("after"),
            )
            if (
                before.get("kind") == "directory"
                and after
                and after.get("kind") == "directory"
            ):
                continue
            try:
                if after and self.snapshots.same_state(before, after):
                    continue
            except (OSError, ValueError):
                # 1. 【文件恢复】【原件缺失】损坏项仍显示在列表，逐项预览明确给出不可恢复原因
                pass
            key = hashlib.sha256(
                (point["point_id"] + "\0" + entry["path"]).encode()
            ).hexdigest()
            result.append(
                {
                    **entry,
                    "before": before,
                    "entry_id": key,
                    "point_id": point["point_id"],
                    "attribution": entry.get(
                        "attribution", point.get("attribution", "unknown")
                    ),
                }
            )
        return result

    def _entries(
        self, payload: Mapping[str, Any], session_id: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """按动作或轮次取真实前态；参数：已知来源、会话；返回：逐路径首次前态及来源点。"""
        points = self._points(session_id)
        if payload.get("point_id"):
            points = [
                point for point in points if point["point_id"] == payload["point_id"]
            ]
        elif payload.get("input_id"):
            points = [
                point
                for point in points
                if point.get("input_id") == payload["input_id"]
            ]
        else:
            raise ValueError("请选择具体轮次或文件动作")
        if not points:
            raise ValueError("恢复来源不存在或不属于当前会话")
        grouped: dict[str, dict[str, Any]] = {}
        for point in sorted(points, key=lambda point: point["started_at"]):
            for entry in self._affected(point):
                key = os.path.normcase(entry["path"])
                previous = grouped.get(key)
                grouped[key] = (
                    entry
                    if previous is None
                    else {
                        **previous,
                        "after": entry.get("after"),
                        "attribution": "exact"
                        if previous["attribution"] == entry["attribution"] == "exact"
                        else "unknown",
                    }
                )
        return list(grouped.values()), points

    def preview(
        self, payload: Mapping[str, Any], session_id: str, *, persist: bool
    ) -> dict[str, Any]:
        """固定选择和当前版本，只预览不改文件；参数：来源、会话、是否生成计划；返回：预览。"""
        workspace = self.workspaces.for_session(session_id)
        authority = self.authority(session_id)
        entries, points = self._entries(payload, session_id)
        choices = selection_choices(
            payload.get("selection", []), {entry["entry_id"] for entry in entries}
        )
        plan_id = "plan-" + uuid4().hex
        rows = [
            self._preview_entry(
                entry, authority, workspace.workspace_id, choices, plan_id=plan_id
            )
            for entry in entries
        ]
        selected = [row for row in rows if row["selected"]]
        blocked = authority.blocked_by
        can_execute = (
            bool(selected)
            and not blocked
            and all(row["executable"] for row in selected)
        )
        warning = (
            "来源待确认的选择可能一并撤销运行期间的外部编辑；"
            if any(
                row["selected"] and row["state"] in {"source_unknown", "conflict"}
                for row in rows
            )
            else ""
        )
        summary = "；".join(f"{row['action']} {row['destination']}" for row in selected)
        row = {
            "plan_id": plan_id,
            "session_id": session_id,
            "workspace_id": workspace.workspace_id,
            "workspace_root": str(workspace.project_root),
            "created_at": utc_now(),
            "entries": rows,
            "authority": authority.evidence(),
            "can_execute": can_execute,
            "blocked_by": list(blocked),
            "confirmation_text": f"{warning}本次仅执行已选 {len(selected)} 项：{summary}",
            "coverage_text": coverage_text(points),
            "source": dict(payload),
        }
        if persist:
            self.records.save("file_restore_plan", plan_id, row, create=True)
        return plan_view(row)

    def _preview_entry(
        self,
        entry: dict[str, Any],
        authority: RestoreAuthority,
        workspace_id: str,
        choices: Mapping[str, str],
        *,
        plan_id: str,
    ) -> dict[str, Any]:
        """验证单项字节和权限；参数：原件、当前权限、归属、选择、计划；返回：固定执行材料。"""
        path, target = Path(entry["path"]), entry["before"]
        sensitive = bool(target.get("sensitive")) or path_security.uses_redacted_files(
            path, authority.lease
        )
        choice = choices.get(entry["entry_id"], "keep")
        destination = (
            path.with_name(path.name + ".restored-" + plan_id[-12:])
            if choice == "copy"
            else path
        )
        if choice == "copy" and sensitive:
            destination = path.parent / ("restored-" + plan_id[-12:]) / path.name
        row = {
            **entry,
            "target": target,
            "choice": choice,
            "path": str(path),
            "destination": str(destination),
            "selected": choice != "keep",
            "sensitive": sensitive,
            "binary": False,
            "state": "unavailable",
            "default_selected": False,
            "executable": False,
            "error": None,
            "action": "另存副本" if choice == "copy" else "恢复",
            "captured_at": target.get("captured_at"),
        }
        try:
            ensure_path(path, self.workspaces.get(workspace_id).project_root)
            if (
                path_security.check_read(
                    path, authority.lease, filtered=row["sensitive"]
                )
                != path_security.Decision.ALLOWED
            ):
                raise PermissionError("当前权限不允许读取此文件的恢复原件")
            self.snapshots.validate_state(target)
            current = self.snapshots.capture_file(path, authority.lease, workspace_id)
            self.snapshots.validate_state(current)
            row["current"] = current
            row["sensitive"] = row["sensitive"] or bool(current.get("sensitive"))
            same = self.snapshots.same_state(current, target)
            trusted = entry["attribution"] == "exact" and entry.get("after") is not None
            matches_after = trusted and self.snapshots.same_state(
                current, entry["after"]
            )
            row["state"] = (
                "unchanged"
                if same
                else "ready"
                if matches_after
                else "conflict"
                if trusted
                else "source_unknown"
            )
            row["default_selected"] = row["state"] == "ready"
            row["expected"] = (
                self.snapshots.capture_file(destination, authority.lease, workspace_id)
                if choice == "copy"
                else current
            )
            if choice == "copy" and (
                target["kind"] != "file" or row["expected"]["kind"] != "missing"
            ):
                raise ValueError("只能把完整历史文件另存到尚不存在的副本路径")
            row["parents"] = parent_versions(
                destination, self.workspaces.get(workspace_id).project_root
            )
            row["action"] = (
                "另存副本"
                if choice == "copy"
                else "无需修改"
                if same
                else (
                    "删除"
                    if target["kind"] == "missing"
                    else "重新创建"
                    if current["kind"] == "missing"
                    else "覆盖"
                )
            )
            row["binary"] = is_binary(self.snapshots, target) or is_binary(
                self.snapshots, current
            )
            row["executable"] = write_allowed(destination, authority, row["sensitive"])
            if not row["executable"]:
                row["error"] = "当前权限不允许恢复此文件"
        except (OSError, ValueError) as exc:
            row.update(
                state="unavailable",
                error=str(exc),
                executable=False,
                default_selected=False,
            )
        return row

    def _owned(
        self, kind: str, identity: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        """读取原件并核对请求归属；参数：类型、身份、请求；返回：固定来源。"""
        row = self.records.require(kind, identity)
        for key in ("session_id", "workspace_id"):
            if key in payload and payload[key] != row[key]:
                raise ValueError("恢复记录不属于所选会话或工作区")
        if (
            self.workspaces.for_session(row["session_id"]).workspace_id
            != row["workspace_id"]
        ):
            raise ValueError("恢复记录工作区归属已失效")
        return row

    def diff(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """读取固定预览差异页；参数：计划与项、分页；返回：准确文本差异或明确元信息。"""
        plan = self._owned(
            "file_restore_plan", require_text(payload, "plan_id"), payload
        )
        identity = require_text(payload, "entry_id")
        entry = next(
            (entry for entry in plan["entries"] if entry["entry_id"] == identity), None
        )
        if entry is None:
            raise ValueError("计划中没有此文件")
        authority = self.authority(plan["session_id"])
        path = Path(entry["path"])
        if (
            path_security.check_read(path, authority.lease, filtered=entry["sensitive"])
            != path_security.Decision.ALLOWED
        ):
            raise PermissionError("当前权限不允许查看此文件")
        text = difference_text(self.snapshots, entry)
        offset = page_number(payload.get("offset", 0))
        limit = page_number(payload.get("limit", DEFAULT_DIFF_PAGE_SIZE), positive=True)
        end = min(offset + limit, len(text))
        return {
            "text": text[offset:end],
            "offset": offset,
            "total_chars": len(text),
            "has_more": end < len(text),
            "next_offset": end if end < len(text) else None,
        }

    def accept(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        """同计划幂等接纳明确选择；参数：计划及确认；返回：公开操作和首次接纳标志。"""
        plan = self._owned(
            "file_restore_plan", require_text(payload, "plan_id"), payload
        )
        with self.snapshots.database.snapshot() as source:
            previous = source.get(
                "file_restore_operation", "restore-" + plan["plan_id"]
            )
        if previous is not None:
            return operation_view(previous), False
        confirmation = payload.get("confirmation")
        if (
            not isinstance(confirmation, Mapping)
            or confirmation.get("accepted") is not True
        ):
            raise ValueError("请明确确认本次预览中的文件选择")
        selected = [entry for entry in plan["entries"] if entry["selected"]]
        if not plan["can_execute"] or not selected:
            raise ValueError("本计划不可执行，请重新预览并选择可恢复文件")
        if (
            any(entry["sensitive"] for entry in selected)
            and confirmation.get("sensitive") is not True
        ):
            raise PermissionError("本次受保护配置恢复需要专门确认")
        authority = self.authority(plan["session_id"])
        if authority.evidence() != plan["authority"] or authority.blocked_by:
            raise ValueError("权限或活动已变化，请重新预览")
        operation, created = self.records.accept(
            plan, {"accepted": True, "sensitive": confirmation.get("sensitive") is True}
        )
        return operation_view(operation), created

    def status(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """按身份或会话读取最近作业；参数：查询范围；返回：公开状态，无作业为空。"""
        if payload.get("operation_id"):
            return operation_view(
                self._owned(
                    "file_restore_operation",
                    require_text(payload, "operation_id"),
                    payload,
                )
            )
        session_id = require_text(payload, "session_id")
        with self.snapshots.database.snapshot() as source:
            rows = source.list("file_restore_operation", session_id=session_id)
        return operation_view(rows[-1]) if rows else {}

    def cancel(self, identity: str) -> dict[str, Any]:
        """标记剩余文件取消；参数：操作身份；返回：实际状态，不回滚已完成文件。"""
        return operation_view(self.records.cancel(identity))

    def run(self, identity: str) -> dict[str, Any]:
        """在后台执行既有计划；参数：唯一操作；返回：实际逐项回执。"""
        from runtime.file_restore_execution import RestoreExecution

        return RestoreExecution(self).run(identity)

    def reconcile(self) -> None:
        """重启只核对历史执行证据，不自动重放；参数：无；返回：无。"""
        from runtime.file_restore_execution import RestoreExecution

        RestoreExecution(self).reconcile()


def require_text(payload: Mapping[str, Any], key: str) -> str:
    """校验公共身份；参数：请求和字段；返回：非空文本。"""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


def page_number(value: Any, *, positive: bool = False) -> int:
    """校验页码和长度；参数：调用方值、是否正数；返回：整数，非法请求明确失败。"""
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError("分页参数必须为有效整数")
    return int(value)


def selection_choices(selection: Any, identities: set[str]) -> dict[str, str]:
    """只接纳后端已核实条目，不接受任意路径或正文；参数：具体选择、可选身份；返回：选择表。"""
    if not isinstance(selection, list):
        raise ValueError("selection must be a list")
    result = {}
    for row in selection:
        if not isinstance(row, dict) or set(row) != {"entry_id", "choice"}:
            raise ValueError("文件选择仅接受 entry_id 和 choice")
        identity, choice = row["entry_id"], row["choice"]
        if identity not in identities or identity in result or choice not in CHOICES:
            raise ValueError("文件选择不存在、重复或动作无效")
        result[identity] = choice
    return result


def ensure_path(path: Path, root: Path) -> None:
    """核对原路径及每个祖先的真实边界；参数：目标和原工作区；返回：无，链接不冒充普通文件。"""
    if (
        not path.is_absolute()
        or not path.is_relative_to(root)
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise PermissionError("文件不在原工作区内")
    for candidate in (path, *path.parents):
        try:
            status = candidate.lstat()
        except FileNotFoundError:
            continue
        if candidate.is_symlink() or bool(getattr(status, "st_reparse_tag", 0)):
            raise PermissionError("路径包含链接或重解析点，不能按普通文件恢复")
        if candidate == root:
            break


def parent_versions(path: Path, root: Path) -> list[dict[str, Any]]:
    """绑定父目录存在状态和权限，不让预览后路径替换绕过校验；参数：文件、根；返回：目录身份。"""
    from tools.restore_protection import read_security

    result: list[dict[str, Any]] = []
    for parent in path.parents:
        try:
            status = parent.stat()
            result.append(
                {
                    "path": str(parent),
                    "identity": f"{status.st_dev}:{status.st_ino}",
                    "security": read_security(parent),
                }
            )
        except FileNotFoundError:
            result.append({"path": str(parent), "identity": None})
        if parent == root:
            break
    return result


def write_allowed(path: Path, authority: RestoreAuthority, sensitive: bool) -> bool:
    """在具体计划确认前检查写入资格；参数：目标、实时权限、敏感标记；返回：允许确认执行。"""
    return (
        authority.mode is not ApprovalMode.READ_ONLY
        and path_security.check_write(path, authority.lease, filtered=sensitive)
        is not path_security.Decision.DENY
    )


def is_binary(store: SnapshotStore, state: Mapping[str, Any]) -> bool:
    """流式识别元信息预览场景；参数：原件服务及状态；返回：是否二进制，不把大文本误判。"""
    if state.get("kind") != "file" or state.get("sensitive"):
        return False
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        for chunk in store.iter_state(state):
            if "\0" in decoder.decode(chunk):
                return True
        decoder.decode(b"", final=True)
        return False
    except UnicodeDecodeError:
        return True


def difference_text(store: SnapshotStore, entry: Mapping[str, Any]) -> str:
    """生成固定原件差异，敏感资料只显示元信息；参数：原件服务、计划项；返回：安全文本。"""
    if entry["state"] == "unavailable":
        return str(entry["error"])
    current, target = entry["current"], entry["target"]
    sizes = f"当前：{current.get('metadata', {}).get('size', 0)} 字节；目标：{target.get('metadata', {}).get('size', 0)} 字节"
    if entry["sensitive"]:
        return f"受保护配置；仅显示元信息，真实秘密不会返回界面。\n{sizes}\n密文仅保证当前 Windows 账户可解密。"
    if entry["binary"]:
        return f"二进制文件；完整原件仍可恢复/另存副本。\n{sizes}\n当前版本：{current['version']}\n目标版本：{target['version']}"
    old = (
        store.read_state(current).decode("utf-8").splitlines(keepends=True)
        if current["kind"] == "file"
        else []
    )
    new = (
        store.read_state(target).decode("utf-8").splitlines(keepends=True)
        if target["kind"] == "file"
        else []
    )
    return (
        "".join(difflib.unified_diff(old, new, fromfile="当前文件", tofile="恢复目标"))
        or "文件与目标版本相同，无需修改"
    )


def coverage_text(points: list[dict[str, Any]]) -> str:
    """说明捕获区间和实际未覆盖范围；参数：来源点；返回：人可读覆盖说明。"""
    excluded = [
        f"{row['path']}：{row.get('reason', row.get('error', '未覆盖'))}"
        for point in points
        for row in point.get("exclusions", [])
    ]
    errors = [
        str(row.get("error", row))
        for point in points
        for row in point.get("errors", [])
    ]
    return (
        "各文件回到首次受影响前的已保存版本；逐文件捕获时间不同。工作区外及远端副作用未覆盖。"
        + ("\n排除：" + "；".join(excluded) if excluded else "")
        + ("\n覆盖错误：" + "；".join(errors) if errors else "")
    )


def plan_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """只返回可展示字段，隔离内部字节和权限材料；参数：固定计划；返回：界面预览。"""
    fields = (
        "entry_id",
        "path",
        "destination",
        "choice",
        "selected",
        "sensitive",
        "binary",
        "state",
        "default_selected",
        "executable",
        "error",
        "action",
        "captured_at",
        "attribution",
        "point_id",
    )
    return {
        key: row[key]
        for key in (
            "plan_id",
            "session_id",
            "workspace_id",
            "workspace_root",
            "created_at",
            "can_execute",
            "blocked_by",
            "confirmation_text",
            "coverage_text",
        )
    } | {
        "entries": [{key: entry.get(key) for key in fields} for entry in row["entries"]]
    }
