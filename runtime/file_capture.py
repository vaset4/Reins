"""【文件恢复】【执行接线】在真实后端窗口捕获前后态，不在审批或模型等待时占锁。

作者：xxx
时间：2026-09-30 21:00:00
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, TYPE_CHECKING, TypeVar
from uuid import uuid4

import path_security
from runtime.cancellation import CancellationToken, ExecutionCancelled
from runtime.file_snapshots import SnapshotStore, missing_state, new_point
from runtime.file_content import verified_content_scope
from runtime.lease import Lease
from runtime.workspaces import WorkspaceStore
from tasks.ids import utc_now
from tools.file_persistence import FileEditConflict, read_file_bytes
from tools.workspace_coordination import workspace_write_window, workspace_write_windows
from tools.types import ToolError, ToolErrorCategory

if TYPE_CHECKING:
    from tools.tool_registry import ToolDefinition

_LOG = logging.getLogger(__name__)
_ACTIVE: ContextVar[FileCapture | None] = ContextVar("reins_file_capture", default=None)
_DATA_TOOLS = frozenset({"memory_note", "memory_manage", "todo", "redact"})
_EXTERNAL_TOOLS = frozenset({"browser_navigate", "browser_type"})
_Result = TypeVar("_Result")


class FileCapture:
    """一次真实执行尝试的捕获上下文，超时返回后仍由实际执行线程持有。"""

    def __init__(
        self,
        store: SnapshotStore,
        lease: Lease,
        identity: Mapping[str, str],
        *,
        cancellation: CancellationToken | None = None,
        execution_scopes: tuple[tuple[Path, bool], ...] = (),
    ) -> None:
        """绑定原件、权限及操作归属；传参：服务、权限、身份；返回：无。"""
        self.store, self.lease, self.identity = store, lease, dict(identity)
        self.points: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.observing = False
        self.execution_started = False
        self.unchanged_after_publication = False
        self.cancellation = cancellation
        self.execution_scopes = execution_scopes

    def before_file(self, path: Path) -> dict[str, Any]:
        """在最终文件锁内发布旧版本；传参：目标；返回：点，捕获失败禁止开始文件动作。"""
        point = new_point(self.identity, scope="files", attribution="exact")
        self.ensure_active()
        state = self.store.capture_file(
            path,
            self.lease,
            self.identity["workspace_id"],
            cancellation=self.cancellation,
        )
        if not state["restorable"]:
            raise FileEditConflict(
                f"cannot protect file before edit: {state.get('error')}"
            )
        point["entries"] = [
            {"path": str(path), "before": state, "after": None, "attribution": "exact"}
        ]
        self.store.publish_point(point, cancellation=self.cancellation)
        self.points.append(point)
        self.ensure_active()
        return point

    def ensure_active(self) -> None:
        """实际写入前核对取消信号；参数：无；返回：无，扫描期间取消不能在之后启动副作用。"""
        if self.cancellation is not None and self.cancellation.cancelled:
            raise ExecutionCancelled(
                "cancelled before starting the protected file action"
            )

    def after_file(self, point: dict[str, Any]) -> None:
        """保存实际后态，登记失败不抹掉已发生的工具结果；传参：前态点；返回：无，错误进入独立证据。"""
        point = next(
            value for value in self.points if value["point_id"] == point["point_id"]
        )
        try:
            after = self.store.capture_file(
                Path(point["entries"][0]["path"]),
                self.lease,
                self.identity["workspace_id"],
            )
            self.unchanged_after_publication = self.store.same_state(
                point["entries"][0]["before"], after, identity=True
            )
            state_error = after.get("error") if not after["restorable"] else None
            errors = [
                *point["errors"],
                *([{"error": state_error}] if state_error else []),
            ]
            finished = {
                **point,
                "status": "incomplete" if errors else "complete",
                "completed_at": utc_now(),
                "errors": errors,
                "entries": [{**point["entries"][0], "after": after}],
            }
            self.store.publish_point(finished)
            self._remember(finished)
        except (OSError, ValueError) as exc:
            self.record_error(point, exc)

    def publish_file(
        self, point: dict[str, Any], original: bytes | None, updated: bytes
    ) -> None:
        """【文件恢复】【准确发布】保留最终被替换字节；参数：前态点、旧字节、新字节；返回：无，竞争明确报错。"""
        from tools.file_persistence import publish_prepared_file, replace_prepared_file
        from tools.restore_protection import (
            create_private_staging,
            protected_replace_prepared_file,
            write_private_file,
        )

        path = Path(point["entries"][0]["path"])
        self.ensure_active()
        staging = create_private_staging(path.parent)
        temporary, backup = staging / "replacement", staging / "displaced"
        # 1. 【文件恢复】【发布意图】临时正文创建前登记同卷私有路径，崩溃后仍可定位
        point.update(
            temporary_path=str(temporary),
            backup_path=str(backup),
            staging_path=str(staging),
        )
        try:
            self.store.publish_point(point, cancellation=self.cancellation)
            write_private_file(temporary, updated)
            self.ensure_active()
            if read_file_bytes(path) != original:
                raise FileEditConflict(
                    "file changed during edit; no replacement was published"
                )
            self.execution_started = True
            if original is None:
                publish_prepared_file(path, temporary)
            elif point["entries"][0]["before"]["sensitive"]:
                protected_replace_prepared_file(path, temporary, backup_path=backup)
            else:
                replace_prepared_file(path, temporary, backup_path=backup)
        finally:
            # 2. 【文件恢复】【真实前态】先归档最终被移走的原件，登记失败保留受保护备份
            try:
                self._archive_displaced(point, backup)
            finally:
                if not backup.exists():
                    try:
                        temporary.unlink(missing_ok=True)
                        staging.rmdir()
                    except OSError as exc:
                        self.record_error(point, exc)

    def _archive_displaced(self, point: dict[str, Any], backup: Path) -> None:
        """归档最终替换前态而不改写原记录；参数：点、备份路径；返回：无，归档失败独立记录。"""
        if not backup.exists():
            return
        try:
            entry = point["entries"][0]
            displaced = self.store.capture_displaced(
                backup,
                entry["before"],
                self.lease,
                workspace_id=self.identity["workspace_id"],
            )
            if not displaced["restorable"]:
                raise ValueError(
                    f"actual replaced bytes remain in staging: {displaced.get('error')}"
                )
            conflict = not self.store.same_state(entry["before"], displaced)
            point["entries"] = [
                {
                    **entry,
                    "displaced_before": displaced,
                    "attribution": "observed" if conflict else "exact",
                }
            ]
            if conflict:
                point["attribution"] = "observed"
                point["errors"] = [
                    *point["errors"],
                    {
                        "error": "external edit replaced at publication; actual bytes preserved"
                    },
                ]
            self.store.publish_point(point)
            self._remember(point)
            backup.unlink()
        except (OSError, ValueError) as exc:
            self.record_error(point, exc)
            return
        if conflict:
            raise FileEditConflict(
                "external edit replaced at publication; actual bytes preserved in restore point"
            )

    def _remember(self, point: dict[str, Any]) -> None:
        """替换同一捕获点的当前回执；参数：新状态；返回：无，不用可变字典相等性定位。"""
        self.points = [
            point if saved["point_id"] == point["point_id"] else saved
            for saved in self.points
        ]

    def record_error(self, point: dict[str, Any], error: Exception) -> None:
        """保留前态并独立报告记录失败；传参：点与真实错误；返回：无，不重放副作用。"""
        message = f"restore record incomplete ({point['point_id']}): {error}"
        self.errors.append(message)
        _LOG.error("【文件恢复】【捕获不完整】%s", message)
        incomplete = {
            **point,
            "status": "incomplete",
            "errors": [*point["errors"], {"error": str(error)}],
        }
        try:
            self.store.publish_point(incomplete)
            self._remember(incomplete)
        except (OSError, ValueError) as commit_error:
            self.errors.append(f"restore record commit failed: {commit_error}")
            _LOG.error("【文件恢复】【登记失败】%s", commit_error)

    def observe(self, execute: Callable[[], _Result]) -> _Result:
        """独占覆盖根直到实际后端返回；传参：真正执行器；返回：原始结果，停止未确认时调用持续等待。"""
        if self.observing:
            return execute()
        root = Path(self.identity["workspace_root"])
        with workspace_write_windows(
            ((root, True), *self.execution_scopes), owner=self.identity["operation_id"]
        ):
            self.ensure_active()
            if self.cancellation is not None:
                self.cancellation.begin_tool_phase("capture_before")
            point = self._before_workspace(root)
            self.ensure_active()
            self.observing = True
            self.execution_started = True
            if self.cancellation is not None:
                self.cancellation.begin_tool_phase("execution")
            started = time.monotonic()
            try:
                result = execute()
            except BaseException as exc:
                point["metrics"]["execution_seconds"] = time.monotonic() - started
                self._after_workspace(root, point)
                if isinstance(exc, Exception):
                    self.errors.append(
                        f"tool execution raised after capture: {type(exc).__name__}"
                    )
                raise
            finally:
                self.observing = False
            point["metrics"]["execution_seconds"] = time.monotonic() - started
            self._after_workspace(root, point)
            return result

    def _before_workspace(self, root: Path) -> dict[str, Any]:
        """建立未知写集执行前完整清单；传参：覆盖根；返回：已发布前态点，IO失败阻止启动。"""
        point = new_point(self.identity, scope="workspace", attribution="observed")
        capture = self.store.capture_workspace(
            root,
            self.lease,
            self.identity["workspace_id"],
            cancellation=self.cancellation,
        )
        point.update(
            exclusions=capture["exclusions"],
            errors=capture["errors"],
            metrics={"before": capture["metrics"]},
        )
        point["entries"] = [
            {"path": path, "before": state, "after": None, "attribution": "observed"}
            for path, state in capture["states"].items()
        ]
        point["scope_rules"] = {
            "root": str(root),
            "data_root": str(self.store.data_root),
            "git_ignore_used": False,
            "atomic_volume_snapshot": False,
            "coordinated_paths": [
                str(path) for path, _subtree in self.execution_scopes
            ],
        }
        point["status"] = "incomplete" if capture["errors"] else "before_saved"
        self.store.publish_point(point, cancellation=self.cancellation)
        self.points.append(point)
        if capture["errors"]:
            raise FileEditConflict(
                f"workspace capture failed before execution: {capture['errors']}"
            )
        return point

    def _after_workspace(self, root: Path, point: dict[str, Any]) -> None:
        """封存真实结束后观察到的差异；传参：根和前态点；返回：无，后态失败不影响前态可用性。"""
        if self.cancellation is not None:
            self.cancellation.begin_tool_phase("capture_after")
        try:
            capture = self.store.capture_workspace(
                root, self.lease, self.identity["workspace_id"]
            )
            before = {entry["path"]: entry for entry in point["entries"]}
            entries = [
                {**entry, "after": capture["states"].get(path, missing_state(path))}
                for path, entry in before.items()
            ]
            entries.extend(
                {
                    "path": path,
                    "before": missing_state(path),
                    "after": state,
                    "attribution": "observed",
                }
                for path, state in capture["states"].items()
                if path not in before
            )
            failed_paths = {row["path"] for row in capture["errors"] if "path" in row}
            entries = [
                {**entry, "after": None}
                if any(
                    Path(entry["path"]).is_relative_to(Path(path))
                    for path in failed_paths
                )
                else entry
                for entry in entries
            ]
            finished = {
                **point,
                "entries": entries,
                "completed_at": utc_now(),
                "errors": capture["errors"],
                "status": "incomplete" if capture["errors"] else "complete",
                "metrics": {**point["metrics"], "after": capture["metrics"]},
            }
            self.store.publish_point(finished)
            self._remember(finished)
        except (OSError, ValueError) as exc:
            self.record_error(point, exc)

    def attach_result(self, result: Any) -> Any:
        """把捕获身份附到真实工具回执；传参：原回执；返回：带恢复证据的新回执。"""
        if not self.points and not self.errors:
            return result
        evidence = {
            "restore_point_ids": [point["point_id"] for point in self.points],
            "restore_record_errors": list(
                dict.fromkeys(
                    [
                        *self.errors,
                        *(
                            str(error["error"])
                            for point in self.points
                            for error in point["errors"]
                        ),
                    ]
                )
            ),
        }
        if isinstance(result, ToolError):
            execution_state = (
                "completed"
                if self.unchanged_after_publication
                else "unknown"
                if self.execution_started
                else "not_started"
            )
            return replace(
                result,
                details={
                    "execution_state": execution_state,
                    **result.details,
                    **evidence,
                },
            )
        if isinstance(result, dict):
            return {**result, "meta": {**result.get("meta", {}), **evidence}}
        return {"content": str(result), "meta": evidence}


def run_captured_tool(definition: ToolDefinition, args: dict[str, object]) -> object:
    """在watchdog实际线程中注入捕获；传参：真实定义和宿主参数；返回：原工具结果及捕获证据。"""
    executor = definition.executor
    if executor is None:
        raise ValueError("tool executor is missing")
    effect = capture_effect(definition, args)
    if effect == "none":
        return executor(args)
    data_root, lease = args.get("__data_root__"), args.get("__lease__")
    if not isinstance(data_root, str | Path) or not isinstance(lease, Lease):
        return ToolError(
            ToolErrorCategory.PERMISSION,
            "file capture requires data root and lease",
            retryable=False,
            details={"execution_state": "not_started"},
        )
    cancellation = args.get("__cancellation__")
    capture = FileCapture(
        SnapshotStore(data_root),
        lease,
        _capture_identity(args, Path(data_root), lease),
        cancellation=cancellation
        if isinstance(cancellation, CancellationToken)
        else None,
        execution_scopes=_execution_scopes(definition, args, lease),
    )
    token = _ACTIVE.set(capture)
    try:
        with verified_content_scope(capture.store.data_root):
            prepared_args = {**args, "__file_capture__": capture}
            try:
                result = (
                    capture.observe(lambda: executor(prepared_args))
                    if effect == "workspace"
                    else executor(prepared_args)
                )
            except ExecutionCancelled as exc:
                result = ToolError(
                    ToolErrorCategory.CANCELLED,
                    str(exc),
                    retryable=False,
                    details={
                        "execution_state": "unknown"
                        if capture.execution_started
                        else "not_started"
                    },
                )
            except (OSError, ValueError) as exc:
                result = ToolError(
                    ToolErrorCategory.UNKNOWN,
                    str(exc),
                    retryable=False,
                    details={
                        "execution_state": "unknown"
                        if capture.execution_started
                        else "not_started"
                    },
                )
            return capture.attach_result(result)
    finally:
        _ACTIVE.reset(token)


def capture_effect(definition: ToolDefinition, args: Mapping[str, object]) -> str:
    """按实际能力分类，编排和纯运行数据不持写窗口；传参：工具定义、参数；返回：捕获类型。"""
    if definition.runtime_action or definition.source == "mcp_reserved":
        return "none"
    if definition.readonly and not definition.exec_boundary:
        return "none"
    if definition.source == "builtin" and definition.name in {
        "file_write",
        "file_patch",
    }:
        return "files"
    if definition.name == "skill_run":
        return "deferred"
    if definition.name == "browser_click":
        return (
            "workspace"
            if str(args.get("expect_download", "")).casefold()
            in {"true", "1", "yes", "on"}
            else "none"
        )
    if definition.name in _EXTERNAL_TOOLS:
        return "none"
    if definition.exec_boundary:
        return "workspace"
    if definition.readonly or definition.name in _DATA_TOOLS:
        return "none"
    return "workspace"


def observe_local_execution(execute: Callable[[], _Result]) -> _Result:
    """在本地技能真正启动进程处观察；传参：后端动作；返回：原结果，无宿主上下文时保持底层API。"""
    capture = _ACTIVE.get()
    return execute() if capture is None else capture.observe(execute)


@contextmanager
def exact_file_window(path: Path, capture: object) -> Iterator[None]:
    """明确文件写共享祖先并独占目标；传参：最终路径和宿主捕获；返回：写入窗口。"""
    if isinstance(capture, FileCapture) and capture.observing:
        yield
        return
    owner = (
        capture.identity["operation_id"]
        if isinstance(capture, FileCapture)
        else "file_tool"
    )
    with workspace_write_window(path, owner=owner):
        yield


def _capture_identity(
    args: Mapping[str, object], data_root: Path, lease: Lease
) -> dict[str, str]:
    """固定原会话工作区而非exec子目录；传参：宿主参数、数据根、租约；返回：完整捕获归属。"""
    supplied = args.get("__capture_identity__")
    identity = dict(supplied) if isinstance(supplied, dict) else {}
    session_id = str(identity.get("session_id") or args.get("__session_id__") or "")
    workspace_store = WorkspaceStore(data_root)
    workspace = workspace_store.find_for_session(session_id) if session_id else None
    if workspace is None:
        root = path_security._project_root(lease)
        if root is None:
            raise ValueError("file capture requires the original workspace root")
        workspace = workspace_store.register(root)
    return {
        "workspace_id": workspace.workspace_id,
        "workspace_root": str(workspace.project_root),
        "session_id": session_id,
        "run_id": str(identity.get("run_id", "")),
        "input_id": str(identity.get("input_id", "")),
        "execution_id": f"execution-{uuid4().hex}",
        "operation_id": str(args.get("__operation_id__") or f"op-{uuid4().hex}"),
    }


def _execution_scopes(
    definition: ToolDefinition, args: Mapping[str, object], lease: Lease
) -> tuple[tuple[Path, bool], ...]:
    """协调真实执行目录和显式目标，恢复仍属于原会话；参数：定义/已审批参数/租约；返回：物理写范围。"""
    if not definition.exec_boundary:
        return ()
    requested = args.get("cwd")
    cwd = (
        path_security.resolve_target(Path(requested), lease)
        if isinstance(requested, str) and requested.strip()
        else path_security.task_workspace(lease)
    )
    if cwd is None:
        return ()
    command = str(args.get("command") or args.get("code") or "")
    targets = path_security.execution_target_paths(command, cwd)
    return ((cwd, True), *((path, path.is_dir()) for path in targets))
