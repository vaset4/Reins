from __future__ import annotations

from typing import cast

from runtime.types import (
    ReadOnlyInspectionRequest,
    ReadOnlyInspectionResult,
    RunToolsRequest,
    RunToolsResult,
)
from runtime.lease import Lease
from runtime.cancellation import CancellationToken
from tools.inspection_parser import parse_inspection_payload
from tools.readonly_inspection import ReadOnlyInspectionExecutor
from tools.redacted_files import RedactedFiles


class ReadOnlyFileToolExecutor:
    def __init__(self, inspection: ReadOnlyInspectionExecutor) -> None:
        self._inspection = inspection

    def execute(self, request: RunToolsRequest) -> RunToolsResult:
        tool_name = request.tool_name or request.action
        lease = _lease_argument(request.arguments)
        if tool_name == "inspect":
            inspection_request = parse_inspection_payload(request.payload)
            if inspection_request is None:
                return RunToolsResult.error_result(
                    action=request.action,
                    tool_name=tool_name,
                    error="INVALID_REQUEST: unsupported inspect payload",
                    summary="invalid request",
                    target_scope=request.target_scope,
                )
            return self._to_run_tools_result(
                request=request,
                inspection_result=self._inspection.execute(
                    inspection_request,
                    lease=lease,
                    cancellation=_cancellation_argument(request.arguments),
                ),
            )

        try:
            inspection_request = self._build_inspection_request(request)
        except ValueError as exc:
            return RunToolsResult.error_result(
                action=tool_name,
                error=str(exc),
                meta={"error_category": "invalid_input"},
            )
        if inspection_request is None:
            return RunToolsResult.error_result(
                action=request.action,
                tool_name=tool_name,
                error=f"INVALID_REQUEST: unsupported readonly file tool: {tool_name}",
                summary="invalid request",
                target_scope=request.target_scope,
            )

        return self._to_run_tools_result(
            request=request,
            inspection_result=self._inspection.execute(
                inspection_request,
                lease=lease,
                cancellation=_cancellation_argument(request.arguments),
                redacted_files=_redacted_argument(request.arguments),
                session_id=str(request.arguments.get("__session_id__", "")),
            ),
        )

    def _build_inspection_request(
        self, request: RunToolsRequest
    ) -> ReadOnlyInspectionRequest | None:
        """将模型文件参数交给同一权限与读取执行器；参数：工具请求；返回：有明确分页单位的请求。"""
        tool = request.tool_name or request.action
        actions = {
            "file_read": "read_file",
            "list": "list_dir",
            "grep": "grep_text",
            "find_path": "find_path",
        }
        if tool not in actions:
            return None
        arguments = request.arguments
        path = arguments.get("path", "." if tool == "find_path" else "")
        query = arguments.get("query")
        if not isinstance(path, str) or not path.strip():
            return None
        if tool in {"grep", "find_path"} and (not isinstance(query, str) or not query):
            return None
        if "offset" in arguments or "start" in arguments or "start_index" in arguments:
            raise ValueError(
                "file_read now uses start_line/line_count or the returned cursor; old character offsets are not line numbers"
            )
        cursor = arguments.get("cursor")
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise ValueError("file cursor must be non-empty text")
        kind = arguments.get("kind", "any" if tool == "find_path" else "file")
        if not isinstance(kind, str) or kind not in {"file", "directory", "any"}:
            raise ValueError("path kind must be file, directory or any")
        for name in ("limit", "start_line", "line_count"):
            value = arguments.get(name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        first = arguments.get(
            "start_line", 1 if tool == "file_read" and cursor is None else None
        )
        return ReadOnlyInspectionRequest(
            action=actions[tool],
            target_path=path.strip(),
            query=cast(str | None, query),
            cursor=cursor,
            limit=cast(int | None, arguments.get("limit")),
            start_line=cast(int | None, first),
            line_count=cast(int | None, arguments.get("line_count")),
            path_kind=kind,
        )

    def _to_run_tools_result(
        self,
        *,
        request: RunToolsRequest,
        inspection_result: ReadOnlyInspectionResult,
    ) -> RunToolsResult:
        tool_name = request.tool_name or request.action
        if inspection_result.status == "ok":
            return RunToolsResult.ok(
                action=request.action,
                tool_name=tool_name,
                content=inspection_result.output,
                summary="tool executed",
                target_scope=request.target_scope,
                meta=dict(inspection_result.meta),
            )
        if inspection_result.status == "rejected":
            if _is_invalid_readonly_request(inspection_result.output):
                return RunToolsResult.error_result(
                    action=request.action,
                    tool_name=tool_name,
                    error=inspection_result.output,
                    summary="invalid request",
                    target_scope=request.target_scope,
                    meta=dict(inspection_result.meta),
                )
            return RunToolsResult.denied(
                action=request.action,
                tool_name=tool_name,
                error=inspection_result.output,
                summary="tool denied",
                target_scope=request.target_scope,
                meta=dict(inspection_result.meta),
            )
        return RunToolsResult.error_result(
            action=request.action,
            tool_name=tool_name,
            error=inspection_result.output,
            summary="tool execution failed",
            target_scope=request.target_scope,
            meta=dict(inspection_result.meta),
        )


def _is_invalid_readonly_request(output: str) -> bool:
    return output.startswith(("INVALID_REQUEST:", "REJECTED_ACTION:"))


def _lease_argument(arguments: dict[str, object]) -> Lease | None:
    value = arguments.get("__lease__")
    return value if isinstance(value, Lease) else None


def _redacted_argument(arguments: dict[str, object]) -> RedactedFiles | None:
    """读取宿主注入的脱敏服务；传参：内部执行参数；返回：服务或无。"""
    value = arguments.get("__redacted_files__")
    return value if isinstance(value, RedactedFiles) else None


def _cancellation_argument(arguments: dict[str, object]) -> CancellationToken | None:
    """取得执行边界注入的协作停止信号；参数：内部参数；返回：取消信号或无。"""
    value = arguments.get("__cancellation__")
    return value if isinstance(value, CancellationToken) else None
