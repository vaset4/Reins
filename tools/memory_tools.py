from __future__ import annotations

from pathlib import Path

from runtime.types import RunToolsRequest, RunToolsResult


class MemoryToolExecutor:
    def __init__(self, data_root: Path | str) -> None:
        self._data_root = Path(data_root)

    def set_session_key(self, session_key: str) -> None:
        del session_key

    def execute(self, request: RunToolsRequest) -> RunToolsResult:
        tool_name = request.tool_name or request.action
        if tool_name == "memory_note":
            return self._memory_note(request)
        return RunToolsResult.error_result(
            action=request.action,
            tool_name=tool_name,
            error=f"INVALID_REQUEST: unsupported memory tool: {tool_name}",
            summary="invalid request",
            target_scope=request.target_scope,
        )

    def _memory_note(self, request: RunToolsRequest) -> RunToolsResult:
        """旧无运行入口不能把便签写成全局事实；传参：旧请求；返回：明确的运行依赖错误。"""
        return RunToolsResult.error_result(
            action=request.action,
            tool_name=request.tool_name,
            error="memory_note requires an active runtime to bind the session and source; use the native tool entry",
            target_scope=request.target_scope,
        )
