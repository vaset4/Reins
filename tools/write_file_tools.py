from __future__ import annotations

from pathlib import Path

from runtime.types import RunToolsRequest, RunToolsResult
from tools.file_persistence import (
    FileEditConflict,
    content_sha256,
    file_edit_lock,
    publish_file,
    read_file_bytes,
)
from tools.config_syntax import is_private_key


class WriteFileToolExecutor:
    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root.resolve()

    def execute(self, request: RunToolsRequest) -> RunToolsResult:
        """在文件互斥窗口中核对版本、构造修改并完整发布。

        传参：request 为已解析工具请求；返回：真实修改数和版本，或明确失败
        """
        tool_name = request.tool_name or request.action
        target_scope = request.target_scope
        try:
            resolved = self._resolve_repo_path(target_scope or "")
            if resolved is None:
                return RunToolsResult.denied(
                    action=request.action,
                    tool_name=tool_name,
                    error=f"REJECTED_PATH: {target_scope}",
                    summary="tool denied",
                    target_scope=target_scope,
                )
            if tool_name not in {"file_write", "file_patch"}:
                raise ValueError(f"unsupported write file tool: {tool_name}")
            if request.arguments.get("__protected_edit__") is not None:
                return _publish_protected(request)
            if any(
                isinstance(request.arguments.get(key), str)
                and is_private_key(str(request.arguments[key]))
                for key in ("content", "new_text")
            ):
                raise ValueError("private key content cannot be written by file tools")
            from runtime.file_capture import FileCapture, exact_file_window

            capture = request.arguments.get("__file_capture__")
            with exact_file_window(resolved, capture), file_edit_lock(resolved):
                original = read_file_bytes(resolved)
                _check_version(request.arguments, original)
                updated, changes = _edit_contents(
                    tool_name, request.arguments, original
                )
                if is_private_key(original or b"") or is_private_key(updated):
                    raise ValueError(
                        "private key content cannot be written by file tools"
                    )
                validate = request.arguments.get("__validate_file__")
                if callable(validate):
                    validate()
                point = (
                    capture.before_file(resolved)
                    if isinstance(capture, FileCapture)
                    else None
                )
                if tool_name == "file_write":
                    resolved.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if isinstance(capture, FileCapture) and point is not None:
                        capture.publish_file(point, original, updated)
                    else:
                        publish_file(resolved, original, updated)
                finally:
                    if isinstance(capture, FileCapture) and point is not None:
                        capture.after_file(point)
            label = "WROTE_FILE" if tool_name == "file_write" else "PATCHED_FILE"
            return RunToolsResult.ok(
                action=request.action,
                tool_name=tool_name,
                content=f"{label}: {resolved.relative_to(self._repo_root)}",
                summary="tool executed",
                target_scope=target_scope,
                meta={
                    "resolved_path": str(resolved),
                    **changes,
                    "content_sha256": content_sha256(updated),
                    "byte_count": len(updated),
                    "previous_sha256": content_sha256(original)
                    if original is not None
                    else None,
                },
            )
        except (OSError, ValueError) as exc:
            prefix = (
                "INVALID_REQUEST" if isinstance(exc, ValueError) else "RUN_TOOLS_ERROR"
            )
            return RunToolsResult.error_result(
                action=request.action,
                tool_name=tool_name,
                error=f"{prefix}: {exc}",
                summary="tool execution failed",
                target_scope=target_scope,
            )

    def _resolve_repo_path(self, target_path: str) -> Path | None:
        """定位工作区内的实际目标；传参：请求路径；返回：路径，越界或空路径为None。"""
        normalized = target_path.strip()
        if not normalized:
            return None
        candidate = (self._repo_root / normalized).resolve()
        try:
            candidate.relative_to(self._repo_root)
        except ValueError:
            return None
        return candidate


def _publish_protected(request: RunToolsRequest) -> RunToolsResult:
    """由宿主候选执行同版发布，正文不进入普通工具参数；传参：已授权请求；返回：脱敏回执。"""
    from tools.redacted_files import ProtectedEdit, RedactedFiles

    files, edit = (
        request.arguments.get("__redacted_files__"),
        request.arguments.get("__protected_edit__"),
    )
    validate = request.arguments.get("__validate_file__")
    if (
        not isinstance(files, RedactedFiles)
        or not isinstance(edit, ProtectedEdit)
        or not callable(validate)
    ):
        raise ValueError(
            "protected edit requires its host and final authorization check"
        )
    result = files.publish(
        edit, validate=validate, capture=request.arguments.get("__file_capture__")
    )
    return RunToolsResult.ok(
        action=request.action,
        tool_name=request.tool_name,
        content=result["content"],
        summary="tool executed",
        target_scope=request.target_scope,
        meta=result["meta"],
    )


def _check_version(arguments: dict[str, object], original: bytes | None) -> None:
    """覆盖已有文件前检查调用者读取的版本；传参：参数和原字节；返回：无，不匹配时抛错。"""
    expected = arguments.get("expected_sha256")
    if original is None and expected is None:
        return
    if expected is None:
        raise FileEditConflict(
            "existing file requires expected_sha256 from file_read before editing"
        )
    actual = content_sha256(original) if original is not None else None
    if expected != actual:
        raise FileEditConflict(
            f"file version conflict: expected {expected}, current {actual}"
        )


def _edit_contents(
    tool_name: str,
    arguments: dict[str, object],
    original: bytes | None,
) -> tuple[bytes, dict[str, object]]:
    """构造精确文本修改，多处匹配只有显式批量操作才接受。

    传参：工具、参数与原内容；返回：新字节和实际修改计数
    """
    if tool_name == "file_write":
        updated = _text_argument(arguments, "content").encode("utf-8")
        return updated, {"changed": updated != original, "created": original is None}
    if original is None:
        raise ValueError("cannot patch a missing file")
    old_text, new_text = (
        _text_argument(arguments, "old_text"),
        _text_argument(arguments, "new_text"),
    )
    if not old_text:
        raise ValueError("old_text must not be empty")
    current = original.decode("utf-8")
    matched_count = current.count(old_text)
    if matched_count == 0:
        raise ValueError("old_text not found; matched_count=0")
    replace_all = arguments.get("replace_all", False)
    if type(replace_all) is not bool:
        raise ValueError("replace_all must be a boolean")
    if matched_count > 1 and not replace_all:
        raise ValueError(
            f"old_text matches {matched_count} locations; provide unique text or replace_all=true"
        )
    updated_text = current.replace(old_text, new_text)
    changed_count = matched_count if old_text != new_text else 0
    return updated_text.encode("utf-8"), {
        "matched_count": matched_count,
        "changed_count": changed_count,
        "changed": changed_count > 0,
    }


def _text_argument(arguments: dict[str, object], name: str) -> str:
    """读取原始文本，不把缺失值或其他类型伪装为空字符串；传参：参数与字段；返回：文本。"""
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value
