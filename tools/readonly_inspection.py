from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from hashlib import sha256
import json
import fnmatch
import os
from collections.abc import Iterator

import path_security
from runtime.lease import Lease
from runtime.cancellation import CancellationToken, ExecutionCancelled
from runtime.types import ReadOnlyInspectionRequest, ReadOnlyInspectionResult
from tools.file_persistence import content_sha256
from tools.redacted_files import RedactedFiles
from tools.config_syntax import is_private_key
from tools.file_paging import FilePageChanged, entry_page, text_page

MATCH_PREVIEW_CHARACTERS = 320
DECODE_EXAMPLE_COUNT = 5
PDF_SUFFIX = ".pdf"
PDF_EXTRACTION_UNAVAILABLE = "PDF_TEXT_EXTRACTION_UNAVAILABLE: pypdf is not installed"
PDF_EMPTY_TEXT = "PDF_TEXT_EMPTY: no extractable text found; OCR may be required"

# file_read 单次返回的字符上限。这个数字真正决定的不是"占多少上下文"，而是"读一个
# 文件要几次模型往返"：45185 字的论文按 4000 分页要 12 次 file_read，而 provider 普遍
# 按分钟限流（2026-09-05 实测 qq1244 是 10 次/分钟），任务在读完之前就被掐断；每次续读
# 还要把整段已积累的对话重新发一遍，实测 11 次调用累计烧掉 109267 prompt token 才读到
# 80%，比一次读完贵一倍多。4 万字约合 1-2 万 token，占 30 万窗口的 5% 上下，同一篇论文
# 降到 2 次读取。模型仍然自己决定要不要接着读，offset / next_offset 契约不变。
# ponytail: 固定值，不随模型窗口自适应；真要接窗口只有 8K 的模型时再改成按窗口推算
DEFAULT_READ_MAX_CHARS = 40000


class ReadOnlyInspectionExecutor:
    def __init__(
        self,
        repo_root: Path,
        max_entries: int,
        max_chars: int,
        max_matches: int,
    ) -> None:
        self._repo_root = repo_root.resolve()
        self._max_entries = max_entries
        self._max_chars = max_chars
        self._max_matches = max_matches

    def execute(
        self,
        request: ReadOnlyInspectionRequest,
        *,
        lease: Lease | None = None,
        redacted_files: RedactedFiles | None = None,
        session_id: str = "",
        cancellation: CancellationToken | None = None,
    ) -> ReadOnlyInspectionResult:
        resolved = self._resolve_repo_path(request.target_path)
        if resolved is None:
            return ReadOnlyInspectionResult(
                action=request.action,
                status="rejected",
                output=f"REJECTED_PATH: {request.target_path}",
            )

        try:
            if request.action == "list_dir":
                return self._list_dir(resolved, request)
            if request.action == "read_file":
                return self._read_file(
                    resolved,
                    request.offset,
                    lease,
                    redacted_files=redacted_files,
                    session_id=session_id,
                    selection=request
                    if request.start_line is not None
                    or request.line_count is not None
                    or request.cursor is not None
                    else None,
                )
            if request.action == "grep_text":
                if not request.query:
                    return ReadOnlyInspectionResult(
                        action=request.action,
                        status="rejected",
                        output="INVALID_REQUEST: missing query",
                    )
                return self._grep_text(
                    resolved,
                    request.query,
                    lease,
                    request=request,
                    cancellation=cancellation,
                )
            if request.action in {"find_file", "find_path"}:
                if not request.query:
                    return ReadOnlyInspectionResult(
                        action=request.action,
                        status="rejected",
                        output="INVALID_REQUEST: missing query",
                    )
                return self._find_file(
                    resolved,
                    request.query,
                    lease,
                    request=request,
                    cancellation=cancellation,
                )
            return ReadOnlyInspectionResult(
                action=request.action,
                status="rejected",
                output=f"REJECTED_ACTION: {request.action}",
            )
        except FilePageChanged as exc:
            return ReadOnlyInspectionResult(
                request.action,
                "error",
                str(exc),
                meta={
                    "error_category": "source_changed",
                    "resolved_path": str(resolved),
                },
            )
        except ValueError as exc:
            return ReadOnlyInspectionResult(
                request.action,
                "error",
                str(exc),
                meta={
                    "error_category": "invalid_input",
                    "resolved_path": str(resolved),
                },
            )
        except ExecutionCancelled:
            assert cancellation is not None
            cancellation.report_backend(execution_state="completed", stopped=True)
            category = "timeout" if cancellation.reason == "timeout" else "cancelled"
            return ReadOnlyInspectionResult(
                action=request.action,
                status="error",
                output=f"{category}: file search stopped",
                meta={"error_category": category, **cancellation.backend_evidence()},
            )
        except Exception as exc:
            return ReadOnlyInspectionResult(
                action=request.action,
                status="error",
                output=f"{request.action} failed: {type(exc).__name__}: {exc}",
                error="readonly_inspection_execution_failed",
                meta={
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )

    def _resolve_repo_path(self, target_path: str) -> Path | None:
        candidate = self._repo_root / target_path
        resolved = candidate.resolve()
        if resolved != self._repo_root and self._repo_root not in resolved.parents:
            return None
        return candidate

    def _list_dir(
        self, path: Path, request: ReadOnlyInspectionRequest
    ) -> ReadOnlyInspectionResult:
        """返回带类型与版本分页的单层目录；参数：实际路径、查询；返回：可直接用于下一操作的条目。"""
        invalid = _path_type_error(path, request, expected="directory")
        if invalid is not None:
            return invalid
        entries = []
        for item in sorted(path.iterdir(), key=lambda value: value.name):
            kind = "directory" if item.is_dir() else "file"
            relative = item.relative_to(self._repo_root).as_posix()
            entries.append(
                {
                    "path": relative,
                    "name": item.name,
                    "kind": kind,
                    "display": f"{'dir' if kind == 'directory' else 'file'} {item.name}",
                    "read_action": {
                        "tool": "list" if kind == "directory" else "file_read",
                        "path": relative,
                    },
                }
            )
        return entry_page(
            entries, request, {"resolved_path": str(path)}, self._max_entries
        )

    def _read_file(
        self,
        path: Path,
        offset: int = 0,
        lease: Lease | None = None,
        *,
        redacted_files: RedactedFiles | None = None,
        session_id: str = "",
        selection: ReadOnlyInspectionRequest | None = None,
    ) -> ReadOnlyInspectionResult:
        """保留真实文件保护与脱敏，再按已声明单位选择正文；参数：路径、旧偏移、权限及新选择；返回：真实文件页。"""
        invalid = _path_type_error(
            path,
            selection or ReadOnlyInspectionRequest("read_file", str(path)),
            expected="file",
        )
        if invalid is not None:
            return invalid
        denied = _check_read_candidate("read_file", path, lease)
        if denied is not None:
            return denied

        if lease is not None and path_security.uses_redacted_files(path, lease):
            if redacted_files is None or not session_id:
                raise ValueError("redacted file access requires an active host session")
            view = redacted_files.read(
                path,
                session_id=session_id,
                offset=offset,
                max_chars=self._max_chars,
                selection=selection,
            )
            return ReadOnlyInspectionResult(
                action="read_file",
                status="ok",
                output=view["content"],
                meta=view["meta"],
            )

        if path.suffix.lower() == PDF_SUFFIX:
            return self._read_pdf(path, offset, selection=selection)

        raw = path.read_bytes()
        if is_private_key(raw):
            return ReadOnlyInspectionResult(
                action="read_file",
                status="ok",
                output="仅提供私钥文件元信息，内容不可编辑",
                meta={
                    "resolved_path": str(path),
                    "metadata_only": True,
                    "reason": "private_key",
                    "redacted": True,
                    "byte_count": len(raw),
                    "content_sha256": content_sha256(raw),
                },
            )
        content = raw.decode("utf-8")
        return self._bounded_read_result(
            content,
            offset,
            {
                "resolved_path": str(path),
                "file_type": "text",
                "content_sha256": content_sha256(raw),
            },
            selection=selection,
        )

    def _read_pdf(
        self,
        path: Path,
        offset: int,
        *,
        selection: ReadOnlyInspectionRequest | None = None,
    ) -> ReadOnlyInspectionResult:
        """以提取文本而非PDF字节分页，并绑定原PDF版本；参数：文件、旧位置及行选择；返回：正文页。"""
        original_hash = content_sha256(path.read_bytes())
        try:
            content, page_count = _extract_pdf_text(path)
        except ImportError:
            return ReadOnlyInspectionResult(
                action="read_file",
                status="error",
                output=PDF_EXTRACTION_UNAVAILABLE,
                meta={"resolved_path": str(path), "file_type": "pdf"},
            )
        except Exception as exc:
            return ReadOnlyInspectionResult(
                action="read_file",
                status="error",
                output=f"PDF_TEXT_EXTRACTION_ERROR: {exc}",
                meta={"resolved_path": str(path), "file_type": "pdf"},
            )
        if not content.strip():
            return ReadOnlyInspectionResult(
                action="read_file",
                status="error",
                output=PDF_EMPTY_TEXT,
                meta={
                    "resolved_path": str(path),
                    "file_type": "pdf",
                    "page_count": page_count,
                },
            )
        from importlib.metadata import version

        if original_hash != content_sha256(path.read_bytes()):
            raise FilePageChanged("PDF changed while text was being extracted")
        return self._bounded_read_result(
            content,
            offset,
            {
                "resolved_path": str(path),
                "file_type": "pdf",
                "page_count": page_count,
                "content_sha256": original_hash,
                "representation": "pdf_extracted_text:pypdf:" + version("pypdf"),
            },
            selection=selection,
        )

    def _bounded_read_result(
        self,
        content: str,
        offset: int,
        meta: dict[str, object],
        *,
        selection: ReadOnlyInspectionRequest | None = None,
    ) -> ReadOnlyInspectionResult:
        """新请求按行选择，保留旧inspect字符读取语义；参数：正文、偏移、来源、选择；返回：有原文出口的页。"""
        if selection is not None:
            return text_page(content, selection, meta, self._max_chars)
        start = max(0, offset)
        output = content[start : start + self._max_chars]
        next_offset = start + len(output)
        truncated = next_offset < len(content)
        result_meta = dict(meta)
        result_meta.update(
            {
                "truncated": truncated,
                "returned_count": len(output),
                "total_count": len(content),
                "offset": start,
                "next_offset": next_offset if truncated else None,
            }
        )
        return ReadOnlyInspectionResult(
            action="read_file",
            status="ok",
            output=output,
            meta=result_meta,
        )

    def _grep_text(
        self,
        path: Path,
        query: str,
        lease: Lease | None = None,
        *,
        request: ReadOnlyInspectionRequest,
        cancellation: CancellationToken | None = None,
    ) -> ReadOnlyInspectionResult:
        """逐文件做字面检索并记录完整查询版本；参数：路径、查询、权限与分页/取消；返回：匹配行目录。"""
        if not path.exists():
            invalid = _path_type_error(path, request, expected="file")
            assert invalid is not None
            return invalid
        paths = (
            [path]
            if path.is_file()
            else sorted(set(_find_candidates(path, "*", cancellation)))
        )
        matches, versions, skipped = [], [], []
        for file_path in paths:
            _check_find_cancellation(cancellation)
            denied = _check_read_candidate("grep_text", file_path, lease)
            if denied is not None:
                return denied
            relative = file_path.relative_to(self._repo_root).as_posix()
            if lease is not None and path_security.uses_redacted_files(
                file_path, lease
            ):
                matches.append(
                    {
                        "path": relative,
                        "kind": "file",
                        "redacted": True,
                        "display": f"{relative}: 内容已脱敏",
                        "read_action": {"tool": "file_read", "path": relative},
                    }
                )
                versions.append([relative, "redacted"])
                continue
            raw = file_path.read_bytes()
            versions.append([relative, content_sha256(raw)])
            if is_private_key(raw):
                matches.append(
                    {
                        "path": relative,
                        "kind": "file",
                        "redacted": True,
                        "display": f"{relative}: 内容已脱敏",
                    }
                )
                continue
            try:
                lines = raw.decode("utf-8").splitlines()
            except UnicodeDecodeError:
                skipped.append(relative)
                continue
            for line_no, line in enumerate(lines, start=1):
                if query in line:
                    preview = line[:MATCH_PREVIEW_CHARACTERS]
                    matches.append(
                        {
                            "path": relative,
                            "kind": "file",
                            "line": line_no,
                            "preview": preview,
                            "preview_only": len(preview) < len(line),
                            "display": f"{relative}:{line_no}: {preview}",
                            "read_action": {
                                "tool": "file_read",
                                "path": relative,
                                "start_line": line_no,
                                "line_count": 1,
                            },
                        }
                    )
        return entry_page(
            matches,
            request,
            {
                "resolved_path": str(path),
                "query": query,
                "source_version": sha256(
                    json.dumps(versions, ensure_ascii=False).encode()
                ).hexdigest(),
                "skipped_decode_count": len(skipped),
                "skipped_decode_examples": skipped[:DECODE_EXAMPLE_COUNT],
            },
            self._max_matches,
        )

    def _find_file(
        self,
        path: Path,
        query: str,
        lease: Lease | None = None,
        *,
        request: ReadOnlyInspectionRequest,
        cancellation: CancellationToken | None = None,
    ) -> ReadOnlyInspectionResult:
        """按名称或glob发现真实文件/目录，保持原范围与权限；参数：路径、查询、权限、分页和取消；返回：路径页。"""
        invalid = _path_type_error(path, request, expected="directory")
        if invalid is not None:
            return invalid
        pattern = query if _looks_like_glob(query) else f"*{query}*"
        if cancellation is not None:
            cancellation.report_backend(supports_stop=True)
        entries = []
        for item in sorted(
            set(
                _find_candidates(
                    path, pattern, cancellation, path_kind=request.path_kind
                )
            )
        ):
            _check_find_cancellation(cancellation)
            denied = _check_read_candidate(request.action, item, lease)
            if denied is not None:
                return denied
            relative = item.relative_to(self._repo_root).as_posix()
            kind = "directory" if item.is_dir() else "file"
            entries.append(
                {
                    "path": relative,
                    "kind": kind,
                    "display": relative,
                    "read_action": {
                        "tool": "list" if kind == "directory" else "file_read",
                        "path": relative,
                    },
                }
            )
        result = entry_page(
            entries,
            request,
            {"resolved_path": str(path), "query": query},
            self._max_matches,
        )
        return (
            result
            if entries
            else replace(
                result, output=f"NO_MATCH: no {request.path_kind} matching {query!r}"
            )
        )


def _check_find_cancellation(cancellation: CancellationToken | None) -> None:
    """在目录和条目边界停止搜索；参数：运行信号；返回：无，已取消时抛停止异常。"""
    if cancellation is not None and cancellation.cancelled:
        raise ExecutionCancelled("file search stopped")


def _find_candidates(
    path: Path,
    pattern: str,
    cancellation: CancellationToken | None,
    *,
    path_kind: str = "file",
) -> Iterator[Path]:
    """单次扫描每个目录，保留递归 glob 的隐藏文件和链接语义；参数：根、模式、信号；返回：命中文件。"""
    query_path = Path(pattern)
    if query_path.is_absolute() or query_path.drive:
        raise ValueError("find_file query must be relative")
    parts = ("**", *query_path.parts)
    if any("**" in part and part != "**" for part in parts):
        raise ValueError("'**' can only be an entire path component")
    if pattern.endswith(("/", os.sep)):
        return
    pending = [(str(path), frozenset({0}))]
    while pending:
        _check_find_cancellation(cancellation)
        directory, positions = pending.pop()
        # 1. 双星号可以匹配零层目录；普通片段只消费一个目录或文件名
        active = set(positions)
        for index in range(len(parts)):
            if index in active and parts[index] == "**":
                active.add(index + 1)
        for index in active:
            if index < len(parts) and parts[index] == "..":
                pending.append((os.path.join(directory, ".."), frozenset({index + 1})))
        with os.scandir(directory) as entries:
            for entry in entries:
                _check_find_cancellation(cancellation)
                states = _find_entry_states(entry, parts, active)
                if len(parts) in states and (
                    (path_kind != "directory" and entry.is_file())
                    or (path_kind != "file" and entry.is_dir())
                ):
                    yield Path(entry.path)
                if entry.is_dir() and any(index < len(parts) for index in states):
                    pending.append((entry.path, frozenset(states - {len(parts)})))


def _find_entry_states(
    entry: os.DirEntry[str], parts: tuple[str, ...], active: set[int]
) -> set[int]:
    """计算路径下一段，递归双星号不穿透目录链接；参数：条目、模式片段、当前位置；返回：下一位置集合。"""
    states: set[int] = set()
    for index in active:
        if index == len(parts):
            continue
        if parts[index] == "**":
            if entry.is_dir(follow_symlinks=False):
                states.add(index)
        elif fnmatch.fnmatch(entry.name, parts[index]):
            states.add(index + 1)
    return states


def _looks_like_glob(query: str) -> bool:
    return any(char in query for char in ("*", "?", "["))


def _check_read_candidate(
    action: str, path: Path, lease: Lease | None
) -> ReadOnlyInspectionResult | None:
    if lease is None:
        return None
    decision = path_security.check_read(path, lease, filtered=True)
    if decision is path_security.Decision.ALLOWED:
        return None
    reason = (
        "path_security_confirm_required"
        if decision is path_security.Decision.CONFIRM
        else "path_security_deny"
    )
    return ReadOnlyInspectionResult(
        action=action,
        status="rejected",
        output=f"PERMISSION_DENIED: {reason}",
        meta={
            "decision": decision.value,
            "denied_path_count": 1,
        },
    )


def _extract_pdf_text(path: Path) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = list(reader.pages)
    texts = [page.extract_text() or "" for page in pages]
    return "\n\n".join(texts), len(pages)


def _path_type_error(
    path: Path, request: ReadOnlyInspectionRequest, *, expected: str
) -> ReadOnlyInspectionResult | None:
    """区分不存在与真实类型不符，不执行替代动作；参数：目标、请求、预期类型；返回：明确错误或类型正确。"""
    actual = "directory" if path.is_dir() else "file" if path.is_file() else "missing"
    if actual == expected:
        return None
    message = (
        f"path not found: {path.name}"
        if actual == "missing"
        else f"not a {expected}: {path.name}"
    )
    meta: dict[str, object] = {
        "actual_kind": actual,
        "expected_kind": expected,
        "resolved_path": str(path),
        "error_category": "not_found" if actual == "missing" else "wrong_resource_kind",
    }
    if actual == "directory":
        meta["read_action"] = {"tool": "list", "path": request.target_path}
    return ReadOnlyInspectionResult(
        request.action, "rejected", "INVALID_REQUEST: " + message, meta=meta
    )
