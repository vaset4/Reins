"""宿主会话内的配置视图、受保护输入和同版本写回。

作者：xxx
时间：2026-09-24 20:30:00
"""

from __future__ import annotations

from runtime.types import ReadOnlyInspectionRequest

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, cast
from uuid import uuid4

from tools.config_syntax import (
    PARSER_VERSION,
    ConfigDocument,
    ConfigSpan,
    UnsupportedConfig,
    is_private_key,
    parse_config,
)
from tools.file_persistence import (
    FileEditConflict,
    content_sha256,
    file_edit_lock,
    publish_file,
    read_file_bytes,
)
from tools.write_file_tools import _edit_contents

_TOKEN = re.compile(r"<redacted:[0-9a-f]{32}>")
_INPUT = re.compile(r"<protected-input:[0-9a-f]{32}>")
_BOM = b"\xef\xbb\xbf"


@dataclass(frozen=True, slots=True)
class ProtectedValue:
    token: str
    span: ConfigSpan
    literal: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class FileView:
    view_id: str
    session_id: str
    path: Path
    sha256: str
    document: ConfigDocument = field(repr=False)
    original: bytes = field(repr=False)
    content: str
    values: tuple[ProtectedValue, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ProtectedEdit:
    edit_id: str
    session_id: str
    path: Path
    view_id: str
    previous_sha256: str | None
    candidate_sha256: str
    replacement_count: int
    candidate: bytes = field(repr=False)


class RedactedFiles:
    """一个实际宿主持有的临时材料，不写入磁盘或审计正文。"""

    def __init__(self) -> None:
        """初始化独立视图和受保护输入；传参：无；返回：无。"""
        self._views: dict[str, FileView] = {}
        self._latest: dict[tuple[str, Path], str] = {}
        self._delivered: dict[str, list[tuple[int, int]]] = {}
        self._inputs: dict[str, tuple[str, str]] = {}
        self._input_ids: dict[tuple[str, str, str, str], str] = {}
        self._lock = RLock()

    def read(
        self,
        path: Path,
        *,
        session_id: str,
        offset: int = 0,
        max_chars: int | None = None,
        selection: ReadOnlyInspectionRequest | None = None,
    ) -> dict[str, Any]:
        """从同一完整字节快照生成可编辑视图；传参：已核对路径及会话；返回：无原秘密的正文和元信息。"""
        path = path.resolve()
        with file_edit_lock(path), self._lock:
            raw = path.read_bytes()
            meta = {
                "resolved_path": str(path),
                "byte_count": len(raw),
                "content_sha256": content_sha256(raw),
                "redacted": True,
            }
            try:
                document = parse_config(path, raw)
            except UnsupportedConfig as exc:
                return {
                    "content": "仅提供文件元信息，内容不可编辑",
                    "meta": {**meta, "metadata_only": True, "reason": str(exc)},
                }
            key = (session_id, path)
            previous_id = self._latest.get(key)
            previous = self._views.get(previous_id or "")
            if previous is not None and previous.original == raw:
                return self._public(
                    previous, offset=offset, max_chars=max_chars, selection=selection
                )
            reusable = (
                {item.span.identity: item for item in previous.values}
                if previous
                else {}
            )
            values = tuple(
                _protected_value(document, span, reusable) for span in document.spans
            )
            content = _replace_spans(document.text, values)
            view = FileView(
                f"view-{uuid4().hex}",
                session_id,
                path,
                content_sha256(raw),
                document,
                raw,
                content,
                values,
            )
            if previous_id:
                self._views.pop(previous_id, None)
                self._delivered.pop(previous_id, None)
            self._views[view.view_id], self._latest[key] = view, view.view_id
            return self._public(
                view, offset=offset, max_chars=max_chars, selection=selection
            )

    def protect_arguments(
        self,
        arguments: dict[str, object],
        *,
        session_id: str,
        request_id: str,
        call_id: str,
    ) -> dict[str, object]:
        """在通用证据保存前暂存候选新值；传参：工具参数及请求身份；返回：只携带受保护引用的参数。"""
        replacements = arguments.get("secret_replacements")
        result = dict(arguments)
        protected: dict[str, object] = {}
        with self._lock:
            if isinstance(replacements, dict):
                for index, (token, raw) in enumerate(replacements.items()):
                    safe_token = (
                        token
                        if isinstance(token, str) and _TOKEN.fullmatch(token)
                        else self.stage_text(
                            str(token),
                            session_id=session_id,
                            identity=(request_id, call_id, f"key.{index}"),
                        )
                    )
                    if not isinstance(raw, str):
                        # 【敏感文件】【非法输入】保留非法形状以暴露校验错误，错误正文只携带暂存引用
                        protected[safe_token] = self._protect_nested_value(
                            raw,
                            session_id=session_id,
                            identity=(request_id, call_id, f"value.{index}"),
                        )
                        continue
                    reference = self.stage_text(
                        raw,
                        session_id=session_id,
                        identity=(request_id, call_id, f"value.{index}"),
                    )
                    protected[safe_token] = reference
                    for name in ("content", "new_text", "old_text"):
                        if isinstance(result.get(name), str):
                            result[name] = _mask_literal(
                                str(result[name]), raw, reference
                            )
                result["secret_replacements"] = protected
            elif replacements is not None:
                result["secret_replacements"] = self._protect_nested_value(
                    replacements,
                    session_id=session_id,
                    identity=(request_id, call_id, "invalid_replacements"),
                )
        return result

    def _protect_nested_value(
        self, value: object, *, session_id: str, identity: tuple[str, str, str]
    ) -> object:
        """保护非法输入的字符串并保持原类型，重复接纳不改变身份；传参：输入树与身份；返回：保护后的树。"""
        if isinstance(value, str):
            return self.stage_text(value, session_id=session_id, identity=identity)
        if isinstance(value, dict):
            return {
                self.stage_text(
                    str(key),
                    session_id=session_id,
                    identity=(identity[0], identity[1], f"{identity[2]}.key.{index}"),
                ): self._protect_nested_value(
                    item,
                    session_id=session_id,
                    identity=(identity[0], identity[1], f"{identity[2]}.{index}"),
                )
                for index, (key, item) in enumerate(value.items())
            }
        if isinstance(value, list):
            return [
                self._protect_nested_value(
                    item,
                    session_id=session_id,
                    identity=(identity[0], identity[1], f"{identity[2]}.{index}"),
                )
                for index, item in enumerate(value)
            ]
        return value

    def stage_text(
        self, text: str, *, session_id: str, identity: tuple[str, str, str]
    ) -> str:
        """在宿主内暂存新秘密或敏感正文；传参：正文、会话及请求/调用/字段身份；返回：不可猜测的引用。"""
        with self._lock:
            if _INPUT.fullmatch(text):
                # 引用丢失仍保持为引用，执行时明确报失效，不能把编号当新密钥写入
                return text
            key = (session_id, *identity)
            reference = self._input_ids.get(key) or f"<protected-input:{uuid4().hex}>"
            if reference in self._inputs and self._inputs[reference] != (
                session_id,
                text,
            ):
                raise ValueError("protected replacement identity changed")
            self._inputs[reference], self._input_ids[key] = (
                (session_id, text),
                reference,
            )
            return reference

    def resolve_text(self, text: str, *, session_id: str) -> str:
        """仅在执行边界读取本会话暂存正文；传参：文本或引用、会话；返回：原文本，引用失效明确失败。"""
        if not _INPUT.fullmatch(text):
            return text
        with self._lock:
            stored = self._inputs.get(text)
            if stored is None or stored[0] != session_id:
                raise ValueError(
                    "protected input is unavailable or belongs to another session; provide it again"
                )
            return stored[1]

    def sanitize_text(self, text: str, *, session_id: str) -> str:
        """在普通证据中遮蔽本轮已接纳的秘密副本；传参：证据文本及会话；返回：只含受保护引用的文本。"""
        with self._lock:
            values = [
                (value, reference)
                for reference, (owner, value) in self._inputs.items()
                if owner == session_id and value
            ]
        for value, reference in sorted(
            values, key=lambda item: len(item[0]), reverse=True
        ):
            variants = {
                value,
                json.dumps(value, ensure_ascii=False)[1:-1],
                json.dumps(value)[1:-1],
            }
            for variant in sorted(variants, key=len, reverse=True):
                if variant:
                    text = text.replace(variant, reference)
        return text

    def protect_file_text(
        self,
        arguments: dict[str, object],
        *,
        session_id: str,
        request_id: str,
        call_id: str,
    ) -> dict[str, object]:
        """将敏感文件的候选正文放进内存，审批仅见引用；传参：参数及调用身份；返回：可落盘的副本。"""
        result = dict(arguments)
        for name in ("content", "new_text", "old_text"):
            value = result.get(name)
            if isinstance(value, str):
                if is_private_key(value):
                    for index, line in enumerate(value.splitlines()):
                        if line.strip() and not line.startswith("-----"):
                            self.stage_text(
                                line.strip(),
                                session_id=session_id,
                                identity=(request_id, call_id, f"{name}.key.{index}"),
                            )
                self._stage_config_literals(
                    value,
                    path=Path(str(arguments.get("path", ""))),
                    session_id=session_id,
                    identity=(request_id, call_id, name),
                )
                result[name] = self.stage_text(
                    value, session_id=session_id, identity=(request_id, call_id, name)
                )
        return result

    def has_private_key_input(
        self, arguments: dict[str, object], *, session_id: str
    ) -> bool:
        """识别候选原文或已暂存的私钥，普通文件同样禁止；传参：参数与会话；返回：是否包含私钥。"""
        for name in ("content", "old_text", "new_text"):
            value = arguments.get(name)
            if not isinstance(value, str):
                continue
            with self._lock:
                stored = self._inputs.get(value)
            if stored is not None and stored[0] == session_id:
                value = stored[1]
            if is_private_key(value):
                return True
        return False

    def _stage_config_literals(
        self, text: str, *, path: Path, session_id: str, identity: tuple[str, str, str]
    ) -> None:
        """过滤完整可解析候选中保护字段的单独副本；传参：正文、格式路径与身份；返回：无。"""
        try:
            document = parse_config(path, text.encode("utf-8"))
        except UnsupportedConfig:
            # 【敏感文件】【候选解析】不能解析的正文仅整段暂存，执行边界仍会明确拒绝该候选
            return
        for index, span in enumerate(document.spans):
            literal = document.text[span.start : span.end].strip()
            values = {literal, literal.strip("\"'")}
            if literal.startswith('"'):
                try:
                    decoded = json.loads(literal)
                except json.JSONDecodeError:
                    decoded = None
                if isinstance(decoded, str):
                    values.add(decoded)
            for variant, value in enumerate(sorted(values)):
                if value and not _TOKEN.fullmatch(value):
                    self.stage_text(
                        value,
                        session_id=session_id,
                        identity=(
                            identity[0],
                            identity[1],
                            f"{identity[2]}.value.{index}.{variant}",
                        ),
                    )

    def prepare(
        self, tool: str, arguments: dict[str, object], *, session_id: str, path: Path
    ) -> ProtectedEdit:
        """在脱敏视图内构造完整候选，审批期间不写文件；传参：工具、参数、会话和路径；返回：不公开正文的固定候选。"""
        with self._lock:
            arguments = {
                key: self.resolve_text(value, session_id=session_id)
                if key in {"content", "new_text", "old_text"} and isinstance(value, str)
                else value
                for key, value in arguments.items()
            }
            if (
                tool == "file_write"
                and not arguments.get("view_id")
                and not path.exists()
            ):
                return self._prepare_new(arguments, session_id=session_id, path=path)
            view = self._require_view(
                str(arguments.get("view_id", "")), session_id, path
            )
            expected = arguments.get("expected_sha256", view.sha256)
            if expected != view.sha256:
                raise FileEditConflict(
                    "redacted view version does not match expected_sha256"
                )
            if tool == "file_write":
                if self._delivered.get(view.view_id) != [(0, len(view.content))]:
                    raise ValueError(
                        "redacted view is incomplete; read all pages or use file_patch"
                    )
                text = arguments.get("content")
                if not isinstance(text, str):
                    raise ValueError("content must be text")
            else:
                edited, _ = _edit_contents(
                    tool, arguments, view.content.encode("utf-8")
                )
                text = edited.decode("utf-8")
            replacements = self._replacement_literals(view, arguments)
            references = arguments.get("secret_replacements", {})
            if isinstance(references, dict):
                for token, reference in references.items():
                    if (
                        token not in text
                        and isinstance(reference, str)
                        and text.count(reference) == 1
                    ):
                        text = text.replace(reference, token)
            redacted_document = parse_config(path, text.encode("utf-8"))
            known = {item.token for item in view.values}
            additions = sum(
                not span.comment and text[span.start : span.end] not in known
                for span in redacted_document.spans
            )
            candidate = _restore_candidate(view, text, replacements)
            if is_private_key(candidate):
                raise ValueError("private key content cannot be written by file tools")
            parse_config(path, candidate)
            return ProtectedEdit(
                f"edit-{uuid4().hex}",
                session_id,
                path,
                view.view_id,
                view.sha256,
                content_sha256(candidate),
                len(replacements) + additions,
                candidate,
            )

    def _prepare_new(
        self, arguments: dict[str, object], *, session_id: str, path: Path
    ) -> ProtectedEdit:
        """固定新建敏感配置并要求对新增保护值确认；传参：正文参数、会话、目标；返回：尚未发布的候选。"""
        content = arguments.get("content")
        if not isinstance(content, str):
            raise ValueError("content must be text")
        if (
            arguments.get("expected_sha256")
            or arguments.get("secret_replacements")
            or _TOKEN.search(content)
            or _INPUT.search(content)
        ):
            raise ValueError(
                "new config cannot borrow a previous version or protected view tokens"
            )
        candidate = content.encode("utf-8")
        document = parse_config(path, candidate)
        return ProtectedEdit(
            f"edit-{uuid4().hex}",
            session_id,
            path,
            "",
            None,
            content_sha256(candidate),
            len(document.spans),
            candidate,
        )

    def publish(
        self,
        edit: ProtectedEdit,
        *,
        validate: Callable[[], None] | None = None,
        capture: object = None,
    ) -> dict[str, Any]:
        """实际执行线程核对原文件和会话视图后完整发布；传参：已获授权的候选；返回：真实前后版本。"""
        from runtime.file_capture import FileCapture, exact_file_window

        with exact_file_window(edit.path, capture), file_edit_lock(edit.path):
            with self._lock:
                view = (
                    self._require_view(edit.view_id, edit.session_id, edit.path)
                    if edit.view_id
                    else None
                )
            original = read_file_bytes(edit.path)
            if (
                original != (view.original if view else None)
                or (content_sha256(original) if original is not None else None)
                != edit.previous_sha256
            ):
                raise FileEditConflict(
                    "redacted file changed after the view was read; read it again"
                )
            if content_sha256(edit.candidate) != edit.candidate_sha256:
                raise ValueError("protected candidate identity changed")
            if validate is not None:
                validate()
            point = (
                capture.before_file(edit.path)
                if isinstance(capture, FileCapture)
                else None
            )
            if original is None:
                edit.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if isinstance(capture, FileCapture) and point is not None:
                    capture.publish_file(point, original, edit.candidate)
                else:
                    publish_file(edit.path, original, edit.candidate)
            finally:
                if isinstance(capture, FileCapture) and point is not None:
                    capture.after_file(point)
            with self._lock:
                self._views.pop(edit.view_id, None)
                self._delivered.pop(edit.view_id, None)
                self._latest.pop((edit.session_id, edit.path), None)
            return {
                "content": f"已更新脱敏配置：{edit.path.name}",
                "meta": {
                    "resolved_path": str(edit.path),
                    "previous_sha256": edit.previous_sha256,
                    "content_sha256": edit.candidate_sha256,
                    "byte_count": len(edit.candidate),
                    "changed": original != edit.candidate,
                    "created": original is None,
                    "redacted": True,
                    "secret_replacements": edit.replacement_count,
                },
            }

    def invalidate_paths(self, paths: tuple[str, ...]) -> None:
        """恢复文件后撤销旧脱敏视图；传参：已发生变化的路径；返回：无，其他文件视图保持。"""
        targets = {Path(path).resolve() for path in paths}
        with self._lock:
            obsolete = {
                identity
                for identity, view in self._views.items()
                if view.path.resolve() in targets
            }
            for identity in obsolete:
                self._views.pop(identity, None)
                self._delivered.pop(identity, None)
            for key in tuple(self._latest):
                if key[1].resolve() in targets:
                    self._latest.pop(key, None)

    def _replacement_literals(
        self, view: FileView, arguments: dict[str, object]
    ) -> dict[str, str]:
        """解析与原编号明确关联的新值，不从缺失编号推断替换；传参：视图和已保护参数；返回：内部新字面量。"""
        replacements = arguments.get("secret_replacements", {})
        if not isinstance(replacements, dict):
            raise ValueError("secret_replacements must be an object")
        known = {item.token: item for item in view.values}
        result = {}
        for token, reference in replacements.items():
            if token not in known or known[token].span.comment:
                raise ValueError(
                    "replacement must reference an original protected value token"
                )
            stored = self._inputs.get(str(reference))
            if stored is None or stored[0] != view.session_id:
                raise ValueError(
                    "protected replacement input is unavailable; provide it again"
                )
            result[token] = _new_literal(stored[1], view.document.format)
        return result

    def _require_view(self, identity: str, session_id: str, path: Path) -> FileView:
        """只接受本宿主、同会话、同路径的当前视图；传参：视图身份和范围；返回：内部快照。"""
        view = self._views.get(identity)
        if view is None or view.session_id != session_id or view.path != path:
            raise FileEditConflict(
                "redacted view is stale or belongs to another file/session; read it again"
            )
        return view

    def _public(
        self,
        view: FileView,
        *,
        offset: int,
        max_chars: int | None,
        selection: ReadOnlyInspectionRequest | None = None,
    ) -> dict[str, Any]:
        """构造只含脱敏正文和版本的视图；传参：内部快照；返回：可安全保存的结果。"""
        if offset < 0 or (max_chars is not None and max_chars <= 0):
            raise ValueError(
                "read offset must be nonnegative and max_chars must be positive"
            )
        start = min(offset, len(view.content))
        end = (
            len(view.content)
            if max_chars is None
            else min(start + max_chars, len(view.content))
        )
        page = None
        if selection is not None:
            from tools.file_paging import text_page

            page = text_page(
                view.content,
                selection,
                {
                    "resolved_path": str(view.path),
                    "content_sha256": view.sha256,
                    "representation": "redacted_config:" + str(PARSER_VERSION),
                },
                max_chars if max_chars is not None else max(1, len(view.content)),
            )
            start = cast(int, page.meta["offset"])
            end = start + cast(int, page.meta["returned_count"])
        # 【敏感文件】【实际交付】只登记本页范围，内部完整脱敏不能冒充模型已经读齐
        ranges = sorted([*self._delivered.get(view.view_id, []), (start, end)])
        merged: list[tuple[int, int]] = []
        for left, right in ranges:
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
            else:
                merged.append((left, right))
        self._delivered[view.view_id] = merged
        return {
            "content": view.content[start:end],
            "meta": {
                "resolved_path": str(view.path),
                "redacted": True,
                "metadata_only": False,
                "view_id": view.view_id,
                "content_sha256": view.sha256,
                "parser_version": PARSER_VERSION,
                "format": view.document.format,
                "offset": start,
                "next_offset": end,
                "total_chars": len(view.content),
                "output_complete": start == 0 and end == len(view.content),
                "has_more": end < len(view.content),
                "protected_occurrences": len(view.values),
                **(page.meta if page is not None else {}),
            },
        }

    def close(self) -> None:
        """宿主结束后撤回所有内存映射；传参：无；返回：无。"""
        with self._lock:
            self._views.clear()
            self._latest.clear()
            self._delivered.clear()
            self._inputs.clear()
            self._input_ids.clear()


def _protected_value(
    document: ConfigDocument,
    span: ConfigSpan,
    reusable: dict[tuple[str, str, int], ProtectedValue],
) -> ProtectedValue:
    """只在字段身份和原字面量均未改变时复用编号；传参：文档、区域及旧编号；返回：该出现位置的值。"""
    literal = document.text[span.start : span.end]
    previous = reusable.get(span.identity)
    token = (
        previous.token
        if previous is not None and previous.literal == literal
        else f"<redacted:{uuid4().hex}>"
    )
    return ProtectedValue(token, span, literal)


def _replace_spans(text: str, values: tuple[ProtectedValue, ...]) -> str:
    """一次线性拼接完整脱敏文档；传参：正文和不重叠区域；返回：模型视图。"""
    parts: list[str] = []
    cursor = 0
    for item in values:
        parts.extend((text[cursor : item.span.start], item.token))
        cursor = item.span.end
    parts.append(text[cursor:])
    return "".join(parts)


def _restore_candidate(
    view: FileView, text: str, replacements: dict[str, str]
) -> bytes:
    """核对每个原编号完整且唯一，再逐位置还原秘密；传参：视图、候选和明确替换；返回：完整新字节。"""
    known = {
        item.token: replacements.get(item.token, item.literal) for item in view.values
    }
    counts = Counter(_TOKEN.findall(text))
    if set(counts) - set(known):
        raise ValueError("candidate contains an unknown protected token")
    if any(counts[token] != 1 for token in known):
        raise ValueError(
            "each protected token must appear exactly once; keep tokens and use secret_replacements"
        )
    if _INPUT.search(text):
        raise ValueError(
            "keep the original token in the view and provide its new value through secret_replacements"
        )
    restored = _TOKEN.sub(lambda match: known[match.group()], text)
    return (_BOM if view.original.startswith(_BOM) else b"") + restored.encode("utf-8")


def _new_literal(value: str, format: str) -> str:
    """把明确新值编码为单个配置值，不能注入新的赋值语句；传参：新值及格式；返回：安全字面量。"""
    if format == "dotenv":
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return "\n    ".join(value.split("\n"))


def _mask_literal(text: str, value: str, reference: str) -> str:
    """保护模型已提供的新值，普通证据中只保留引用；传参：参数正文、新值和引用；返回：已隐藏新值的正文。"""
    variants = [
        json.dumps(value, ensure_ascii=False),
        "'" + value.replace("'", "\\'") + "'",
    ]
    if value:
        variants.append(value)
    for variant in sorted(set(variants), key=len, reverse=True):
        text = text.replace(variant, reference)
    return text
