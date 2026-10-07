"""完整识别可原样往返的 dotenv 与 INI 配置，不执行变量或命令展开。

作者：xxx
时间：2026-09-24 20:30:00
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PARSER_VERSION = 1
_DOTENV_KEY = re.compile(r"(?:export[ \t]+)?([A-Za-z_][A-Za-z0-9_.-]*)[ \t]*=[ \t]*")
_INI_KEY = re.compile(r"([^\s:=;#\[\]][^:=\r\n]*?)[ \t]*[=:][ \t]*")
_SECTION = re.compile(r"\[([^\]\r\n]+)\][ \t]*(?:[;#].*)?$")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----|openssh-key-v1", re.IGNORECASE
)
_SAFE_KEYS = frozenset(
    {
        "port",
        "host",
        "hostname",
        "debug",
        "log_level",
        "loglevel",
        "region",
        "aws_region",
        "output",
        "environment",
        "node_env",
    }
)
_CREDENTIALS = re.compile(
    r"://[^/\s]*@|(?:password|passwd|secret|token|api[_-]?key)\s*[=:]", re.IGNORECASE
)


class UnsupportedConfig(ValueError):
    """整份配置无法证明已解析或安全过滤，只能返回元信息。"""


@dataclass(frozen=True, slots=True)
class ConfigSpan:
    """一处受保护的完整原始字面量；相同值在不同位置仍是不同出现项。"""

    start: int
    end: int
    identity: tuple[str, str, int]
    comment: bool = False


@dataclass(frozen=True, slots=True)
class ConfigDocument:
    """已完整解析的文档及受保护区域，原文字节由会话宿主管理。"""

    format: str
    text: str
    spans: tuple[ConfigSpan, ...]


def is_private_key(content: bytes | str) -> bool:
    """识别私钥原件与容器头，不依赖文件扩展名；传参：原字节或正文；返回：是否为私钥内容。"""
    text = (
        content.decode("ascii", errors="ignore")
        if isinstance(content, bytes)
        else content
    )
    return _PRIVATE_KEY.search(text) is not None


def parse_config(path: Path, content: bytes) -> ConfigDocument:
    """完整解析支持的配置，任一未识别片段使整份只读元信息；传参：路径、完整字节；返回：文档与保护区域。"""
    if is_private_key(content):
        raise UnsupportedConfig("private_key")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnsupportedConfig("unsupported_encoding") from exc
    if "\x00" in text:
        raise UnsupportedConfig("binary_content")
    if (
        path.name == ".env"
        or path.name.startswith(".env.")
        or path.suffix.casefold() == ".env"
    ):
        return ConfigDocument("dotenv", text, _dotenv_spans(text))
    if path.suffix.casefold() in {".ini", ".cfg", ".conf"} or path.name.casefold() in {
        "credentials",
        "config",
    }:
        return ConfigDocument("ini", text, _ini_spans(text))
    raise UnsupportedConfig("unsupported_format")


def _protected(key: str, literal: str) -> bool:
    """仅保留明确的普通配置值，未知字段和凭据连接串均保护；传参：字段与字面量；返回：是否保护。"""
    value = literal.strip().strip("\"'")
    if key.casefold() not in _SAFE_KEYS or _CREDENTIALS.search(value):
        return True
    if key.casefold() == "port":
        return not value.isdecimal()
    if key.casefold() == "debug":
        return value.casefold() not in {"true", "false", "0", "1", "yes", "no"}
    return any(char in value for char in "\r\n")


def _line_end(text: str, start: int) -> int:
    """定位当前行正文末尾，不含换行；传参：全文与起点；返回：末尾偏移。"""
    newline = text.find("\n", start)
    end = len(text) if newline < 0 else newline
    return end - 1 if end > start and text[end - 1] == "\r" else end


def _next_line(text: str, start: int) -> int:
    """跨过一个完整物理行；传参：全文与起点；返回：下一行或全文末尾。"""
    newline = text.find("\n", start)
    return len(text) if newline < 0 else newline + 1


def _quoted_end(text: str, start: int) -> int:
    """识别可跨行的完整引号字面量，转义不触发执行；传参：全文与引号位置；返回：闭引号后的偏移。"""
    quote, cursor = text[start], start + 1
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
        elif text[cursor] == quote:
            return cursor + 1
        else:
            cursor += 1
    raise UnsupportedConfig("unterminated_quoted_value")


def _dotenv_spans(text: str) -> tuple[ConfigSpan, ...]:
    """识别每一行及跨行值，保护未知字段和注释；传参：完整正文；返回：不重叠的保护区域。"""
    spans: list[ConfigSpan] = []
    counts: dict[str, int] = {}
    cursor, comments = 0, 0
    while cursor < len(text):
        end = _line_end(text, cursor)
        first = cursor + len(text[cursor:end]) - len(text[cursor:end].lstrip(" \t"))
        if first == end:
            cursor = _next_line(text, cursor)
            continue
        if text[first] == "#":
            spans.append(
                ConfigSpan(first + 1, end, ("", "#comment", comments), comment=True)
            )
            comments += 1
            cursor = _next_line(text, cursor)
            continue
        match = _DOTENV_KEY.match(text, first)
        if match is None:
            raise UnsupportedConfig("unrecognized_dotenv_statement")
        key, start = match.group(1), match.end()
        value_end, line_end, comment = _dotenv_value(text, start)
        occurrence = counts.get(key, 0)
        counts[key] = occurrence + 1
        if _protected(key, text[start:value_end]):
            spans.append(ConfigSpan(start, value_end, ("", key, occurrence)))
        if comment is not None:
            spans.append(
                ConfigSpan(comment, line_end, ("", "#comment", comments), comment=True)
            )
            comments += 1
        cursor = _next_line(text, line_end)
    return tuple(spans)


def _dotenv_value(text: str, start: int) -> tuple[int, int, int | None]:
    """识别值与行尾注释，拒绝引号后的额外语句；传参：全文与值起点；返回：值末尾、行末和注释起点。"""
    if start < len(text) and text[start] in "\"'":
        value_end = _quoted_end(text, start)
        end = _line_end(text, value_end)
        suffix = text[value_end:end]
        stripped = suffix.lstrip(" \t")
        if stripped and not stripped.startswith("#"):
            raise UnsupportedConfig("unexpected_text_after_quoted_value")
        comment = end - len(stripped) + 1 if stripped else None
        return value_end, end, comment
    end = _line_end(text, start)
    value = text[start:end]
    marker = re.search(r"(?:^|[ \t])#", value)
    comment = start + value.index("#", marker.start()) + 1 if marker else None
    value_end = (comment - 1) if comment is not None else end
    value_end -= len(text[start:value_end]) - len(text[start:value_end].rstrip(" \t"))
    return value_end, end, comment


def _ini_spans(text: str) -> tuple[ConfigSpan, ...]:
    """识别分节、赋值和缩进续行，未识别语句不部分透传；传参：完整正文；返回：保护区域。"""
    spans: list[ConfigSpan] = []
    counts: dict[str, int] = {}
    section, cursor, comments = "", 0, 0
    while cursor < len(text):
        end = _line_end(text, cursor)
        line = text[cursor:end]
        stripped = line.lstrip(" \t")
        first = end - len(stripped)
        if not stripped:
            cursor = _next_line(text, cursor)
            continue
        if stripped[0] in "#;":
            spans.append(
                ConfigSpan(
                    first + 1, end, (section, "#comment", comments), comment=True
                )
            )
            comments += 1
            cursor = _next_line(text, cursor)
            continue
        header = _SECTION.fullmatch(stripped)
        if header is not None:
            section = header.group(1)
            tail = first + stripped.index("]") + 1
            marker = re.search(r"[;#]", text[tail:end])
            if marker:
                spans.append(
                    ConfigSpan(
                        tail + marker.start() + 1,
                        end,
                        (section, "#comment", comments),
                        comment=True,
                    )
                )
                comments += 1
            cursor = _next_line(text, cursor)
            continue
        assignment = _INI_KEY.match(text, first, end)
        if assignment is None or not section:
            raise UnsupportedConfig("unrecognized_ini_statement")
        key, start = assignment.group(1).rstrip(), assignment.end()
        value_end, next_cursor = _ini_value_end(text, end, first - cursor)
        identity = f"{section}:{key}"
        occurrence = counts.get(identity, 0)
        counts[identity] = occurrence + 1
        if _protected(key, text[start:value_end]):
            spans.append(ConfigSpan(start, value_end, (section, key, occurrence)))
        cursor = next_cursor
    return tuple(spans)


def _ini_value_end(text: str, end: int, indent: int) -> tuple[int, int]:
    """把跨空行、注释的缩进续行纳入原值；传参：全文、当前行末和缩进；返回：值末尾和下一语句。"""
    cursor = _next_line(text, end)
    next_statement = cursor
    while cursor < len(text):
        next_end = _line_end(text, cursor)
        line = text[cursor:next_end]
        stripped = line.lstrip(" \t")
        if not stripped or stripped[0] in "#;":
            cursor = _next_line(text, next_end)
            continue
        if len(line) - len(stripped) <= indent:
            break
        end = next_end
        cursor = _next_line(text, next_end)
        next_statement = cursor
    return end, next_statement
