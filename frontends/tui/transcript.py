from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from threading import Lock
from typing import Literal

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.utils import get_cwidth

ItemRole = Literal["user", "assistant", "tool", "approval", "system", "status", "error"]
ItemState = Literal["info", "pending", "success", "error"]

DEFAULT_RENDER_WIDTH = 88
MIN_BLOCK_WIDTH = 24
MAX_BLOCK_WIDTH = 120
BLOCK_MARGIN = 2
BLOCK_PADDING = 2
TEXT_INDENT = "  "
ELLIPSIS = "..."
EMPTY_BODY = "(empty)"
BORDER_CHAR = "-"
STATE_LABELS: dict[ItemState, str] = {
    "info": "info",
    "pending": "running",
    "success": "done",
    "error": "error",
}


@dataclass(frozen=True, slots=True)
class TranscriptItem:
    role: ItemRole
    title: str
    body: str
    timestamp: str
    state: ItemState = "info"


class TranscriptBuffer:
    def __init__(self) -> None:
        self._items: list[TranscriptItem] = []
        self._lock = Lock()

    def append(
        self, role: ItemRole, title: str, body: str, *, state: ItemState = "info"
    ) -> None:
        item = TranscriptItem(
            role=role,
            title=title,
            body=body,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            state=state,
        )
        with self._lock:
            self._items.append(item)

    def append_to_last(
        self,
        role: ItemRole,
        title: str,
        text: str,
        *,
        state: ItemState = "info",
    ) -> None:
        """把文本追加进同角色同标题的最后一条，没有就新起一条。

        作者：LKX
        时间：2026-09-05 00:00:00
        传参：role 为条目角色；title 为标题，同时也是合并判据；text 为要追加的文本；
              state 仅在新起条目时生效
        返回：无

        流式增量一轮会来几十上百片，每片新起一条会把 transcript 刷爆，连续增量必须
        并进同一条里增长。合并要同时看标题：status 这个角色被 lifecycle、lease、state
        好几类共用，只看角色会把思考链并进上一条 lifecycle 里去。
        """
        if not text:
            return
        with self._lock:
            if self._items and self._merges_with_last(role, title):
                item = self._items[-1]
                self._items[-1] = replace(item, body=item.body + text)
                return
            self._items.append(
                TranscriptItem(
                    role=role,
                    title=title,
                    body=text,
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    state=state,
                )
            )

    def _merges_with_last(self, role: ItemRole, title: str) -> bool:
        last = self._items[-1]
        return last.role == role and last.title == title

    def append_to_last_assistant(self, text: str) -> None:
        self.append_to_last("assistant", "assistant", text)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def snapshot(self) -> list[TranscriptItem]:
        with self._lock:
            return list(self._items)

    def fragments(
        self,
        width: int = DEFAULT_RENDER_WIDTH,
        *,
        scroll_offset: int = 0,
        height: int | None = None,
        tail: StyleAndTextTuples | None = None,
    ) -> StyleAndTextTuples:
        lines = self.fragment_lines(width)
        if tail:
            lines.extend(_fragments_to_lines(tail))
        if height is not None:
            offset = min(max(0, scroll_offset), max(0, len(lines) - height))
            lines = lines[offset : offset + height]
        return _join_lines(lines)

    def line_count(
        self, width: int = DEFAULT_RENDER_WIDTH, *, tail_lines: int = 0
    ) -> int:
        return len(self.fragment_lines(width)) + tail_lines

    def fragment_lines(
        self, width: int = DEFAULT_RENDER_WIDTH
    ) -> list[StyleAndTextTuples]:
        items = self.snapshot()
        if not items:
            return [[("", "")]]
        fragments: StyleAndTextTuples = []
        for item in items:
            fragments.extend(_item_fragments(item, width))
        return _fragments_to_lines(fragments)


def _item_fragments(item: TranscriptItem, width: int) -> StyleAndTextTuples:
    if item.role == "user":
        return _user_fragments(item, width)
    if item.role == "assistant":
        return _assistant_fragments(item)
    if item.role in {"tool", "approval", "error"}:
        return _execution_fragments(item, width)
    return _compact_fragments(item)


def _user_fragments(item: TranscriptItem, width: int) -> StyleAndTextTuples:
    block_width = _block_width(width)
    inner_width = max(1, block_width - BLOCK_PADDING * 2)
    fragments: StyleAndTextTuples = [("", "\n")]
    for line in _body_lines(item.body):
        body = _pad_text(_fit_text(line, inner_width), inner_width)
        fragments.append(
            ("class:user.block", f"{' ' * BLOCK_PADDING}{body}{' ' * BLOCK_PADDING}")
        )
        fragments.append(("", "\n"))
    fragments.append(("", "\n"))
    return fragments


def _assistant_fragments(item: TranscriptItem) -> StyleAndTextTuples:
    fragments: StyleAndTextTuples = [("", "\n")]
    for line in _body_lines(item.body):
        fragments.append(("class:assistant.body", f"{TEXT_INDENT}{line}"))
        fragments.append(("", "\n"))
    fragments.append(("", "\n"))
    return fragments


def _execution_fragments(item: TranscriptItem, width: int) -> StyleAndTextTuples:
    block_width = _block_width(width)
    header = _execution_header(item, block_width)
    border = BORDER_CHAR * block_width
    border_style = f"class:{item.role}.{item.state}.border"
    fragments: StyleAndTextTuples = [
        ("", "\n"),
        (border_style, border),
        ("", "\n"),
        (f"class:{item.role}.{item.state}.title", header),
        ("", "\n"),
    ]
    fragments.extend(_execution_body_fragments(item, block_width))
    fragments.extend([(border_style, border), ("", "\n\n")])
    return fragments


def _execution_body_fragments(
    item: TranscriptItem, block_width: int
) -> StyleAndTextTuples:
    inner_width = max(1, block_width - len(TEXT_INDENT))
    style = f"class:{item.role}.{item.state}.body"
    fragments: StyleAndTextTuples = []
    for line in _body_lines(item.body):
        fragments.append((style, f"{TEXT_INDENT}{_fit_text(line, inner_width)}"))
        fragments.append(("", "\n"))
    return fragments


def _compact_fragments(item: TranscriptItem) -> StyleAndTextTuples:
    title = f"{item.title}"
    body = item.body.rstrip()
    if body:
        text = f"{TEXT_INDENT}{title}: {body}"
    else:
        text = f"{TEXT_INDENT}{title}"
    return [("class:status.line", text), ("", "\n\n")]


def _execution_header(item: TranscriptItem, block_width: int) -> str:
    label = STATE_LABELS[item.state]
    text = f"{TEXT_INDENT}{item.title}  {label}"
    return _fit_text(text, block_width)


def _body_lines(body: str) -> list[str]:
    stripped = body.rstrip()
    if not stripped:
        return [EMPTY_BODY]
    return stripped.splitlines()


def _block_width(width: int) -> int:
    usable_width = max(1, width - BLOCK_MARGIN)
    if usable_width < MIN_BLOCK_WIDTH:
        return usable_width
    return min(usable_width, MAX_BLOCK_WIDTH)


def _fit_text(text: str, width: int) -> str:
    if get_cwidth(text) <= width:
        return text
    if width <= len(ELLIPSIS):
        return ELLIPSIS[:width]
    return _clip_text(text, width - len(ELLIPSIS)) + ELLIPSIS


def _clip_text(text: str, width: int) -> str:
    result = ""
    used = 0
    for char in text:
        char_width = get_cwidth(char)
        if used + char_width > width:
            return result
        result += char
        used += char_width
    return result


def _pad_text(text: str, width: int) -> str:
    padding = max(0, width - get_cwidth(text))
    return text + (" " * padding)


def _fragments_to_lines(fragments: StyleAndTextTuples) -> list[StyleAndTextTuples]:
    lines: list[StyleAndTextTuples] = []
    current: StyleAndTextTuples = []
    for fragment in fragments:
        style = fragment[0]
        text = fragment[1]
        parts = text.split("\n")
        for index, part in enumerate(parts):
            if part:
                current.append((style, part))
            if index < len(parts) - 1:
                lines.append(current)
                current = []
    if current or not lines:
        lines.append(current)
    return lines


def _join_lines(lines: list[StyleAndTextTuples]) -> StyleAndTextTuples:
    fragments: StyleAndTextTuples = []
    for line in lines:
        fragments.extend(line)
        fragments.append(("", "\n"))
    return fragments


def truncate_text(value: object, limit: int) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 4] + " ..."


__all__ = ["TranscriptBuffer", "TranscriptItem", "truncate_text"]
