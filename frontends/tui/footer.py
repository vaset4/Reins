from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.formatted_text import StyleAndTextTuples

FOOTER_FALLBACK_WIDTH = 88
FOOTER_MIN_PADDING = 2
FOOTER_ELLIPSIS = "..."


@dataclass(frozen=True, slots=True)
class FooterView:
    project_root: Path
    session_id: str
    state: str
    run_id: str | None
    provider: str
    model: str


def render_footer(view: FooterView, width: int) -> StyleAndTextTuples:
    left = _compact_path(view.project_root)
    right = f"session {view.session_id[-12:]}"
    status = _status_label(view)
    return [
        ("class:footer", _fit_footer_line(left, right, width)),
        ("", "\n"),
        ("class:footer.active", _fit_footer_line(status, _model_label(view), width)),
    ]


def render_pending_approval_footer() -> StyleAndTextTuples:
    return [
        ("class:footer.pending", " approval pending"),
        ("", "\n"),
        ("class:footer", " type /approve once, /approve task, or /deny"),
    ]


def _model_label(view: FooterView) -> str:
    if view.provider:
        return f"{view.provider} {view.model}"
    return view.model


def _status_label(view: FooterView) -> str:
    run = f" run {view.run_id[-12:]}" if view.run_id else ""
    return f"{view.state}{run} | scroll Wheel/PgUp/PgDn | /help /status /exit"


def _compact_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(Path.home().resolve())
    except ValueError:
        return str(path)
    if str(relative) == ".":
        return "~"
    return f"~\\{relative}"


def _fit_footer_line(left: str, right: str, width: int) -> str:
    right_width = len(right)
    available_left = max(0, width - right_width - FOOTER_MIN_PADDING)
    clipped_left = _truncate_simple(left, available_left)
    padding = max(FOOTER_MIN_PADDING, width - len(clipped_left) - right_width)
    line = f"{clipped_left}{' ' * padding}{right}"
    return _truncate_simple(line, width)


def _truncate_simple(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    if limit <= len(FOOTER_ELLIPSIS):
        return FOOTER_ELLIPSIS[:limit]
    return text[: limit - len(FOOTER_ELLIPSIS)] + FOOTER_ELLIPSIS


__all__ = [
    "FOOTER_FALLBACK_WIDTH",
    "FooterView",
    "render_footer",
    "render_pending_approval_footer",
]
