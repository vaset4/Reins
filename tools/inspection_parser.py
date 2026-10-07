from __future__ import annotations

import re

from runtime.types import ReadOnlyInspectionRequest

_WORKSPACE_ALIASES = {"workspace"}
_PLAIN_PATH_PATTERN = re.compile(r"^[A-Za-z0-9._/\-\\]+$")


def parse_inspection_payload(payload: str) -> ReadOnlyInspectionRequest | None:
    text = payload.strip()
    if text.startswith("dir "):
        target_path = text.removeprefix("dir ").strip()
        return _request_or_none("list_dir", _normalize_list_target(target_path))

    if text.startswith("file "):
        target_path = text.removeprefix("file ").strip()
        return _request_or_none("read_file", target_path)

    if text.startswith("search ") and " for " in text:
        target_path, query = text.removeprefix("search ").split(" for ", 1)
        target_path = target_path.strip()
        query = query.strip()
        if target_path and query:
            return ReadOnlyInspectionRequest(
                action="grep_text",
                target_path=target_path,
                query=query,
            )
        return None

    if text and " " not in text and _is_plain_path_token(text):
        return ReadOnlyInspectionRequest(
            action="list_dir",
            target_path=_normalize_workspace_alias(text),
            query=None,
        )

    return None


def _request_or_none(
    action: str,
    target_path: str,
) -> ReadOnlyInspectionRequest | None:
    if not target_path:
        return None
    return ReadOnlyInspectionRequest(
        action=action,
        target_path=target_path,
        query=None,
    )


def _normalize_list_target(target_path: str) -> str:
    if not target_path:
        return target_path
    return _normalize_workspace_alias(target_path)


def _normalize_workspace_alias(target_path: str) -> str:
    normalized = target_path.strip()
    if normalized in _WORKSPACE_ALIASES:
        return "."
    return normalized


def _is_plain_path_token(text: str) -> bool:
    return bool(_PLAIN_PATH_PATTERN.fullmatch(text))
