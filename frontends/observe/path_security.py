from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath


def resolve_relative_under(root: Path, relative_path: str) -> Path | None:
    if not relative_path or _is_unsafe_relative_path(relative_path):
        return None
    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError:
        return None
    return candidate


def _is_unsafe_relative_path(raw_path: str) -> bool:
    posix_path = PurePosixPath(raw_path)
    windows_path = PureWindowsPath(raw_path)
    if posix_path.is_absolute() or windows_path.is_absolute():
        return True
    return ".." in posix_path.parts or ".." in windows_path.parts


__all__ = ["resolve_relative_under"]
