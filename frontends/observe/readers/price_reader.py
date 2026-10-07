from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_HAS_YAML = False
try:
    import yaml

    _HAS_YAML = True
except ImportError:
    pass


class PriceReader:
    """Reads the observe_prices.yaml price table.

    Reloads on mtime change. Returns empty dict if file missing or malformed.
    """

    def __init__(self, config_root: Path) -> None:
        self._path = config_root / "observe_prices.yaml"
        self._mtime: float = 0.0
        self._data: dict[str, Any] = {}
        self._loaded = False

    def prices(self) -> dict[str, Any]:
        self._maybe_reload()
        return self._data

    def _maybe_reload(self) -> None:
        if not self._path.is_file():
            self._data = {}
            self._loaded = True
            return
        try:
            current_mtime = os.path.getmtime(self._path)
        except OSError:
            return
        if self._loaded and current_mtime == self._mtime:
            return
        self._mtime = current_mtime
        self._data = self._parse()
        self._loaded = True

    def _parse(self) -> dict[str, Any]:
        if not _HAS_YAML:
            return {"_error": "pyyaml not installed"}
        try:
            text = self._path.read_text(encoding="utf-8")
            parsed = yaml.safe_load(text)
        except Exception as exc:
            return {"_error": f"parse_error: {exc}"}
        if not isinstance(parsed, dict):
            return {"_error": "root must be a mapping"}
        return parsed


__all__ = ["PriceReader"]
