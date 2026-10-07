from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

_PARAM_RE = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True, slots=True)
class HttpError(Exception):
    status: int
    body: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Route:
    method: str
    pattern: str
    regex: re.Pattern[str]
    param_names: list[str]
    handler: Callable[..., Any]


class Router:
    """Simple path-based router for the observe API."""

    def __init__(self) -> None:
        self._routes: list[Route] = []

    def get(self, pattern: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._add("GET", pattern, fn)
            return fn

        return decorator

    def resolve(
        self, method: str, path: str
    ) -> tuple[Callable[..., Any], dict[str, str]] | None:
        for route in self._routes:
            if route.method != method:
                continue
            match = route.regex.fullmatch(path)
            if match is None:
                continue
            params = {name: match.group(name) for name in route.param_names}
            return route.handler, params
        return None

    def _add(self, method: str, pattern: str, handler: Callable[..., Any]) -> None:
        param_names = _PARAM_RE.findall(pattern)
        regex_str = _PARAM_RE.sub(r"(?P<\1>[^/]+)", pattern)
        self._routes.append(
            Route(
                method=method,
                pattern=pattern,
                regex=re.compile(regex_str),
                param_names=param_names,
                handler=handler,
            )
        )


__all__ = ["HttpError", "Router"]
