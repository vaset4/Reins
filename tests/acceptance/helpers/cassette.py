from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class Cassette:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict[str, Any]] = []
        if path.is_file():
            self.records = json.loads(path.read_text(encoding="utf-8"))

    def replay_or_record(
        self,
        *,
        method: str,
        url: str,
        body: object,
        response_factory: Callable[[], object],
    ) -> object:
        key = _request_key(method=method, url=url, body=body)
        for record in self.records:
            if record.get("key") == key:
                return record["response"]
        response = response_factory()
        self.records.append(
            {
                "key": key,
                "request": {"method": method, "url": url, "body": body},
                "response": response,
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return response


def cassette(name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:  # noqa: ANN002, ANN003
            path = Path(kwargs.pop("cassette_path", Path("tests") / "cassettes" / name))
            kwargs["cassette"] = Cassette(path)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _request_key(*, method: str, url: str, body: object) -> str:
    body_text = json.dumps(body, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(body_text.encode("utf-8")).hexdigest()
    return f"{method.upper()} {url} {digest}"
