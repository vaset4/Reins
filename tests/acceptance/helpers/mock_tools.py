from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallRecorder:
    result: Any = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, *args, **kwargs) -> Any:  # noqa: ANN002, ANN003
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.result
