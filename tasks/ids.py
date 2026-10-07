from __future__ import annotations

import secrets
from datetime import datetime, timezone

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    timestamp_ms = int(current.timestamp() * 1000) & ((1 << 48) - 1)
    value = (timestamp_ms << 80) | int.from_bytes(secrets.token_bytes(10), "big")
    chars = ["0"] * 26
    for index in range(25, -1, -1):
        value, remainder = divmod(value, 32)
        chars[index] = _ALPHABET[remainder]
    return "".join(chars)


def new_task_id(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    return f"{current.date().isoformat()}-{new_ulid(current)}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
