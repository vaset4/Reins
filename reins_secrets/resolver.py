from __future__ import annotations

import re
from typing import Any

from reins_secrets.store import SecretNotFoundError, SecretsVault

_SECRET_REF = re.compile(r"\$\{secret:([A-Z_][A-Z0-9_]*)\}")


def resolve_secret_refs(value: str, vault: Any | None = None) -> str:
    if _SECRET_REF.search(value) is None:
        return value
    resolved_vault = vault or SecretsVault()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        secret_value = resolved_vault.get(name)
        if secret_value is None:
            raise SecretNotFoundError(f"secret not found: {name}")
        return secret_value

    return _SECRET_REF.sub(replace, value)
