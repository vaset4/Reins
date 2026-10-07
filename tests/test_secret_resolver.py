from __future__ import annotations

import pytest

from reins_secrets.resolver import SecretNotFoundError, resolve_secret_refs


def test_resolve_single_secret_ref() -> None:
    vault = _Vault({"API_TOKEN": "token-value"})

    assert resolve_secret_refs("Bearer ${secret:API_TOKEN}", vault=vault) == (
        "Bearer token-value"
    )


def test_resolve_multiple_secret_refs() -> None:
    vault = _Vault({"A": "one", "B_2": "two"})

    assert resolve_secret_refs("${secret:A}_${secret:B_2}", vault=vault) == "one_two"


def test_missing_secret_raises() -> None:
    with pytest.raises(SecretNotFoundError):
        resolve_secret_refs("${secret:MISSING}", vault=_Vault({}))


def test_lowercase_secret_ref_is_not_matched() -> None:
    assert resolve_secret_refs("${secret:lower}") == "${secret:lower}"


class _Vault:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def get(self, name: str) -> str | None:
        return self.values.get(name)
