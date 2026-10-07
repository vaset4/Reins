from __future__ import annotations

import pytest

from llm.provider_connection import (
    ConnectionProfile,
    HeaderPolicy,
    ProviderConnectionError,
    resolve_connection,
)


class Credentials:
    def resolve(self, credential_ref: str) -> str:
        assert credential_ref == "fixture/openai"
        return "synthetic-test-token"


def test_openai_header_policy_resolves_secret_only_at_send_boundary() -> None:
    profile = ConnectionProfile(
        key="fixture",
        base_url="https://api.openai.com/v1",
        timeout_seconds=30,
        credential_ref="fixture/openai",
        project="project-fixture",
        extra_headers={"X-Fixture": "yes"},
    )
    resolved = resolve_connection(profile, Credentials(), HeaderPolicy("openai_chat"))
    assert resolved.headers["Authorization"] == "Bearer synthetic-test-token"
    assert resolved.headers["OpenAI-Project"] == "project-fixture"
    assert resolved.headers["User-Agent"].startswith("Reins/")
    assert "synthetic-test-token" not in repr(profile)


def test_reserved_header_override_is_case_insensitive_and_fails_closed() -> None:
    profile = ConnectionProfile(
        key="fixture",
        base_url="https://api.anthropic.com",
        timeout_seconds=30,
        credential_ref="fixture/openai",
        extra_headers={"X-Api-Key": "override"},
    )
    with pytest.raises(ProviderConnectionError, match="reserved_header_override"):
        resolve_connection(profile, Credentials(), HeaderPolicy("anthropic_messages"))
