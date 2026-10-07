from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.startup import StartupIdentity, resolve_startup_identity


@pytest.mark.parametrize(
    ("environment", "expected_name"),
    [
        ({"REINS_PROJECT_ROOT": "canonical"}, "canonical"),
        ({"XIANGMU_PROJECT_ROOT": "legacy"}, "legacy"),
        (
            {
                "REINS_PROJECT_ROOT": "canonical",
                "XIANGMU_PROJECT_ROOT": "legacy",
            },
            "canonical",
        ),
        ({}, None),
    ],
)
def test_resolver_uses_one_project_root_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expected_name: str | None,
) -> None:
    monkeypatch.chdir(tmp_path)
    values = {key: str(tmp_path / value) for key, value in environment.items()}

    identity = resolve_startup_identity(environ=values, user_home=tmp_path / "profile")
    expected_project = (
        tmp_path / expected_name if expected_name is not None else tmp_path
    ).resolve()

    assert identity.project_root == expected_project
    assert identity.data_root == tmp_path / "profile" / ".reins" / "data"


def test_explicit_roots_override_environment_and_are_normalized(tmp_path: Path) -> None:
    explicit_project = tmp_path / "project" / ".." / "project"
    explicit_data = tmp_path / "data" / ".." / "data"

    identity = resolve_startup_identity(
        project_root=explicit_project,
        data_root=explicit_data,
        environ={"REINS_PROJECT_ROOT": str(tmp_path / "env")},
    )

    assert identity == StartupIdentity(
        project_root=explicit_project.resolve(),
        data_root=explicit_data.resolve(),
    )
    with pytest.raises(FrozenInstanceError):
        identity.project_root = tmp_path  # type: ignore[misc]
