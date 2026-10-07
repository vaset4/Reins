from __future__ import annotations

from pathlib import Path

from path_security import Decision, check_read, check_write
from runtime.lease import from_trigger


def test_check_read_denies_sensitive_globs_and_allows_whitelist(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    lease = _lease(project)

    assert check_read(project / "secret.pem", lease) is Decision.DENY
    assert check_read(project / "README.md", lease) is Decision.ALLOWED
    assert check_read(outside / "notes.txt", lease) is Decision.CONFIRM


def test_check_read_uses_explicit_lease_whitelist_without_default_broadening(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    allowed = project / "allowed"
    allowed.mkdir(parents=True)
    lease = from_trigger(
        "user",
        task_id="task-1",
        capabilities={
            "fs": {
                "project_root": str(project),
                "read": [str(allowed)],
                "deny_read": ["*.pem", "*.key", ".env"],
            }
        },
    )

    assert check_read(allowed / "note.txt", lease) is Decision.ALLOWED
    assert check_read(project / "README.md", lease) is Decision.CONFIRM


def test_check_write_workspace_project_data_and_outside_rules(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    project = tmp_path / "project"
    lease = _lease(project)

    assert (
        check_write(
            project / ".reins" / "workspace" / "task-1" / "scratch" / "out.txt",
            lease,
        )
        is Decision.ALLOWED
    )
    assert check_write(project / "src" / "app.py", lease) is Decision.CONFIRM
    assert (
        check_write(home / ".reins" / "data" / "cache.txt", lease) is Decision.ALLOWED
    )
    assert check_write(tmp_path / "other" / "out.txt", lease) is Decision.DENY
    assert check_write(project / ".env", lease) is Decision.DENY


def test_check_write_resolves_dotdot_before_workspace_allow(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    lease = _lease(project)
    escaped = (
        project
        / ".reins"
        / "workspace"
        / "task-1"
        / "scratch"
        / ".."
        / ".."
        / ".."
        / ".."
        / ".."
        / "outside.txt"
    )

    assert check_write(escaped, lease) is Decision.DENY


def test_check_write_denies_explicit_deny_write_before_workspace_allow(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    lease = _lease(project, fs_extra={"deny_write": ["blocked.out"]})

    target = project / ".reins" / "workspace" / "task-1" / "blocked.out"

    assert check_write(target, lease) is Decision.DENY


def test_check_write_falls_back_to_deny_read_when_deny_write_absent(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    lease = _lease(project)

    assert check_write(project / ".env", lease) is Decision.DENY


def test_check_read_ignores_deny_write(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    lease = _lease(project, fs_extra={"deny_write": ["README.md"]})

    assert check_read(project / "README.md", lease) is Decision.ALLOWED


def test_check_write_explicit_empty_deny_write_disables_deny_read_fallback(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    lease = _lease(project, fs_extra={"deny_write": []})

    assert check_write(project / ".env", lease) is Decision.CONFIRM


def _lease(project: Path, *, fs_extra: dict[str, object] | None = None):
    fs = {
        "project_root": str(project),
        "read": [
            str(Path.home() / ".reins" / "data"),
            str(project / ".reins" / "workspace"),
            str(project),
        ],
        "write": [
            str(project / ".reins" / "workspace"),
            str(Path.home() / ".reins" / "data"),
        ],
        "deny_read": ["*.pem", "*.key", ".env"],
    }
    fs.update(fs_extra or {})
    return from_trigger(
        "user",
        task_id="task-1",
        capabilities={"fs": fs},
    )
