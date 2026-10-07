from __future__ import annotations

import pytest

from scripts.check_tracked_artifacts import classify_tracked_path, find_violations


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        (".reins/data/index.db", "Reins runtime data"),
        ("app/__pycache__/cli.cpython-311.pyc", "Python bytecode cache directory"),
        ("llm/parser.cpython-311.pyc", "Python bytecode artifact"),
        ("sessions/run/raw/model_request_1.json", "raw model evidence"),
        ("sessions/run/raw/model_response.json", "raw model evidence"),
        ("sessions/run/raw/parsed_plan.json", "raw model evidence"),
        ("llm.json", "local LLM config"),
        (".reins/config/observe_prices.yaml", "local observe price config"),
    ],
)
def test_classifies_blocked_tracked_artifacts(path: str, reason: str) -> None:
    violation = classify_tracked_path(path)

    assert violation is not None
    assert violation.path == path
    assert reason in violation.reason


@pytest.mark.parametrize(
    "path",
    [
        "llm.example.json",
        "README.md",
        "runtime/run_evidence.py",
        ".trellis/tasks/archive/raw-model-evidence/prd.md",
    ],
)
def test_allows_source_docs_and_templates(path: str) -> None:
    assert classify_tracked_path(path) is None


def test_find_violations_returns_only_blocked_paths() -> None:
    violations = find_violations(
        [
            "README.md",
            ".reins/data/index.db",
            "llm.example.json",
            "tools/__pycache__/terminal_tool.cpython-311.pyc",
        ]
    )

    assert [violation.path for violation in violations] == [
        ".reins/data/index.db",
        "tools/__pycache__/terminal_tool.cpython-311.pyc",
    ]
