from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON_CACHE_EXTENSIONS = (".pyc", ".pyo", ".pyd")
RAW_EVIDENCE_FRAGMENTS = (
    "/raw/model_request",
    "/raw/model_response",
    "/raw/parsed_plan",
)
BLOCKED_EXACT_PATHS = {
    ".reins/config/observe_prices.yaml": "local observe price config",
    "llm.json": "local LLM config; use llm.example.json as the tracked template",
}
BLOCKED_PREFIXES = {
    ".reins/data/": "Reins runtime data",
}


@dataclass(frozen=True, slots=True)
class ArtifactViolation:
    path: str
    reason: str


def normalize_git_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def classify_tracked_path(path: str) -> ArtifactViolation | None:
    normalized = normalize_git_path(path)
    lowered = normalized.lower()

    if normalized in BLOCKED_EXACT_PATHS:
        return ArtifactViolation(normalized, BLOCKED_EXACT_PATHS[normalized])
    for prefix, reason in BLOCKED_PREFIXES.items():
        if normalized.startswith(prefix):
            return ArtifactViolation(normalized, reason)
    if "__pycache__" in normalized.split("/"):
        return ArtifactViolation(normalized, "Python bytecode cache directory")
    if lowered.endswith(PYTHON_CACHE_EXTENSIONS):
        return ArtifactViolation(normalized, "Python bytecode artifact")
    if any(fragment in f"/{normalized}" for fragment in RAW_EVIDENCE_FRAGMENTS):
        return ArtifactViolation(normalized, "raw model evidence")
    return None


def find_violations(paths: Iterable[str]) -> list[ArtifactViolation]:
    violations = []
    for path in paths:
        violation = classify_tracked_path(path)
        if violation:
            violations.append(violation)
    return violations


def read_tracked_paths(project_root: Path = PROJECT_ROOT) -> list[str]:
    process = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=project_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        message = process.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git ls-files failed: {message}")
    decoded = process.stdout.decode("utf-8", errors="replace")
    return [path for path in decoded.split("\0") if path]


def format_violations(violations: Iterable[ArtifactViolation]) -> str:
    lines = ["tracked runtime/local artifacts found:"]
    for violation in violations:
        lines.append(f"- {violation.path}: {violation.reason}")
    lines.append("remove them from the Git index without deleting local files")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail when runtime artifacts or local config are tracked by Git."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root to inspect.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        paths = read_tracked_paths(args.project_root.resolve())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    violations = find_violations(paths)
    if not violations:
        return 0
    print(format_violations(violations), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
