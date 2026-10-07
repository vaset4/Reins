from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


LEGACY_CHECKPOINT_DIR = "checkpoints"
LEGACY_TRAJECTORY_FILE = "trajectory.jsonl"


@dataclass(frozen=True, slots=True)
class CleanupTarget:
    task_dir: Path
    checkpoints_dir: Path | None
    trajectory_file: Path | None
    bytes_before: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove legacy task checkpoints and trajectory mirrors."
    )
    parser.add_argument(
        "--tasks-root",
        default=".reins/data/tasks",
        help="Path to the Reins task data root.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete files. Omit for dry-run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks_root = Path(args.tasks_root).resolve()
    targets = collect_targets(tasks_root)
    total_bytes = sum(target.bytes_before for target in targets)
    mode = "apply" if args.apply else "dry-run"
    print(f"mode={mode} tasks_root={tasks_root}")
    print(f"targets={len(targets)} bytes_before={total_bytes}")
    for target in targets:
        print(format_target(target))
        if args.apply:
            delete_target(tasks_root, target)
    if args.apply:
        remaining = collect_targets(tasks_root)
        print(f"remaining={len(remaining)}")
        return 1 if remaining else 0
    return 0


def collect_targets(tasks_root: Path) -> list[CleanupTarget]:
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"tasks root not found: {tasks_root}")
    roots = [path for path in tasks_root.iterdir() if path.is_dir()]
    inbox_root = tasks_root / "_inbox"
    if inbox_root.is_dir():
        roots.extend(path for path in inbox_root.iterdir() if path.is_dir())
    targets: list[CleanupTarget] = []
    for task_dir in sorted(set(roots)):
        checkpoints_dir = _legacy_dir(task_dir)
        trajectory_file = _legacy_file(task_dir)
        if checkpoints_dir is None and trajectory_file is None:
            continue
        bytes_before = _path_size(checkpoints_dir) + _path_size(trajectory_file)
        targets.append(
            CleanupTarget(
                task_dir=task_dir,
                checkpoints_dir=checkpoints_dir,
                trajectory_file=trajectory_file,
                bytes_before=bytes_before,
            )
        )
    return targets


def delete_target(tasks_root: Path, target: CleanupTarget) -> None:
    for path in (target.checkpoints_dir, target.trajectory_file):
        if path is None:
            continue
        resolved = path.resolve()
        _assert_inside(tasks_root, resolved)
        if resolved.is_dir():
            shutil.rmtree(resolved)
        elif resolved.is_file():
            resolved.unlink()


def format_target(target: CleanupTarget) -> str:
    kinds = []
    if target.checkpoints_dir is not None:
        kinds.append(LEGACY_CHECKPOINT_DIR)
    if target.trajectory_file is not None:
        kinds.append(LEGACY_TRAJECTORY_FILE)
    return (
        f"task={target.task_dir.relative_to(target.task_dir.parents[1])} "
        f"kinds={','.join(kinds)} bytes_before={target.bytes_before}"
    )


def _legacy_dir(task_dir: Path) -> Path | None:
    path = task_dir / LEGACY_CHECKPOINT_DIR
    return path if path.is_dir() else None


def _legacy_file(task_dir: Path) -> Path | None:
    path = task_dir / LEGACY_TRAJECTORY_FILE
    return path if path.is_file() else None


def _path_size(path: Path | None) -> int:
    if path is None:
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _assert_inside(root: Path, path: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"refusing to delete outside {root}: {path}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
