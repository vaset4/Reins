"""Inspect durable inbox records and preview promotion.

Usage:
    python -m scripts.inbox_scan [--data-dir PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contextlib import closing

from runtime.persistence import INDEX_NAME, SPACE_ID_FILE
from tasks.store import TaskStore


@dataclass(slots=True)
class ScanIssue:
    type: str
    task_id: str
    detail: str


@dataclass(slots=True)
class PromoteCandidate:
    task_id: str
    goal: str
    blocker: str | None = None


@dataclass(slots=True)
class PromotePlan:
    action: str
    from_path: str
    to_path: str
    new_task_id: str | None = None


@dataclass(slots=True)
class ScanReport:
    scanned_at: str
    data_root: str
    summary: dict[str, int] = field(default_factory=dict)
    issues: list[ScanIssue] = field(default_factory=list)
    promote_candidates: list[PromoteCandidate] = field(default_factory=list)
    index_available: bool = True
    promote_plan: list[PromotePlan] = field(default_factory=list)


def scan(data_root: Path, *, dry_run: bool = False) -> ScanReport:
    """【Reins】【收件箱查询】读取规范任务记录并预览转正式事项。

    参数：data_root 为已初始化数据根，dry_run 表示包含操作预览；返回：只读扫描报告
    """
    if (
        not (data_root / SPACE_ID_FILE).is_file()
        or not (data_root / "commits.jsonl").is_file()
    ):
        raise FileNotFoundError(
            "runtime source identity or commit journal missing; restore originals before inspecting inbox"
        )
    with closing(TaskStore(data_root)) as store:
        records = store.list_tasks()
    inbox = [record for record in records if record.is_inbox]
    report = ScanReport(
        scanned_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        data_root=str(data_root.resolve()),
        summary={
            "inbox_count": len(inbox),
            "formal_count": len(records) - len(inbox),
            "issues_count": 0,
        },
        index_available=(data_root / INDEX_NAME).is_file(),
    )
    for record in inbox:
        report.promote_candidates.append(
            PromoteCandidate(task_id=record.task_id, goal=record.goal)
        )
        if dry_run:
            report.promote_plan.append(
                PromotePlan(
                    action="promote",
                    from_path=f"inbox:{record.task_id}",
                    to_path=f"task:{record.task_id}",
                )
            )
    return report


def write_report(report: ScanReport, data_root: Path) -> Path:
    reports_dir = data_root / "assets" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"inbox_scan_{stamp}.json"
    path.write_text(
        json.dumps(_report_to_dict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _report_to_dict(report: ScanReport) -> dict[str, Any]:
    return {
        "scanned_at": report.scanned_at,
        "data_root": report.data_root,
        "summary": report.summary,
        "index_available": report.index_available,
        "issues": [asdict(i) for i in report.issues],
        "promote_candidates": [asdict(c) for c in report.promote_candidates],
        "promote_plan": [_plan_to_dict(p) for p in report.promote_plan],
    }


def _plan_to_dict(plan: PromotePlan) -> dict[str, Any]:
    # PRD R3 specifies the JSON keys "from" and "to"; map from the
    # dataclass field names (which avoid the Python reserved word "from").
    return {
        "action": plan.action,
        "from": plan.from_path,
        "to": plan.to_path,
        "new_task_id": plan.new_task_id,
    }


def main() -> int:
    """查询已有文件空间并写显式报告；参数：命令行数据根和预览选项；返回：退出状态。"""
    parser = argparse.ArgumentParser(
        description="Inspect committed inbox originals and generate a promotion preview."
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path.home() / ".reins" / "data"),
        help="Path to data root (default: ~/.reins/data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Include promote plan in report without executing",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    if not data_root.is_dir():
        print(f"Error: data root does not exist: {data_root}", file=sys.stderr)
        return 1

    try:
        report = scan(data_root, dry_run=args.dry_run)
    except (ValueError, OSError) as exc:
        print(f"Inbox scan failed: {exc}", file=sys.stderr)
        return 1
    report_path = write_report(report, data_root)

    print("Inbox scan complete.")
    print(f"  Inbox tasks: {report.summary.get('inbox_count', 0)}")
    print(f"  Formal tasks: {report.summary.get('formal_count', 0)}")
    print(f"  Issues found: {report.summary.get('issues_count', 0)}")
    print(f"  Promote candidates: {len(report.promote_candidates)}")
    if report.promote_plan:
        print(f"  Dry-run plan entries: {len(report.promote_plan)}")
    print(f"  Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
