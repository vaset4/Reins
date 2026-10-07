"""汇总已落盘的长上下文实验，保留失败、未完成尝试和未知用量。

作者：xxx
时间：2026-09-26 19:35:00
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.eval_long_context import write_json

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_input_tokens",
    "cache_write_input_tokens",
)


def collect_group(directory: Path) -> dict[str, Any]:
    """按逻辑调用及实际尝试汇总一组实验；传参：冻结证据目录；返回：成本、行为结果和失败路径。"""
    attempts: list[dict[str, Any]] = []
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_rounds = (
        1 if manifest.get("strategy") == "full" else manifest.get("rounds")
    )
    calls = []
    for path in sorted(directory.glob("*/call-*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        calls.append(
            {
                "path": str(path.relative_to(directory)),
                "purpose": value["purpose"],
                "finished": "elapsed_seconds" in value,
                "elapsed_seconds": value.get("elapsed_seconds"),
                "error": value.get("error"),
            }
        )
        attempts.extend(value.get("attempts", []))
    runtime_reads = []
    runtime_requests = 0
    for path in sorted(directory.glob("*/data/sessions/*/runs/*/facts.jsonl")):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        attempts.extend(row for row in rows if row.get("event") == "llm:attempt")
        runtime_requests += sum(row.get("event") == "llm:request" for row in rows)
    results: list[dict[str, Any]] = []
    samples = []
    for path in sorted(directory.glob("*/progress.json")):
        progress = json.loads(path.read_text(encoding="utf-8"))
        samples.append(
            {
                "sample": path.parent.name,
                "completed_rounds": progress["next_round"],
                "expected_rounds": expected_rounds,
                "complete": progress["next_round"] == expected_rounds
                if expected_rounds is not None
                else None,
                "passed_completed_rounds": all(
                    row["result"]["passed"] for row in progress["rounds"]
                )
                if progress["rounds"]
                else None,
            }
        )
    for path in sorted(directory.glob("*/result.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "rounds" in value:
            results.extend(
                {"sample": path.parent.name, "round": row["round"], **row["result"]}
                for row in value["rounds"]
            )
        else:
            complete = (
                value.get("state") in {None, "DONE", "done"}
                and value.get("error_type") is None
                and value.get("reason")
                not in {"runtime_failed", "model_call_failed_or_not_final"}
            )
            samples.append(
                {
                    "sample": path.parent.name,
                    "complete": complete,
                    "passed": value["passed"],
                }
            )
            results.append(
                {
                    "sample": path.parent.name,
                    **{
                        key: value[key]
                        for key in (
                            "passed",
                            "reason",
                            "fields",
                            "error",
                            "error_type",
                            "runtime_error",
                            "state",
                            "elapsed_seconds",
                        )
                        if key in value
                    },
                }
            )
        runtime_reads.extend(value.get("history_reads", []))
    failures = [
        {
            "path": str(path.relative_to(directory)),
            **json.loads(path.read_text(encoding="utf-8")),
        }
        for path in sorted(directory.glob("*/failure-*.json"))
    ]
    return {
        "directory": str(directory),
        "manifest": manifest,
        "samples": samples,
        "calls": calls,
        "runtime_requests": runtime_requests,
        "recorded_attempts": len(attempts),
        "usage": usage_totals(attempts),
        "attempt_elapsed_seconds": sum(
            row["elapsed_ms"] / 1000
            for row in attempts
            if row.get("elapsed_ms") is not None
        ),
        "history_reads": len(runtime_reads),
        "history_read_output_chars": sum(
            len(row.get("result", {}).get("output", "")) for row in runtime_reads
        ),
        "behavior_results": results,
        "failures": failures,
        "interrupted": (directory / "INTERRUPTED.md").exists(),
        "note": "Reported or derived usage is summed separately from unknown measurements; unfinished calls may incur additional unknown usage.",
    }


def usage_totals(attempts: list[dict[str, Any]]) -> dict[str, object]:
    """未知用量单列，不把未报告当零；传参：真实尝试；返回：各指标已知小计及未知数量。"""
    totals: dict[str, object] = {}
    for field in USAGE_FIELDS:
        known = 0
        unknown = 0
        statuses: dict[str, int] = {}
        for attempt in attempts:
            usage = attempt.get("usage", {})
            measurement: Mapping[str, Any] = usage.get(
                field, {"status": "unknown", "value": None}
            )
            status, value = measurement["status"], measurement["value"]
            statuses[status] = statuses.get(status, 0) + 1
            if status == "unknown":
                unknown += 1
            if value is not None:
                known += value
        totals[field] = {
            "known_subtotal": known,
            "unknown_measurements": unknown,
            "statuses": statuses,
        }
    return totals


def main() -> int:
    """将多组现有证据汇总为独立JSON，不调用模型；传参：目录列表与输出路径；返回：零。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = [collect_group(directory) for directory in args.directories]
    write_json(args.output, {"groups": groups})
    for group in groups:
        results = group["behavior_results"]
        print(
            json.dumps(
                {
                    "directory": group["directory"],
                    "passed_results": sum(bool(row["passed"]) for row in results),
                    "recorded_results": len(results),
                    "calls": len(group["calls"]) + group["runtime_requests"],
                    "attempts": group["recorded_attempts"],
                    "complete_samples": sum(
                        row["complete"] is True for row in group["samples"]
                    ),
                    "samples_seen": len(group["samples"]),
                    "known_total_tokens": group["usage"]["total_tokens"][
                        "known_subtotal"
                    ],
                    "history_reads": group["history_reads"],
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
