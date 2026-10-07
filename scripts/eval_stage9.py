"""【阶段九评估】【同条件对照】冻结整份源码并串行执行有总费用额度的正式运行样本。

作者：xxx
时间：2026-10-01 14:30:00
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from scripts.eval_long_context import write_json
from scripts.long_context_cases import CASES
from scripts.long_context_evidence import evaluation_source_files, snapshot_sources

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = PROJECT_ROOT / ".tmp/stage9-baseline-20261001-110355"
PROBE_CALL_LIMIT = 16
DEFAULT_OUTPUT_PER_CALL = 16384
CONTROLLED_WINDOW = 65536
SUMMARY_OUTPUT_RESERVES = 2
VARIANTS = ("baseline", "candidate")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析评估预算，默认只展示计划；参数：CLI参数；返回：已验证选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--profile", default="happy:deepseek-v4.1-flash")
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case.name for case in CASES],
        default=["conditions", "correction"],
    )
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--mode", choices=("controlled", "natural"), default="controlled"
    )
    parser.add_argument("--window", type=int, default=CONTROLLED_WINDOW)
    parser.add_argument("--max-logical-calls", type=int, default=PROBE_CALL_LIMIT)
    parser.add_argument("--max-attempts", type=int, default=PROBE_CALL_LIMIT)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=PROBE_CALL_LIMIT * DEFAULT_OUTPUT_PER_CALL,
    )
    parser.add_argument(
        "--per-call-output-tokens", type=int, default=DEFAULT_OUTPUT_PER_CALL
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Dispatch paid requests after reviewing the printed plan.",
    )
    args = parser.parse_args(argv)
    if any(
        getattr(args, name) < 1
        for name in (
            "rounds",
            "repeats",
            "window",
            "max_logical_calls",
            "max_attempts",
            "max_output_tokens",
            "per_call_output_tokens",
        )
    ):
        parser.error("all counts and token limits must be positive")
    if (
        args.per_call_output_tokens * SUMMARY_OUTPUT_RESERVES >= args.window
        and args.mode == "controlled"
    ):
        parser.error(
            "controlled window must leave room beyond both generation and audit output allowances"
        )
    if args.baseline_root.resolve() != BASELINE_ROOT.resolve():
        parser.error(
            "this comparison requires the frozen 20261001-110355 full baseline"
        )
    if args.output.resolve().is_relative_to(BASELINE_ROOT.resolve()):
        parser.error("evaluation output cannot modify the frozen baseline")
    if len(set(args.cases)) != len(args.cases):
        parser.error(
            "each case can be selected once; use --repeats for independent repetitions"
        )
    if len(set(args.variants)) != len(args.variants):
        parser.error(
            "each variant can be selected once; use --repeats for independent repetitions"
        )
    return args


def experiment_plan(args: argparse.Namespace) -> dict[str, Any]:
    """生成可复核的样本矩阵而不读凭据或调用模型；参数：选项；返回：完整执行计划。"""
    samples = [
        {"case": case, "repeat": repeat + 1, "variant": variant}
        for case in args.cases
        for repeat in range(args.repeats)
        for variant in args.variants
    ]
    return {
        "schema_version": 1,
        "profile": args.profile,
        "samples": samples,
        "rounds": args.rounds,
        "mode": args.mode,
        "window": args.window if args.mode == "controlled" else "profile_unchanged",
        "max_logical_calls": args.max_logical_calls,
        "max_output_tokens": args.max_output_tokens,
        "max_attempts": args.max_attempts,
        "attempt_limit_semantics": "every local dispatch including retries and reduced-output attempts",
        "per_call_output_tokens": args.per_call_output_tokens,
        "baseline_root": str(BASELINE_ROOT.resolve()),
        "output": str(args.output.resolve()),
        "price": "unknown: no local provider unit prices",
        "output_limit_semantics": "sum of per-attempt maximum output reservations; retries count, unknown usage never becomes zero",
        "scope": "formal AgentLoop and ProductionContextBuilder; controlled /compact then semantic continuation",
        "background_scope": "not exercised; admission disabled and no scheduler started",
        "natural_scope": "unforced profile window; short seeded samples do not demonstrate a naturally filled 300k window",
    }


def source_hashes(root: Path) -> dict[str, str]:
    """记录整份冻结源码的现存文件身份；参数：源码目录；返回：哈希清单。"""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not {"__pycache__", ".pytest_cache"}.intersection(path.parts)
    }


def execute(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    """从两个独立源码启动串行样本，共用总额度；参数：已审阅计划；返回：失败状态。"""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    candidate = output / "candidate-source"
    candidate_hashes = snapshot_sources(
        PROJECT_ROOT, evaluation_source_files(PROJECT_ROOT), candidate, resume=False
    )
    harness = output / "evaluation-harness"
    harness.mkdir()
    for name in ("eval_stage9_worker.py", "eval_stage9_meter.py"):
        (harness / name).write_bytes((candidate / "scripts" / name).read_bytes())
    baseline_hashes = source_hashes(BASELINE_ROOT)
    if not baseline_hashes or "runtime/agent_loop.py" not in baseline_hashes:
        raise ValueError("complete frozen baseline is unavailable")
    for name in ("scripts/long_context_cases.py", "scripts/eval_long_context.py"):
        if baseline_hashes[name] != candidate_hashes[name]:
            raise ValueError(f"comparison input or seed implementation differs: {name}")
    budget = output / "budget.json"
    write_json(
        budget,
        {
            "logical_calls": 0,
            "attempts": 0,
            "output_tokens": 0,
            "limit_exceeded": None,
            "max_logical_calls": args.max_logical_calls,
            "max_attempts": args.max_attempts,
            "max_output_tokens": args.max_output_tokens,
        },
    )
    manifest = {
        **plan,
        "candidate_source_sha256": candidate_hashes,
        "baseline_source_sha256": baseline_hashes,
    }
    write_json(output / "manifest.json", manifest)
    results = []
    for sample in plan["samples"]:
        if json.loads(budget.read_text(encoding="utf-8"))["limit_exceeded"] is not None:
            break
        name = f"{sample['case']}-{sample['repeat']}-{sample['variant']}"
        config = {
            **sample,
            **{
                key: plan[key]
                for key in (
                    "profile",
                    "rounds",
                    "mode",
                    "window",
                    "per_call_output_tokens",
                )
            },
            "source_root": str(
                BASELINE_ROOT.resolve()
                if sample["variant"] == "baseline"
                else candidate
            ),
            "sample_output": str(output / name),
            "budget": str(budget),
        }
        path = output / "configs" / f"{name}.json"
        write_json(path, config)
        command = [
            sys.executable,
            "-I",
            "-B",
            str(harness / "eval_stage9_worker.py"),
            "--config",
            str(path),
        ]
        sample_output = Path(config["sample_output"])
        sample_output.mkdir()
        with (sample_output / "process.log").open("wb") as transcript:
            result = subprocess.run(
                command,
                cwd=config["source_root"],
                check=False,
                stdout=transcript,
                stderr=subprocess.STDOUT,
            )
        results.append(
            {
                **sample,
                "exit_code": result.returncode,
                "output": config["sample_output"],
            }
        )
        write_json(
            output / "progress.json",
            {"results": results, "planned_samples": len(plan["samples"])},
        )
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    changed = source_hashes(BASELINE_ROOT) != baseline_hashes
    models = [
        json.loads((Path(row["output"]) / "model.json").read_text(encoding="utf-8"))
        for row in results
        if (Path(row["output"]) / "model.json").is_file()
    ]
    matched = all(model == models[0] for model in models) if models else False
    final = {
        "results": results,
        "baseline_changed": changed,
        "budget": json.loads(budget.read_text(encoding="utf-8")),
        "complete": len(results) == len(plan["samples"]),
        "same_model_conditions": matched,
    }
    write_json(output / "result.json", final)
    return int(
        changed
        or not matched
        or not final["complete"]
        or any(row["exit_code"] for row in results)
    )


def main(argv: Sequence[str] | None = None) -> int:
    """默认输出无付费计划，显式execute执行；参数：CLI；返回：状态码。"""
    args = parse_args(argv)
    plan = experiment_plan(args)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    return execute(args, plan) if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
