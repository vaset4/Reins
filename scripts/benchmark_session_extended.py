"""在明确源码根执行隔离进程基准，保留失败和导入身份。

作者：xxx
时间：2026-09-28 23:00:00
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

REPETITIONS = 5
TIMEOUT_SECONDS = 300
SCALES = {
    "background": (1, 100, 1000),
    "prepare": (100, 1000, 10000),
    "facts": (3, 30, 300),
}
SCRIPTS = (
    "benchmark_session_extended.py",
    "benchmark_session_scenarios.py",
    "benchmark_session_runtime.py",
    "benchmark_session_metrics.py",
)


def fingerprints(root: Path) -> dict[str, str]:
    """记录完整输入字节身份；传参：目录；返回：相对路径和SHA256。"""
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def source_identity(source: Path) -> dict[str, dict[str, str]]:
    """拒绝混用工作区生产模块；传参：明确源码根；返回：实际导入路径和哈希。"""
    rows = {}
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in {
            "app",
            "runtime",
            "llm",
            "context",
            "tasks",
            "tools",
            "memory",
            "skills",
            "schedules",
            "approval",
            "triggers",
        } and not name.startswith("scripts.testing"):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(source):
            raise ValueError(f"mixed production import: {name}: {path}")
        rows[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return rows


def worker(args: argparse.Namespace) -> dict[str, object]:
    """先选源码再装载测量入口；传参：命令参数；返回：原始计量和实际源码身份。"""
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import scripts

    # 【阶段四基准】【源码归属】测试Adapter属于受测版本；先绑定冻结包，避免测量目录覆盖它
    if (source / "scripts/testing").is_dir():
        importlib.import_module("scripts.testing")
    scripts.__path__.insert(0, str(Path(__file__).resolve().parent))
    from scripts.benchmark_session_scenarios import (
        background_rows,
        preparation_rows,
        forbidden_dispatch,
        generate_scenario,
        fact_density_rows,
    )
    from scripts.benchmark_session_metrics import WorkingSetSampler, measure_reads

    imports = source_identity(source)
    if args.generate:
        generate_scenario(args.input, args.scenario, args.size)
        return {"imports": imports, "inputs": fingerprints(args.input)}
    # 【阶段四基准】【样本隔离】可写副本与固定输入同盘，保留现场供核对；由进程退出释放SQLite句柄
    with nullcontext(
        tempfile.mkdtemp(prefix="worker-data-", dir=args.input.parent)
    ) as temporary:
        root = Path(temporary) / "data"
        shutil.copytree(args.input, root)
        if fingerprints(root) != fingerprints(args.input):
            raise ValueError("worker fixture copy differs")
        action = {
            "background": background_rows,
            "prepare": preparation_rows,
            "facts": fact_density_rows,
        }[args.scenario]
        measurement: dict[str, object] = {}
        with (
            patch("socket.create_connection", forbidden_dispatch),
            patch("socket.socket.connect", forbidden_dispatch),
        ):
            if args.metric == "reads":
                with measure_reads(root, allow_writes=True) as meter:
                    rows = action(root, args.size)
                measurement["reads"] = asdict(meter.snapshot())
            elif args.metric == "memory":
                sampler = WorkingSetSampler()
                sampler.start()
                try:
                    rows = action(root, args.size)
                finally:
                    measurement["memory"] = sampler.finish()
            else:
                rows = action(root, args.size)
        return {
            "rows": rows,
            **measurement,
            "imports": source_identity(source),
            "work_directory": temporary,
            "error": None,
        }


def child(args: argparse.Namespace, *, generate: bool = False) -> dict[str, object]:
    """运行有独立硬超时的基准子进程；传参：样本参数；返回：成功或失败原始证据。"""
    command = [
        sys.executable,
        "-X",
        "utf8",
        str(Path(__file__).resolve()),
        "--worker",
        "--source",
        str(args.source.resolve()),
        "--input",
        str(args.input.resolve()),
        "--scenario",
        args.scenario,
        "--size",
        str(args.size),
        "--metric",
        args.metric,
    ]
    if generate:
        command.append("--generate")
    started = perf_counter()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(result.stderr or result.stdout)
        return {"worker_seconds": perf_counter() - started, **json.loads(result.stdout)}
    except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as exc:
        return {
            "worker_seconds": perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "command": command,
        }


def main() -> int:
    """运行指定场景各规模，生成与测量分离；传参：CLI；返回：所有样本成功为0。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scenario", choices=tuple(SCALES), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument(
        "--metric", choices=("timing", "reads", "memory"), default="timing"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.size not in SCALES[args.scenario]:
        parser.error("size is outside the planned scales")
    if args.worker:
        print(json.dumps(worker(args), ensure_ascii=False))
        return 0
    if args.output is None:
        parser.error("--output required")
    args.output.mkdir(parents=True, exist_ok=False)
    scripts = {}
    for name in SCRIPTS:
        content = Path(__file__).with_name(name).read_bytes()
        (args.output / name).write_bytes(content)
        scripts[name] = hashlib.sha256(content).hexdigest()
    if args.generate:
        result = child(args, generate=True)
        (args.output / "generation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if result.get("error"):
            return 1
    inputs = fingerprints(args.input)
    if not inputs:
        raise ValueError("empty fixture")
    rows = []
    for repetition in range(REPETITIONS):
        row = {"repetition": repetition, **child(args)}
        rows.append(row)
        (args.output / f"sample-{repetition}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"{args.scenario}-{args.size}/{args.metric}: {repetition + 1}/5 error={bool(row.get('error'))}",
            flush=True,
        )
    if inputs != fingerprints(args.input):
        raise ValueError("immutable source fixture changed")
    for name, digest in scripts.items():
        if (
            hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            != digest
        ):
            raise ValueError(f"measurement script changed: {name}")
    source_manifest = args.source.parent / "source-manifest.json"
    manifest = {
        "command": sys.argv,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "worker_timeout_seconds": TIMEOUT_SECONDS,
        "source": str(args.source.resolve()),
        "script_sha256": scripts,
        "inputs": inputs,
        "source_manifest_sha256": hashlib.sha256(
            source_manifest.read_bytes()
        ).hexdigest()
        if source_manifest.is_file()
        else None,
        "source_identity": "each sample records and verifies every loaded production module path and SHA256",
        "scenario": args.scenario,
        "size": args.size,
        "metric": args.metric,
        "scope": "imports and fixture copy excluded; timing steps exclude cleanup; read/memory scope includes action construction and cleanup",
        "not_measured": [
            "network",
            "model_tokens",
            "model_wait",
            "external_tools",
            "human_wait",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return int(any(row.get("error") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
