"""离线测量会话存储成本，原始重复结果供结构迁移前后比较。

作者：xxx
时间：2026-09-28 18:30:00
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, replace
from contextlib import ExitStack
from itertools import count
from unittest.mock import patch
from uuid import UUID

from runtime.file_records import record_key
from pathlib import Path
from time import perf_counter
from llm.messages import (
    AgentMessage,
    AssistantMessage,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
)
from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionEntry, SessionMessageStore
from runtime.session_runtime import SessionRun, SessionRuntime
from runtime.tool_operations import ToolOperation, ToolOperationStore
from runtime.types import RunToolsRequest
from scripts.benchmark_session_metrics import WorkingSetSampler, measure_reads

SESSION_ID = "session-benchmark"
FIXED_TIME = "2026-09-28T18:30:00+08:00"
BODY = "合成会话内容用于测量读取成本。" * 16
REPETITIONS = 5
WORKER_TIMEOUT_SECONDS = 60
MESSAGE_GROUP_SIZE = 4
SCALES = {
    "messages": (100, 1000, 10000),
    "runs": (10, 100, 1000),
    "operations": (100, 1000, 10000),
}


def write_json(path: Path, payload: object) -> None:
    """独占保存测量证据；传参：新路径和可序列化值；返回：无，已有文件拒绝覆盖。"""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def message_entry(index: int) -> SessionEntry:
    """生成固定工具配对的合法消息；传参：零起始序号；返回：生产消息Entry。"""
    entry_id, call_id = f"entry-{index:05d}", f"call-{index // MESSAGE_GROUP_SIZE:05d}"
    position = index % MESSAGE_GROUP_SIZE
    if position == 0:
        message: AgentMessage = UserMessage(entry_id, (TextPart(BODY),))
    elif position == 1:
        message = AssistantMessage(
            entry_id, (ToolCallPart(call_id, "probe", {"text": BODY}),)
        )
    elif position == 2:
        message = ToolResultMessage(
            entry_id, call_id, "probe", (TextPart(BODY),), "success"
        )
    else:
        message = AssistantMessage(entry_id, (TextPart(BODY),))
    return SessionEntry(
        type="inbound" if position == 0 else "message",
        entry_id=entry_id,
        parent_id=f"entry-{index - 1:05d}" if index else None,
        session_id=SESSION_ID,
        timestamp=FIXED_TIME,
        message=message,
        input_source="user" if position == 0 else None,
    )


def generate_fixture(
    root: Path, dimension: str, size: int, *, visible_inputs: bool = False
) -> dict[str, str]:
    """生成隔离且确定的正式存储样本；参数：新根、维度、规模、输入是否已投影；返回：原件哈希。"""
    root.mkdir(parents=True, exist_ok=False)
    identities = count(1)
    # 1. 【会话基准】【确定性输入】仅冻结合成样本的身份和时间，所有字节通过正式提交生成
    with ExitStack() as patches:
        for module in ("runtime.persistence", "runtime.file_journal"):
            patches.enter_context(
                patch(f"{module}.uuid4", side_effect=lambda: UUID(int=next(identities)))
            )
        patches.enter_context(
            patch(
                "runtime.run_facts.new_ulid",
                side_effect=lambda: f"fact-{next(identities):08d}",
            )
        )
        patches.enter_context(
            patch("runtime.tool_operations.utc_now", return_value=FIXED_TIME)
        )
        messages = SessionMessageStore(root)
        messages.create_session(SESSION_ID, created_at=FIXED_TIME)
        total = size if dimension == "messages" else MESSAGE_GROUP_SIZE
        with messages.database.transaction() as batch:
            for index in range(total):
                entry = message_entry(index)
                if visible_inputs and entry.type == "inbound":
                    entry = replace(entry, type="message", input_source=None)
                batch.put(
                    "session_entry",
                    record_key(SESSION_ID, entry.entry_id),
                    entry.to_mapping(),
                    session_id=SESSION_ID,
                )
            header = batch.get("session", SESSION_ID)
            assert header is not None
            batch.put(
                "session",
                SESSION_ID,
                {**header, "leaf_id": entry.entry_id},
                session_id=SESSION_ID,
            )
        facts = RunFactStore(root)
        operations = ToolOperationStore(root)
        with messages.database.transaction():
            for index in range(size if dimension == "runs" else 1):
                identity = {
                    "session_id": SESSION_ID,
                    "run_id": f"run-{index:05d}",
                    "ts": FIXED_TIME,
                }
                facts.append({**identity, "event": "run:start"})
                facts.append(
                    {**identity, "event": "input:handled", "input_ids": ["entry-00000"]}
                )
                facts.append(
                    {**identity, "event": "run:lifecycle", "lifecycle": "done"}
                )
            for index in range(size if dimension == "operations" else 1):
                call = ToolOperation(
                    RunToolsRequest(
                        "probe", tool_name="probe", arguments={"text": BODY}
                    ),
                    f"call-{index:05d}",
                    "probe",
                    {"text": BODY},
                    operation_id=f"op-{index:05d}",
                )
                operations.write(
                    {
                        "session_id": SESSION_ID,
                        "run_id": "run-00000",
                        "operation_id": f"op-{index:05d}",
                    },
                    {
                        "state": ("completed", "not_started", "unknown")[index % 3],
                        "call": asdict(call),
                        "text": BODY,
                    },
                )
        # 2. 【会话基准】【核验夹具】正式读取器核对完整会话及调用配对，派生索引不属于冻结输入
        assert len(messages.materialize(SESSION_ID).entries) == total
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_relative_to(root / "runtime")
    }


def unexpected_run(request: SessionRun) -> None:
    """禁止读取基准意外启动执行；传参：意外运行；返回：无，直接暴露错误。"""
    raise AssertionError(f"benchmark unexpectedly dispatched {request.input_id}")


def measure_worker(root: Path, dimension: str, size: int) -> list[dict[str, object]]:
    """在新进程测首次及重复读取；传参：固定夹具/维度/规模；返回：含失败的原始观测。"""
    messages, facts = SessionMessageStore(root), RunFactStore(root)
    operations = ToolOperationStore(root)
    runtime = SessionRuntime(
        SESSION_ID, messages=messages, facts=facts, run=unexpected_run
    )
    rows: list[dict[str, object]] = []
    for phase in ("first_in_process", "repeat_in_process"):
        started = perf_counter()
        error: str | None = None
        count = -1
        try:
            if dimension == "messages":
                count = len(messages.materialize(SESSION_ID).entries)
                expected = size
            elif dimension == "runs":
                count = len(runtime._unhandled_inputs())
                expected = 0
            else:
                count = len(operations.for_session(SESSION_ID))
                expected = size
            if count != expected:
                raise AssertionError(f"expected {expected}, got {count}")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "phase": phase,
                "seconds": perf_counter() - started,
                "result_count": count,
                "error": error,
            }
        )
    runtime.close()
    return rows


def collect_samples(
    root: Path, dimension: str, size: int, *, metric: str = "timing"
) -> list[dict[str, object]]:
    """每次启动独立进程并保留超时；传参：夹具/维度/规模；返回：全部成功失败样本。"""
    rows: list[dict[str, object]] = []
    for repetition in range(REPETITIONS):
        command = [
            sys.executable,
            "-m",
            "scripts.benchmark_session_runtime",
            "--worker",
            str(root.resolve()),
            "--dimension",
            dimension,
            "--size",
            str(size),
            "--metric",
            metric,
        ]
        started = perf_counter()
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=WORKER_TIMEOUT_SECONDS,
                check=False,
                cwd=Path(__file__).resolve().parents[1],
            )
            if result.returncode:
                raise RuntimeError(result.stderr or result.stdout)
            observations = json.loads(result.stdout)
            rows.extend({"repetition": repetition, **row} for row in observations)
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "repetition": repetition,
                    "phase": "worker_failure",
                    "seconds": perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return rows


def instrumented_worker(
    root: Path, dimension: str, size: int, metric: str
) -> list[dict[str, object]]:
    """分别运行读量或内存采样；传参：固定夹具/维度/规模/指标；返回：含原观测的计量证据。"""
    if metric == "reads":
        with measure_reads(root) as meter:
            observations = measure_worker(root, dimension, size)
        measurement: dict[str, object] = {"counts": asdict(meter.snapshot())}
    else:
        sampler = WorkingSetSampler()
        sampler.start()
        try:
            observations = measure_worker(root, dimension, size)
        finally:
            memory = sampler.finish()
        measurement = {"memory": memory}
    return [
        {
            "phase": f"{metric}_two_reads",
            **measurement,
            "error": next((row["error"] for row in observations if row["error"]), None),
            "observations": observations,
        }
    ]


def collect_read_measurements(
    source: Path,
    output: Path,
    *,
    metric: str,
    selection: tuple[str | None, int | None],
) -> int:
    """复用已冻结数据采集读取量；传参：计时数据目录/新输出目录；返回：存在失败时为1。"""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    selected = {
        label: fixture
        for label, fixture in manifest["inputs"].items()
        if (selection[0] is None or label.rsplit("-", 1)[0] == selection[0])
        and (selection[1] is None or int(label.rsplit("-", 1)[1]) == selection[1])
    }
    if not selected:
        raise ValueError(f"selection matches no frozen input: {selection}")
    output.mkdir(parents=True, exist_ok=False)
    script_fingerprints = {}
    for name in ("benchmark_session_runtime.py", "benchmark_session_metrics.py"):
        content = Path(__file__).with_name(name).read_bytes()
        (output / name).write_bytes(content)
        script_fingerprints[name] = hashlib.sha256(content).hexdigest()
    failed = False
    for label, fixture in selected.items():
        dimension, size = label.rsplit("-", 1)
        root = source / label
        for name, digest in fixture["sha256"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"benchmark fixture changed: {label}/{name}")
        rows = collect_samples(root, dimension, int(size), metric=metric)
        write_json(output / f"{label}-{metric}.json", rows)
        failed = failed or any(row.get("error") for row in rows)
        print(f"{label}: {len(rows)} independent {metric} samples", flush=True)
        for name, digest in script_fingerprints.items():
            if (
                hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                != digest
            ):
                raise ValueError(f"measurement code changed during collection: {name}")
    write_json(
        output / "manifest.json",
        {
            "source_manifest_sha256": hashlib.sha256(
                (source / "manifest.json").read_bytes()
            ).hexdigest(),
            "source": str(source.resolve()),
            "command": sys.argv,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "script_sha256": script_fingerprints,
            "metric": metric,
            "selection": selection,
            "scope": (
                "two Store reads per sample; constructors excluded"
                if metric == "timing"
                else "two Store reads per sample; constructors included; all json.loads attempts counted"
            ),
            "bytes": "raw file bytes before decoding and newline translation; not physical disk IO",
            "buffer_reads": "raw buffer fills including EOF reads; instrumentation uses default IO buffer size",
            "timing": (
                "uninstrumented Store reading and projection"
                if metric == "timing"
                else "instrumented observations only; do not merge with uninstrumented latency"
            ),
        },
    )
    return int(failed)


def main() -> int:
    """运行离线存储基准；传参：CLI；返回：全样本成功为0，失败为1。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--input",
        type=Path,
        help="reuse immutable fixtures for timing, reads or memory",
    )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dimension", choices=tuple(SCALES))
    parser.add_argument("--size", type=int)
    parser.add_argument(
        "--metric", choices=("timing", "reads", "memory"), default="timing"
    )
    args = parser.parse_args()
    if args.worker:
        if args.dimension is None or args.size is None:
            parser.error("worker requires dimension and size")
        if args.metric != "timing":
            print(
                json.dumps(
                    instrumented_worker(
                        args.worker, args.dimension, args.size, args.metric
                    )
                )
            )
        else:
            print(json.dumps(measure_worker(args.worker, args.dimension, args.size)))
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.input is not None:
        return collect_read_measurements(
            args.input,
            args.output,
            metric=args.metric,
            selection=(args.dimension, args.size),
        )
    if args.metric != "timing":
        parser.error("instrumented metrics require --input")
    if args.size is not None and (
        args.dimension is None or args.size not in SCALES[args.dimension]
    ):
        parser.error("--size requires a dimension and one of its planned scales")
    args.output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    inputs: dict[str, object] = {}
    for dimension, scales in SCALES.items():
        if args.dimension and args.dimension != dimension:
            continue
        for size in scales:
            if args.size and args.size != size:
                continue
            label = f"{dimension}-{size}"
            started = perf_counter()
            fingerprints = generate_fixture(args.output / label, dimension, size)
            inputs[label] = {
                "sha256": fingerprints,
                "generation_seconds": perf_counter() - started,
            }
            samples = collect_samples(args.output / label, dimension, size)
            rows.extend(
                {"dimension": dimension, "size": size, **sample} for sample in samples
            )
            write_json(args.output / f"{label}-results.json", samples)
            print(f"{label}: {len(samples)} observations", flush=True)
    write_json(
        args.output / "manifest.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "command": sys.argv,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "inputs": inputs,
            "repetitions": REPETITIONS,
            "body_utf8_bytes": len(BODY.encode("utf-8")),
            "seed": "deterministic index sequence; no RNG",
            "facts_per_run": 3,
            "timing": "store read and projection only; constructors/imports/generation excluded",
            "not_measured": [
                "model",
                "external_tools",
                "human_wait",
                "request_preparation",
                "background_sessions",
                "file_read_counts",
                "process_memory",
            ],
            "cache_note": "first process read does not imply empty OS file cache",
        },
    )
    return int(any(row.get("error") for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
