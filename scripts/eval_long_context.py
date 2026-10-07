"""通过生产请求适配器留存长会话压缩和接续的真实模型对照。

作者：xxx
时间：2026-09-25 12:00:00
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from app.cli import build_llm_client
from context.compaction import CompactionMaterial
from context.production_builder import ProductionContextBundle
from context.window import request_budget
from llm.client import RealLLMClient
from llm.messages import agent_message_to_mapping
from llm.model_request import ComposedRequest, model_request_to_mapping
from llm.resolved_target import ResolvedModelTarget
from llm.types import LLMPlan, usage_to_mapping
from runtime.context_preparation import ContextCompactor
from runtime.session_compaction import SessionCompactionStore, SummarySource
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import (
    ToolExchange,
    append_assistant_message,
    append_tool_exchange,
    append_user_message,
)
from scripts.long_context_cases import CASES, ContextCase
from scripts.long_context_baseline import load_baseline
from scripts.long_context_evidence import (
    evaluation_source_files,
    model_manifest,
    snapshot_sources,
)
from tools.tool_registry import ToolRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = evaluation_source_files(PROJECT_ROOT)
MODEL_TIMEOUT_SECONDS = 90
MODEL_OUTPUT_TOKENS = 16384
DEFAULT_REPEATS = 2
DEFAULT_ROUNDS = 2
DISTRACTION_PARAGRAPHS = 12


def write_json(path: Path, value: object) -> None:
    """完整写入后发布评测证据，失败保留上一进度；传参：文件和内容；返回：无，IO错误直接暴露。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class RecordedModel:
    """包装已配置的生产客户端；传参：客户端和独立证据目录；返回：可记录真实调用的实例。"""

    def __init__(self, client: RealLLMClient, output: Path) -> None:
        """注入模型与证据位置；传参：生产客户端、目录；返回：无。"""
        self.client = client
        self.output = output
        self.calls = max(
            (int(path.stem.split("-")[1]) for path in output.glob("call-*.json")),
            default=0,
        )

    def invoke(self, bundle: ProductionContextBundle) -> LLMPlan:
        """保存实际请求、响应和已报告用量；传参：请求材料；返回：真实计划，错误不伪装为成功。"""
        self.calls += 1
        context = bundle.model_context
        prepared = context.get("prepared_request")
        if not isinstance(prepared, ComposedRequest):
            prepared = self.client.prepare_request(bundle.model_task, context)
        request_path = self.output / f"call-{self.calls:03d}.json"
        started = time.perf_counter()
        evidence = {
            "purpose": context.get("context_purpose", "continuation"),
            "request": model_request_to_mapping(prepared.request),
            "budget": request_budget(
                prepared.request, prepared.context_window
            ).evidence(),
        }
        write_json(request_path, evidence)
        plan = self.client.plan(
            bundle.model_task, context={**context, "prepared_request": prepared}
        )
        write_json(
            request_path,
            {
                **evidence,
                "elapsed_seconds": time.perf_counter() - started,
                "request_id": plan.request_id,
                "output": plan.final_output,
                "response": plan.raw_model_response,
                "error": asdict(plan.model_error) if plan.model_error else None,
                "attempts": [
                    {
                        "attempt_id": event.attempt_id,
                        "phase": event.phase,
                        "usage": usage_to_mapping(event.usage),
                        "elapsed_ms": event.elapsed_ms,
                    }
                    for event in plan.model_attempts
                ],
            },
        )
        return plan


def seed_case(root: Path, case: ContextCase) -> SessionMessageStore:
    """按真实角色保存合成输入与工具配对；传参：隔离目录和场景；返回：会话所有者。"""
    for index, (role, text) in enumerate(case.turns):
        if role == "user":
            append_user_message(root, case.name, text)
        elif role == "assistant":
            append_assistant_message(root, case.name, text)
        else:
            append_tool_exchange(
                root,
                case.name,
                ToolExchange(
                    f"synthetic-{index}",
                    "inspect_local",
                    rendered=text,
                    status="error" if role == "tool_error" else "ok",
                ),
            )
    append_assistant_message(
        root,
        case.name,
        "\n".join(
            f"附件页{index}：这是已完成目录核对的过程说明，没有新增操作或决定。" * 3
            for index in range(DISTRACTION_PARAGRAPHS)
        ),
    )
    append_user_message(root, case.name, continuation_question(case))
    return SessionMessageStore(root)


def continuation_question(case: ContextCase) -> str:
    """明确机器判据的字段类型而不泄露答案；传参：场景；返回：追问正文。"""
    return (
        case.question
        + "是否类字段必须使用JSON布尔值true或false，数字字段使用数字，不使用字符串代替。"
    )


def result_fields(plan: LLMPlan, expected: Mapping[str, object]) -> dict[str, object]:
    """按预先登记字段比较后续行为；传参：模型结果与独立判据；返回：逐字段结果与明确失败。"""
    if (
        plan.model_error is not None
        or plan.run_tools_request is not None
        or not plan.final_output
    ):
        return {
            "passed": False,
            "reason": "model_call_failed_or_not_final",
            "model_error": asdict(plan.model_error)
            if plan.model_error is not None
            else None,
        }
    try:
        parsed = json.loads(plan.final_output)
    except json.JSONDecodeError:
        return {
            "passed": False,
            "reason": "continuation_not_json",
            "output": plan.final_output,
        }
    if not isinstance(parsed, dict):
        return {"passed": False, "reason": "continuation_not_object"}
    comparisons = {
        key: {
            "expected": value,
            "actual": parsed.get(key),
            "matched": parsed.get(key) == value,
        }
        for key, value in expected.items()
    }
    return {
        "passed": all(item["matched"] for item in comparisons.values()),
        "fields": comparisons,
    }


def run_case(
    client: RealLLMClient,
    case: ContextCase,
    output: Path,
    *,
    strategy: str,
    rounds: int,
    baseline_source: Path | None = None,
    resume: bool = False,
) -> bool:
    """执行或接续同一独立样本，已发布摘要不重做；传参：模型、场景、策略与证据目录；返回：所有已完成轮次是否正确。"""
    target = client.resolved_target
    if target is None:
        raise ValueError("evaluation requires a resolved production target")
    progress_path = output / "progress.json"
    if resume and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        data_root = Path(progress["data_root"])
        owner = SessionMessageStore(data_root)
    else:
        data_root = Path(tempfile.mkdtemp(prefix=f"reins-context-{case.name}-"))
        owner = seed_case(data_root, case)
        progress = {
            "data_root": str(data_root),
            "next_round": 0,
            "phase": "compact",
            "rounds": [],
            "elapsed_seconds": 0.0,
        }
        write_json(progress_path, progress)
    store = SessionCompactionStore(owner)
    model = RecordedModel(client, output)
    context = {
        "session_id": case.name,
        "run_id": "synthetic-evaluation",
        "segment_id": "evaluation",
        "tool_registry": ToolRegistry(),
        "system_prompt": "根据可见证据继续当前工作，保留未知与要求的区别。",
    }
    snapshots: list[dict[str, Any]] = progress["rounds"]
    total_rounds = rounds if strategy != "full" else 1
    for index in range(progress["next_round"], total_rounds):
        started = time.perf_counter()
        view = owner.materialize(case.name)
        current = store.current(view)
        # 1. 【评测】【断点接续】摘要可能先于进度落盘，以已提交覆盖范围识别完成，避免重复生成
        published = (
            resume
            and current is not None
            and current.message_ids
            == tuple(message.message_id for message in view.messages[:-1])
        )
        if strategy != "full" and progress["phase"] == "compact":
            if not published:
                _compact_round(
                    store,
                    model,
                    context,
                    case_name=case.name,
                    strategy=strategy,
                    baseline_source=baseline_source,
                )
        progress["phase"] = "continuation"
        write_json(progress_path, progress)
        current = store.current(view)
        messages = (
            view.messages[len(current.message_ids) :] if current else view.messages
        )
        prepared_context = {
            **context,
            "conversation_history": messages,
            "input_message_id": view.messages[-1].message_id,
        }
        if current:
            prepared_context["session_summary"] = current.model_view()
        plan = model.invoke(
            ProductionContextBundle(continuation_question(case), prepared_context, ())
        )
        if plan.model_error is not None:
            raise ValueError(
                f"continuation model failed: {plan.model_error.category}: {plan.model_error.summary}"
            )
        evaluation = result_fields(plan, dict(case.expected))
        snapshots.append(
            {
                "round": index + 1,
                "result": evaluation,
                "summary": asdict(current) if current else None,
                "call_count": model.calls,
            }
        )
        progress.update(
            next_round=index + 1,
            phase="compact",
            elapsed_seconds=progress["elapsed_seconds"] + time.perf_counter() - started,
        )
        write_json(output / f"round-{index + 1:03d}.json", snapshots[-1])
        print(
            json.dumps(
                {
                    "case": case.name,
                    "strategy": strategy,
                    "round": index + 1,
                    "passed": evaluation["passed"],
                    "calls": model.calls,
                }
            ),
            flush=True,
        )
        if index + 1 < total_rounds:
            if plan.final_output is not None:
                append_assistant_message(data_root, case.name, plan.final_output)
            append_assistant_message(
                data_root,
                case.name,
                "继续核对过程记录，尚无新决定。\n" * DISTRACTION_PARAGRAPHS,
            )
            append_user_message(data_root, case.name, continuation_question(case))
        write_json(progress_path, progress)
        write_json(
            output / "result.json",
            {
                "strategy": strategy,
                "case": asdict(case),
                "data_root": str(data_root),
                "rounds": snapshots,
                "elapsed_seconds": progress["elapsed_seconds"],
            },
        )
    return all(bool(snapshot["result"]["passed"]) for snapshot in snapshots)


def _compact_round(
    store: SessionCompactionStore,
    model: RecordedModel,
    context: Mapping[str, object],
    *,
    case_name: str,
    strategy: str,
    baseline_source: Path | None,
) -> None:
    """只发布一次完整摘要，失败保留原快照；传参：真实所有者、模型与冻结策略；返回：无。"""
    view = store.messages.materialize(case_name)
    previous = store.current(view)
    begin = len(previous.message_ids) if previous else 0
    count = len(view.messages) - 1
    text = json.dumps(
        [agent_message_to_mapping(item) for item in view.messages[begin:count]],
        ensure_ascii=False,
    )
    source = SummarySource(view, count, previous)
    if strategy == "baseline":
        if baseline_source is None:
            raise ValueError("baseline requires frozen source")
        baseline = load_baseline(baseline_source)(
            store, model.client.prepare_request, model.invoke
        )
        summary_text, ids = baseline._summarize(
            CompactionMaterial(source, text),
            context,
            context_window=model.client.context_window,
        )
        store.publish(source, summary_text, request_ids=ids)
        return
    compactor = ContextCompactor(store, model.client.prepare_request, model.invoke)
    content, ids = compactor._summarize(
        CompactionMaterial(source, text),
        context,
        context_window=model.client.context_window,
        verify_originals=strategy != "incremental",
    )
    store.publish(source, content.render(), request_ids=ids, content=content)


def main() -> int:
    """运行指定案例并保存代码/输入指纹；传参：CLI 参数；返回：工具执行状态，行为成败见各结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        help="Named provider:model profile; does not change the active profile.",
    )
    parser.add_argument(
        "--strategy",
        choices=("current", "incremental", "full", "baseline"),
        required=True,
    )
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same frozen sample after an explicitly recorded failed call.",
    )
    parser.add_argument("--case", choices=tuple(case.name for case in CASES))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.rounds < 1
        or (args.output.exists() and not args.resume)
    ):
        raise ValueError(
            "positive repeats/rounds and a new evidence directory are required"
        )
    if args.strategy == "baseline" and args.baseline_source is None:
        raise ValueError(
            "baseline requires --baseline-source from the pre-change capture"
        )
    overrides: dict[str, object] = {
        "timeout_seconds": MODEL_TIMEOUT_SECONDS,
        "max_output_tokens": MODEL_OUTPUT_TOKENS,
    }
    if args.profile is not None:
        overrides["profile_name"] = args.profile
    client = build_llm_client(overrides, project_root=PROJECT_ROOT)
    if not isinstance(client, RealLLMClient) or client.resolved_target is None:
        raise RuntimeError("real model is not configured")
    target = client.resolved_target
    _evaluation_manifest(args, target)
    failures = 0
    for case in CASES:
        if args.case and case.name != args.case:
            continue
        for repeat in range(args.repeats):
            output = args.output / f"{case.name}-{repeat + 1}"
            try:
                passed = run_case(
                    client,
                    case,
                    output,
                    strategy=args.strategy,
                    rounds=args.rounds,
                    baseline_source=args.baseline_source,
                    resume=args.resume,
                )
                failures += int(not passed)
            except Exception as exc:
                failures += 1
                identity = len(list(output.glob("failure-*.json"))) + 1
                write_json(
                    output / f"failure-{identity:03d}.json",
                    {"error_type": type(exc).__name__, "error": str(exc)},
                )
                print(
                    json.dumps(
                        {
                            "case": case.name,
                            "repeat": repeat + 1,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return int(failures > 0)


def _evaluation_manifest(args: argparse.Namespace, target: ResolvedModelTarget) -> None:
    """固定模型、输入和代码指纹，接续实验不能混入新策略；传参：CLI与解析目标；返回：无。"""
    fingerprints = snapshot_sources(
        PROJECT_ROOT, SOURCE_FILES, args.output / "source", resume=args.resume
    )
    baseline_fingerprints = {}
    if args.baseline_source:
        load_baseline(args.baseline_source)
        for name in ("context/compaction.py", "runtime/context_preparation.py"):
            data = (args.baseline_source / name).read_bytes()
            baseline_fingerprints[name] = hashlib.sha256(data).hexdigest()
            destination = args.output / "frozen_baseline" / name
            if not args.resume:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
    manifest = {
        **model_manifest(target),
        "strategy": args.strategy,
        "source_sha256": fingerprints,
        "cases_sha256": hashlib.sha256(
            json.dumps([asdict(case) for case in CASES], ensure_ascii=False).encode()
        ).hexdigest(),
        "synthetic_only": True,
        "scope": "production summary generation/publication and continuation; fit policy tested separately",
        "baseline_source": str(args.baseline_source) if args.baseline_source else None,
        "baseline_sha256": baseline_fingerprints,
        "rounds": args.rounds,
        "repeats": args.repeats,
        "selected_case": args.case,
        "output_allowance_note": "4096-token runs hit the actual reasoning/output ceiling; comparison groups share 16384 and retain earlier failures.",
    }
    path = args.output / "manifest.json"
    if args.resume:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                "resume requires identical model, source, inputs and comparison parameters"
            )
        return
    write_json(path, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
