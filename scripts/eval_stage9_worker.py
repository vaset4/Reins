"""【阶段九评估】【隔离执行】从指定整份源码运行正式主循环，禁止混用旧新生产模块。

作者：xxx
时间：2026-10-01 14:30:00
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

EVALUATION_RUN_MAX_STEPS = 100
EVALUATION_RUN_MAX_TOKENS = 10000000
ROUND_PROCESS_PARAGRAPHS = 100
MODEL_TIMEOUT_SECONDS = 90


def install_source(source: Path) -> None:
    """在导入生产包前固定整份源码；参数：快照根；返回：无。"""
    source = source.resolve()
    harness = Path(__file__).resolve().parent
    checkout = harness.parent
    sys.path[:] = [
        str(source),
        str(harness),
        *(
            item
            for item in sys.path
            if item and Path(item).resolve() not in {source, harness, checkout}
        ),
    ]
    sys.dont_write_bytecode = True


def imported_sources(source: Path) -> dict[str, str]:
    """核对所有已导入生产模块来自同一源码；参数：快照；返回：实际模块位置。"""
    packages = {
        "app",
        "approval",
        "artifacts",
        "context",
        "llm",
        "memory",
        "runtime",
        "schedules",
        "scripts",
        "skills",
        "tasks",
        "tools",
    }
    result = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.split(".")[0] not in packages or not filename:
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(source.resolve()):
            raise RuntimeError(f"mixed production import: {name}: {path}")
        result[name] = path.relative_to(source.resolve()).as_posix()
    return result


def run_turn(
    client: Any, registry: Any, root: Path, *, session: str, task_id: str, text: str
) -> dict[str, Any]:
    """通过真实AgentLoop提交隔离输入；参数：运行依赖与输入；返回：终态、耗时和来源身份。"""
    from runtime.agent_loop import AgentLoop
    from runtime.lease import from_trigger
    from runtime.session_message_store import SessionMessageStore
    from runtime.types import RunContext, Trigger

    lease = from_trigger(
        "user",
        task_id=task_id,
        capabilities={
            "fs": {"project_root": str(root.parent / "workspace"), "read": [str(root)]},
            "background_run": {"enabled": False},
        },
        max_steps=EVALUATION_RUN_MAX_STEPS,
        max_tokens=EVALUATION_RUN_MAX_TOKENS,
    )
    context = RunContext(
        task_id=task_id,
        session_id=session,
        trigger=Trigger.USER,
        payload={"message": text},
        capability_lease=lease,
    )
    # 1. 【阶段九评估】【输入接纳】每轮输入先保存原件，再由正式主循环交付；摘要辅助调用不生成输入
    context.payload["input_message_id"] = (
        SessionMessageStore(root)
        .accept_input(
            session,
            text,
            run_id=context.run_id,
            task_id=task_id,
        )
        .entry_id
    )
    loop = AgentLoop(root, llm_client=client, tool_registry=registry)
    started = time.perf_counter()
    state = loop.run(context)
    return {
        "state": state.value,
        "output": loop.last_output,
        "run_id": context.run_id,
        "input_message_id": context.payload["input_message_id"],
        "foreground_wait_seconds": time.perf_counter() - started,
    }


def collect(root: Path, session: str, *, candidate: bool) -> dict[str, Any]:
    """采集真实摘要、原文回查和后台状态；参数：隔离来源；返回：可核对证据。"""
    from runtime.session_compaction import SessionCompactionStore
    from runtime.session_message_store import SessionMessageStore
    from runtime.tool_operations import ToolOperationStore

    owner = SessionMessageStore(root)
    summaries = SessionCompactionStore(owner).chain(owner.materialize(session))
    operations = ToolOperationStore(root).for_session(session)
    reads = [
        row
        for row in operations
        if row["call"]["tool_name"]
        in {"read_history", "read_artifact", "memory_query", "skill_read"}
    ]
    result = {
        "summaries": [asdict(item) for item in summaries],
        "original_read_count": len(reads),
        "original_reads": reads,
        "background": {
            "status": "not_exercised",
            "seconds": None,
            "reason": "controlled comparison disables background admission; no scheduler is started",
        },
    }
    if candidate:
        result["four_level_summary_count"] = sum(
            bool(item.content and item.content.segments) for item in summaries
        )
    return result


def evaluate(config: dict[str, Any], client: Any) -> dict[str, Any]:
    """运行固定语义案例与多轮正式压缩；参数：实验配置和计量客户端；返回：逐轮判据。"""
    from scripts.eval_long_context import (
        continuation_question,
        result_fields,
        seed_case,
    )
    from scripts.eval_long_context_scenarios import history_registry
    from scripts.long_context_cases import CASES
    from llm.types import LLMPlan
    from runtime.session_messages import append_assistant_message
    from runtime.workspaces import WorkspaceStore
    from tasks.store import TaskStore

    save = importlib.import_module("eval_stage9_meter").save

    output = Path(config["sample_output"])
    root, workspace = output / "data", output / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    case = next(item for item in CASES if item.name == config["case"])
    WorkspaceStore(root).bind_session(case.name, workspace)
    seed_case(root, case)
    tasks = TaskStore(root)
    task = tasks.create_task("验证对话中的条件与决定")
    tasks.close()
    rounds: list[dict[str, Any]] = []
    registry = history_registry()
    try:
        for number in range(config["rounds"]):
            compact = None
            if config["mode"] == "controlled":
                compact = run_turn(
                    client,
                    registry,
                    root,
                    session=case.name,
                    task_id=task.task_id,
                    text="/compact",
                )
                save(output / f"compact-{number + 1:03d}.json", compact)
                if compact["state"] != "DONE":
                    raise RuntimeError(
                        "formal compaction run failed; inspect compact and call evidence"
                    )
            turn = run_turn(
                client,
                registry,
                root,
                session=case.name,
                task_id=task.task_id,
                text=continuation_question(case),
            )
            evidence = collect(
                root, case.name, candidate=config["variant"] == "candidate"
            )
            fields = result_fields(
                LLMPlan(final_output=turn["output"]), dict(case.expected)
            )
            structural = (
                config["mode"] != "controlled"
                or config["variant"] != "candidate"
                or evidence["four_level_summary_count"] >= number + 1
            )
            item = {
                "round": number + 1,
                "turn": turn,
                "compaction_turn": compact,
                "evaluation": fields,
                "passed": turn["state"] == "DONE"
                and bool(fields["passed"])
                and structural,
                **evidence,
            }
            rounds.append(item)
            save(output / f"round-{number + 1:03d}.json", item)
            if number + 1 < config["rounds"]:
                append_assistant_message(
                    root,
                    case.name,
                    "继续核对已有资料，没有新的决定或执行结果。\n"
                    * ROUND_PROCESS_PARAGRAPHS,
                )
    finally:
        registry.close()
    return {
        "passed": all(item["passed"] for item in rounds),
        "rounds": rounds,
        "case": asdict(case),
    }


def main() -> int:
    """读取父进程冻结配置后加载唯一源码；参数：CLI；返回：真实评估状态。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source, output = Path(config["source_root"]), Path(config["sample_output"])
    install_source(source)
    from app.cli import build_llm_client
    from scripts.long_context_evidence import model_manifest

    meter = importlib.import_module("eval_stage9_meter")
    save = meter.save

    started = time.perf_counter()
    try:
        options = {
            "profile_name": config["profile"],
            "max_output_tokens": config["per_call_output_tokens"],
            "timeout_seconds": MODEL_TIMEOUT_SECONDS,
        }
        if config["mode"] == "controlled":
            options["context_window"] = config["window"]
        client = build_llm_client(options, project_root=source)
        target = getattr(client, "resolved_target", None)
        if target is None:
            raise ValueError("a real configured production profile is required")
        save(
            output / "model.json",
            {**model_manifest(target), "reasoning_effort": target.reasoning_effort},
        )
        measured = meter.MeteredClient(
            client,
            output=output / "calls",
            budget=meter.EvaluationBudget(Path(config["budget"])),
        )
        result = evaluate(config, measured)
    except Exception as exc:
        result = {"passed": False, "error_type": type(exc).__name__, "error": str(exc)}
    try:
        result["imported_sources"] = imported_sources(source)
    except RuntimeError as exc:
        result.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    result.update(
        elapsed_seconds=time.perf_counter() - started,
        metrics=meter.call_metrics(output / "calls"),
    )
    save(output / "result.json", result)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
