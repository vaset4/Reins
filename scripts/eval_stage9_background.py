"""【阶段九评估】【后台联合验收】默认只生成计划，显式执行才从完整冻结源码调用模型。

作者：xxx
时间：2026-10-01 20:10:00
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATTEMPTS = 28
DEFAULT_OUTPUT_TOKENS = 16384
CONTROLLED_WINDOW = 65536
MODEL_TIMEOUT_SECONDS = 90
GATE_TIMEOUT_SECONDS = 120
GATE_POLL_SECONDS = 0.05
RUN_MAX_STEPS = 100
RUN_MAX_TOKENS = 10000000
SCENARIOS = ("history", "knowledge")
KNOWLEDGE_STAGES = (
    (
        "new",
        "knowledge",
        "请长期记住：本项目对外服务端口固定为8000，属于项目约定。此次只简短确认收到。",
    ),
    (
        "no_op",
        "casual",
        "临时聊一句：刚才在窗边看到一只鸟。这不形成任务、偏好、决定或长期事实，不需要保存。只简短回应。",
    ),
    (
        "correction",
        "knowledge",
        "更正本项目服务端口约定：今后固定使用9000，之前8000的约定不再有效。请保留这项长期更正，此次只简短确认收到。",
    ),
)


@dataclass(frozen=True)
class Environment:
    """固定单个隔离案例的来源；参数：目录、会话、任务；返回：不可变运行位置。"""

    output: Path
    root: Path
    workspace: Path
    session: str
    task_id: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """解析明确模型与独立空间，默认不执行；参数：CLI参数；返回：验证后的计划选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS)
    )
    parser.add_argument("--window", type=int, default=CONTROLLED_WINDOW)
    parser.add_argument("--max-logical-calls", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument(
        "--per-call-output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_ATTEMPTS * DEFAULT_OUTPUT_TOKENS,
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if any(
        getattr(args, name) < 1
        for name in (
            "max_logical_calls",
            "max_attempts",
            "per_call_output_tokens",
            "max_output_tokens",
        )
    ):
        parser.error("all evaluation limits must be positive")
    if (
        args.window < CONTROLLED_WINDOW
        or args.per_call_output_tokens * 2 >= args.window
    ):
        parser.error(
            "window must be at least 65536 and leave room beyond generation plus audit output reserves"
        )
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("each scenario can be selected once")
    if not args.profile.strip():
        parser.error("an explicit named model profile is required")
    return args


def experiment_plan(args: argparse.Namespace) -> dict[str, Any]:
    """公布调用、证据和局限，不读凭据、不建目录；参数：选项；返回：可审阅计划。"""
    return {
        "schema_version": 1,
        "profile": args.profile,
        "scenarios": args.scenarios,
        "window": args.window,
        "output": str(args.output.resolve()),
        "source_snapshot": "all installable Python source and pyproject.toml",
        "max_logical_calls": args.max_logical_calls,
        "max_attempts": args.max_attempts,
        "per_call_output_tokens": args.per_call_output_tokens,
        "max_output_tokens": args.max_output_tokens,
        "budget_semantics": "one locked durable total across foreground, workers and every retry; full output reservation per attempt",
        "authorization": "proposed separate batch; does not reuse or increase a previously authorized batch",
        "estimated_calls_without_retries": {
            "history": "5-9",
            "knowledge": "not fixed; prior run used 23 before maintenance and fresh-session verification completed",
        },
        "history": "controlled admission through ContextCompactionJobs and real scheduler; hold first non-empty provider output delta while foreground input runs, then publish and verify adoption",
        "knowledge": "automatic new project rule, explicit no-op conversation, source correction and a fresh-session answer",
        "not_covered": [
            "natural window pressure",
            "uninterrupted network overlap timing",
            "host process death",
            "multi-workspace permissions",
            "cancellation during paid IO",
            "idle file-change verification",
            "real TUI interaction",
            "comparative speedup or provider cache benefit",
        ],
        "price": "unknown",
        "cost_amount": None,
    }


def execute_plan(args: argparse.Namespace, plan: dict[str, Any]) -> int:
    """冻结完整源码并逐案例启动独立进程；参数：明确执行选项/计划；返回：联合失败状态。"""
    from scripts.eval_stage9_meter import save
    from scripts.long_context_evidence import evaluation_source_files, snapshot_sources

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    hashes = snapshot_sources(
        PROJECT_ROOT, evaluation_source_files(PROJECT_ROOT), source, resume=False
    )
    save(output / "manifest.json", {**plan, "source_sha256": hashes})
    budget = output / "budget.json"
    save(
        budget,
        {
            "logical_calls": 0,
            "attempts": 0,
            "output_tokens": 0,
            "limit_exceeded": None,
            **{
                key: plan[key]
                for key in ("max_logical_calls", "max_attempts", "max_output_tokens")
            },
        },
    )
    results = []
    for scenario in args.scenarios:
        if json.loads(budget.read_text(encoding="utf-8"))["limit_exceeded"] is not None:
            break
        sample = output / scenario
        sample.mkdir()
        config = {
            **plan,
            "scenario": scenario,
            "source_root": str(source),
            "sample_output": str(sample),
            "budget": str(budget),
        }
        path = output / f"{scenario}-config.json"
        save(path, config)
        command = [
            sys.executable,
            "-I",
            "-B",
            str(source / "scripts/eval_stage9_background.py"),
            "--worker-config",
            str(path),
            "--execute",
        ]
        with (sample / "process.log").open("wb") as transcript:
            result = subprocess.run(
                command,
                cwd=source,
                stdout=transcript,
                stderr=subprocess.STDOUT,
                check=False,
            )
        results.append(
            {
                "scenario": scenario,
                "exit_code": result.returncode,
                "output": str(sample),
            }
        )
        save(output / "progress.json", {"results": results})
    complete = len(results) == len(args.scenarios)
    save(
        output / "result.json",
        {
            "results": results,
            "complete": complete,
            "budget": json.loads(budget.read_text(encoding="utf-8")),
        },
    )
    return int(not complete or any(row["exit_code"] for row in results))


class EvaluationClients:
    """每个执行者独立计量目录，全部角色共用既有持久总额度。"""

    def __init__(
        self, config: dict[str, Any], creator: Callable[[str, dict[str, object]], Any]
    ) -> None:
        """注入唯一模型装配器；参数：冻结配置/角色装配；返回：无，不提前调用模型。"""
        from scripts.eval_stage9_meter import EvaluationBudget

        self.config, self.creator = config, creator
        self.budget = EvaluationBudget(Path(config["budget"]))
        self.clients: dict[str, Any] = {}
        self.models: dict[str, Any] = {}

    def get(self, role: str, frozen: dict[str, object] | None = None) -> Any:
        """保存实际模型身份并保留所有辅助调用；参数：角色/调度冻结选型；返回：计量客户端。"""
        from scripts.eval_stage9_meter import MeteredClient, save
        from scripts.long_context_evidence import model_manifest

        if role in self.clients:
            return self.clients[role]
        options = {
            "profile_name": self.config["profile"],
            **(frozen or {}),
            "max_output_tokens": self.config["per_call_output_tokens"],
            "context_window": self.config["window"],
            "timeout_seconds": MODEL_TIMEOUT_SECONDS,
        }
        client = self.creator(role, options)
        target = client.resolved_target
        if target is None:
            raise ValueError("evaluation requires an explicit resolved model target")
        manifest = {
            **model_manifest(target),
            "reasoning_effort": target.reasoning_effort,
        }
        if self.models and manifest != next(iter(self.models.values())):
            raise ValueError("foreground and auxiliary model conditions differ")
        self.models[role] = manifest
        save(Path(self.config["sample_output"]) / "models.json", self.models)
        measured = MeteredClient(
            client,
            output=Path(self.config["sample_output"]) / role / "calls",
            budget=self.budget,
        )
        self.clients[role] = measured
        return measured


class StreamingGate:
    """只在评估中暂停首个供应商正文/思考增量，确定性观察前台能否独立继续。"""

    def __init__(self, client: Any, *, evidence_path: Path | None = None) -> None:
        """绑定已计量客户端；参数：后台模型；返回：无。"""
        self.client = client
        self.entered, self.release = Event(), Event()
        self.pause_seconds = 0.0
        self.evidence: dict[str, object] | None = None
        self.evidence_path = evidence_path

    def __getattr__(self, name: str) -> Any:
        """透传正式准备接口；参数：属性名；返回：原属性。"""
        return getattr(self.client, name)

    def plan_stream(self, task: str, context: Any = None) -> Any:
        """供应商已返回非空增量后等待前台完成，派发前通知不触发；参数：真实请求；返回：原模型流。"""
        from llm.types import ModelOutputDelta
        from scripts.eval_stage9_meter import save

        stream = self.client.plan_stream(task, context)
        try:
            while True:
                try:
                    event = next(stream)
                except StopIteration as done:
                    return done.value
                if (
                    self.entered.is_set()
                    or not isinstance(event, ModelOutputDelta)
                    or not event.text
                ):
                    yield event
                    continue
                started = time.perf_counter()
                self.evidence = {
                    "event": "ModelOutputDelta",
                    "channel": event.channel,
                    "observed_characters": len(event.text),
                }
                self.entered.set()
                try:
                    if not self.release.wait(GATE_TIMEOUT_SECONDS):
                        raise TimeoutError(
                            "foreground did not release the controlled background gate"
                        )
                finally:
                    self.pause_seconds = time.perf_counter() - started
                    if self.evidence_path is not None:
                        save(
                            self.evidence_path,
                            {
                                "trigger": self.evidence,
                                "controlled_pause_seconds": self.pause_seconds,
                                "released": self.release.is_set(),
                                "timing_note": "call/attempt elapsed includes this client-side pause",
                            },
                        )
                yield event
        finally:
            stream.close()


def environment(output: Path, session: str) -> Environment:
    """创建单案例隔离工作区和任务；参数：新证据目录/会话；返回：真实运行位置。"""
    from runtime.workspaces import WorkspaceStore
    from tasks.store import TaskStore

    root, workspace = output / "data", output / "workspace"
    workspace.mkdir(parents=True, exist_ok=False)
    WorkspaceStore(root).bind_session(session, workspace)
    with closing(TaskStore(root)) as tasks:
        task = tasks.create_task("阶段九后台联合验收", is_inbox=True)
    return Environment(output, root, workspace, session, task.task_id)


def primary_turn(
    env: Environment, client: Any, text: str
) -> tuple[dict[str, Any], Any]:
    """执行正式输入、主循环和自动接纳，前台无记忆写者避免抢做后台工作；参数：环境/模型/输入；返回：真实回执及运行。"""
    from app.run_task import execute_context
    from runtime.lease import from_trigger
    from runtime.session_message_store import SessionMessageStore
    from runtime.types import RunContext, Trigger
    from scripts.eval_long_context_scenarios import history_registry

    lease = from_trigger(
        "user",
        task_id=env.task_id,
        capabilities={
            "background_run": {"enabled": True},
            "fs": {
                "project_root": str(env.workspace),
                "read": [str(env.workspace)],
                "write": [],
            },
        },
        max_steps=RUN_MAX_STEPS,
        max_tokens=RUN_MAX_TOKENS,
    )
    context = RunContext(
        task_id=env.task_id,
        session_id=env.session,
        trigger=Trigger.USER,
        payload={"message": text},
        capability_lease=lease,
    )
    entry = SessionMessageStore(env.root).accept_input(
        env.session, text, run_id=context.run_id, task_id=env.task_id
    )
    context.payload["input_message_id"] = entry.entry_id
    started = time.perf_counter()
    with closing(history_registry()) as registry:
        response = execute_context(
            context, data_root=env.root, llm_client=client, registry=registry
        )
    return {
        **asdict(response),
        "input_message_id": entry.entry_id,
        "foreground_wait_seconds": time.perf_counter() - started,
    }, context


def wait_for_stream(gate: StreamingGate, future: Future[Any]) -> None:
    """等待后台供应商非空输出，已结束却无增量的工作明确失败；参数：事件门/调度future；返回：无。"""
    deadline = time.monotonic() + GATE_TIMEOUT_SECONDS
    while not gate.entered.wait(GATE_POLL_SECONDS):
        if future.done():
            raise RuntimeError(
                f"background ended before its first provider output delta: {future.result()}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "background produced no provider output delta within the evaluation deadline"
            )


def evaluate_history(
    config: dict[str, Any], clients: EvaluationClients
) -> dict[str, Any]:
    """通过正式调度整理冻结旧源，同时继续主输入并核对发布后采用；参数：配置/模型池；返回：原件判据。"""
    from app.scheduled_run import create_scheduler
    from context.compaction import CompactionMaterial
    from llm.types import LLMPlan
    from runtime.context_compaction_jobs import ContextCompactionJobs
    from runtime.knowledge_maintenance import KnowledgeMaintenance
    from runtime.session_compaction import SessionCompactionStore, SummarySource
    from schedules.store import ScheduleStore
    from scripts.eval_long_context import (
        continuation_question,
        result_fields,
        seed_case,
    )
    from scripts.eval_stage9_meter import save
    from scripts.long_context_cases import CASES

    case = CASES[0]
    env = environment(Path(config["sample_output"]), case.name)
    KnowledgeMaintenance(env.root).configure(enabled=False)
    owner = seed_case(env.root, case)
    foreground = clients.get("foreground")
    initial, origin = primary_turn(env, foreground, continuation_question(case))
    view = owner.materialize(env.session)
    originals = [entry.to_mapping() for entry in owner.read_entries(env.session)]
    save(
        env.output / "initial.json",
        {"turn": initial, "originals": originals, "case": asdict(case)},
    )
    if initial["status"] != "done":
        raise RuntimeError("initial foreground request failed")
    source = SummarySource(view, len(view.messages) - 1, None)
    jobs = ContextCompactionJobs(env.root)
    accepted = jobs.accept(
        CompactionMaterial(source, ""), {}, run=origin, client=foreground
    )
    if accepted["status"] != "queued":
        raise RuntimeError(f"controlled history work not accepted: {accepted}")
    with closing(ScheduleStore(env.root)) as schedules:
        schedule = schedules.load_schedule(accepted["schedule_id"])
        if schedule is None:
            raise RuntimeError("accepted background schedule is missing")
        frozen = schedule.model_config
    gate = StreamingGate(
        clients.get("background-summary", frozen),
        evidence_path=env.output / "gate.json",
    )
    with (
        closing(
            create_scheduler(
                project_root=env.workspace,
                data_root=env.root,
                llm_factory=lambda _options: gate,
            )
        ) as scheduler,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        future = pool.submit(scheduler.run_due_jobs, now=datetime.now(timezone.utc))
        try:
            wait_for_stream(gate, future)
            during, _ = primary_turn(env, foreground, continuation_question(case))
            concurrent = (
                jobs.load(accepted["job_id"])["status"] == "running"
                and not future.done()
            )
            save(
                env.output / "during.json",
                {
                    "turn": during,
                    "background_still_running": concurrent,
                    "gate_evidence": gate.evidence,
                },
            )
        finally:
            gate.release.set()
        outcomes = future.result()
    published = jobs.load(accepted["job_id"])
    save(
        env.output / "background.json",
        {"work": published, "outcomes": [asdict(item) for item in outcomes]},
    )
    if published["status"] != "published":
        raise RuntimeError("background history was not published")
    after, _ = primary_turn(env, foreground, continuation_question(case))
    summary = SessionCompactionStore(owner).current(owner.materialize(env.session))
    calls = sorted((env.output / "foreground/calls").glob("*/call.json"))
    adoption = history_adoption(env.root, calls[-1], summary)
    semantic = result_fields(LLMPlan(final_output=after["output"]), dict(case.expected))
    during_semantic = result_fields(
        LLMPlan(final_output=during["output"]), dict(case.expected)
    )
    checks = {
        "initial_done": initial["status"] == "done",
        "foreground_during_done": during["status"] == "done",
        "foreground_completed_before_gate_release": concurrent,
        "after_done": after["status"] == "done",
        "originals_unchanged": [
            entry.to_mapping()
            for entry in owner.read_entries(env.session)[: len(originals)]
        ]
        == originals,
        "published_version_adopted": adoption["passed"],
        "continuation_correct": bool(semantic["passed"]),
        "concurrent_answer_correct": bool(during_semantic["passed"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "initial": initial,
        "during": during,
        "after": after,
        "summary": asdict(summary) if summary else None,
        "semantic": semantic,
        "during_semantic": during_semantic,
        "adoption": adoption,
        "controlled_pause_seconds": gate.pause_seconds,
        "gate_evidence": gate.evidence,
        "concurrency_mode": "controlled pause after first non-empty provider output delta; not natural network timing",
        "timing_note": "background call and attempt elapsed include controlled client-consumption pause; no provider-only latency is inferred",
    }


def payload_strings(value: Any) -> list[str]:
    """展开供应商实际输入字段中的文本块；参数：协议输入值；返回：正文字符串，不读取metadata。"""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in payload_strings(item)]
    if isinstance(value, dict):
        return [
            text
            for key in ("content", "text")
            if key in value
            for text in payload_strings(value[key])
        ]
    return []


def history_adoption(root: Path, call_path: Path, summary: Any) -> dict[str, Any]:
    """按请求身份、持久材料版本与实际wire正文核验采用；参数：原件根/调用/摘要；返回：精确证据。"""
    from context.history_segments import HISTORY_LEVELS
    from runtime.persistence import RuntimeStore

    if summary is None or summary.content is None:
        raise ValueError("published history has no inspectable content")
    call = json.loads(call_path.read_text(encoding="utf-8"))
    request_id = call["request_id"]
    candidate = call["composition"]["context_baseline"]
    with RuntimeStore(root).snapshot() as source:
        baseline = next(
            (
                row
                for row in source.list(
                    "context_baseline", session_id=summary.session_id
                )
                if row["adopted_request_id"] == request_id
                and row["baseline_id"] == candidate["baseline_id"]
                and row["delta_id"] == candidate["delta_id"]
            ),
            None,
        )
    materials = [
        row
        for row in candidate["materials"]
        if row["source"] == "history"
        and row["identity"].startswith(f"segment:{summary.summary_id}:")
        and row["version"].startswith(f"{summary.summary_id}/")
        and row["representation"] in {"full", *HISTORY_LEVELS}
    ]
    stored = [] if baseline is None else [*baseline["baseline"], *baseline["delta"]]
    versions = {(row["identity"], row["version"]): row for row in stored}
    expected = []
    for material in materials:
        row = versions.get((material["identity"], material["version"]))
        segment_id = material["identity"].removeprefix(f"segment:{summary.summary_id}:")
        segment = next(
            (
                item
                for item in summary.content.segments
                if item.segment_id == segment_id
            ),
            None,
        )
        if (
            row is None
            or segment is None
            or row["representation"] != material["representation"]
        ):
            continue
        # 1. 【阶段九评估】【实际采用】材料身份版本保持默认正文身份，选档以本次采用记录为准
        level = (
            material["version"].rsplit("/", 1)[-1]
            if material["representation"] == "full"
            else material["representation"]
        )
        expected.append((row["text"], segment.render(level)))
    attempts = []
    for path in sorted(call_path.parent.glob("*-started.json")):
        attempt = json.loads(path.read_text(encoding="utf-8"))
        if attempt["request_id"] != request_id:
            continue
        inputs = [
            text
            for field in ("messages", "instructions", "system", "input")
            for text in payload_strings(attempt["request"].get(field))
        ]
        wire = "\n".join(inputs)
        attempts.append(
            {
                "attempt_id": attempt["attempt_id"],
                "path": str(path),
                "bodies_present": bool(expected)
                and all(text in wire and body in wire for text, body in expected),
            }
        )
    checks = {
        "persisted_request_matches": baseline is not None,
        "exact_material_versions": bool(materials) and len(expected) == len(materials),
        "actual_input_contains_bodies": bool(attempts)
        and all(row["bodies_present"] for row in attempts),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "request_id": request_id,
        "summary_id": summary.summary_id,
        "baseline_id": candidate["baseline_id"],
        "delta_id": candidate["delta_id"],
        "attempts": attempts,
        "materials": [
            {key: row[key] for key in ("identity", "version", "representation")}
            for row in materials
        ],
    }


def memory_snapshot(root: Path) -> list[dict[str, Any]]:
    """从原记忆写者核对真实版本和替代关系；参数：隔离数据根；返回：当前原件视图。"""
    from memory.store import MemoryStore

    with closing(MemoryStore(root)) as store:
        return store.record_views(store.list_memories())


def knowledge_checks(
    stage: str,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    work: dict[str, Any],
    turn: dict[str, Any],
    *,
    root: Path,
) -> dict[str, bool]:
    """核对真实处理结果与来源，不把终稿当保存成功；参数：阶段/前后原件/工作/输入；返回：结构判据。"""
    current = [row for row in after if row["effective_state"] == "active"]
    checks = {"foreground_done": turn["status"] == "done"}
    if stage == "no_op":
        previous = {
            (row["memory_id"], row["version"], row["effective_state"]) for row in before
        }
        versions = {
            (row["memory_id"], row["version"], row["effective_state"]) for row in after
        }
        return {
            **checks,
            "verified_no_op": work["state"] == "no_op",
            "no_commits": not work.get("commits"),
            "versions_unchanged": previous == versions,
        }
    port = "8000" if stage == "new" else "9000"
    supported = [
        row
        for row in current
        if port in row["content"]
        and row["details"]["scope"].startswith("project:")
        and any(
            source["kind"] == "user_input"
            and source["reference"] == turn["input_message_id"]
            for source in row["details"]["sources"]
        )
    ]
    checks.update(
        completed=work["state"] == "completed",
        actual_commits=bool(work.get("commits")),
        current_rule_has_real_source=bool(supported),
    )
    if stage == "correction":
        from memory.store import MemoryStore

        old = {
            (row["memory_id"], row["version"])
            for row in before
            if "8000" in row["content"]
        }
        checks["old_version_not_active"] = not old.intersection(
            (row["memory_id"], row["version"]) for row in current
        )
        with closing(MemoryStore(root)) as memories:
            checks["old_originals_readable"] = all(
                memories.load_memory(row["memory_id"], version=row["version"]).content
                == row["content"]
                for row in before
            )
    return checks


def evaluate_knowledge(
    config: dict[str, Any], clients: EvaluationClients
) -> dict[str, Any]:
    """自动新来源提炼、明确no-op及更正均走真实调度；参数：配置/模型池；返回：真实来源与版本判据。"""
    from app.scheduled_run import create_scheduler
    from llm.types import LLMPlan
    from runtime.context_compaction_jobs import ContextCompactionJobs
    from runtime.knowledge_maintenance import KnowledgeMaintenance
    from runtime.workspaces import WorkspaceStore
    from scripts.eval_long_context import result_fields
    from scripts.eval_stage9_meter import save

    env = environment(Path(config["sample_output"]), "knowledge")
    ContextCompactionJobs(env.root).configure(enabled=False)
    manager = KnowledgeMaintenance(env.root)
    manager.configure(enabled=True)
    stages = []
    for stage, session, text in KNOWLEDGE_STAGES:
        WorkspaceStore(env.root).bind_session(session, env.workspace)
        current = replace(env, session=session)
        before = memory_snapshot(env.root)
        known = {work["work_id"] for work in manager.status()["works"]}
        turn, _ = primary_turn(current, clients.get(f"foreground-{stage}"), text)
        save(env.output / f"{stage}-input.json", turn)
        works = [
            work for work in manager.status()["works"] if work["work_id"] not in known
        ]
        if turn["status"] != "done" or len(works) != 1:
            raise RuntimeError(
                f"{stage}: expected one automatically admitted work after a completed foreground turn"
            )
        with closing(
            create_scheduler(
                project_root=env.workspace,
                data_root=env.root,
                llm_factory=lambda options: clients.get(f"worker-{stage}", options),
            )
        ) as scheduler:
            outcomes = scheduler.run_due_jobs(now=datetime.now(timezone.utc))
        # 1. 【阶段九评估】【失败对账】工作失败不撤销此前记忆提交，按工作身份读取正式回执汇总
        work = next(
            row
            for row in manager.status(session_id=session)["works"]
            if row["work_id"] == works[0]["work_id"]
        )
        after = memory_snapshot(env.root)
        checks = knowledge_checks(stage, before, after, work, turn, root=env.root)
        item = {
            "stage": stage,
            "passed": all(checks.values()),
            "checks": checks,
            "turn": turn,
            "work": work,
            "before": before,
            "after": after,
            "outcomes": [asdict(outcome) for outcome in outcomes],
        }
        stages.append(item)
        save(env.output / f"{stage}-result.json", item)
        if not item["passed"]:
            return {
                "passed": False,
                "stages": stages,
                "remaining_stages": "not executed after failed acceptance",
            }
    manager.configure(enabled=False)
    WorkspaceStore(env.root).bind_session("verification", env.workspace)
    final, _ = primary_turn(
        replace(env, session="verification"),
        clients.get("foreground-verification"),
        '按本项目现行约定回答服务端口，只返回JSON：{"port": 端口数字}，不知道时填null。',
    )
    semantic = result_fields(LLMPlan(final_output=final["output"]), {"port": 9000})
    return {
        "passed": final["status"] == "done" and bool(semantic["passed"]),
        "stages": stages,
        "fresh_session_turn": final,
        "semantic": semantic,
    }


def worker_main(path: Path) -> int:
    """只从冻结源码装配正式模型，失败及所有辅助费用仍落盘；参数：父进程配置；返回：真实状态码。"""
    config = json.loads(path.read_text(encoding="utf-8"))
    source, output = (
        Path(config["source_root"]).resolve(),
        Path(config["sample_output"]),
    )
    sys.path.insert(0, str(source))
    sys.dont_write_bytecode = True
    from app.cli import build_llm_client
    from scripts.eval_stage9_meter import call_metrics, save
    from scripts.eval_stage9_worker import imported_sources

    clients = EvaluationClients(
        config, lambda _role, options: build_llm_client(options, project_root=source)
    )
    started = time.perf_counter()
    try:
        result = (
            evaluate_history(config, clients)
            if config["scenario"] == "history"
            else evaluate_knowledge(config, clients)
        )
    except Exception as exc:
        result = {"passed": False, "error_type": type(exc).__name__, "error": str(exc)}
    try:
        result["imported_sources"] = imported_sources(source)
    except RuntimeError as exc:
        result.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    result.update(
        elapsed_seconds=time.perf_counter() - started,
        metrics={
            role: call_metrics(output / role / "calls") for role in clients.clients
        },
        metrics_timing="observed end-to-end call time; history background includes the separately recorded controlled pause",
        budget=json.loads(Path(config["budget"]).read_text(encoding="utf-8")),
    )
    save(output / "result.json", result)
    return int(not result["passed"])


def main(argv: Sequence[str] | None = None) -> int:
    """默认打印计划；内部worker只使用父进程冻结配置；参数：命令行；返回：执行状态。"""
    selected = list(sys.argv[1:] if argv is None else argv)
    if selected and selected[0] == "--worker-config":
        if len(selected) != 3 or selected[2] != "--execute":
            raise ValueError("worker execution also requires explicit --execute")
        return worker_main(Path(selected[1]))
    args = parse_args(selected)
    plan = experiment_plan(args)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    return execute_plan(args, plan) if args.execute else 0


if __name__ == "__main__":
    raise SystemExit(main())
