"""以真实模型验证受控遗漏、历史导航和未保存事实的恢复接续。

作者：xxx
时间：2026-09-26 19:20:00
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path

from app.background.sessions import BackgroundSession, SessionRecord, SessionServices
from app.cli import build_llm_client
from context.compaction import CompactionMaterial, original_sources, source_groups
from context.production_builder import ProductionContextBundle
from context.summary_entries import (
    SummaryCitation,
    SummaryContent,
    SummaryEntry,
    SummaryTopic,
)
from llm.client import RealLLMClient
from llm.types import LLMPlan
from llm.public_config import public_model_config
from memory.records import MemoryDetails, MemorySource
from memory.store import MemoryStore
from runtime.agent_loop import AgentLoop, State
from runtime.context_preparation import ContextCompactor
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_compaction import SessionCompactionStore, SummarySource
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_assistant_message, append_user_message
from runtime.types import RunContext, Trigger
from scripts.eval_long_context import (
    MODEL_OUTPUT_TOKENS,
    MODEL_TIMEOUT_SECONDS,
    PROJECT_ROOT,
    SOURCE_FILES,
    RecordedModel,
    _compact_round,
    continuation_question,
    result_fields,
    seed_case,
    write_json,
)
from scripts.long_context_cases import CASES
from scripts.long_context_evidence import model_manifest, snapshot_sources
from skills.store import SkillStore, build_skill_markdown
from tasks.store import TaskStore
from tools.native_actions import register_native_actions
from tools.tool_registry import ToolRegistry

SCENARIOS = (
    "controlled_omission",
    "navigation",
    "missing_directory",
    "working_fact_recovery",
)
BACKGROUND_WAIT_SECONDS = 600
NAVIGATION_QUESTION = (
    "回到之前个人笔记的数据库方案。请按已经讨论的内容回答，只返回JSON：database（数据库名）、"
    "cloud（是否使用云端）、encoding（已讨论的导入编码）、pending（尚未确定的是去重规则还是主题颜色）。"
)
NAVIGATION_EXPECTED = {
    "database": "SQLite",
    "cloud": False,
    "encoding": "UTF-16LE",
    "pending": "去重规则",
}


def history_registry() -> ToolRegistry:
    """复用生产历史工具，只暴露本场景所需的只读能力；传参：无；返回：独立注册表。"""
    with closing(ToolRegistry()) as catalog:
        register_native_actions(catalog)
        registry = ToolRegistry()
        for name in ("capabilities", "read_history"):
            definition = catalog.get(name)
            assert definition is not None
            registry.register(definition)
        return registry


def controlled_omission(client: RealLLMClient, output: Path) -> dict[str, object]:
    """故意遗漏费用例外，让真实核对模型对原文修正后接续；传参：模型和输出；返回：独立行为判据。"""
    case = CASES[0]
    root = output / "data"
    owner = seed_case(root, case)
    view = owner.materialize(case.name)
    source = SummarySource(view, len(view.messages) - 1)
    original = view.messages[0].message_id
    candidate = SummaryContent(
        entries=(
            SummaryEntry(
                "limits",
                "requirement",
                "总费用最多500元。不要上传原文，也不要联网。",
                (
                    SummaryCitation(
                        original,
                        "总费用最多500元。不要上传原文，也不要联网。",
                        "user_input",
                    ),
                ),
            ),
        )
    )
    write_json(
        output / "controlled_candidate.json",
        {
            "content": asdict(candidate),
            "intentional_omission": "只有本地离线转换费单项不超过80可先付，其他费用先询问",
        },
    )
    context = {
        "session_id": case.name,
        "run_id": "controlled",
        "segment_id": "audit",
        "tool_registry": ToolRegistry(),
        "context_purpose": "compaction",
    }
    model = RecordedModel(client, output)
    compactor = ContextCompactor(
        SessionCompactionStore(owner), client.prepare_request, model.invoke
    )
    requests: list[str] = []
    corrected = compactor._audit(
        candidate,
        source_groups(CompactionMaterial(source, "")),
        context,
        requests=requests,
        base=None,
        sources=original_sources(source),
        auxiliary=[],
        related=[],
        published_content=SummaryContent(),
    )
    saved = compactor.store.publish(
        source, corrected.render(), request_ids=tuple(requests), content=corrected
    )
    plan = model.invoke(
        ProductionContextBundle(
            continuation_question(case),
            {
                **context,
                "context_purpose": "continuation",
                "session_summary": saved.model_view(),
                "conversation_history": view.messages[-1:],
                "input_message_id": view.messages[-1].message_id,
            },
            (),
        )
    )
    result = result_fields(plan, dict(case.expected))
    write_json(output / "corrected_summary.json", asdict(saved))
    return {
        **result,
        "controlled_candidate": True,
        "checks": corrected.checks,
        "calls": model.calls,
    }


def seed_navigation(root: Path, *, directory: bool) -> tuple[str, str]:
    """保存A/B/C原话及仅含线索的快照，检验补读而非摘要生成；传参：目录及是否有话题目录；返回：任务与输入ID。"""
    with closing(TaskStore(root)) as tasks:
        task = tasks.create_task("接续个人笔记的数据库方案")
    owner = SessionMessageStore(root)
    store = SessionCompactionStore(owner)
    originals = (
        (
            "笔记数据库",
            "个人笔记只用本地SQLite，云端方案被否决，因为要完全离线。导入文件编码已定UTF-16LE，去重规则还没决定。",
        ),
        ("日志数据库", "另一个日志服务用PostgreSQL，与个人笔记无关。"),
        ("界面主题", "界面主题已选浅色，这和笔记导入方案分开。"),
    )
    previous = None
    for index, (title, body) in enumerate(originals):
        identity = append_user_message(root, "navigation", body)
        append_assistant_message(root, "navigation", "已记录当前讨论，未执行导入。")
        append_user_message(root, "navigation", "先切换到下一个话题")
        content = (
            SummaryContent(
                topics=(
                    SummaryTopic(
                        f"topic-{index}", title, "决定理由与未决事项见原话", (identity,)
                    ),
                )
            )
            if directory
            else None
        )
        previous = store.publish(
            SummarySource(
                owner.materialize("navigation"),
                len(owner.materialize("navigation").messages) - 1,
                previous,
            ),
            content.render() if content else "历史讨论的细节需回查原话",
            request_ids=(f"controlled-fixture-{index}",),
            content=content,
        )
    input_id = append_user_message(root, "navigation", NAVIGATION_QUESTION)
    return task.task_id, input_id


def navigation(
    client: RealLLMClient, output: Path, *, directory: bool
) -> dict[str, object]:
    """让生产主循环由模型决定如何找回旧话题；传参：真实模型和目录场景；返回：答案与实际补读证据。"""
    root = output / "data"
    task_id, input_id = seed_navigation(root, directory=directory)
    context = RunContext(
        task_id=task_id,
        session_id="navigation",
        trigger=Trigger.USER,
        payload={"message": NAVIGATION_QUESTION, "input_message_id": input_id},
        capability_lease=from_trigger(
            "user", task_id=task_id, capabilities={"fs": {"read": [str(root)]}}
        ),
    )
    with closing(history_registry()) as registry:
        loop = AgentLoop(root, llm_client=client, tool_registry=registry)
        state = loop.run(context)
        operations = loop.operations.for_session("navigation")
    reads = [row for row in operations if row["call"]["tool_name"] == "read_history"]
    result = (
        result_fields(LLMPlan(final_output=loop.last_output), NAVIGATION_EXPECTED)
        if state is State.DONE
        else {
            "passed": False,
            "reason": "runtime_failed",
            "runtime_error": loop.last_output,
        }
    )
    return {
        **result,
        "passed": result["passed"] and state is State.DONE and bool(reads),
        "state": state.value,
        "output": loop.last_output,
        "controlled_directory_fixture": True,
        "directory_provided": directory,
        "history_reads": reads,
        "run_id": context.run_id,
        "facts": RunFactStore(root).read_run(context.run_id),
    }


def working_fact_recovery(client: RealLLMClient, output: Path) -> dict[str, object]:
    """真实旧记忆在两次模型压缩后保持未改，再由后台恢复接续；传参：模型和目录；返回：事实与权限结果。"""
    case = next(item for item in CASES if item.name == "working_fact")
    root = output / "data"
    owner = seed_case(root, case)
    from runtime.workspaces import WorkspaceStore

    WorkspaceStore(root).bind_session(case.name, output)
    with closing(TaskStore(root)) as tasks:
        task = tasks.create_task("继续本地生产服务诊断")
        tasks.update_task_refs(task.task_id, skill_refs=["diagnostic"])
    first = owner.materialize(case.name).messages[0].message_id
    with closing(MemoryStore(root)) as memories:
        memories.create_memory(
            "fact",
            "本地生产服务监听8000",
            ["服务", "端口"],
            memory_id="port",
            details=MemoryDetails(
                subject="本地生产服务",
                fact_key="端口",
                scope=f"session:{case.name}",
                sources=(MemorySource("user_input", first, session_id=case.name),),
            ),
        )
    SkillStore(root).create_skill(
        "diagnostic",
        build_skill_markdown(name="诊断指南", body="只依据当前证据诊断。\n" * 40000),
        meta={},
    )
    with closing(MemoryStore(root)) as memories:
        before = memories.load_memory("port")
    model = RecordedModel(client, output)
    context = {
        "session_id": case.name,
        "run_id": "fact-original-run",
        "segment_id": "summary",
        "tool_registry": ToolRegistry(),
    }
    store = SessionCompactionStore(owner)
    _compact_round(
        store,
        model,
        context,
        case_name=case.name,
        strategy="current",
        baseline_source=None,
    )
    append_assistant_message(
        root, case.name, "继续核对过程材料，未修改长期记忆。" * 100
    )
    owner.accept_input(
        case.name, continuation_question(case), input_id="fact-original-input"
    )
    owner.deliver_inputs(case.name, run_id="fact-original-run", task_id=task.task_id)
    facts = RunFactStore(root)
    facts.append(
        {"event": "run:start", "session_id": case.name, "run_id": "fact-original-run"}
    )
    facts.append(
        {
            "event": "input:handled",
            "session_id": case.name,
            "run_id": "fact-original-run",
            "input_ids": ["fact-original-input"],
        }
    )
    _compact_round(
        store,
        model,
        context,
        case_name=case.name,
        strategy="current",
        baseline_source=None,
    )
    inbound = sum(entry.type == "inbound" for entry in owner.read_entries(case.name))
    lease = from_trigger(
        "user", task_id=task.task_id, capabilities={"fs": {"read": [str(root)]}}
    )
    record = SessionRecord(
        case.name,
        compatibility_task_id=task.task_id,
        status="running",
        intent={
            "run_id": "fact-original-run",
            "root_run_id": "fact-original-run",
            "input_id": "fact-original-input",
            "task_id": task.task_id,
            "focus_task_id": task.task_id,
            "trigger": "user",
            "lease": asdict(lease),
        },
    )
    session = BackgroundSession(
        record, SessionServices(output, root, lambda _options: client, history_registry)
    )
    session.records.save_input(
        record.session_id,
        "fact-original-input",
        public_model_config(client),
        needs_ephemeral_key=False,
    )
    try:
        session.recover()
        if not session.runtime.wait_idle(BACKGROUND_WAIT_SECONDS):
            raise TimeoutError(
                "real recovery evaluation did not finish within the recorded experiment window"
            )
        view = owner.materialize(case.name)
        answer = next(
            message
            for message in reversed(view.messages)
            if message.kind == "assistant"
        )
        from llm.messages import model_visible_text

        result = (
            result_fields(
                LLMPlan(final_output=model_visible_text(answer)), dict(case.expected)
            )
            if session.record.status == "done"
            else {
                "passed": False,
                "reason": "runtime_failed",
                "runtime_error": session.record.error,
            }
        )
        with closing(MemoryStore(root)) as memories:
            after = memories.load_memory("port")
            # 【上下文评测】【记忆核对】1. 使用时间是召回统计，不代表模型修订正文、来源或历史版本
            memory_unchanged = (
                replace(after, last_used_at=before.last_used_at) == before
            )
        no_fake_input = (
            sum(entry.type == "inbound" for entry in owner.read_entries(case.name))
            == inbound
        )
        return {
            **result,
            "passed": result["passed"]
            and session.record.status == "done"
            and memory_unchanged
            and no_fake_input,
            "state": session.record.status,
            "memory_unchanged": memory_unchanged,
            "no_fake_input": no_fake_input,
            "memory_last_used_at": after.last_used_at,
            "summary_ids": [item.summary_id for item in store.chain(view)],
            "output": model_visible_text(answer),
            "session_record": asdict(session.record),
        }
    finally:
        session.close()


def main() -> int:
    """在独立目录逐例调用真实模型，失败保留原始证据；传参：命令行；返回：是否有失败。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        help="Named provider:model profile; does not change the active profile.",
    )
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        raise ValueError("new output directory and positive repeats required")
    overrides: dict[str, object] = {
        "timeout_seconds": MODEL_TIMEOUT_SECONDS,
        "max_output_tokens": MODEL_OUTPUT_TOKENS,
    }
    if args.profile is not None:
        overrides["profile_name"] = args.profile
    client = build_llm_client(overrides, project_root=PROJECT_ROOT)
    if not isinstance(client, RealLLMClient) or client.resolved_target is None:
        raise ValueError("a real configured model is required")
    fingerprints = snapshot_sources(
        PROJECT_ROOT, SOURCE_FILES, args.output / "source", resume=False
    )
    target = client.resolved_target
    write_json(
        args.output / "manifest.json",
        {
            "scenario": args.scenario,
            "repeats": args.repeats,
            "synthetic_only": True,
            **model_manifest(target),
            "source_sha256": fingerprints,
        },
    )
    os.environ["REINS_TRACE_LEVEL"] = "debug"
    failures = 0
    for index in range(args.repeats):
        output = args.output / f"{args.scenario}-{index + 1}"
        started = time.perf_counter()
        try:
            if args.scenario == "controlled_omission":
                result = controlled_omission(client, output)
            elif args.scenario == "working_fact_recovery":
                result = working_fact_recovery(client, output)
            else:
                result = navigation(
                    client, output, directory=args.scenario == "navigation"
                )
        except Exception as exc:
            result = {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "result.json", result)
        failures += int(not result["passed"])
        print(
            json.dumps(
                {
                    "scenario": args.scenario,
                    "repeat": index + 1,
                    "passed": result["passed"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "error": result.get("error"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
