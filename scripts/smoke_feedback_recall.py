"""用合成材料验证生产模型接线，所有运行资料保留在独立临时目录。

作者：xxx
时间：2026-09-22 10:30:00
"""

from __future__ import annotations

import json
import tempfile
from contextlib import closing
from pathlib import Path

from app.cli import build_llm_client
from context.production_builder import ProductionContextBuilder
from llm.client import RealLLMClient
from llm.model_request import composed_request_evidence, model_request_to_mapping
from memory.store import MemoryStore
from runtime.lease import from_trigger
from runtime.session_message_store import SessionMessageStore
from runtime.session_messages import append_user_message
from runtime.types import RunContext, Trigger
from skills.store import SkillStore, build_skill_markdown
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry


def _synthetic_context(root: Path) -> RunContext:
    """创建账单后转向晚餐的合成会话；传参：隔离数据目录；返回：当前运行上下文。"""
    with closing(TaskStore(root)) as tasks:
        task = tasks.create_task("核对本月账单")
    with closing(MemoryStore(root)) as memories:
        memories.create_memory(
            "preference", "用户饮食偏好：不吃辣，晚餐清淡", [], memory_id="meal"
        )
    SkillStore(root).create_skill(
        "meal", build_skill_markdown(name="meal", body="按照饮食偏好安排清淡晚餐")
    )
    context = RunContext(
        task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": "根据已知饮食偏好，用一句话给出晚餐建议"},
        capability_lease=from_trigger("user", task_id=task.task_id),
    )
    append_user_message(root, context.session_id, "先核对账单")
    append_user_message(root, context.session_id, str(context.payload["message"]))
    owner = SessionMessageStore(root)
    owner.accept_input(
        context.session_id, "自动恢复：检查账单任务", input_source="agent"
    )
    owner.deliver_inputs(
        context.session_id, run_id=context.run_id, task_id=task.task_id
    )
    return context


def main() -> int:
    """调用本机配置的生产适配器并落盘请求证据；传参：无；返回：成功为零，失败为一。"""
    root = Path(tempfile.mkdtemp(prefix="reins-feedback-recall-smoke-"))
    print(f"evidence_dir={root}", flush=True)
    client = build_llm_client(
        {"timeout_seconds": 15, "max_output_tokens": 2048},
        project_root=Path(__file__).resolve().parents[1],
    )
    if not isinstance(client, RealLLMClient) or client.resolved_target is None:
        raise RuntimeError("real model is not configured")
    context = _synthetic_context(root)
    bundle = ProductionContextBuilder(
        root, system_prompt_provider=lambda: "Reins synthetic smoke test"
    ).build(
        task=str(context.payload["message"]),
        context=context,
        toolset_policy={},
        tool_registry=ToolRegistry(),
        recoverable_error_notice="Synthetic rehearsal: previous tool argument value must be a string",
        no_progress_observation="Synthetic rehearsal: NO_PROGRESS_OBSERVED; repeated same action and output",
    )
    prepared = client.prepare_request(bundle.model_task, bundle.model_context)
    text = "\n".join(part.text for part in prepared.request.instructions)
    for expected in (
        "不吃辣",
        "清淡晚餐",
        "[recoverable_error_notice]",
        "[no_progress_observation]",
    ):
        if expected not in text:
            raise RuntimeError(f"smoke request missing expected material: {expected}")
    evidence = {
        "request": model_request_to_mapping(prepared.request),
        "composition": composed_request_evidence(prepared),
    }
    (root / "request.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = client.plan(
        bundle.model_task,
        context={**bundle.model_context, "prepared_request": prepared},
    )
    target = client.resolved_target
    report = {
        "provider": target.provider,
        "model": target.model,
        "api_mode": target.api_mode,
        "context_window": target.context_window,
        "timeout_seconds": target.timeout_seconds,
        "max_output_tokens": target.output_token_limit,
        "synthetic_data_only": True,
        "success": result.model_error is None
        and bool(result.final_output)
        and result.run_tools_request is None,
        "error_category": result.model_error.category if result.model_error else None,
        "output": result.final_output,
        "request_id": result.request_id,
        "attempt_count": len(result.model_attempts),
        "evidence_dir": str(root),
    }
    (root / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
