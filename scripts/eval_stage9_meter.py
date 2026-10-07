"""【阶段九评估】【费用边界】在评估调用外记录配额和实际尝试，不改变产品策略。

作者：xxx
时间：2026-10-01 14:30:00
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any


class EvaluationLimitExceeded(RuntimeError):
    """本次评估授权耗尽，停止新请求并保留已发生用量。"""


def save(path: Path, payload: object) -> None:
    """沿用评估原子写者；参数：证据位置与内容；返回：无。"""
    from scripts.eval_long_context import write_json

    write_json(path, payload)


def call_metrics(output: Path) -> dict[str, Any]:
    """汇总全部尝试并保留未知值；参数：本样本调用证据；返回：分类耗时与用量，不估造金额。"""
    phases: dict[str, Any] = {}
    for path in sorted(output.glob("*/call.json")):
        call = json.loads(path.read_text(encoding="utf-8"))
        row = phases.setdefault(
            call["evaluation_phase"],
            {
                "logical_calls": 0,
                "seconds": 0.0,
                "started_attempts": 0,
                "finished_attempts": 0,
                "failed_attempts": 0,
                "usage": {},
            },
        )
        row["logical_calls"] += 1
        row["seconds"] += call.get("elapsed_seconds", 0.0)
        row["started_attempts"] += len(list(path.parent.glob("*-started.json")))
        for attempt_path in path.parent.glob("*-finished.json"):
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            row["finished_attempts"] += 1
            row["failed_attempts"] += int(attempt["error"] is not None)
            for field, measurement in attempt["usage"].items():
                counts = row["usage"].setdefault(
                    field,
                    {
                        "reported_sum": 0,
                        "unknown_attempts": 0,
                        "not_applicable_attempts": 0,
                    },
                )
                if measurement["value"] is not None:
                    counts["reported_sum"] += measurement["value"]
                elif measurement["status"] == "not_applicable":
                    counts["not_applicable_attempts"] += 1
                else:
                    counts["unknown_attempts"] += 1
        row["unfinished_attempts"] = row["started_attempts"] - row["finished_attempts"]
    return {
        "by_phase": phases,
        "price": "unknown",
        "cost_amount": None,
        "usage_note": "reported sums are partial when unknown or unfinished attempts exist",
    }


class EvaluationBudget:
    """前后台评估执行者共享一个持久总账，重试也预留完整输出额度。"""

    def __init__(self, path: Path) -> None:
        """读取父进程冻结的总额度；参数：账本路径；返回：无。"""
        self.path = path

    def reserve(self, resources: Mapping[str, int], *, identity: str) -> dict[str, Any]:
        """派发前一起核对并预留调用/尝试/输出；参数：资源及身份；返回：已持久状态。"""
        from tools.file_persistence import file_edit_lock

        # 1. 【阶段九评估】【并发额度】同一Windows锁覆盖读取、全部核对与一次发布，拒绝不部分消费资源
        with file_edit_lock(self.path.with_suffix(".budget-lock"), wait=True):
            state: dict[str, Any] = json.loads(self.path.read_text(encoding="utf-8"))
            for kind, amount in resources.items():
                if (
                    kind not in {"logical_calls", "attempts", "output_tokens"}
                    or amount < 1
                ):
                    raise ValueError("invalid evaluation reservation")
                if state[kind] + amount > state[f"max_{kind}"]:
                    state["limit_exceeded"] = {
                        "kind": kind,
                        "requested": amount,
                        "identity": identity,
                    }
                    save(self.path, state)
                    raise EvaluationLimitExceeded(
                        f"evaluation {kind} allowance exhausted before dispatch"
                    )
            state.update(
                {kind: state[kind] + amount for kind, amount in resources.items()}
            )
            save(self.path, state)
            return state


class MeteredClient:
    """透传真实客户端，记录所有主调用、生成、核对以及失败尝试。"""

    def __init__(self, client: Any, *, output: Path, budget: EvaluationBudget) -> None:
        """接入真实客户端与独立证据；参数：客户端、输出、总账；返回：无。"""
        self.client, self.output, self.budget = client, output, budget
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        """保留模型身份及准备接口；参数：属性名；返回：原客户端属性。"""
        return getattr(self.client, name)

    def plan(self, task: str, context: Any = None) -> Any:
        """同步调用消费同一计量流；参数：任务与上下文；返回：真实计划。"""
        return self._drain(self.plan_stream(task, context))

    def continue_from_run_tools(
        self, task: str, result: Any, context: Any = None
    ) -> Any:
        """同步工具接续消费同一计量流；参数：任务、真实结果、上下文；返回：计划。"""
        return self._drain(self.continue_stream(task, result, context))

    @staticmethod
    def _drain(stream: Any) -> Any:
        """取生产生成器返回值；参数：模型流；返回：真实计划，不制造输出。"""
        while True:
            try:
                next(stream)
            except StopIteration as done:
                return done.value

    def plan_stream(self, task: str, context: Any = None) -> Any:
        """透传首次请求事件；参数：任务与上下文；返回：计量后的流。"""
        return self._stream(task, context, result=None)

    def continue_stream(self, task: str, result: Any, context: Any = None) -> Any:
        """透传工具接续事件；参数：任务、结果、上下文；返回：计量后的流。"""
        return self._stream(task, context, result=result)

    def _stream(self, task: str, context: Any, *, result: Any) -> Any:
        """先保留配额与来源，再执行真实客户端；参数：请求信息；返回：原始事件与计划。"""
        from llm.model_request import composed_request_evidence
        from llm.messages import model_visible_text

        current = dict(context or {})
        self.calls += 1
        identity = f"{self.output.parent.name}/call-{self.calls:03d}"
        self.budget.reserve({"logical_calls": 1}, identity=identity)
        path = self.output / f"call-{self.calls:03d}"
        prepared = current.get("prepared_request")
        phase = str(current.get("context_purpose", "main"))
        if phase == "compaction":
            if prepared is None:
                raise ValueError(
                    "summary evaluation requires its actual prepared request"
                )
            # 1. 【阶段九评估】【调用分类】读取固定模式行，原文JSON里的引文不参与判断；confirmation是前台整理答复
            mode = model_visible_text(prepared.request.messages[-1]).rsplit("\n", 2)[-2]
            if mode.startswith("独立原文核对："):
                phase = "summary_audit"
            elif mode == "生成最小必要差量。":
                phase = "summary_generation"
            else:
                raise ValueError(
                    "evaluation cannot identify the actual summary generation/audit mode"
                )
        details = {
            "purpose": current.get("context_purpose", "main"),
            "status": "started",
            "evaluation_phase": phase,
            "input_message_id": current.get("input_message_id"),
            "composition": composed_request_evidence(prepared)
            if prepared is not None
            else None,
        }
        save(path / "call.json", details)
        recorder = current.get("model_attempt_recorder")
        current["model_attempt_recorder"] = self._recorder(path, recorder)
        started = time.perf_counter()
        try:
            stream = (
                self.client.plan_stream(task, current)
                if result is None
                else self.client.continue_stream(task, result, current)
            )
            plan = yield from stream
            save(
                path / "call.json",
                {
                    **details,
                    "status": "finished",
                    "elapsed_seconds": time.perf_counter() - started,
                    "request_id": plan.request_id,
                    "output": plan.final_output,
                    "error": asdict(plan.model_error) if plan.model_error else None,
                },
            )
            return plan
        except BaseException as exc:
            save(
                path / "call.json",
                {
                    **details,
                    "status": "failed",
                    "elapsed_seconds": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise

    def _recorder(self, path: Path, downstream: Any) -> Any:
        """组合评估与产品证据，不覆盖产品结算；参数：目录及原回调；返回：尝试记录器。"""
        from llm.messages import thaw_json_value
        from llm.types import usage_to_mapping

        def record(event: Any) -> None:
            """按每次真实尝试预留输出并保存unknown；参数：事件；返回：无，超额拒绝。"""
            request = thaw_json_value(event.request)
            if not isinstance(request, dict):
                raise ValueError("model attempt request must be an object")
            if event.phase == "started":
                maximum = next(
                    (
                        request[key]
                        for key in (
                            "max_output_tokens",
                            "max_completion_tokens",
                            "max_tokens",
                        )
                        if key in request
                    ),
                    None,
                )
                if type(maximum) is not int or maximum < 1:
                    raise ValueError(
                        "evaluation requires an explicit output ceiling on every attempt"
                    )
                # 1. 【阶段九评估】【派发额度】重试和缩小输出后的尝试也占一次派发，不能借小输出突破总次数
                self.budget.reserve(
                    {"attempts": 1, "output_tokens": maximum}, identity=event.attempt_id
                )
            save(
                path / f"{event.attempt_id}-{event.phase}.json",
                {
                    "phase": event.phase,
                    "attempt_id": event.attempt_id,
                    "request_id": event.request_id,
                    "attempt_index": event.attempt_index,
                    "api_family": event.api_family,
                    "request": request,
                    "response": thaw_json_value(event.response),
                    "sources": thaw_json_value(event.sources),
                    "elapsed_ms": event.elapsed_ms,
                    "usage": usage_to_mapping(event.usage),
                    "error": asdict(event.error) if event.error else None,
                },
            )
            if downstream is not None:
                downstream(event)

        return record
