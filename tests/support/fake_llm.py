from __future__ import annotations

from llm.types import LLMPlan
from runtime.types import RunToolsRequest, RunToolsResult


class FakeLLMClient:
    """测试专用的稳定模型桩，避免把假实现继续留在生产目录里。"""

    def __init__(self) -> None:
        self._continue_calls: int = 0

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        del context
        if task.startswith("inspect "):
            return LLMPlan(
                run_tools_request=RunToolsRequest(
                    action="inspect",
                    payload=task.removeprefix("inspect "),
                )
            )
        if task.startswith("echo "):
            return LLMPlan(
                run_tools_request=RunToolsRequest(
                    action="echo",
                    payload=task.removeprefix("echo "),
                )
            )
        return LLMPlan(final_output=f"FAKE_MODEL_RESPONSE: {task}")

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: RunToolsResult,
        context: object | None = None,
    ) -> LLMPlan:
        del context
        self._continue_calls += 1
        if task == "inspect workspace twice" and self._continue_calls == 1:
            return LLMPlan(
                run_tools_request=RunToolsRequest(
                    action="inspect",
                    payload="workspace again",
                )
            )
        return LLMPlan(
            final_output=f"FAKE_MODEL_RESPONSE: run_tools output => {run_tools_result.output}"
        )
