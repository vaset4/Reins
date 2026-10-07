from __future__ import annotations

from typing import Protocol

from llm.types import LLMPlan
from runtime.types import RunToolsResult


class LLMClient(Protocol):
    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        """Return either a direct response or a run_tools request."""

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: RunToolsResult,
        context: object | None = None,
    ) -> LLMPlan:
        """Continue the loop after one tool result has been written back."""
