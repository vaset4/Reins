"""扩展观察与材料贡献的共同执行边界；作者：xxx；时间：2026-09-28 18:00:00。"""

from __future__ import annotations
import logging
from functools import partial
from typing import Any
from context.production_builder import ContextReadError
from runtime.cancellation import CancellationToken, ExecutionCancelled
from runtime.extensions import RuntimeExtensions, RuntimeObservation, run_hook
from runtime.run_facts import RunFactStore
from runtime.types import RunContext
from runtime.watchdog import Watchdog

_LOG = logging.getLogger(__name__)


class ExtensionExecution:
    """持有共享扩展与取消依赖，观察错误写入既有运行事实。"""

    def __init__(
        self,
        extensions: RuntimeExtensions,
        cancellation: CancellationToken,
        facts: RunFactStore,
    ) -> None:
        """绑定现有扩展、取消与事实写者；返回：无。"""
        self.extensions = extensions
        self.cancellation = cancellation
        self.run_facts = facts

    def observe(
        self,
        context: RunContext,
        event: str,
        detail: dict[str, Any],
        *,
        watchdog: Watchdog,
    ) -> None:
        """持久提交后才通知观察者；传参：运行、事件和事实快照；返回：无，观察错误独立记录。"""
        if not self.extensions.observers:
            return
        observation = RuntimeObservation(
            event,
            {"session_id": context.session_id, "run_id": context.run_id, **detail},
        )
        try:
            errors = run_hook(
                partial(self.extensions.observe, observation),
                watchdog,
                self.cancellation,
            )
        except Exception as exc:
            errors = (str(exc),)
        for error in errors:
            self.record_error(context, "observer", error)

    def record_error(self, context: RunContext, stage: str, error: str) -> None:
        """独立记录扩展故障，不覆写工具或运行的实际结果；传参：运行/阶段/原因；返回：无。"""
        _LOG.error("【扩展】【执行故障】%s: %s", stage, error)
        self.run_facts.append(
            {
                "event": "extension:error",
                "session_id": context.session_id,
                "run_id": context.run_id,
                "stage": stage,
                "error": error,
            }
        )

    def context_contributions(
        self, context: RunContext, watchdog: Watchdog
    ) -> list[str]:
        """扩展只贡献有来源的材料，不修改真实消息或授权；传参：当前运行；返回：待纳入模型预算的材料。"""
        observation = RuntimeObservation(
            "context_request",
            {
                "session_id": context.session_id,
                "run_id": context.run_id,
                "task_id": context.material_task_id,
            },
        )
        contributions: list[str] = []
        for hook in self.extensions.context_sources:
            try:
                text = run_hook(partial(hook, observation), watchdog, self.cancellation)
                if not isinstance(text, str):
                    raise TypeError("context hook must return text")
            except ExecutionCancelled:
                raise
            except Exception as exc:
                self.record_error(context, "context_source", str(exc))
                raise ContextReadError(
                    "extension_context", context.storage_task_id, exc
                ) from exc
            contributions.append(text)
        return contributions
