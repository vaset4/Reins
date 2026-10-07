"""运行所有者向模型和工具边界传递的运行视图；作者：xxx；时间：2026-09-28 18:00:00。"""

from dataclasses import dataclass
from pathlib import Path
from runtime.types import RunContext
from runtime.watchdog import Watchdog
from tasks.store import TaskStore


@dataclass(frozen=True, slots=True)
class TurnExecutionContext:
    task_dir: Path
    context: RunContext
    store: TaskStore
    watchdog: Watchdog
    storage_task_id: str
    task: str


@dataclass(frozen=True, slots=True)
class OperationOrigin:
    """迟到结果的固定归属；不持有可变焦点、会话宿主或取消后的主循环。"""

    session_id: str
    run_id: str
    task_id: str | None
    focus_task_id: str | None
    compatibility_task_id: str | None
    segment_id: str
    material_task_id: str

    @classmethod
    def capture(cls, context: RunContext) -> "OperationOrigin":
        """冻结派发时证据归属；传参：当前运行；返回：不可变的迟到提交身份。"""
        return cls(
            context.session_id,
            context.run_id,
            context.task_id,
            context.focus_task_id,
            context.compatibility_task_id,
            context.segment_id,
            context.material_task_id,
        )
