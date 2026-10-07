from __future__ import annotations

from contextlib import closing
from pathlib import Path

from runtime.lease import Lease
from runtime.session_messages import append_user_message
from runtime.types import RunContext, Trigger
from tasks.ids import new_ulid
from tasks.store import TaskStore


def make_run_context(
    message: str,
    *,
    data_root: Path | str,
    task_id: str | None = None,
    inbox: bool = False,
    formal_task: bool | None = None,
    session_id: str = "",
    run_id: str = "",
    lease: Lease | None = None,
    input_id: str | None = None,
) -> RunContext:
    """接纳用户正文或引用已接纳动作；传参：正文、数据根和运行身份；返回：可执行运行上下文。"""
    with closing(TaskStore(data_root)) as store:
        has_formal_task = (
            formal_task if formal_task is not None else task_id is not None or not inbox
        )
        if task_id is None:
            task = store.create_task(message, is_inbox=(inbox or not has_formal_task))
        else:
            task = store.require_task(task_id)
        context = RunContext(
            session_id=session_id,
            run_id=run_id,
            task_id=task.task_id if has_formal_task else None,
            focus_task_id=task.task_id if has_formal_task else None,
            focus_task=(
                {"task_id": task.task_id, "goal": task.goal, "status": task.status}
                if has_formal_task
                else {}
            ),
            compatibility_task_id=None if has_formal_task else task.task_id,
            trigger=Trigger.USER,
            payload={
                "message": message,
                "compatibility_task_id": None if has_formal_task else task.task_id,
                "inbox_compatibility": not has_formal_task,
            },
            capability_lease=lease or Lease(),
            segment_id=f"user-{new_ulid()}",
        )
        # 用户输入落成 canonical 消息；session_id 由 RunContext 兜底生成，因此必须在其之后提交
        if input_id is None:
            context.payload["input_message_id"] = append_user_message(
                data_root,
                context.session_id,
                message,
                run_id=context.run_id,
                task_id=task.task_id,
            )
        else:
            from llm.messages import UserMessage, model_visible_text
            from runtime.session_message_store import SessionMessageStore

            # 【用户入口】【已接纳动作】明确确认已有唯一正文，只引用原输入，不能再写一条普通用户消息
            entry = next(
                (
                    row
                    for row in SessionMessageStore(data_root).read_entries(
                        context.session_id
                    )
                    if row.entry_id == input_id
                ),
                None,
            )
            if (
                entry is None
                or entry.type != "inbound"
                or entry.input_source != "user"
                or entry.input_kind is not None
                or not isinstance(entry.message, UserMessage)
                or model_visible_text(entry.message) != message
            ):
                raise ValueError(
                    "user input identity does not match the accepted message"
                )
            context.payload["input_message_id"] = input_id
    return context
