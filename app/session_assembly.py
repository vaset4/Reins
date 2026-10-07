"""将已接纳输入与中立会话身份装配为运行上下文。

作者：xxx
时间：2026-09-28 16:50:00
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, replace
from pathlib import Path

from llm.messages import UserMessage, model_visible_text
from runtime.default_capabilities import build_local_agent_capabilities
from runtime.lease import Lease, from_trigger
from runtime.run_evidence import RunEvidenceStore
from runtime.session_message_store import SessionMessageStore
from runtime.session_state import SessionStateStore
from runtime.types import RunContext, Trigger, new_session_id
from tasks.ids import new_ulid, utc_now
from tasks.store import TaskStore


@dataclass(frozen=True, slots=True)
class SessionBinding:
    """入口对会话身份的不可变投影；焦点权威仍是SessionStateStore。"""

    session_id: str
    focus_task_id: str | None = None
    compatibility_task_id: str | None = None


def ensure_session_binding(
    binding: SessionBinding, store: TaskStore, first_message: str
) -> SessionBinding:
    """复用有效目标或收件箱；传参：入口身份/任务存储/首次正文；返回：新绑定，不接纳正文。"""
    session_id = binding.session_id or new_session_id()
    focus = binding.focus_task_id
    compatibility = binding.compatibility_task_id
    if focus is not None and store.load_task(focus) is not None:
        return replace(binding, session_id=session_id)
    if compatibility is not None and store.load_task(compatibility) is not None:
        return SessionBinding(session_id, compatibility_task_id=compatibility)
    record = store.create_task(first_message, is_inbox=True)
    return SessionBinding(session_id, compatibility_task_id=record.task_id)


def current_session_binding(binding: SessionBinding, data_root: Path) -> SessionBinding:
    """读取本轮持久焦点；传参：入口投影与存储根；返回：最新绑定，保留兼容存储。"""
    saved = SessionStateStore(data_root).load(binding.session_id)
    if saved is None:
        return binding
    return SessionBinding(
        binding.session_id,
        saved.focus_task_id,
        saved.compatibility_task_id or binding.compatibility_task_id,
    )


def user_lease(*, task_id: str, project_root: Path, data_root: Path) -> Lease:
    """构造普通用户的既有本地能力；传参：存储身份与目录；返回：USER租约，不用于定时恢复。"""
    return from_trigger(
        "user",
        task_id=task_id,
        capabilities=build_local_agent_capabilities(project_root, data_root),
    )


def accepted_input_text(data_root: Path, session_id: str, input_id: str) -> str:
    """从当前分支核对真实已接纳正文；传参：数据根/会话/输入引用；返回：原文，失效引用报错。"""
    entries = SessionMessageStore(data_root).materialize(session_id).entries
    entry = next((item for item in entries if item.entry_id == input_id), None)
    if (
        entry is None
        or entry.type != "inbound"
        or not isinstance(entry.message, UserMessage)
    ):
        raise ValueError("accepted input is missing from current session branch")
    return model_visible_text(entry.message)


def user_context(
    binding: SessionBinding,
    lease: Lease,
    *,
    data_root: Path,
    run_id: str,
    input_id: str,
) -> RunContext:
    """装配已接纳输入，不新增正文；传参：身份/权限/目录/运行和输入引用；返回：USER上下文。"""
    compatibility = (
        None if binding.focus_task_id is not None else binding.compatibility_task_id
    )
    text = accepted_input_text(data_root, binding.session_id, input_id)
    return RunContext(
        session_id=binding.session_id,
        run_id=run_id,
        task_id=binding.focus_task_id,
        focus_task_id=binding.focus_task_id,
        compatibility_task_id=compatibility,
        trigger=Trigger.USER,
        payload={
            "message": text,
            "input_message_id": input_id,
            "compatibility_task_id": compatibility,
            "inbox_compatibility": binding.focus_task_id is None,
        },
        capability_lease=lease,
        segment_id=f"user-{new_ulid()}",
    )


def record_run_error(data_root: Path, context: RunContext, exc: BaseException) -> None:
    """保存归属明确的运行错误；传参：数据根/运行/异常；返回：无，证据写失败直接暴露。"""
    RunEvidenceStore(data_root).append_error(
        session_id=context.session_id,
        run_id=context.run_id,
        error={
            "ts": utc_now(),
            "category": type(exc).__name__,
            "message": str(exc),
            "segment_id": context.segment_id,
            "stage": "session_execution",
            "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
            "cause": str(exc.__cause__) if exc.__cause__ else None,
            "context": str(exc.__context__) if exc.__context__ else None,
        },
    )
