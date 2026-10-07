"""目标完成声明与实际证据、修订和持久化边界的验证。

作者：xxx
时间：2026-09-13 20:00:00
"""

from __future__ import annotations

from pathlib import Path

import pytest
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

from llm.messages import (
    AssistantMessage,
    TextPart,
    ToolCallPart,
    ToolResultMessage,
    UserMessage,
)
from runtime.goal_manager import GoalManager
from runtime.persistence import RuntimeStore
from runtime.session_message_store import SessionMessageStore
from tasks.persistence import TaskRevisionConflict, task_write_lock
from tasks.store import TaskStore


def _environment(root: Path) -> tuple[TaskStore, SessionMessageStore, GoalManager]:
    """创建可核对的目标和原始输入；传参：隔离根；返回：任务、消息与目标服务。"""
    tasks = TaskStore(root)
    tasks.create_task("核对两份资料", task_id="goal")
    messages = SessionMessageStore(root)
    messages.append_message(
        "session",
        UserMessage("input", (TextPart("两份资料一致才算完成"),)),
        task_id="goal",
    )
    return tasks, messages, GoalManager(tasks, messages=messages)


def _result(
    messages: SessionMessageStore, *, task_id: str = "goal", failed: bool = False
) -> None:
    """记录配对的工具与真实结果；传参：消息库、归属及结果状态；返回：无。"""
    messages.append_message(
        "session",
        AssistantMessage("assistant", (ToolCallPart("call", "compare", {}),)),
        task_id=task_id,
        run_id="run-source",
    )
    messages.append_message(
        "session",
        ToolResultMessage(
            message_id="result",
            call_id="call",
            tool_name="compare",
            content=(TextPart("两份资料核对记录"),),
            status="error" if failed else "success",
            error="第二份文件无法读取" if failed else None,
        ),
        task_id=task_id,
        run_id="run-source",
    )


def _complete(
    manager: GoalManager,
    *,
    revision: int = 1,
    kind: str = "tool_result",
    reference: str = "call",
) -> None:
    """提交带明确依据的完成声明；传参：服务与所见版本/引用；返回：无。"""
    manager.complete_goal(
        "goal",
        expected_revision=revision,
        summary="两份资料一致",
        evidence=[
            {
                "kind": kind,
                "reference": reference,
                "reason": "已核对全部资料并满足目标",
            }
        ],
        session_id="session",
        run_id="run-complete",
    )


@pytest.mark.parametrize(
    "invalid", ["other_goal", "failed_tool", "old_branch", "missing"]
)
def test_invalid_completion_evidence_preserves_active_goal(
    tmp_path: Path, invalid: str
) -> None:
    """他人、失败、旧分支或不存在的证据不能关闭目标；传参：根与场景；返回：无。"""
    tasks, messages, manager = _environment(tmp_path)
    original = messages.materialize("session").entries[-1].entry_id
    _result(
        messages,
        task_id="other" if invalid == "other_goal" else "goal",
        failed=invalid == "failed_tool",
    )
    if invalid == "old_branch":
        messages.branch("session", original)
    before = tasks.load_task_payload("goal")
    with pytest.raises(ValueError, match="evidence"):
        _complete(manager, reference="missing" if invalid == "missing" else "call")
    assert tasks.load_task_payload("goal") == before
    assert tasks.require_task("goal").status == "active"
    tasks.close()


@pytest.mark.parametrize("writer", ["status", "refs", "sediment", "grant"])
def test_completion_rejects_revision_changed_by_any_metadata_writer(
    tmp_path: Path, writer: str
) -> None:
    """所有元数据写者都使旧完成声明过期；传参：根与写者类型；返回：无。"""
    tasks, messages, manager = _environment(tmp_path)
    _result(messages)
    if writer == "status":
        tasks.update_task_status("goal", "active")
    elif writer == "refs":
        tasks.update_task_refs("goal", spec_refs=["additional-material"])
    elif writer == "sediment":
        tasks.update_sediment_status("goal", attempts=1)
    else:
        tasks.append_grant("goal", {"tool": "compare", "scope": "task", "args": {}})
    with pytest.raises(TaskRevisionConflict, match="revision changed"):
        _complete(manager)
    assert tasks.require_task("goal").status == "active"
    _complete(manager, revision=tasks.require_task("goal").revision)
    assert tasks.require_task("goal").status_source == "goal_completion"
    tasks.close()


def test_completion_write_failure_never_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """完成记录写入失败向调用方暴露，旧目标不被改为成功；传参：根和替换器；返回：无。"""
    tasks, messages, manager = _environment(tmp_path)
    _result(messages)
    from runtime import file_journal

    before = tasks.load_task_payload("goal")
    append = file_journal.append_synced

    def fail_publish(path: Path, content: bytes) -> int:
        """模拟任务原子发布失败；传参：文件和内容；返回：无。"""
        if path.name == "events.jsonl" and b'"kind":"task"' in content:
            raise OSError("disk full")
        return append(path, content)

    monkeypatch.setattr(file_journal, "append_synced", fail_publish)
    with pytest.raises(OSError, match="disk full"):
        _complete(manager)
    assert tasks.load_task_payload("goal") == before
    tasks.close()


def test_writers_cannot_overwrite_metadata_during_completion_window(
    tmp_path: Path,
) -> None:
    """其他线程的元数据写入等待当前提交，两个修改都保留；传参：隔离根；返回：无。"""
    tasks, _, _ = _environment(tmp_path)
    entered = Event()

    def concurrent_grant():
        """模拟独立写者在目标提交期间更新授权；传参：无；返回：新记录。"""
        entered.set()
        return TaskStore(tmp_path).append_grant("goal", {"tool": "write"})

    with ThreadPoolExecutor(max_workers=1) as executor:
        with task_write_lock(tmp_path):
            tasks.update_task_refs("goal", spec_refs=["first"])
            future = executor.submit(concurrent_grant)
            assert entered.wait(5)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.2)
        updated = future.result(timeout=5)
    assert updated.spec_refs == ["first"] and updated.grants == [{"tool": "write"}]
    tasks.close()


def test_inbox_promotion_keeps_unknown_fields_and_rejects_duplicate_identity(
    tmp_path: Path,
) -> None:
    """转正保留扩展字段，重复建目标不能覆盖旧工作；传参：隔离根；返回：无。"""
    tasks = TaskStore(tmp_path)
    tasks.create_task("普通会话中的事项", task_id="inbox", is_inbox=True)
    payload = {**tasks.load_task_payload("inbox"), "custom": {"source": "original"}}
    with RuntimeStore(tmp_path).transaction() as batch:
        batch.put("task", "inbox", payload)
    promoted = tasks.promote_inbox_to_task("inbox", "formal")
    assert promoted.revision == 2
    assert tasks.load_task_payload("formal")["custom"] == payload["custom"]
    assert tasks.load_task("inbox") is None
    assert [row.task_id for row in tasks.list_tasks()] == ["formal"]
    with pytest.raises(FileExistsError):
        tasks.create_task("试图覆盖", task_id="formal")
    assert tasks.require_task("formal").goal == "普通会话中的事项"
    tasks.close()


@pytest.mark.parametrize("reply", ["还没做完", "继续", "好", "我已经核对完成"])
def test_ordinary_user_reply_cannot_impersonate_completion_confirmation(
    tmp_path: Path, reply: str
) -> None:
    """普通正文不能冒充明确确认动作；传参：隔离目录和回复；返回：无。"""
    tasks, messages, manager = _environment(tmp_path)
    try:
        messages.append_message(
            "session", UserMessage("ordinary", (TextPart(reply),)), task_id="goal"
        )
        with pytest.raises(ValueError, match="confirmation"):
            _complete(manager, kind="user_confirmation", reference="ordinary")
        assert tasks.require_task("goal").status == "active"
    finally:
        tasks.close()


def test_published_answer_can_complete_without_manual_confirmation(
    tmp_path: Path,
) -> None:
    """已发布文字成果可以作为完成依据；传参：隔离目录；返回：无。"""
    tasks, messages, manager = _environment(tmp_path)
    try:
        messages.append_message(
            "session",
            AssistantMessage(
                "report", (TextPart("资料逐项比较：数值、日期和来源均一致"),)
            ),
            task_id="goal",
            run_id="report-run",
        )
        _complete(manager, kind="answer", reference="report")
        assert tasks.require_task("goal").status == "done"
    finally:
        tasks.close()
