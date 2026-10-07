from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from runtime.lease import from_trigger
from runtime.watchdog import _notify_pause
from runtime.shared_budget import BudgetOwner
from schedules.notifications import NotificationStore
from tools.clipboard import (
    clipboard_read,
    clipboard_write,
    register_tools as register_clipboard,
)
from tools.notification import (
    register_tools as register_notification,
)
from tools.tool_registry import Idempotent, ToolRegistry, ToolRisk


def test_clipboard_tools_register_expected_risk_and_idempotency() -> None:
    registry = ToolRegistry()

    register_clipboard(registry)

    read = registry.get("clipboard_read")
    write = registry.get("clipboard_write")
    assert read is not None
    assert read.risk is ToolRisk.SAFE
    assert read.idempotent is Idempotent.YES
    assert write is not None
    assert write.risk is ToolRisk.CONFIRM
    assert write.idempotent is Idempotent.CONDITIONAL


def test_clipboard_read_and_write_use_pyperclip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "pyperclip",
        SimpleNamespace(
            paste=lambda: "clip text", copy=lambda text: copied.append(text)
        ),
    )

    assert clipboard_read() == {"text": "clip text"}
    assert clipboard_write("new text") == {"written": True, "chars": 8}
    assert copied == ["new text"]


def test_notification_tool_requires_runtime_identity() -> None:
    """通知需要持久操作归属，不能在无运行的旧入口假报送达；传参：无；返回：无。"""
    registry = ToolRegistry()
    register_notification(registry)
    definition = registry.get("notification_send")
    assert definition is not None and definition.risk is ToolRisk.SAFE
    assert definition.runtime_action and not definition.readonly
    assert definition.idempotent is Idempotent.NO


def test_watchdog_pause_notification_is_durable_and_idempotent(tmp_path) -> None:
    """相同运行的暂停只有一份通知，尚未送达；传参：隔离目录；返回：无。"""
    owner = BudgetOwner("session-pause", "run-pause", from_trigger("user"))
    for _ in range(2):
        _notify_pause("step_limit", data_root=tmp_path, owner=owner)
    records = NotificationStore(tmp_path).list_all()
    assert len(records) == 1 and records[0].delivery_status == "pending"
    assert records[0].source["run_id"] == "run-pause" and records[0].read_at is None


def test_watchdog_pause_notification_storage_failure_is_visible(
    tmp_path, monkeypatch
) -> None:
    """必要通知接纳失败向调用方暴露，不能假装已提醒；传参：目录与替换器；返回：无。"""

    def fail(*_args, **_kwargs):
        """重现实际持久化错误；传参：通知内容；返回：无。"""
        raise OSError("outbox unavailable")

    monkeypatch.setattr(NotificationStore, "enqueue", fail)
    owner = BudgetOwner("session-pause", "run-pause", from_trigger("user"))
    with pytest.raises(OSError, match="outbox unavailable"):
        _notify_pause("step_limit", data_root=tmp_path, owner=owner)


def test_clipboard_write_is_confirm_in_cron_without_grant() -> None:
    registry = ToolRegistry()
    register_clipboard(registry)

    result = registry.execute_tool(
        "clipboard_write",
        {"text": "cron"},
        from_trigger("cron", task_id="task-1"),
    )

    assert getattr(result, "message", "") == "requires_permanent_grant_in_cron_mode"
