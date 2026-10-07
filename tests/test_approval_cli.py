from __future__ import annotations

from pathlib import Path

import pytest

from approval import ApprovalDecision, ApprovalRequest
from approval.cli import cli_request_approval
from runtime.lease import from_trigger


def test_cli_request_approval_accepts_menu_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    decision = cli_request_approval(_request())

    assert decision is ApprovalDecision.TASK


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        tool="file_write",
        args={"path": "out.txt"},
        risk="confirm",
        lease=from_trigger("user", task_id="task-1"),
        data_root=Path("."),
        message="write file",
    )


def test_make_cli_approval_backend_is_honestly_stateless() -> None:
    # S1-34: docstring 谎称 context-aware/captures state/store，实则 `del state, store`。
    # 诚实化 = 去死参 + docstring 如实（无状态，仅渲染 ApprovalRequest 字段）。
    import inspect

    from app.repl import _make_cli_approval_backend

    sig = inspect.signature(_make_cli_approval_backend)
    assert list(sig.parameters) == []  # 死参 state/store 已移除

    doc = (_make_cli_approval_backend.__doc__ or "").lower()
    assert "context-aware" not in doc  # 不再谎称上下文感知
    assert "captures" not in doc

    backend = _make_cli_approval_backend()
    assert callable(backend)
