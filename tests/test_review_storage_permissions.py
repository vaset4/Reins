"""【审查修复】【存储与权限】验证跨提交坏源检测和只读维护接纳。

作者：xxx
时间：2026-10-03 12:00:00
"""

from contextlib import closing
from pathlib import Path

import pytest

from app.run_task import execute_context
from approval.session import ApprovalMode, ApprovalSession
from runtime.file_journal import FileJournal
from runtime.knowledge_maintenance import KnowledgeMaintenance
from runtime.lease import from_trigger
from runtime.persistence import RuntimeStore, SourceCorruptionError, SourceSnapshot
from runtime.session_message_store import SessionMessageStore
from runtime.types import RunContext, Trigger
from runtime.workspaces import WorkspaceStore
from scripts.testing.llm import from_test_stub
from tools.builtin_tools import build_tool_registry


@pytest.mark.parametrize("damaged", [False, True])
def test_unrelated_commit_does_not_certify_cached_source(
    tmp_path: Path, damaged: bool
) -> None:
    """无关新提交不能把旧文件损坏变成可信缓存；参数：隔离根/是否损坏；返回：无。"""
    store = RuntimeStore(tmp_path)
    with store.transaction() as batch:
        batch.put(
            "tool_operation", "op-a", {"value": "ORIGINAL"}, session_id="session-a"
        )
    reader = FileJournal(tmp_path)
    assert SourceSnapshot(store, reader.load()).get("tool_operation", "op-a") == {
        "value": "ORIGINAL"
    }
    if damaged:
        path = store.source_path("tool_operation", "op-a")
        path.write_bytes(path.read_bytes().replace(b"ORIGINAL", b"CORRUPT!"))
    with store.transaction() as batch:
        batch.put("tool_operation", "op-b", {"value": "new"}, session_id="session-b")
    if damaged:
        with pytest.raises(SourceCorruptionError, match="committed source corrupt"):
            SourceSnapshot(store, reader.load()).get("tool_operation", "op-a")
        with pytest.raises(SourceCorruptionError, match="committed source corrupt"):
            FileJournal(tmp_path).load()
    else:
        warm = SourceSnapshot(store, reader.load())
        cold = SourceSnapshot(store, FileJournal(tmp_path).load())
        assert warm.get("tool_operation", "op-a") == cold.get("tool_operation", "op-a")
        assert warm.get("tool_operation", "op-b") == {"value": "new"}


def _chat(root: Path, session: ApprovalSession, text: str) -> None:
    """经真实宿主入口提交新一轮输入；参数：隔离根/宿主权限/输入；返回：无。"""
    session_id = "permission-session"
    WorkspaceStore(root).bind_session(session_id, root)
    lease = from_trigger(
        "user",
        task_id="permission-task",
        capabilities={
            "fs": {
                "project_root": str(root),
                "read": [str(root)],
                "write": [str(root)],
            },
            "background_run": {"enabled": True},
        },
    )
    context = RunContext(
        session_id=session_id,
        trigger=Trigger.USER,
        payload={"message": text},
        capability_lease=lease,
    )
    entry = SessionMessageStore(root).accept_input(
        session_id, text, run_id=context.run_id
    )
    context.payload["input_message_id"] = entry.entry_id
    with closing(build_tool_registry(repo_root=root, data_root=root)) as registry:
        result = execute_context(
            context,
            data_root=root,
            registry=registry,
            llm_client=from_test_stub("已了解"),
            approval_session=session,
        )
    assert result.status == "done"


def test_readonly_chat_neither_creates_nor_extends_maintenance(tmp_path: Path) -> None:
    """只读时不接纳新维护或扩展旧来源，恢复权限后正常接纳；参数：隔离根；返回：无。"""
    manager = KnowledgeMaintenance(tmp_path)
    manager.configure(enabled=True)
    session = ApprovalSession()
    session.set_mode(ApprovalMode.READ_ONLY)
    _chat(tmp_path, session, "只读期间核对金额规则")
    assert manager.status()["works"] == []
    session.set_mode(ApprovalMode.WORKSPACE)
    _chat(tmp_path, session, "金额保留两位小数")
    accepted = manager.status()["works"]
    assert len(accepted) == 1 and accepted[0]["state"] == "queued"
    session.set_mode(ApprovalMode.READ_ONLY)
    _chat(tmp_path, session, "只读期间继续讨论")
    assert manager.status()["works"] == accepted
    session.set_mode(ApprovalMode.WORKSPACE)
    _chat(tmp_path, session, "继续核对新的金额规则")
    updated = manager.status()["works"]
    assert len(updated) == 1 and updated[0]["work_id"] == accepted[0]["work_id"]
    assert set(updated[0]["message_ids"]) > set(accepted[0]["message_ids"])
