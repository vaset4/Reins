"""验证会话装配失败仍释放宿主拥有的连接。

作者：xxx
时间：2026-09-28 16:45:00
"""

from pathlib import Path

import pytest

from app.background.sessions import BackgroundSession, SessionRecord, SessionServices
from app.repl.session_host import local_session_resources
from app.repl.slash_commands import ReplState
from llm.base import LLMClient
from llm.client import MissingConfigurationLLMClient
from runtime.file_records import SourceCorruptionError, record_key
from runtime.session_message_store import SessionMessageStore
from tasks.store import TaskStore
from tools.tool_registry import ToolRegistry


class OwnedRegistry(ToolRegistry):
    """记录真实注册表的关闭次数，检测装配尚未进入循环时的资源泄漏。"""

    def __init__(self) -> None:
        """初始化真实注册表；传参：无；返回：无。"""
        super().__init__()
        self.close_count = 0

    def close(self) -> None:
        """关闭实际连接并记录所有者清理；传参：无；返回：无。"""
        self.close_count += 1
        super().close()


def test_background_client_assembly_failure_closes_owned_registry(
    tmp_path: Path,
) -> None:
    """模型装配失败不能泄漏先创建的工具连接；传参：隔离根；返回：无。"""
    registry = OwnedRegistry()

    def make_registry() -> ToolRegistry:
        """提供本次运行拥有的连接；传参：无；返回：真实注册表。"""
        return registry

    def fail_client(options: dict[str, object]) -> LLMClient:
        """模拟模型配置解析的真实失败边界；传参：公开配置；返回：不返回。"""
        raise ValueError("model assembly failed")

    services = SessionServices(tmp_path, tmp_path, fail_client, make_registry)
    session = BackgroundSession(SessionRecord("session-assembly-failure"), services)
    try:
        session.submit(
            "保存本地结果", input_id="input-assembly-failure", model_config={}
        )
        assert session.runtime.wait_idle(10)
        assert session.record.status == "failed"
        assert session.record.error == "ValueError: model assembly failed"
        assert registry.close_count == 1
        entries = session.messages.read_entries(session.record.session_id)
        assert sum(entry.entry_id == "input-assembly-failure" for entry in entries) == 1
    finally:
        session.close()


def test_local_close_failure_still_closes_owned_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """审批关闭故障不能跳过工具连接清理；传参：隔离根/替换器；返回：无。"""
    registry = OwnedRegistry()
    closed_stores: list[TaskStore] = []
    original_close = TaskStore.close

    def close_store(store: TaskStore) -> None:
        """观察入口任务连接释放；传参：存储；返回：无。"""
        closed_stores.append(store)
        original_close(store)

    def fail_close() -> None:
        """暴露关闭故障；传参：无；返回：不返回。"""
        raise OSError("approval close failed")

    monkeypatch.setattr(TaskStore, "close", close_store)
    with pytest.raises(OSError, match="approval close failed"):
        with local_session_resources(
            ReplState(),
            project_root=tmp_path,
            data_root=tmp_path,
            llm_client=MissingConfigurationLLMClient(),
            tool_registry=registry,
        ) as (store, _, host):
            monkeypatch.setattr(host.approvals, "close", fail_close)
    assert registry.close_count == 1
    assert closed_stores.count(store) == 1


def test_local_initialization_failure_closes_owned_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """确认状态损坏使宿主初始化失败时仍关闭连接；传参：隔离根；返回：无。"""
    registry = OwnedRegistry()
    state = ReplState(session_id="session-invalid-history")
    owned_store = TaskStore(tmp_path)
    messages = SessionMessageStore(tmp_path)
    entry = messages.accept_input(
        state.session_id, "已保存输入", input_id="broken-input"
    )
    assert messages.read_entries(state.session_id)[-1] == entry
    source = messages.database.source_path(
        "session_entry", record_key(entry.session_id, entry.entry_id)
    )
    original = source.read_bytes()
    corrupted = original.replace(
        "已保存输入".encode("utf-8"), "已篡改输入".encode("utf-8")
    )
    assert corrupted != original
    source.write_bytes(corrupted)
    closed_stores: list[TaskStore] = []
    original_close = TaskStore.close

    def close_store(store: TaskStore) -> None:
        """观察真实任务连接关闭；传参：存储；返回：无。"""
        closed_stores.append(store)
        original_close(store)

    monkeypatch.setattr(TaskStore, "close", close_store)
    monkeypatch.setattr("app.repl.session_host.TaskStore", lambda _root: owned_store)
    with pytest.raises(SourceCorruptionError, match="committed source corrupt"):
        with local_session_resources(
            state,
            project_root=tmp_path,
            data_root=tmp_path,
            llm_client=MissingConfigurationLLMClient(),
            tool_registry=registry,
        ):
            pytest.fail("invalid history must fail initialization")
    assert registry.close_count == 1
    assert closed_stores.count(owned_store) == 1
    assert source.read_bytes() == corrupted
