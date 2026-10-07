"""【会话】【全局工作区】跨目录会话选择、执行边界和查询并发回归。

作者：xxx
时间：2026-09-30 11:00:00
"""

from threading import Event, RLock, Thread
from pathlib import Path

import pytest

from app.background.frontend import BackgroundSessionHost
from app.background.client import BackgroundUnavailable
from app.background.server import BackgroundServer
from app.background.service import BackgroundService
from app.background.sessions import SessionServices
from app.startup import resolve_startup_identity
from runtime.persistence import RuntimeStore
from runtime.file_content import ContentFiles
from runtime.session_directory import directory_page
from runtime.workspaces import WorkspaceStore
from llm.messages import ToolCallPart
from scripts.testing.llm import from_test_sequence, from_test_native_tool_then_final
from tools.builtin_tools import build_tool_registry
from tools.tool_registry import ToolRegistry


def services_for(root, data, calls):
    """提供每个目录独立的惰性执行依赖；参数：目录、空间与观测列表；返回：会话工厂。"""

    def model(options):
        """记录真实执行工厂的目录；参数：冻结配置；返回：确定性模型。"""
        calls.append((root, options))
        return from_test_sequence([f"工作区：{root.name}"])

    return SessionServices(
        root,
        data,
        model,
        lambda: build_tool_registry(repo_root=root, data_root=data),
        lambda selected: services_for(selected, data, calls),
    )


def test_default_data_space_is_independent_of_launch_directory(tmp_path):
    """A/B正常启动共享空间，显式根仍隔离；参数：临时用户目录；返回：无。"""
    first = resolve_startup_identity(
        project_root=tmp_path / "a", user_home=tmp_path / "home", environ={}
    )
    second = resolve_startup_identity(
        project_root=tmp_path / "b", user_home=tmp_path / "home", environ={}
    )
    assert first.data_root == second.data_root == tmp_path / "home/.reins/data"
    assert first.project_root != second.project_root
    separate = resolve_startup_identity(
        project_root=first.project_root, data_root=tmp_path / "other"
    )
    assert separate.data_root != first.data_root


def test_global_list_retains_original_workspaces_and_startup_selection(tmp_path):
    """两个目录全局可见，默认选择仅取当前目录最近会话；参数：临时空间；返回：无。"""
    a, b, data = tmp_path / "a", tmp_path / "b", tmp_path / "data"
    a.mkdir()
    b.mkdir()
    calls = []
    service = BackgroundService(services_for(a, data, calls))
    first, second = service.attach(project_root=a), service.attach(project_root=b)
    assert calls == []
    assert first.record.session_id != second.record.session_id
    assert service.attach(project_root=a) is first
    assert service.attach(second.record.session_id).services.project_root == b
    rows = directory_page(data)["sessions"]
    assert {row["project_root"] for row in rows} == {str(a), str(b)}
    assert len(directory_page(data, str(a))["sessions"]) == 1
    assert len({row["workspace_id"] for row in rows}) == 2
    with pytest.raises(FileNotFoundError, match="session does not exist"):
        service.attach("session-unknown")
    service.close()


def test_missing_workspace_keeps_history_but_refuses_new_execution(tmp_path):
    """原目录消失时历史仍可读，发送不借用启动目录；参数：临时空间；返回：无。"""
    workspace, data = tmp_path / "original", tmp_path / "data"
    workspace.mkdir()
    calls = []
    service = BackgroundService(services_for(workspace, data, calls))
    session = service.attach()
    session.submit("原始内容", input_id="input-original", model_config={})
    assert session.runtime.wait_idle(10)
    session.close()
    workspace.rename(tmp_path / "moved")
    restarted = BackgroundService(services_for(tmp_path, data, calls))
    selected = restarted.attach(session.record.session_id)
    snapshot = selected.snapshot(history=True)
    assert (
        snapshot["project_root"] == str(workspace)
        and not snapshot["workspace_available"]
    )
    assert snapshot["history"][0]["text"] == "原始内容"
    with pytest.raises(FileNotFoundError, match="原工作区不可用"):
        selected.submit("继续", input_id="input-next", model_config={})
    assert len(calls) == 1
    restarted.close()


def test_session_binding_is_immutable_and_space_survives_restart(tmp_path):
    """绑定禁止改写且实例重启不改变空间身份；参数：临时根；返回：无。"""
    store = WorkspaceStore(tmp_path / "data")
    original = store.bind_session("session-a", tmp_path / "a")
    with pytest.raises(ValueError, match="cannot be rebound"):
        store.bind_session("session-a", tmp_path / "b")
    assert store.for_session("session-a") == original
    identity = store.database.data_space_id
    assert RuntimeStore(tmp_path / "data").data_space_id == identity
    assert RuntimeStore(tmp_path / "other").data_space_id != identity


def test_long_browse_does_not_hold_control_or_poll_lock():
    """长查询仍允许控制通道取得锁；参数：无；返回：无。"""
    entered, release = Event(), Event()

    class Client:
        """只阻塞详情读取的认证连接替身。"""

        def call(self, method, **params):
            """模拟长详情查询；参数：方法及参数；返回：完成页。"""
            entered.set()
            assert release.wait(5)
            return {"done": True}

    host = BackgroundSessionHost.__new__(BackgroundSessionHost)
    host.client = Client()
    host._connection_lock = RLock()
    worker = Thread(target=host.browse, args=("query_evidence",))
    worker.start()
    try:
        assert entered.wait(2)
        acquired = host._connection_lock.acquire(timeout=0.2)
        assert acquired
        host._connection_lock.release()
    finally:
        release.set()
        worker.join(5)


def test_concurrent_workspaces_write_only_their_original_files(tmp_path):
    """A/B并发文件工具和模型配置各归原会话，不改变进程cwd；参数：临时根；返回：无。"""
    a, b, data = tmp_path / "same" / "a", tmp_path / "same" / "b", tmp_path / "data"
    a.mkdir(parents=True)
    b.mkdir()
    observed = []

    def factory(root):
        """冻结目录专属模型与文件工具；参数：工作区；返回：生产会话依赖。"""

        def model(options):
            """记录本次真实使用的参数；参数：冻结模型配置；返回：写入并完成的模型响应。"""
            observed.append((root, options["model"]))
            return from_test_native_tool_then_final(
                [
                    ToolCallPart(
                        "write",
                        "file_write",
                        {"path": "result.txt", "content": root.name},
                    ),
                ],
                "写入完成",
            )

        return SessionServices(
            root,
            data,
            model,
            lambda: build_tool_registry(repo_root=root, data_root=data),
            factory,
        )

    previous = Path.cwd()
    service = BackgroundService(factory(b))
    try:
        first, second = service.attach(project_root=a), service.attach(project_root=b)
        first.submit("写A文件", input_id="input-a", model_config={"model": "model-a"})
        second.submit("写B文件", input_id="input-b", model_config={"model": "model-b"})
        assert first.runtime.wait_idle(15) and second.runtime.wait_idle(15)
        assert (a / "result.txt").read_text(encoding="utf-8") == "a"
        assert (b / "result.txt").read_text(encoding="utf-8") == "b"
        assert set(observed) == {(a, "model-a"), (b, "model-b")}
        assert Path.cwd() == previous
    finally:
        service.close()


def test_global_directory_pages_and_search_without_decoding_large_bodies(
    tmp_path, monkeypatch
):
    """全局翻页及工作区检索不解码百万字符原件；参数：临时根与解码观测器；返回：无。"""
    store = WorkspaceStore(tmp_path / "data")
    total = 105
    with store.database.transaction():
        for index in range(total):
            identity = f"session-{index:03}"
            store.bind_session(identity, tmp_path / ("alpha" if index % 2 else "beta"))
            body = "完整原文" * 250000 if index == 0 else f"会话 {index}"
            store.messages.accept_input(identity, body, input_id=f"input-{index}")

    # 1. 【会话】【目录分页】首次全文索引允许读取新正文，之后的普通目录只读取派生摘要
    store.database.rebuild_index()

    def forbidden_decode(*_args):
        """目录只能读取已提交摘要列；参数：解码调用；返回：无，正文解码立即失败。"""
        raise AssertionError("session directory decoded message body")

    monkeypatch.setattr(ContentFiles, "read", forbidden_decode)
    first = directory_page(tmp_path / "data")
    second = directory_page(tmp_path / "data", before=first["next_before"])
    assert len(first["sessions"]) == 100 and len(second["sessions"]) == total - 100
    identities = {row["session_id"] for row in first["sessions"] + second["sessions"]}
    assert len(identities) == total and not second["has_more"]
    filtered = directory_page(tmp_path / "data", "alpha")
    assert len(filtered["sessions"]) == total // 2


def test_queued_inputs_keep_the_first_accepted_model_selection(tmp_path):
    """执行排队时后续选择不改写原输入的模型；参数：隔离根；返回：无。"""
    calls = []
    service = BackgroundService(services_for(tmp_path, tmp_path / "data", calls))
    session = service.attach()
    try:
        # 【会话】【排队配置】1. 在后台取得执行资源前接纳两条不同模型选择的输入
        with session._lock:
            session.submit(
                "先核对甲", input_id="first", model_config={"model": "accepted-a"}
            )
            session.submit(
                "追加乙", input_id="second", model_config={"model": "later-b"}
            )
        assert session.runtime.wait_idle(15)
        assert calls[0][1]["model"] == "accepted-a"
        assert (
            session.records.load_input(session.record.session_id, "second")[
                "model_config"
            ]["model"]
            == "later-b"
        )
    finally:
        service.close()


def test_repl_can_read_session_with_invalid_original_model_config(
    tmp_path, monkeypatch
):
    """打开配置失效的原工作区仍读历史，但不能借用上一会话模型；参数：隔离根及配置替换器；返回：无。"""
    from app.repl.session_host import SessionHostConfig
    from app.repl.slash_commands import ReplState

    a, b, data = tmp_path / "a", tmp_path / "b", tmp_path / "data"
    a.mkdir()
    b.mkdir()
    service = BackgroundService(services_for(a, data, []))
    target = service.create_session(b)
    target.messages.accept_input(
        target.record.session_id, "保留的原文", input_id="saved"
    )
    server = BackgroundServer(service, "isolated-token")

    class Connection:
        """直接调用实际后台路由，隔离网络和后台进程。"""

        def call(self, method, **params):
            """转发当前RPC；参数：方法和参数；返回：实际后台投影。"""
            return server.dispatch(method, params)

    def invalid_model(*_args, **_options):
        """模拟原项目配置损坏；参数：模型工厂输入；返回：无，明确报错。"""
        raise ValueError("项目模型配置格式错误")

    monkeypatch.setattr(BackgroundSessionHost, "_connect", lambda _: Connection())
    monkeypatch.setattr("app.cli.build_llm_client", invalid_model)
    host = BackgroundSessionHost(
        SessionHostConfig(
            ReplState(), ToolRegistry(), from_test_sequence(["unused"]), a, data
        )
    )
    try:
        host.attach_session(target.record.session_id)
        assert host.config.project_root == b and host.config.llm_client is None
        assert host._snapshot["history"][0]["text"] == "保留的原文"
        with pytest.raises(BackgroundUnavailable, match="项目模型配置格式错误"):
            host.submit("继续")
        assert not target.runtime.active
    finally:
        host.close()
        server.server_close()
        service.close()
