"""【TUI】【工作区筛选】真实SQL分页、完整路径选择与执行身份隔离。

作者：xxx
时间：2026-09-30 23:30:00
"""

import asyncio
from threading import Event, Thread

import pytest
from textual.widgets import Button, OptionList, Select, Static

from app.background.client import BackgroundClient
from app.background.server import BackgroundServer
from app.background.service import BackgroundService
from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui
from runtime.file_content import ContentFiles
from runtime.workspaces import WorkspaceStore
from tests.frontends.tui.test_interactive import DisplayBridge
from tests.test_global_workspaces import services_for

WAIT_SECONDS = 5


@pytest.fixture
def directory_rpc(tmp_path):
    """创建独立会话库与真实认证RPC；参数：临时目录；返回：客户端、工作区和会话身份。"""
    first, second = tmp_path / "部门[甲]" / "项目", tmp_path / "部门[乙]" / "项目"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    data = tmp_path / "data"
    store = WorkspaceStore(data)
    a = store.bind_session("session-a", first)
    b = store.bind_session("session-b", second)
    store.messages.accept_input("session-a", "甲会话[保留方括号]", input_id="input-a")
    store.messages.accept_input("session-b", "乙会话[保留方括号]", input_id="input-b")
    service = BackgroundService(services_for(first, data, []))
    server = BackgroundServer(service, "isolated-directory-token")
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = BackgroundClient(
        server.server_port,
        server.token,
        server.instance,
        data_space_id=server.data_space_id,
    )
    try:
        yield client, store, a, b
    finally:
        server.shutdown()
        server.server_close()
        service.close()
        thread.join(WAIT_SECONDS)
        assert not thread.is_alive()


def test_workspace_sql_filter_precedes_paging_and_combines_search(
    directory_rpc, monkeypatch
):
    """筛选跨越全局首屏且不读取正文，搜索与分页共同生效；参数：真实RPC与观测器；返回：无。"""
    client, store, a, b = directory_rpc
    with store.database.transaction():
        for index in range(105):
            identity = f"a-{index:03}"
            store.bind_session(identity, a.project_root)
            store.messages.accept_input(
                identity, f"筛选目标 {index:03}", input_id=f"input-a-{index}"
            )
        for index in range(110):
            identity = f"b-{index:03}"
            store.bind_session(identity, b.project_root)
            store.messages.accept_input(
                identity, "大正文" * 350000 if index == 0 else "其他工作区"
            )

    # 1. 【会话】【查询索引】先完成新原件的首次索引，日常分页不能再次展开已有正文
    store.database.rebuild_index()

    def forbidden(*_args):
        """目录只读标量摘要，解码正文即失败；参数：任意；返回：无。"""
        raise AssertionError("workspace filter decoded message bodies")

    monkeypatch.setattr(ContentFiles, "read", forbidden)
    first = client.call("sessions", workspace_id=a.workspace_id)
    second = client.call(
        "sessions", workspace_id=a.workspace_id, before=first["next_before"]
    )
    rows = first["sessions"] + second["sessions"]
    assert len(first["sessions"]) == 100 and len(second["sessions"]) == 6
    assert len({row["session_id"] for row in rows}) == 106
    assert {row["workspace_id"] for row in rows} == {a.workspace_id} and second[
        "next_before"
    ] is None
    found = client.call("sessions", workspace_id=a.workspace_id, query="筛选目标 104")
    assert [row["session_id"] for row in found["sessions"]] == ["a-104"]
    assert (
        client.call("sessions", workspace_id=b.workspace_id, query="筛选目标")[
            "sessions"
        ]
        == []
    )
    assert {row["project_root"] for row in first["workspaces"]} == {
        str(a.project_root),
        str(b.project_root),
    }
    assert len(client.call("sessions")["sessions"]) == 100


async def wait_for(pilot, predicate):
    """等待当前界面的可观察变化；参数：驾驶器与条件；返回：无，超时失败。"""
    async with asyncio.timeout(WAIT_SECONDS):
        while not predicate():
            await pilot.pause(0.02)


def test_workspace_picker_keyboard_mouse_width_and_new_session_identity(directory_rpc):
    """完整路径可选，筛选不改草稿和执行根，新建恢复可见列表；参数：真实目录RPC；返回：无。"""
    client, store, a, b = directory_rpc

    class DirectoryBridge(DisplayBridge):
        """目录使用真实RPC，仅记录界面执行控制是否被误调用。"""

        def __init__(self):
            """固定原执行根；参数：无；返回：无。"""
            super().__init__()
            self.project_root = a.project_root
            self.attached = []

        def browse(self, method, **params):
            """转发真实认证目录查询；参数：方法和筛选；返回：后台目录。"""
            return client.call(method, **params)

        def attach_session(self, identity):
            """记录明确的新建动作；参数：会话身份；返回：无。"""
            self.attached.append(identity)

    async def scenario():
        """通过键盘鼠标切换目录并检查宽窄屏；参数：无；返回：无。"""
        bridge = DirectoryBridge()
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            app.receive(
                "snapshot",
                {
                    "session_id": "session-a",
                    "data_space_id": client.data_space_id,
                    "workspace_id": a.workspace_id,
                    "project_root": str(a.project_root),
                    "history": [],
                },
            )
            await wait_for(pilot, lambda: len(app._workspace_options) == 3)
            composer = app.query_one(Composer)
            composer.load_text("原工作区未发送草稿")
            picker = app.query_one("#workspace-filter", Select)
            assert picker.value == "" and app.query_one("#sidebar").region.width == 38
            assert {label for label, value in app._workspace_options if value} == {
                str(a.project_root),
                str(b.project_root),
            }
            picker.focus()
            await pilot.press("enter", "end", "enter")
            selected = picker.value
            await wait_for(
                pilot, lambda: not app.query_one("#sessions", OptionList).disabled
            )
            assert selected in {a.workspace_id, b.workspace_id}
            assert app.query_one("#sessions", OptionList).option_count == 1
            assert "[保留方括号]" in str(
                app.query_one("#sessions", OptionList).get_option_at_index(0).prompt
            )
            assert (
                str(app.query_one("#workspace-filter-path", Static).render())
                == app._workspace_paths[str(selected)]
            )
            await pilot.click("#workspace-filter")
            overlay = picker.query_one("SelectOverlay", OptionList)
            await pilot.click(overlay, offset=(2, 1))
            await wait_for(
                pilot,
                lambda: (
                    picker.value == ""
                    and not app.query_one("#sessions", OptionList).disabled
                ),
            )
            assert app.query_one("#sessions", OptionList).option_count == 2
            picker.value = b.workspace_id
            await wait_for(pilot, lambda: app._session_workspace == b.workspace_id)
            assert not bridge.attached and app.projection.session_id == "session-a"
            assert (
                bridge.project_root == a.project_root
                and composer.text == "原工作区未发送草稿"
            )
            assert str(a.project_root) in str(
                app.query_one("#new-session", Button).tooltip
            )
            app.action_new_session()
            await wait_for(
                pilot,
                lambda: bridge.attached == [None] and app._session_workspace is None,
            )
            assert (
                bridge.project_root == a.project_root
                and composer.text == "原工作区未发送草稿"
            )
            await pilot.resize_terminal(70, 24)
            assert not app.query_one("#sidebar").display
            app.action_sidebar()
            await pilot.pause()
            picker.value = b.workspace_id
            await wait_for(
                pilot, lambda: not app.query_one("#sessions", OptionList).disabled
            )
            assert app.query_one("#sidebar").size.width <= 32
            assert app.query_one("#chat").size.width >= 38
            assert app.query_one("#sessions", OptionList).region.height >= 3
            composer.focus()
            composer.move_cursor(composer.document.end)
            await pilot.press("x")
            assert composer.text.endswith("x") and not bridge.submitted

    asyncio.run(scenario())


@pytest.mark.parametrize("old_error", [False, True])
def test_previous_workspace_reply_does_not_replace_new_filter(old_error):
    """旧筛选成功和失败都不得覆盖新选择；参数：旧响应是否失败；返回：无。"""
    entered, release = Event(), Event()

    class DirectoryBridge(DisplayBridge):
        """提供可控的慢工作区查询。"""

        def browse(self, method, **params):
            """按工作区返回目录并隔离旧响应；参数：方法和选择；返回：摘要。"""
            selected = params.get("workspace_id")
            if selected == "old":
                entered.set()
                assert release.wait(WAIT_SECONDS)
                if old_error:
                    raise OSError("旧工作区失败")
            return {
                "sessions": [
                    {
                        "session_id": selected or "all",
                        "title": selected or "全部",
                        "status": "idle",
                    }
                ],
                "workspaces": [
                    {
                        "workspace_id": value,
                        "name": value,
                        "project_root": f"D:/{value}",
                    }
                    for value in ("old", "new")
                ],
                "next_before": None,
            }

    async def scenario():
        """快速改选后释放旧请求；参数：无；返回：无。"""
        app = InteractiveTui(DirectoryBridge())
        try:
            async with app.run_test(size=(120, 40)) as pilot:
                await wait_for(pilot, lambda: len(app._workspace_options) == 3)
                picker = app.query_one("#workspace-filter", Select)
                picker.value = "old"
                await wait_for(pilot, entered.is_set)
                picker.value = "new"
                await wait_for(
                    pilot, lambda: not app.query_one("#sessions", OptionList).disabled
                )
                release.set()
                await app.workers.wait_for_complete()
                await pilot.pause()
                assert picker.value == "new" and app._session_workspace == "new"
                assert (
                    app.query_one("#sessions", OptionList).get_option_at_index(0).id
                    == "new"
                )
                assert "旧工作区失败" not in str(
                    app.query_one("#notice", Static).render()
                )
        finally:
            release.set()

    asyncio.run(scenario())
