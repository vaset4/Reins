"""【会话体验】【真实目录与样式】验证维护归属、线性历史和受限终端尺寸。

作者：xxx
时间：2026-10-02 11:30:00
"""

import asyncio

import pytest
from textual.widgets import OptionList, Tree

from app.background.navigation import session_tree
from frontends.tui.branches import BranchScreen
from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui
from llm.messages import AssistantMessage, TextPart, UserMessage
from runtime.persistence import RuntimeStore
from runtime.session_directory import directory_page
from runtime.session_message_store import SessionMessageStore
from runtime.workspaces import WorkspaceStore
from tests.frontends.tui.test_interactive import DisplayBridge


def test_default_directory_groups_eleven_auxiliary_sessions_without_deletion(tmp_path):
    """十二个原会话默认显示一个聊天，维护可按所属会话检索；参数：隔离根；返回：无。"""
    workspaces = WorkspaceStore(tmp_path)
    workspaces.bind_session("chat", tmp_path)
    with workspaces.database.transaction() as batch:
        for index in range(11):
            identity = f"auxiliary-{index}"
            workspaces.bind_session(identity, tmp_path)
            workspaces.messages.append_message(
                identity, UserMessage(f"m-{index}", (TextPart("维护查询标记"),))
            )
            batch.put(
                "knowledge_maintenance",
                f"work-{index}",
                {
                    "record_type": "work",
                    "work_id": f"work-{index}",
                    "source_session_id": "chat",
                    "worker_session_id": identity,
                    "state": "completed",
                },
                session_id="chat",
            )
    page = directory_page(tmp_path)
    assert [row["session_id"] for row in page["sessions"]] == ["chat"]
    assert page["sessions"][0]["maintenance_count"] == 11
    auxiliary = directory_page(
        tmp_path, "维护查询标记", purpose="maintenance", owner_session_id="chat"
    )
    assert len(auxiliary["sessions"]) == 11
    assert all(row["owner_session_id"] == "chat" for row in auxiliary["sessions"])
    assert len(directory_page(tmp_path, purpose="all")["sessions"]) == 12
    with RuntimeStore(tmp_path).snapshot() as source:
        assert len(source.list("session")) == 12


@pytest.mark.parametrize("size", [(120, 30), (120, 40), (160, 50), (80, 24)])
def test_production_css_keeps_directory_body_and_draft_usable(tmp_path, size):
    """正式样式按宽高留下列表与输入，分支导航填满容器；参数：隔离根/终端尺寸；返回：无。"""
    messages = SessionMessageStore(tmp_path)
    for index in range(4):
        messages.append_message(
            "chat", UserMessage(f"input-{index}", (TextPart(f"第{index}轮用户问题"),))
        )
        messages.append_message(
            "chat", AssistantMessage(f"answer-{index}", (TextPart("已保存的回答"),))
        )
    snapshot = session_tree(tmp_path, "chat")

    async def scenario():
        """运行正式应用及浮层，检查可阅读区域与关闭后的草稿；参数：无；返回：无。"""
        app = InteractiveTui(DisplayBridge())
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            if not app.query_one("#sidebar").display:
                app.action_sidebar()
            app.update_workspace_options(
                [
                    {
                        "workspace_id": "workspace",
                        "project_root": str(tmp_path),
                        "name": "资料",
                    }
                ]
            )
            app._session_workspace = "workspace"
            app.update_workspace_options(
                [
                    {
                        "workspace_id": "workspace",
                        "project_root": str(tmp_path),
                        "name": "资料",
                    }
                ]
            )
            composer = app.query_one(Composer)
            composer.load_text("保留草稿")
            await pilot.pause()
            height = app.query_one(
                "#sessions", OptionList
            ).scrollable_content_region.height
            assert height >= (9 if size == (120, 30) else 3)
            assert composer.region.height >= 3
            assert app.query_one("#conversation").region.height > 0
            screen = BranchScreen(
                snapshot,
                browse=lambda method, **params: session_tree(
                    tmp_path, params["session_id"]
                ),
            )
            app.push_screen(screen)
            await pilot.pause()
            tree = screen.query_one(Tree)
            assert (
                tree.region.width
                >= screen.query_one("#branch-directory").content_region.width - 2
            )
            assert len(tree.root.children) == len(snapshot["entries"])
            screen.dismiss(None)
            await pilot.pause()
            assert composer.text == "保留草稿"

    asyncio.run(scenario())
