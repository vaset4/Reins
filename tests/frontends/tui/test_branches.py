"""真实会话数据的分支浏览与明确继续交互。

作者：xxx
时间：2026-09-29 21:30:00
"""

import asyncio
from functools import partial
from threading import Event

import pytest
from textual.app import App
from textual.widgets import Button, Static, Tree

from app.background.navigation import history_page, session_tree
from frontends.tui.branches import BranchScreen
from frontends.tui.paged_text import PagedText, TEXT_PAGE_CHARACTERS
from llm.messages import TextPart, UserMessage
from runtime.session_message_store import DEFAULT_HISTORY_PAGE_SIZE, SessionMessageStore


def browse(root, method, **params):
    """通过实际分页后端读取隔离数据；参数：根、方法、查询；返回：真实页。"""
    return {"session_tree": session_tree, "history_page": history_page}[method](
        root, **params
    )


def node_with_identity(tree, identity):
    """按身份找到当前页的实际节点；参数：树、条目身份；返回：可操作节点。"""
    pending = [tree.root]
    while pending:
        node = pending.pop()
        if node.data == identity:
            return node
        pending.extend(node.children)
    raise AssertionError(f"missing branch node: {identity}")


def test_branch_browser_preserves_original_path_until_explicit_continue(tmp_path):
    """浏览旧分支不改叶，明确继续返回原会话节点；传参：隔离目录；返回：无。"""
    store = SessionMessageStore(tmp_path)
    first = store.append_message(
        "session-tree", UserMessage("one", (TextPart("共同开头"),))
    )
    old = store.append_message(
        "session-tree", UserMessage("two", (TextPart("旧分支正文"),))
    )
    store.branch("session-tree", first.entry_id)
    current = store.append_message(
        "session-tree", UserMessage("three", (TextPart("当前分支正文"),))
    )
    before = store.read_entries("session-tree")
    snapshot = session_tree(tmp_path, "session-tree")
    assert snapshot["leaf_id"] == current.entry_id
    assert len(snapshot["entries"]) == len(before)

    async def scenario():
        app = App()
        choices = []
        async with app.run_test(size=(120, 40)) as pilot:
            screen = BranchScreen(snapshot, browse=partial(browse, tmp_path))
            app.push_screen(screen, choices.append)
            await pilot.pause()
            tree = screen.query_one(Tree)
            old_node = tree.root.children[0].children[0]
            tree.select_node(old_node)
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            detail = screen.query_one(PagedText).text
            assert "共同开头" in detail and "旧分支正文" in detail
            assert "当前分支正文" not in detail
            assert store.read_entries("session-tree") == before
            await pilot.click("#close-branch")
            assert choices == [None]
            screen = BranchScreen(snapshot, browse=partial(browse, tmp_path))
            app.push_screen(screen, choices.append)
            await pilot.pause()
            tree = screen.query_one(Tree)
            tree.select_node(tree.root.children[0].children[0])
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#continue-branch")
            assert choices[-1] == ("session-tree", old.entry_id)
            assert store.read_entries("session-tree") == before

    asyncio.run(scenario())


def test_branch_directory_and_history_pages_are_bounded_and_keep_original_body(
    tmp_path, monkeypatch
):
    """跨目录页时父节点仍可辨认，按需原文不受摘要长度影响；参数：隔离根；返回：无。"""
    store = SessionMessageStore(tmp_path)
    large_body = "完整历史" * TEXT_PAGE_CHARACTERS + "正文末尾"
    entries = [
        store.append_message("s", UserMessage(f"m-{index}", (TextPart(str(index)),)))
        for index in range(DEFAULT_HISTORY_PAGE_SIZE + 2)
    ]
    latest = store.append_message("s", UserMessage("large", (TextPart(large_body),)))
    before = store.read_entries("s")

    def reject_body_read(*args, **kwargs):
        """目录阶段禁止解码正文；参数：底层读参数；返回：断言失败。"""
        raise AssertionError("branch directory must not decode message bodies")

    with monkeypatch.context() as patch:
        from runtime.file_content import ContentFiles

        patch.setattr(ContentFiles, "read", reject_body_read)
        snapshot = session_tree(tmp_path, "s")
    assert len(snapshot["entries"]) == DEFAULT_HISTORY_PAGE_SIZE
    assert "text" not in snapshot["entries"][0]

    def read_page(method, **params):
        """正文使用短页检验前后翻阅；参数：查询；返回：真实页。"""
        if method == "history_page":
            params["limit"] = 2
        return browse(tmp_path, method, **params)

    async def scenario():
        app = App()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = BranchScreen(snapshot, browse=read_page)
            app.push_screen(screen)
            await pilot.pause()
            await pilot.click("#branch-tree-next")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(screen.entries) == 3
            tree = screen.query_one(Tree)
            assert (
                tree.root.children[0].data
                == entries[DEFAULT_HISTORY_PAGE_SIZE - 1].entry_id
            )
            assert "前页父节点" in str(tree.root.children[0].label)
            assert "← 当前" in str(node_with_identity(tree, latest.entry_id).label)
            tree.select_node(node_with_identity(tree, latest.entry_id))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            text = screen.query_one(PagedText)
            assert large_body in text.text
            assert len(text.page_text) == TEXT_PAGE_CHARACTERS
            assert "正文末尾" not in text.page_text
            await pilot.click("#branch-history-older")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "正文末尾" not in text.text
            assert screen.history_page["leaf_id"] == latest.entry_id
            await pilot.click("#branch-history-newer")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert large_body in text.text
            await pilot.click("#branch-tree-previous")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert len(screen.entries) == DEFAULT_HISTORY_PAGE_SIZE
            assert latest.entry_id not in screen.entries
            assert screen.selected_entry == latest.entry_id
            assert large_body in text.text
            assert not screen.query_one("#continue-branch", Button).disabled
            assert store.read_entries("s") == before

    asyncio.run(scenario())


@pytest.mark.parametrize("old_query_fails", [False, True])
def test_late_selected_branch_query_does_not_replace_new_selection(
    tmp_path, old_query_fails
):
    """旧节点查询迟到不覆盖新节点正文或继续位置；参数：隔离根；返回：无。"""
    store = SessionMessageStore(tmp_path)
    first = store.append_message("s", UserMessage("first", (TextPart("第一处原文"),)))
    second = store.append_message("s", UserMessage("second", (TextPart("第二处原文"),)))
    entered, release = Event(), Event()

    def delayed_browse(method, **params):
        """只延迟第一处读取，保持真实历史内容；参数：查询；返回：真实页。"""
        if params.get("leaf_id") == first.entry_id:
            entered.set()
            assert release.wait(5)
            if old_query_fails:
                raise ValueError("旧分支读取失败")
        return browse(tmp_path, method, **params)

    async def scenario():
        app = App()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = BranchScreen(session_tree(tmp_path, "s"), browse=delayed_browse)
            app.push_screen(screen)
            await pilot.pause()
            tree = screen.query_one(Tree)
            tree.select_node(node_with_identity(tree, first.entry_id))
            assert await asyncio.to_thread(entered.wait, 2)
            tree.select_node(node_with_identity(tree, second.entry_id))
            await pilot.pause()
            assert screen.selected_entry == second.entry_id
            release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "第二处原文" in screen.query_one(PagedText).text
            assert screen.history_page["leaf_id"] == second.entry_id
            assert not screen.query_one("#continue-branch", Button).disabled
            assert "读取失败" not in str(
                screen.query_one("#branch-detail", Static).render()
            )

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_failed_history_page_keeps_reading_position_and_can_retry(tmp_path):
    """翻页失败保留已读页和节点，明确重试成功后才换页；参数：隔离根；返回：无。"""
    store = SessionMessageStore(tmp_path)
    store.append_message("s", UserMessage("first", (TextPart("更早正文"),)))
    latest = store.append_message(
        "s", UserMessage("last", (TextPart("最近正文" * TEXT_PAGE_CHARACTERS),))
    )
    attempts = []

    def failing_browse(method, **params):
        """首次更早查询失败，后续读取真实页；参数：查询；返回：真实历史。"""
        if method == "history_page":
            params["limit"] = 1
            if params.get("before"):
                attempts.append(params["before"])
                if len(attempts) == 1:
                    raise OSError("历史存储暂不可读")
        return browse(tmp_path, method, **params)

    async def scenario():
        """通过真实控件翻页并重试，不重新选择节点；参数：无；返回：无。"""
        app = App()
        async with app.run_test(size=(120, 40)) as pilot:
            screen = BranchScreen(session_tree(tmp_path, "s"), browse=failing_browse)
            app.push_screen(screen)
            await pilot.pause()
            tree = screen.query_one(Tree)
            tree.select_node(node_with_identity(tree, latest.entry_id))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            body = screen.query_one(PagedText)
            body.page = 1
            body.render_page()
            previous = body.text
            await pilot.click("#branch-history-older")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert body.text == previous and body.page == 1
            assert screen.selected_entry == latest.entry_id
            assert screen.history_cursors == [None]
            assert "历史存储暂不可读" in str(
                screen.query_one("#branch-detail", Static).render()
            )
            assert not screen.query_one("#branch-history-older", Button).disabled
            await pilot.pause(0.4)
            await pilot.click("#branch-history-older")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "更早正文" in body.text and "最近正文" not in body.text
            assert body.page == 0
            assert attempts == [latest.entry_id, latest.entry_id]

    asyncio.run(scenario())


def test_backend_branch_keeps_old_entries_and_working_files(tmp_path):
    """原后台分支动作只追加会话事实，不恢复或改写工作文件；传参：隔离目录；返回：无。"""
    from scripts.testing.llm import from_test_sequence
    from tests.test_background_sessions import session_for

    session = session_for(tmp_path, from_test_sequence(["unused"]))
    store = session.messages
    identity = session.record.session_id
    first = store.append_message(identity, UserMessage("one", (TextPart("起点"),)))
    old = store.append_message(identity, UserMessage("two", (TextPart("旧后续"),)))
    working = tmp_path / "work.txt"
    working.write_text("当前文件", encoding="utf-8")
    try:
        session.branch(first.entry_id)
        view = store.materialize(identity)
        assert old not in view.entries and old in store.read_entries(identity)
        assert working.read_text(encoding="utf-8") == "当前文件"
        assert session_tree(tmp_path, identity)["leaf_id"] == view.leaf_id
    finally:
        session.close()


def test_fullscreen_branch_browse_and_continue_preserve_draft_and_original_entries(
    tmp_path,
):
    """输入焦点按 F3 可浏览、关闭和真实继续，全程保留草稿；参数：隔离根；返回：无。"""
    from frontends.tui.composer import Composer
    from frontends.tui.interactive import InteractiveTui
    from scripts.testing.llm import from_test_sequence
    from tests.frontends.tui.test_interactive import DisplayBridge
    from tests.test_background_sessions import session_for

    session = session_for(tmp_path, from_test_sequence(["unused"]))
    identity = session.record.session_id
    first = session.messages.append_message(
        identity, UserMessage("first", (TextPart("分支起点"),))
    )
    session.messages.append_message(
        identity, UserMessage("old", (TextPart("原分支后续"),))
    )
    original = session.messages.read_entries(identity)
    working = tmp_path / "branch-working.txt"
    working.write_text("保留工作文件", encoding="utf-8")

    class BranchBridge(DisplayBridge):
        """仅用真实隔离后台代替连接传输，不替换分支动作。"""

        def browse(self, method, **params):
            """目录用测试空列表，分支读写委托后台；参数：方法与查询；返回：后台结果。"""
            if method == "list_sessions":
                return {"sessions": []}
            if method == "branch":
                assert params["session_id"] == identity
                session.branch(params["entry_id"])
                return {"ok": True}
            return browse(tmp_path, method, **params)

    async def scenario():
        app = InteractiveTui(BranchBridge())
        async with app.run_test(size=(120, 40)) as pilot:
            app.receive("snapshot", session.snapshot(history=True))
            await pilot.pause()
            composer = app.query_one(Composer)
            composer.load_text("尚未发送的中文\n第二行草稿")
            composer.focus()
            await pilot.press("f3")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, BranchScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert composer.text == "尚未发送的中文\n第二行草稿"
            assert session.messages.read_entries(identity) == original
            composer.focus()
            await pilot.press("f3")
            await app.workers.wait_for_complete()
            await pilot.pause()
            tree = app.screen.query_one(Tree)
            tree.select_node(node_with_identity(tree, first.entry_id))
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await pilot.click("#continue-branch")
            await app.workers.wait_for_complete()
            await pilot.pause()
            app.receive("snapshot", session.snapshot(history=True))
            await pilot.pause()
            assert composer.text == "尚未发送的中文\n第二行草稿"
            assert session.messages.read_entries(identity)[:-1] == original
            assert session.messages.materialize(identity).messages == (first.message,)
            assert working.read_text(encoding="utf-8") == "保留工作文件"

    try:
        asyncio.run(scenario())
    finally:
        session.close()


def test_branch_resets_stream_view_and_rejects_late_old_snapshot(tmp_path):
    """切换后不显示旧分支未提交输出，迟到旧快照不会复活旧路径；传参：隔离根；返回：无。"""
    from scripts.testing.llm import from_test_sequence
    from tests.test_background_sessions import session_for
    from frontends.tui.projection import ConversationProjection
    from runtime.stream_events import AssistantTextDelta

    session = session_for(tmp_path, from_test_sequence(["unused"]))
    identity = session.record.session_id
    first = session.messages.append_message(
        identity, UserMessage("first", (TextPart("共同起点"),))
    )
    session.messages.append_message(identity, UserMessage("old", (TextPart("旧分支"),)))
    old_events = session.events
    old_events.emit(
        AssistantTextDelta("旧分支尚未保存的回答", message_id="request-old"),
        run_id="run-old",
    )
    old_snapshot = session.snapshot(history=True)
    projection = ConversationProjection()
    projection.apply(old_snapshot)
    assert any("尚未保存" in card.text for card in projection.cards.values())
    try:
        session.branch(first.entry_id)
        current = session.snapshot(history=True)
        projection.apply(current)
        assert current["event_epoch"] != old_snapshot["event_epoch"]
        assert not current["streams"]
        assert [card.text for card in projection.cards.values()] == ["共同起点"]
        old_events.emit(
            AssistantTextDelta("迟到", message_id="request-old"), run_id="run-old"
        )
        projection.apply(old_snapshot)
        assert [card.text for card in projection.cards.values()] == ["共同起点"]
        assert any(
            entry.message and "旧分支" in str(entry.message)
            for entry in session.messages.read_entries(identity)
        )
    finally:
        session.close()
