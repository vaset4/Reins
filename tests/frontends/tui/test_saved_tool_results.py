"""正式工具执行器保存的产物经后台身份核验和 TUI 懒读保持完整。

作者：xxx
时间：2026-09-29 20:00:00
"""

import asyncio
from dataclasses import asdict
from threading import Event
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Collapsible, Static

from app.background.server import BackgroundServer
from app.background.sessions import session_history_page
from app.background.tool_results import tool_result_detail
from frontends.tui.composer import Composer
from frontends.tui.interactive import InteractiveTui, LIVE_CARD_WINDOW
from frontends.tui.paged_text import PagedText
from llm.messages import AssistantMessage, TextPart, ToolCallPart
from runtime.workspaces import WorkspaceStore
from runtime.session_message_store import SessionMessageStore, SessionMessageStoreError
from runtime.stream_events import AssistantTurnComplete
from runtime.tool_operations import ToolOperationStore
from scripts.testing.llm import from_test_native_tool_then_final
from tests.frontends.tui.test_interactive import DisplayBridge
from tests.frontends.tui.test_retention_ui import settle
from tests.test_background_sessions import session_for
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk

FULL_RESULT = "起始\r\n" + "原件" * 500000 + "中后段必须完整保留\r\n末尾"
RESULT_RENDER_TIMEOUT_SECONDS = 5


async def wait_for_saved_body(widget, pilot):
    """等待异步展开真正挂载原件，读取失败立即暴露；参数：卡片和驾驶器；返回：无。"""
    async with asyncio.timeout(RESULT_RENDER_TIMEOUT_SECONDS):
        while not widget.query(".tool-output"):
            statuses = widget.query(".saved-result-status").results(Static)
            for status in statuses:
                assert "原件读取失败" not in str(status.render()), str(status.render())
            await pilot.pause()


@pytest.fixture
def saved_result(tmp_path):
    """经正式后台、工具执行器和产物所有者保存大结果；参数：隔离根；返回：真实快照。"""
    effects = []

    def execute(arguments):
        """受控外部工具仅执行一次并返回大结果；参数：已校验入参；返回：完整原文。"""
        effects.append("executed")
        return FULL_RESULT

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "large_result",
            "读取大结果",
            {},
            "agent",
            ToolRisk.SAFE,
            True,
            "logical_scope",
            "builtin",
            idempotent=Idempotent.YES,
            executor=execute,
        )
    )
    client = from_test_native_tool_then_final(
        [ToolCallPart("large-call", "large_result", {})], "结果已保存"
    )
    session = session_for(tmp_path, client, registry=registry)
    try:
        session.submit("读取", input_id="large-input", model_config={})
        assert session.runtime.wait_idle(10)
        snapshot = session.snapshot(history=True)
        assert effects == ["executed"]
        source = next(
            row["result_source"] for row in snapshot["history"] if row["role"] == "tool"
        )
        assert source is not None
        yield snapshot, source, effects, session.snapshot(after=0)
    finally:
        session.close()


class ResultBridge(DisplayBridge):
    """仅替换进程传输，原件查询使用真实后台函数。"""

    def __init__(self, data_root):
        """保存查询根和故障控制；参数：隔离根；返回：无。"""
        super().__init__()
        self.data_root = data_root
        self.reads = 0
        self.fail = False
        self.started = Event()
        self.read_release = Event()
        self.read_release.set()

    def browse(self, method, **params):
        """查询正式原件或受控目录；参数：后台方法和来源；返回：已核验正文。"""
        if method == "history_page":
            return session_history_page(self.data_root, **params)
        if method != "tool_result_detail":
            return super().browse(method, **params)
        self.reads += 1
        self.started.set()
        if not self.read_release.wait(5):
            raise TimeoutError("测试没有释放原件读取")
        if self.fail:
            raise FileNotFoundError("保存的原件已丢失")
        return tool_result_detail(self.data_root, **params)


def test_tool_artifact_route_verifies_persisted_identity(
    tmp_path, saved_result, monkeypatch
):
    """完整原件只能由匹配的消息和操作读取，分页不预读；参数：隔离根和真实结果；返回：无。"""
    snapshot, source, effects, _ = saved_result
    tool = next(row for row in snapshot["history"] if row["role"] == "tool")
    assert len(tool["text"]) < len(FULL_RESULT)
    from app.background import tool_results

    reads = []
    original = tool_results.read_artifact

    def record_read(*args, **kwargs):
        """记录实际产物读取而不替换内容；参数：读取参数；返回：真实原件。"""
        reads.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(tool_results, "read_artifact", record_read)
    session_history_page(tmp_path, source["session_id"])
    assert not reads
    server = BackgroundServer(
        SimpleNamespace(
            services=SimpleNamespace(data_root=tmp_path),
            workspaces=WorkspaceStore(tmp_path),
        ),
        "test-token",
    )
    try:
        assert server.dispatch("tool_result_detail", source)["text"] == FULL_RESULT
        assert len(reads) == 1
        assert effects == ["executed"]
        with pytest.raises(ValueError, match="运行或调用"):
            server.dispatch("tool_result_detail", {**source, "run_id": "other-run"})
        with pytest.raises(ValueError, match="运行或调用"):
            server.dispatch("tool_result_detail", {**source, "call_id": "other-call"})
        with pytest.raises(SessionMessageStoreError):
            server.dispatch(
                "tool_result_detail", {**source, "session_id": "other-session"}
            )
        operation_store = ToolOperationStore(tmp_path)
        operation = operation_store.for_session(source["session_id"])[0]
        identity = {
            key: operation[key] for key in ("session_id", "run_id", "operation_id")
        }
        altered = {
            **operation["result"],
            "meta": {
                **operation["result"]["meta"],
                "result_artifact_id": "other-artifact",
            },
        }
        operation_store.write(identity, {**operation, "result": altered})
        with pytest.raises(ValueError, match="引用与已提交操作不一致"):
            server.dispatch("tool_result_detail", source)
    finally:
        server.server_close()


def saved_completions(data_root, source, cursor):
    """保存足以移出实时窗口的新消息；参数：根、所属来源与事件游标；返回：对应收尾事件。"""
    store = SessionMessageStore(data_root)
    events = []
    with store.database.transaction():
        for index in range(LIVE_CARD_WINDOW + 1):
            message = AssistantMessage(
                f"later-{index}", (TextPart(f"后续已保存回答{index}"),)
            )
            entry = store.append_message(
                source["session_id"], message, run_id=source["run_id"]
            )
            event = AssistantTurnComplete(
                f"后续已保存回答{index}",
                message_id=message.message_id,
                entry_id=entry.entry_id,
            )
            events.append(
                {
                    "sequence": cursor + index + 1,
                    "run_id": source["run_id"],
                    "type": type(event).__name__,
                    "data": asdict(event),
                }
            )
    return events


def test_tui_reads_full_artifact_only_on_expand_and_releases_on_collapse(
    tmp_path, saved_result, monkeypatch
):
    """正式大结果可看末页、复制全文且收起释放，后续投影不写回预览；参数：真实快照；返回：无。"""
    snapshot, source, _, _ = saved_result
    events = saved_completions(tmp_path, source, snapshot["cursor"])

    async def scenario():
        """驱动生产卡片与原件读取；参数：无；返回：无。"""
        bridge = ResultBridge(tmp_path)
        app = InteractiveTui(bridge)
        copied = []
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.receive("snapshot", snapshot)
            await settle(app, pilot)
            tool_key = next(
                key for key, card in app.projection.cards.items() if card.role == "tool"
            )
            app.receive(
                "snapshot",
                {
                    "session_id": source["session_id"],
                    "event_epoch": snapshot["event_epoch"],
                    "data_space_id": snapshot["data_space_id"],
                    "events": events,
                },
            )
            await settle(app, pilot)
            assert app.projection.cursor == events[-1]["sequence"]
            app.return_to_live()
            await settle(app, pilot)
            assert tool_key not in app.projection.cards
            app.action_history()
            await app.workers.wait_for_complete()
            await settle(app, pilot)
            assert app._history_page["next_before"] is not None
            await pilot.click("#history-older")
            await app.workers.wait_for_complete()
            await settle(app, pilot)
            widget = next(
                widget for widget in app.cards.values() if widget.card.role == "tool"
            )
            assert asdict(widget.card.result_source) == source
            assert bridge.reads == 0 and not widget.query(PagedText)
            app.query_one(Composer).load_text("仍可编辑的草稿")
            details = widget.query_one(Collapsible)
            details.collapsed = False
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await wait_for_saved_body(widget, pilot)
            body = widget.query_one(".tool-output", PagedText)
            assert body.text == FULL_RESULT
            body.page = body.page_count - 1
            body.render_page()
            assert body.page_text.endswith("中后段必须完整保留\r\n末尾")
            next(
                button for button in body.query(Button) if button.name == "copy"
            ).press()
            await pilot.pause()
            assert copied == [FULL_RESULT]
            await widget.update_card(widget.card)
            assert body.text == FULL_RESULT
            details.collapsed = True
            await pilot.pause()
            assert not widget.query(".tool-output")
            details.collapsed = False
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            await wait_for_saved_body(widget, pilot)
            assert widget.query_one(".tool-output", PagedText).page == body.page
            assert bridge.reads == 2
            assert app.query_one(Composer).text == "仍可编辑的草稿"

    asyncio.run(scenario())


def test_tui_failed_or_late_artifact_read_does_not_replace_draft(
    tmp_path, saved_result
):
    """读取失败明确显示，切会话后的迟到原件不会串入；参数：真实快照；返回：无。"""
    snapshot, _, _, _ = saved_result

    async def scenario():
        """在原件IO边界注入失败和等待；参数：无；返回：无。"""
        bridge = ResultBridge(tmp_path)
        bridge.fail = True
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.receive("snapshot", snapshot)
            await settle(app, pilot)
            widget = next(
                widget for widget in app.cards.values() if widget.card.role == "tool"
            )
            details = widget.query_one(Collapsible)
            details.collapsed = False
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "原件读取失败" in str(
                widget.query_one(".saved-result-status", Static).render()
            )
            assert not widget.query(".tool-output")
            details.collapsed = True
            await pilot.pause()
            bridge.fail = False
            bridge.started.clear()
            bridge.read_release.clear()
            details.collapsed = False
            await pilot.pause()
            assert await asyncio.to_thread(bridge.started.wait, 2)
            app.receive("snapshot", {"session_id": "other-session", "history": []})
            await settle(app, pilot)
            composer = app.query_one(Composer)
            composer.load_text("新会话草稿")
            bridge.read_release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert composer.text == "新会话草稿"
            assert not app.cards

    asyncio.run(scenario())


def test_live_executor_result_expands_complete_without_history_query(
    tmp_path, saved_result
):
    """实时完成事件直接提供完整工具正文，无需先切历史或重连；参数：真实执行；返回：无。"""
    snapshot, _, effects, live = saved_result

    async def scenario():
        """从空连接接收真实流，再读工具末页；参数：无；返回：无。"""
        bridge = ResultBridge(tmp_path)
        app = InteractiveTui(bridge)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            app.receive(
                "snapshot",
                {
                    "session_id": snapshot["session_id"],
                    "event_epoch": snapshot["event_epoch"],
                    "history": [],
                    "cursor": 0,
                },
            )
            app.receive("snapshot", live)
            await settle(app, pilot)
            widget = next(
                widget for widget in app.cards.values() if widget.card.role == "tool"
            )
            assert widget.card.text == FULL_RESULT
            widget.query_one(Collapsible).collapsed = False
            await pilot.pause()
            body = widget.query_one(".tool-output", PagedText)
            body.page = body.page_count - 1
            body.render_page()
            assert body.page_text.endswith("中后段必须完整保留\r\n末尾")
            assert bridge.reads == 0 and effects == ["executed"]

    asyncio.run(scenario())
