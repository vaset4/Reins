"""标准连接的惰性启动、复用、失败证据与宿主关闭。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

import logging
import sys
from builtins import ExceptionGroup
from contextlib import asynccontextmanager, closing
from pathlib import Path

import pytest

from tools.mcp_client.server_proc import ServerProc, ServerUnavailableError
from tools.mcp_client.transport import ConnectionSettings
from tools.types import ToolError, ToolErrorCategory

FIXTURE = Path(__file__).parent / "fixtures" / "standard_mcp_server.py"


def test_server_reuses_standard_session_and_closes_real_process(tmp_path, caplog):
    """读取目录及多次调用只启动同一服务，诊断不混入正文；传参：隔离根；返回：无。"""
    settings = ConnectionSettings(
        name="local",
        command=(sys.executable, str(FIXTURE.resolve())),
        env={"MCP_TEST_TRACE_DIR": str(tmp_path)},
    )
    with (
        closing(ServerProc(settings)) as server,
        caplog.at_level(logging.INFO, logger="tools.mcp_client.session"),
    ):
        assert (
            server.status == "unstarted" and not (tmp_path / "server_started").exists()
        )
        assert "echo" in {item["name"] for item in server.list_tools()}
        first = server.call_tool("echo", {"text": "first"})
        second = server.call_tool("echo", {"text": "second"})
        assert first["structuredContent"]["pid"] == second["structuredContent"]["pid"]
        assert first["mcp"]["request_id"] != second["mcp"]["request_id"]
        assert (
            "fixture diagnostic" not in str(first)
            and "fixture diagnostic" in caplog.text
        )
    assert server.status == "closed" and (tmp_path / "server_exited").exists()


def test_initialization_failure_keeps_leaf_reason_and_does_not_retry(monkeypatch):
    """SDK任务组失败仍显示根因，普通再次调用不重启连接；传参：故障注入；返回：无。"""
    attempts = []

    @asynccontextmanager
    async def fail_session(_settings):
        """模拟SDK初始化阶段失败；传参：连接设置；返回：不产生会话。"""
        attempts.append("start")
        raise ExceptionGroup(
            "SDK task group", [FileNotFoundError("server executable missing")]
        )
        yield

    monkeypatch.setattr("tools.mcp_client.session.open_session", fail_session)
    with closing(
        ServerProc(ConnectionSettings(name="broken", command=("unused",)))
    ) as server:
        for _ in range(2):
            with pytest.raises(
                ServerUnavailableError, match="server executable missing"
            ):
                server.start()
        assert server.status == "unavailable"
    assert attempts == ["start"]


def test_disconnected_call_retains_unknown_effect_and_requires_explicit_reconnect(
    tmp_path,
):
    """服务收到调用后断开时不盲目重放，连接明确失效；传参：隔离根；返回：无。"""
    settings = ConnectionSettings(
        name="local",
        command=(sys.executable, str(FIXTURE.resolve())),
        env={"MCP_TEST_TRACE_DIR": str(tmp_path)},
    )
    with closing(ServerProc(settings)) as server:
        result = server.call_tool("crash_connection", {})
        assert (
            isinstance(result, ToolError)
            and result.category is ToolErrorCategory.TRANSPORT
        )
        assert result.details["execution_state"] == "unknown"
        assert (tmp_path / "call_before_crash").exists()
        assert server.check_available()[0] is False
        with pytest.raises(ServerUnavailableError):
            server.call_tool("echo", {"text": "must not reconnect"})
