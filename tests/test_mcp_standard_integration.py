"""标准SDK服务从保存配置、产品入口到真实工具往返的集成验证。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_sequence

import json
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from pathlib import Path

import pytest
import yaml

from app.run_task import run_task
from approval import ApprovalDecision
from tests.support.approval import install_approval
from llm.messages import ToolResultMessage
from runtime.cancellation import CancellationToken
from runtime.default_capabilities import build_local_agent_capabilities
from runtime.lease import from_trigger
from runtime.session_messages import materialize_messages
from runtime.watchdog import Watchdog
from tools.mcp_client.registry import attach_mcp_registry
from tools.tool_registry import ToolRegistry
from tools.types import ToolError, ToolErrorCategory

FIXTURE = Path(__file__).parent / "fixtures" / "standard_mcp_server.py"
STARTUP_TIMEOUT_SECONDS = 15


def _wait_for(predicate):
    """等待真实IO就绪并限定测试时间；传参：就绪判据；返回：无。"""
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(
                "standard MCP fixture did not reach the expected state"
            )
        time.sleep(0.02)


def _port_ready(port):
    """检查本地服务已监听；传参：端口；返回：是否可连接。"""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.1):
            return True
    except OSError:
        return False


@contextmanager
def _configured_service(tmp_path, monkeypatch, transport):
    """保存真实用户配置，按传输准备官方SDK服务；传参：临时根及传输；返回：配置路径。"""
    server = {
        "transport": transport,
        "timeout_seconds": 10,
        "env": {"MCP_TEST_TRACE_DIR": str(tmp_path)},
    }
    process = None
    log = (tmp_path / "server.log").open("wb")
    try:
        if transport == "stdio":
            server["command"] = [sys.executable, str(FIXTURE.resolve())]
        else:
            with closing(socket.socket()) as reserved:
                reserved.bind(("127.0.0.1", 0))
                port = reserved.getsockname()[1]
            mode = "streamable-http" if transport == "streamable_http" else "sse"
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(FIXTURE.resolve()),
                    "--transport",
                    mode,
                    "--port",
                    str(port),
                ],
                stdout=log,
                stderr=log,
                env={**os.environ, "MCP_TEST_TRACE_DIR": str(tmp_path)},
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            server["url"] = f"http://127.0.0.1:{port}/" + (
                "mcp" if transport == "streamable_http" else "sse"
            )
            _wait_for(lambda: _port_ready(port))
        config = tmp_path / "mcp.yaml"
        config.write_text(
            yaml.safe_dump(
                {"allow_servers": ["test_server"], "servers": {"test_server": server}}
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr("tools.mcp_client.config.MCP_CONFIG_PATH", config)
        install_approval(monkeypatch, lambda _request: ApprovalDecision.ONCE)
        yield config
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log.close()


@pytest.mark.parametrize("transport", ["stdio", "streamable_http", "sse"])
def test_saved_config_drives_product_entry_to_standard_sdk_server(
    tmp_path, monkeypatch, transport
):
    """三类声明支持传输均从产品入口实际往返，业务失败不变成成功；传参：配置环境；返回：无。"""
    with _configured_service(tmp_path, monkeypatch, transport):
        names = ["mcp_test_server_echo", "mcp_test_server_business_error"]
        responses = [
            json.dumps(
                {
                    "type": "run_tools",
                    "tool": "capabilities",
                    "arguments": {"action": "load", "name": names[0]},
                }
            ),
            json.dumps(
                {
                    "type": "run_tools",
                    "tool": names[0],
                    "arguments": {"text": "实际SDK往返"},
                }
            ),
            json.dumps(
                {
                    "type": "run_tools",
                    "tool": "capabilities",
                    "arguments": {"action": "load", "name": names[1]},
                }
            ),
            json.dumps({"type": "run_tools", "tool": names[1], "arguments": {}}),
            json.dumps({"type": "final", "content": "保留成功资料，账单撤回已说明"}),
        ]
        client = from_test_sequence(responses, protocol_mode="text_json")
        response = run_task(
            "查询资料并核对账单",
            tmp_path,
            data_root=tmp_path / "data",
            llm_client=client,
            session_id="session-mcp",
        )
        assert response.status == "done", response.output
        results = {
            item.tool_name: json.loads(item.content[0].text)
            for item in materialize_messages(tmp_path / "data", "session-mcp")
            if isinstance(item, ToolResultMessage)
        }
        echo = json.loads(results[names[0]]["output"])
        assert echo["structuredContent"]["text"] == "实际SDK往返"
        assert [item["type"] for item in echo["content"]] == [
            "text",
            "image",
            "resource",
        ]
        assert echo["mcp"]["protocol_version"] and isinstance(
            echo["mcp"]["request_id"], int
        )
        failed = results[names[1]]
        assert (
            failed["status"] == "error"
            and failed["meta"]["tool_error_category"] == "business"
        )
        assert failed["meta"]["mcp_result"]["structuredContent"] == {
            "reason": "withdrawn"
        }
        assert (
            echo["mcp"]["request_id"]
            != failed["meta"]["mcp_result"]["mcp"]["request_id"]
        )
        if transport == "stdio":
            assert (tmp_path / "server_exited").exists()


def test_reused_connection_multiplexes_calls_and_refreshes_standard_notification(
    tmp_path, monkeypatch
):
    """并发调用共享一个会话但保留独立ID，目录通知只改变新快照；传参：隔离环境；返回：无。"""
    with (
        _configured_service(tmp_path, monkeypatch, "stdio"),
        closing(ToolRegistry()) as registry,
    ):
        lease = from_trigger(
            "user", capabilities=build_local_agent_capabilities(tmp_path, tmp_path)
        )
        owner = attach_mcp_registry(lease, registry)
        assert not owner.failures, owner.describe()
        server = owner.servers["test_server"]
        with ThreadPoolExecutor(max_workers=2) as executor:
            a = executor.submit(server.call_tool, "echo", {"text": "A", "delay": 0.1})
            b = executor.submit(server.call_tool, "echo", {"text": "B", "delay": 0})
            first, second = a.result(timeout=5), b.result(timeout=5)
        assert (
            first["structuredContent"]["text"] == "A"
            and second["structuredContent"]["text"] == "B"
        )
        assert first["structuredContent"]["pid"] == second["structuredContent"]["pid"]
        assert first["mcp"]["request_id"] != second["mcp"]["request_id"]
        assert attach_mcp_registry(lease, registry) is owner
        before = registry.snapshot()
        server.call_tool("publish_capability", {})
        registry.refresh_sources(lease)
        assert before.get("mcp_test_server_late_capability") is None
        assert registry.get("mcp_test_server_late_capability") is not None
        assert before.version != registry.version


@pytest.mark.parametrize("transport", ["stdio", "streamable_http", "sse"])
def test_user_stop_reaches_standard_server_request_handler(
    tmp_path, monkeypatch, transport
):
    """通过真实执行器取消，验证服务端处理收到通知；传参：隔离环境；返回：无，未知副作用仍如实记录。"""
    with (
        _configured_service(tmp_path, monkeypatch, transport),
        closing(ToolRegistry()) as registry,
    ):
        lease = from_trigger(
            "user", capabilities=build_local_agent_capabilities(tmp_path, tmp_path)
        )
        attach_mcp_registry(lease, registry)
        token = CancellationToken()
        prepared = registry.prepare_tool_execution(
            "mcp_test_server_wait_for_cancel",
            {},
            lease,
            cancellation=token,
            watchdog=Watchdog(lease, data_root=tmp_path),
            operation_id="cancelled-mcp-operation",
        )
        assert not isinstance(prepared, ToolError)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(registry.execute_prepared_tool, prepared)
            _wait_for(lambda: (tmp_path / "call_started").exists())
            token.cancel("user_stop")
            result = pending.result(timeout=5)
        _wait_for(lambda: (tmp_path / "call_cancelled").exists())
        assert (
            isinstance(result, ToolError)
            and result.category is ToolErrorCategory.CANCELLED
        )
        assert token.backend_evidence()["mcp_cancel_sent"] is True


def test_mcp_timeout_cancels_original_request_and_keeps_session_usable(
    tmp_path, monkeypatch
):
    """工具超时只取消原请求，未知副作用不重试；传参：隔离环境；返回：无，后续只读请求仍能实际执行。"""
    with (
        _configured_service(tmp_path, monkeypatch, "stdio"),
        closing(ToolRegistry()) as registry,
    ):
        lease = from_trigger(
            "user", capabilities=build_local_agent_capabilities(tmp_path, tmp_path)
        )
        owner = attach_mcp_registry(lease, registry)
        server = owner.servers["test_server"]
        result = server.call_tool("wait_for_cancel", {}, timeout_seconds=0.1)
        assert (
            isinstance(result, ToolError)
            and result.category is ToolErrorCategory.TIMEOUT
        )
        assert (
            result.details["execution_state"] == "unknown" and result.retryable is False
        )
        _wait_for(lambda: (tmp_path / "call_cancelled").exists())
        following = server.call_tool("echo", {"text": "after timeout"})
        assert following["structuredContent"]["text"] == "after timeout"
