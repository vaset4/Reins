"""脚本客户端保留生产请求边界，同时隔离配置、密钥和网络。

作者：xxx；时间：2026-09-28 18:00:00
"""

from __future__ import annotations

import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from llm.client import RealLLMClient
from scripts.testing import llm
from tools.tool_registry import ToolRegistry


@pytest.mark.parametrize(
    "factory",
    [
        lambda: llm.from_test_stub("完成"),
        lambda: llm.from_test_text_json_stub('{"type":"final","content":"完成"}'),
        lambda: llm.from_test_streaming_turn(("依据",), answer=("完", "成")),
        lambda: llm.from_test_native_tool_calls([], content="完成"),
        lambda: llm.from_test_native_tool_then_final([], "完成"),
        lambda: llm.from_test_error("受控失败"),
        lambda: llm.from_test_sequence(["完成"]),
        lambda: llm.from_test_turns(["完成"]),
    ],
)
def test_factories_do_not_resolve_credentials_or_connect(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], RealLLMClient],
) -> None:
    """全部工厂仅使用合成依赖；传参：替换器和工厂；返回：无。"""

    def forbidden(*_args: object, **_kwargs: object) -> None:
        """真实配置或网络访问立即失败；传参：拦截调用；返回：不返回。"""
        pytest.fail(
            "scripted client attempted production configuration or network access"
        )

    monkeypatch.setattr("llm.client.production_connection", forbidden)
    monkeypatch.setattr("llm.client.production_model_registry", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    client = factory()
    assert type(client) is RealLLMClient
    plan = client.plan("独立合成输入", {"tool_registry": ToolRegistry()})
    assert plan.observation is not None
    assert plan.observation.model == "stub-model"
    assert plan.observation.attempt_count == 1


def test_script_exhaustion_remains_a_provider_failure() -> None:
    """用尽的脚本不能虚构后续成功；传参：无；返回：无。"""
    client = llm.from_test_sequence(["一次完成"])
    assert (
        client.plan("输入", {"tool_registry": ToolRegistry()}).final_output
        == "一次完成"
    )
    exhausted = client.plan("下一输入", {"tool_registry": ToolRegistry()})
    assert exhausted.model_error is not None
    assert exhausted.model_error.category == "provider_error"
    assert "sequence exhausted" in exhausted.model_error.summary


def test_production_entry_imports_do_not_load_scripted_support() -> None:
    """新进程导入真实入口不加载测试实现；传参：无；返回：无。"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import app.cli, app.run_task, runtime.agent_loop, sys; "
            "assert not any(n.startswith('scripts.testing') for n in sys.modules)",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
