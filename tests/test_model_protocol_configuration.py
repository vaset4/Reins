"""从配置保存与生产装配到真实SDK HTTP请求的协议接线验证。

作者：xxx
时间：2026-09-14 20:00:00
"""

from __future__ import annotations

import json
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from app.cli import build_llm_client
from llm.config import load_saved_config, save_user_config
from llm.profiles import save_model_profile
from llm.resolved_target import resolve_model_target
from runtime.agent_loop import AgentLoop, State
from runtime.workspaces import WorkspaceStore
from runtime.lease import from_trigger
from runtime.run_facts import RunFactStore
from runtime.session_messages import append_user_message
from runtime.types import RunContext, Trigger
from tasks.store import TaskStore
from tools.tool_registry import Idempotent, ToolDefinition, ToolRegistry, ToolRisk

FIXTURES = Path(__file__).parent / "fixtures" / "llm" / "providers"
PROTOCOL_CASES = (
    (
        "chat_completions",
        "openai_chat",
        "/v1/chat/completions",
        "max_completion_tokens",
    ),
    ("responses", "openai_responses", "/v1/responses", "max_output_tokens"),
    ("anthropic_messages", "anthropic_messages", "/v1/messages", "max_tokens"),
)


def _protocol_events(family, *, first):
    """复用已由官方SDK类型校验的事件，首轮提供一个工具调用；传参：协议和轮次；返回：SSE正文。"""
    name = (
        "stream_tool_fragmented.jsonl"
        if first and family != "anthropic_messages"
        else "stream_text.jsonl"
    )
    rows = [
        json.loads(line)
        for line in (FIXTURES / family / name).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if first and family == "anthropic_messages":
        rows[1] = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "c1",
                "name": "read",
                "input": {},
            },
        }
        rows[2] = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"path":"a.txt"}'},
        }
        rows[4]["delta"]["stop_reason"] = "tool_use"
    return "".join(
        (f"event: {row['type']}\n" if "type" in row else "")
        + "data: "
        + json.dumps(row)
        + "\n\n"
        for row in rows
    )


class _ProtocolHandler(BaseHTTPRequestHandler):
    """仅在测试回环端点接收真实SDK请求，保存协议及正文而不输出凭据。"""

    def do_POST(self):
        """按首轮/后续轮交付标准SSE；传参：HTTP请求；返回：无。"""
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.packets.append((self.path, body))
        payload = _protocol_events(
            self.server.family, first=len(self.server.packets) == 1
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format, *args):
        """测试服务器不向终端输出请求日志；传参：标准HTTP日志；返回：无。"""


@contextmanager
def _protocol_server(family):
    """启动独立的回环HTTP边界并关闭线程；传参：协议；返回：服务器与端点。"""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ProtocolHandler)
    server.family, server.packets = family, []
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("mode,family,path,output_key", PROTOCOL_CASES)
def test_saved_profile_drives_production_sdk_tool_roundtrip(
    tmp_path, monkeypatch, mode, family, path, output_key
):
    """保存的协议经过真实装配、SDK和工具结果回填；传参：隔离配置与协议；返回：无。"""
    data = tmp_path / "data"
    profiles = tmp_path / "models.yaml"
    monkeypatch.setattr("llm.profiles.MODELS_CONFIG_PATH", profiles)
    monkeypatch.setattr(
        "llm.profiles.MODELS_JSON_CONFIG_PATH", tmp_path / "models.json"
    )
    monkeypatch.setattr("llm.config.SAVED_CONFIG_PATH", tmp_path / "config.yaml")
    monkeypatch.setattr("app.cli.SecretsVault", lambda: {})
    monkeypatch.setenv("XIANGMU_LLM_API_KEY", "synthetic-protocol-configuration-token")
    (tmp_path / "a.txt").write_text("material-from-real-file", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            "read",
            "读取已授权资料",
            {"path": {"type": "string", "required": True}},
            "file",
            ToolRisk.SAFE,
            True,
            "path",
            "builtin",
            idempotent=Idempotent.YES,
            executor=lambda args: (tmp_path / args["path"]).read_text(encoding="utf-8"),
        )
    )
    with (
        _protocol_server(family) as (server, endpoint),
        closing(TaskStore(data)) as tasks,
    ):
        base = endpoint if family == "anthropic_messages" else endpoint + "/v1"
        save_model_profile(
            "configured",
            {
                "provider": "fixture",
                "model": "configured-model",
                "base_url": base,
                "api_mode": mode,
                "context_window": 12000,
                "max_output_tokens": 512,
            },
            profiles,
        )
        client = build_llm_client({}, tmp_path)
        task = tasks.create_task("读取资料")
        lease = from_trigger(
            "user",
            task_id=task.task_id,
            capabilities={
                "fs": {"project_root": str(tmp_path), "read": [str(tmp_path)]}
            },
        )
        context = RunContext(
            task_id=task.task_id,
            trigger=Trigger.USER,
            payload={"message": "读取资料"},
            capability_lease=lease,
        )
        WorkspaceStore(data).bind_session(context.session_id, tmp_path)
        context.payload["input_message_id"] = append_user_message(
            data, context.session_id, "读取资料"
        )
        loop = AgentLoop(data, llm_client=client, tool_registry=registry)
        assert loop.run(context) is State.DONE
        assert len(server.packets) == 2
        assert all(
            url == path
            and body[output_key] == 512
            and body["model"] == "configured-model"
            for url, body in server.packets
        )
        assert "material-from-real-file" in json.dumps(server.packets[1][1])
        assert "c1" in json.dumps(server.packets[1][1])
        attempts = [
            row
            for row in RunFactStore(data).read_run(context.run_id)
            if row.get("event") == "llm:attempt"
        ]
        assert len(attempts) == 2
        assert client.resolved_target.api_mode == mode


def test_saved_api_mode_is_retained_and_invalid_update_does_not_replace_it(tmp_path):
    """普通保存入口保留协议且拒绝拼错模式；传参：临时配置；返回：无。"""
    path = tmp_path / "config.yaml"
    save_user_config({"api_mode": "responses"}, path)
    assert load_saved_config(path)["api_mode"] == "responses"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="unsupported.*api_mode"):
        save_user_config({"api_mode": "unknown"}, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", [12.5, "12.5", False, -1, float("inf"), "NaN", ""])
@pytest.mark.parametrize("source", ["saved", "profile", "env"])
def test_output_budget_rejects_invalid_values_without_rounding(
    tmp_path, monkeypatch, value, source
):
    """三种配置入口不能把无效输出额度截断或改成默认值；传参：配置入口及值；返回：无。"""
    with pytest.raises(ValueError, match="max_output_tokens.*positive integer"):
        if source == "saved":
            save_user_config({"max_output_tokens": value}, tmp_path / "config.yaml")
        elif source == "profile":
            save_model_profile(
                "profile",
                {
                    "provider": "fixture",
                    "base_url": "http://localhost/v1",
                    "model": "fixture",
                    "max_output_tokens": value,
                },
                tmp_path / "models.yaml",
            )
        else:
            monkeypatch.setenv("XIANGMU_LLM_MAX_OUTPUT_TOKENS", str(value))
            resolve_model_target()


def test_protocol_and_output_budget_respect_config_precedence(monkeypatch):
    """协议与输出预算采用同一配置优先级；传参：隔离环境；返回：无。"""
    monkeypatch.setenv("XIANGMU_LLM_API_MODE", "anthropic_messages")
    monkeypatch.setenv("XIANGMU_LLM_MAX_OUTPUT_TOKENS", "256")
    target = resolve_model_target(
        saved_config={"api_mode": "responses", "max_output_tokens": 512},
        cli_overrides={"max_output_tokens": 128},
        file_defaults={"api_mode": "chat_completions"},
    )
    assert target.api_mode == "responses"
    assert target.output_token_limit == 128
