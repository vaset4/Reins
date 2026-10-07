from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from app.gateway import (
    ControlMessage,
    GatewayRuntime,
    InboundMessage,
    run_gateway_stdio,
)
from app.run_task import RunTaskResponse
from app.startup import resolve_startup_identity
from llm.client import MissingConfigurationLLMClient
from llm.types import LLMPlan
from runtime.types import RunToolsResult
from tests.support.fake_llm import FakeLLMClient


class RecordingLLMClient:
    """Wraps FakeLLMClient to prove the injected client (not Missing) is used."""

    def __init__(self) -> None:
        self._delegate = FakeLLMClient()
        self.plan_calls: list[str] = []

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        self.plan_calls.append(task)
        return self._delegate.plan(task, context)

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: RunToolsResult,
        context: object | None = None,
    ) -> LLMPlan:
        return self._delegate.continue_from_run_tools(task, run_tools_result, context)


def test_inbound_uses_injected_llm_client(tmp_path: Path) -> None:
    client = RecordingLLMClient()
    gateway = GatewayRuntime(
        identity=resolve_startup_identity(
            project_root=tmp_path, data_root=tmp_path / "data"
        ),
        llm_client=client,
    )
    gateway.submit(InboundMessage(task="hello world", source="gateway"))

    outbound = gateway.process_next()

    assert outbound is not None
    assert client.plan_calls[0] == "hello world"
    assert outbound.status == "done"
    assert outbound.output == "FAKE_MODEL_RESPONSE: hello world"
    assert "missing provider configuration" not in outbound.output


def test_missing_client_surfaces_status(tmp_path: Path) -> None:
    gateway = GatewayRuntime(
        identity=resolve_startup_identity(
            project_root=tmp_path, data_root=tmp_path / "data"
        ),
        llm_client=MissingConfigurationLLMClient(),
    )
    gateway.submit(InboundMessage(task="hello world", source="gateway"))

    outbound = gateway.process_next()

    assert outbound is not None
    assert outbound.status == "failed"
    assert "missing provider configuration" in outbound.output


def test_gateway_passes_identity_roots_to_run_task(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    data_root = tmp_path / "isolated-data"
    identity = resolve_startup_identity(
        project_root=project_root,
        data_root=data_root,
    )
    captured: dict[str, object] = {}

    def fake_run_task(task: str, root: Path, **kwargs: object) -> RunTaskResponse:
        captured.update(task=task, project_root=root, **kwargs)
        return RunTaskResponse(
            task_id="task",
            segment_id="segment",
            status="done",
            output="ok",
        )

    monkeypatch.setattr("app.gateway.run_task", fake_run_task)
    client = RecordingLLMClient()
    gateway = GatewayRuntime(identity=identity, llm_client=client)
    gateway.submit(InboundMessage(task="hello", source="gateway"))

    gateway.process_next()

    assert captured["task"] == "hello"
    assert captured["project_root"] == project_root.resolve()
    assert captured["data_root"] == data_root.resolve()
    assert captured["llm_client"] is client


def test_unsupported_control_action_returns_error_not_crash(tmp_path: Path) -> None:
    gateway = GatewayRuntime(
        identity=resolve_startup_identity(
            project_root=tmp_path, data_root=tmp_path / "data"
        ),
        llm_client=MissingConfigurationLLMClient(),
    )
    gateway.submit_control(ControlMessage(action="sessions", source="gateway"))

    drained = gateway.drain_once()

    assert len(drained) == 1
    outbound = drained[0]
    assert outbound.status == "error"
    assert "unsupported control action: sessions" in outbound.output


def test_control_tasks_ok(tmp_path: Path) -> None:
    gateway = GatewayRuntime(
        identity=resolve_startup_identity(
            project_root=tmp_path, data_root=tmp_path / "data"
        ),
        llm_client=MissingConfigurationLLMClient(),
    )
    gateway.submit_control(ControlMessage(action="tasks", source="gateway"))

    outbound = gateway.process_next()

    assert outbound is not None
    assert outbound.status == "ok"
    assert outbound.output.startswith("TASKS: ")
    assert outbound.data.get("total") == 0


def test_stdio_bad_json_returns_structured_error(tmp_path: Path) -> None:
    output = StringIO()

    code = run_gateway_stdio(
        identity=resolve_startup_identity(
            project_root=tmp_path, data_root=tmp_path / "data"
        ),
        llm_client=MissingConfigurationLLMClient(),
        input_stream=StringIO("{bad json}\n"),
        output_stream=output,
    )

    assert code == 1
    payload = json.loads(output.getvalue())
    assert payload["status"] == "error"
    assert payload["data"]["error_type"] == "invalid_json"
    assert "GATEWAY_INPUT_ERROR" in payload["output"]


def test_stdio_non_object_json_returns_structured_error(tmp_path: Path) -> None:
    output = StringIO()

    code = run_gateway_stdio(
        identity=resolve_startup_identity(
            project_root=tmp_path, data_root=tmp_path / "data"
        ),
        llm_client=MissingConfigurationLLMClient(),
        input_stream=StringIO("[]\n"),
        output_stream=output,
    )

    assert code == 1
    payload = json.loads(output.getvalue())
    assert payload["status"] == "error"
    assert payload["data"]["error_type"] == "invalid_payload"
