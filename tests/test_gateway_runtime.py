from __future__ import annotations


def test_gateway_runtime_processes_inbound_message(tmp_path) -> None:
    from app.gateway import GatewayRuntime, InboundMessage
    from app.startup import resolve_startup_identity
    from llm.client import MissingConfigurationLLMClient

    gateway = GatewayRuntime(
        identity=resolve_startup_identity(
            project_root=tmp_path, data_root=tmp_path / "data"
        ),
        llm_client=MissingConfigurationLLMClient(),
    )

    gateway.submit(InboundMessage(task="inspect workspace", source="gateway"))
    outbound = gateway.process_next()

    assert outbound is not None
    assert outbound.status == "failed"
    assert outbound.task_id
    assert outbound.output.startswith(
        "MODEL_PROVIDER_ERROR: missing provider configuration"
    )


def test_gateway_runtime_processes_tasks_query(tmp_path) -> None:
    from app.gateway import ControlMessage, GatewayRuntime, InboundMessage
    from app.startup import resolve_startup_identity
    from llm.client import MissingConfigurationLLMClient

    gateway = GatewayRuntime(
        identity=resolve_startup_identity(
            project_root=tmp_path, data_root=tmp_path / "data"
        ),
        llm_client=MissingConfigurationLLMClient(),
    )

    gateway.submit(InboundMessage(task="hello", source="gateway"))
    gateway.process_next()
    gateway.submit_control(ControlMessage(action="tasks", source="gateway"))
    outbound = gateway.process_next()

    assert outbound is not None
    assert outbound.status == "ok"
    assert outbound.output == "TASKS: 1"
    assert outbound.data["total"] == 1
