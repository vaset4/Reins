from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from typing import TextIO

from app.run_task import run_task
from app.startup import StartupIdentity
from llm.base import LLMClient
from runtime.schema_meta import UnsupportedSchemaError, ensure_current_schema
from tasks.store import TaskStore


@dataclass(slots=True)
class InboundMessage:
    task: str
    source: str
    message_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "InboundMessage":
        return cls(
            task=str(data.get("task", "")),
            source=str(data.get("source", "gateway")),
            message_id=str(data.get("message_id", "")),
        )


@dataclass(slots=True)
class ControlMessage:
    action: str
    source: str = "gateway"
    message_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ControlMessage":
        return cls(
            action=str(data.get("action", "")),
            source=str(data.get("source", "gateway")),
            message_id=str(data.get("message_id", "")),
        )


@dataclass(slots=True)
class OutboundMessage:
    status: str
    output: str
    source: str = "gateway"
    message_id: str = ""
    task_id: str | None = None
    data: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class GatewayRuntime:
    def __init__(self, *, identity: StartupIdentity, llm_client: LLMClient) -> None:
        """初始化使用统一启动身份的网关运行时

        参数：identity 为入口已解析的项目与数据根；llm_client 为模型客户端
        返回：无
        """
        ensure_current_schema(identity.data_root)

        self._identity = identity
        self._llm_client = llm_client
        self._inbound: list[InboundMessage] = []
        self._control: list[ControlMessage] = []
        self._outbound: list[OutboundMessage] = []

    def submit(self, message: InboundMessage) -> None:
        self._inbound.append(message)

    def submit_control(self, message: ControlMessage) -> None:
        self._control.append(message)

    def process_next(self) -> OutboundMessage | None:
        if self._control:
            outbound = self._execute_control(self._control.pop(0))
        elif self._inbound:
            message = self._inbound.pop(0)
            response = run_task(
                message.task,
                self._identity.project_root,
                data_root=self._identity.data_root,
                llm_client=self._llm_client,
            )
            outbound = OutboundMessage(
                message_id=message.message_id,
                source=message.source,
                status=response.status,
                output=response.output,
                task_id=response.task_id,
            )
        else:
            return None
        self._outbound.append(outbound)
        return outbound

    def drain_once(self) -> list[OutboundMessage]:
        drained: list[OutboundMessage] = []
        while (outbound := self.process_next()) is not None:
            drained.append(outbound)
        return drained

    def _execute_control(self, message: ControlMessage) -> OutboundMessage:
        if message.action == "tasks":
            tasks = TaskStore(self._identity.data_root).list_tasks()
            return OutboundMessage(
                message_id=message.message_id,
                source=message.source,
                status="ok",
                output=f"TASKS: {len(tasks)}",
                data={"total": len(tasks)},
            )
        return OutboundMessage(
            message_id=message.message_id,
            source=message.source,
            status="error",
            output=f"unsupported control action: {message.action}",
        )


def run_gateway_stdio(
    *,
    identity: StartupIdentity,
    llm_client: LLMClient,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    **_kwargs: object,
) -> int:
    """运行使用统一启动身份的 stdin/stdout 网关

    参数：identity 为入口解析结果；llm_client 为模型客户端；input_stream/output_stream 为可选 IO
    返回：网关退出码，输入错误返回 1，正常结束返回 0
    """
    reader = input_stream or sys.stdin
    writer = output_stream or sys.stdout
    try:
        gateway = GatewayRuntime(identity=identity, llm_client=llm_client)
    except UnsupportedSchemaError as exc:
        _write_outbound(
            writer,
            OutboundMessage(
                status="error",
                output=str(exc),
                data={
                    "error_type": "unsupported_schema",
                    "schema_code": exc.code,
                },
            ),
        )
        return 1
    for raw_line in reader:
        line = raw_line.strip()
        if not line:
            continue
        if line == "/exit":
            return 0
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_outbound(writer, _invalid_input_message("invalid_json", str(exc)))
            return 1
        if not isinstance(payload, dict):
            _write_outbound(
                writer, _invalid_input_message("invalid_payload", "expected object")
            )
            return 1
        if str(payload.get("kind", "inbound")).lower() == "control":
            gateway.submit_control(ControlMessage.from_dict(payload))
        else:
            gateway.submit(InboundMessage.from_dict(payload))
        for outbound in gateway.drain_once():
            _write_outbound(writer, outbound)
    return 0


def _invalid_input_message(error_type: str, detail: str) -> OutboundMessage:
    return OutboundMessage(
        status="error",
        output=f"GATEWAY_INPUT_ERROR: {error_type}: {detail}",
        data={"error_type": error_type},
    )


def _write_outbound(writer: TextIO, outbound: OutboundMessage) -> None:
    writer.write(json.dumps(outbound.to_dict(), ensure_ascii=False) + "\n")
    writer.flush()
