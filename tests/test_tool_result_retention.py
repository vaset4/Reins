"""工具原文、分页与诊断受众的实际调用验证。

作者：xxx
时间：2026-09-13 22:00:00
"""

from __future__ import annotations
from scripts.testing.llm import from_test_sequence, from_test_turns

import json
from contextlib import closing
from pathlib import Path

import pytest

from artifacts.store import ArtifactStore
from context.artifact_ref import store_large_output
from llm.messages import ToolResultMessage
from runtime.tool_results import _render_tool_conversation
from runtime.agent_loop import AgentLoop
from runtime.tool_results import MAX_TOOL_OUTPUT_CHARS
from runtime.lease import from_trigger
from runtime.run_evidence import RunEvidenceStore
from runtime.session_messages import append_user_message, materialize_messages
from runtime.types import RunContext, RunToolsResult, Trigger
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tools.read_artifact import read_artifact, read_artifact_page
from tools.tool_registry import Idempotent, ToolDefinition, ToolRisk


@pytest.mark.parametrize("result_kind", ["typed", "mapping"])
def test_long_result_survives_session_and_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result_kind: str
) -> None:
    """真实执行一次并在后续请求中补读中段，诊断只进证据；传参：临时根与替换器；返回：无。"""
    data_root = tmp_path / ".reins" / "data"
    with closing(TaskStore(data_root)) as store:
        task = store.create_task("读取完整工具结果", is_inbox=True)
    content = (
        "开头\r\n"
        + "甲" * MAX_TOOL_OUTPUT_CHARS
        + "必须保留的中段"
        + "乙" * MAX_TOOL_OUTPUT_CHARS
    )
    effects: list[str] = []
    registry = build_tool_registry(repo_root=tmp_path, data_root=data_root)

    def execute(_args: dict[str, object]) -> object:
        """返回带独立诊断的大结果；传参：已校验参数；返回：业务结果。"""
        effects.append("executed")
        if result_kind == "mapping":
            return {
                "content": content,
                "diagnostics": {"detail": "diagnostic-only-marker"},
            }
        return RunToolsResult.ok(
            action="long_result",
            content=content,
            diagnostics={"detail": "diagnostic-only-marker"},
        )

    registry.register(
        ToolDefinition(
            "long_result",
            "读取资料",
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
    client = from_test_sequence(
        [
            '{"type":"run_tools","tool":"long_result","arguments":{}}',
            '{"type":"final","content":"已有原文引用"}',
        ],
        protocol_mode="text_json",
    )
    adapter = client._adapter_registry.require("scripted_test")
    original_stream = adapter.stream
    requests: list[object] = []

    def capture(request: object, **kwargs: object):
        """记录实际供应商请求；传参：已组装请求；返回：原脚本响应流。"""
        requests.append(request)
        return original_stream(request, **kwargs)

    monkeypatch.setattr(adapter, "stream", capture)
    lease = from_trigger(
        "user",
        task_id=task.task_id,
        capabilities={"fs": {"project_root": str(tmp_path), "read": [str(tmp_path)]}},
    )
    context = RunContext(
        task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": "读取资料"},
        capability_lease=lease,
    )
    append_user_message(data_root, context.session_id, "读取资料")
    list(
        AgentLoop(data_root, llm_client=client, tool_registry=registry).run_stream(
            context
        )
    )
    result = next(
        message
        for message in materialize_messages(data_root, context.session_id)
        if isinstance(message, ToolResultMessage)
    )
    assert effects == ["executed"]
    assert len(result.artifact_refs) == 1
    artifact_id = result.artifact_refs[0]
    assert artifact_id in str(requests[-1])
    assert "diagnostic-only-marker" not in str(requests)
    assert read_artifact(data_root, artifact_id, mode="full", lease=lease) == content
    page = read_artifact_page(
        data_root, artifact_id, lease=lease, offset=MAX_TOOL_OUTPUT_CHARS, limit=100
    )
    assert "必须保留的中段" in page["content"]
    diagnostics = RunEvidenceStore(data_root).list_records(
        session_id=context.session_id, run_id=context.run_id, kind="tool_diagnostic"
    )
    assert len(diagnostics) == 1
    assert diagnostics[0]["payload"]["call_id"] == result.call_id


def test_artifact_pages_preserve_unicode_versions_and_scope(tmp_path: Path) -> None:
    """分页拼回原文，旧游标版本与越权读取明确失败；传参：临时根；返回：无。"""
    with closing(TaskStore(tmp_path)) as store:
        task = store.create_task("inbox artifact", is_inbox=True)
    content = "甲🙂\r\n乙\n" * 1000
    ref = store_large_output(tmp_path, task.task_id, content)
    assert ref is not None
    assert ArtifactStore(tmp_path).load_artifact(ref.artifact_id).content_sha256
    offset, revision, pages = 0, None, []
    while offset is not None:
        page = read_artifact_page(
            tmp_path,
            ref.artifact_id,
            lease=None,
            offset=offset,
            limit=601,
            expected_sha256=revision,
        )
        pages.append(page["content"])
        offset = page["meta"]["next_offset"]
        revision = page["meta"]["content_sha256"]
    assert "".join(pages) == content
    with pytest.raises(ValueError, match="version changed"):
        read_artifact_page(
            tmp_path, ref.artifact_id, lease=None, expected_sha256="0" * 64
        )
    lease = from_trigger(
        "user", capabilities={"fs": {"read": [str(tmp_path / "elsewhere")]}}
    )
    with pytest.raises(PermissionError):
        read_artifact_page(tmp_path, ref.artifact_id, lease=lease)


def test_published_inbox_artifact_survives_source_deletion(tmp_path: Path) -> None:
    """收件箱产物发布后不依赖原文件；传参：临时根；返回：无。"""
    with closing(TaskStore(tmp_path)) as store:
        task = store.create_task("old inbox", is_inbox=True)
    old_dir = tmp_path / "tasks" / task.task_id / "artifacts"
    old_dir.mkdir(parents=True)
    (old_dir / "old.txt").write_text("历史原文", encoding="utf-8")
    with closing(ArtifactStore(tmp_path)) as store:
        record = store.create_artifact(
            task.task_id,
            "output",
            f"tasks/{task.task_id}/artifacts/old.txt",
            "旧产物",
            12,
        )
    (old_dir / "old.txt").unlink()
    assert read_artifact(tmp_path, record.artifact_id, mode="full") == "历史原文"


def test_artifact_metadata_rejects_escape_before_writing(tmp_path: Path) -> None:
    """不允许外部产物身份改写元数据目录外文件；传参：临时根；返回：无。"""
    with closing(ArtifactStore(tmp_path)) as store:
        with pytest.raises(ValueError, match="single storage name"):
            store.create_artifact(
                "task", "output", "body.txt", "bad", 1, artifact_id="../escape"
            )
    assert not (tmp_path / "tasks" / "task" / "escape.json").exists()


@pytest.mark.parametrize(
    "factory", [RunToolsResult.error_result, RunToolsResult.denied]
)
def test_failure_result_keeps_diagnostics_separate(factory) -> None:
    """失败结果也保留独立诊断而非塞入正文；传参：结果工厂；返回：无。"""
    result = factory(
        action="test", error="业务失败", diagnostics={"detail": "执行诊断"}
    )
    assert result.output == "业务失败"
    assert result.diagnostics == {"detail": "执行诊断"}


def test_model_receipt_contains_one_copy_of_output(tmp_path: Path) -> None:
    """模型回执不重复携带同一正文；传参：隔离目录；返回：无。"""
    result = RunToolsResult.ok(action="probe", content="单份正文标记")
    rendered = _render_tool_conversation(result)
    assert rendered.count("单份正文标记") == 1
    result = RunToolsResult.ok(
        action="probe",
        content=json.dumps({"value": "structured-marker"}),
        meta={"value": "structured-marker", "operation_id": "actual-operation"},
    )
    rendered = _render_tool_conversation(result)
    assert rendered.count("structured-marker") == 1
    assert json.loads(rendered)["meta"]["operation_id"] == "actual-operation"


@pytest.mark.parametrize("window", [8000, 16000])
def test_large_result_group_fits_actual_window_and_retains_full_receipts(
    tmp_path, monkeypatch, window
):
    """同一批大回执按实际窗口送入模型，原文和大元数据可完整补读；传参：根、替换器及窗口；返回：无。"""
    from scripts.testing.llm import ScriptedTurnOptions
    from llm.messages import agent_message_to_mapping, model_visible_text
    from context.window import request_budget
    from runtime.agent_loop import State
    from runtime.session_messages import (
        ToolExchange,
        append_tool_calls,
        append_tool_exchange,
    )
    from tests.test_tool_batch_execution import make_run
    from tests.test_session_runtime import capture_requests
    from tools.tool_registry import ToolRegistry

    loop, context, _ = make_run(tmp_path, ToolRegistry(), [])
    original_text = "客户资料🙂\r\n" * 4000 + "中段凭据必须完整保留"
    original_meta = {
        "execution_request": {
            "tool": "probe",
            "arguments": {"material": "元数据" * 20000},
        }
    }
    calls = [
        ToolExchange(
            f"receipt-{index}",
            "probe",
            {"index": index},
            json.dumps(
                {"output": original_text, "meta": original_meta}, ensure_ascii=False
            ),
            "ok",
        )
        for index in range(2)
    ]
    append_tool_calls(tmp_path, context.session_id, calls)
    for call in calls:
        append_tool_exchange(tmp_path, context.session_id, call)
    original = materialize_messages(tmp_path, context.session_id)
    client = from_test_turns(
        ["继续处理原件"], options=ScriptedTurnOptions(context_window=window)
    )
    requests = capture_requests(client, monkeypatch)
    loop.llm_client = client
    assert loop.run(context) is State.DONE
    assert len(requests) == 1
    assert request_budget(requests[0], window).required_total <= window
    projected = [
        message
        for message in requests[0].messages
        if isinstance(message, ToolResultMessage)
    ]
    assert [message.call_id for message in projected] == [
        call.call_id for call in calls
    ]
    for message in projected:
        payload = json.loads(model_visible_text(message))
        assert payload["prompt_truncated"] and not payload["output_complete"]
        source = next(
            item for item in original if item.message_id == message.message_id
        )
        ref = payload["read_full_message"]["artifact_id"]
        restored = json.loads(
            read_artifact(tmp_path, ref, mode="full", lease=context.capability_lease)
        )
        assert restored == agent_message_to_mapping(source)
    assert (
        materialize_messages(tmp_path, context.session_id)[: len(original)] == original
    )


def test_result_projection_keeps_media_error_and_stable_source(tmp_path):
    """缩短文字不丢媒体与失败状态，重复组装不重写原件；传参：隔离根；返回：无。"""
    from scripts.testing.llm import ScriptedTurnOptions
    from llm.messages import (
        AssistantMessage,
        DocumentRefPart,
        TextPart,
        ToolCallPart,
        UserMessage,
    )
    from llm.image_input import image_from_bytes
    from tests.test_image_attachments import image_bytes
    from runtime.tool_result_views import ToolResultViews
    from tools.tool_registry import ToolRegistry

    with closing(TaskStore(tmp_path)) as store:
        task = store.create_task("保留媒体与错误")
    media = (
        image_from_bytes(image_bytes()),
        DocumentRefPart("artifact:document-source", "原始文档"),
    )
    original = ToolResultMessage(
        "large-receipt",
        "read-source",
        "probe",
        (TextPart("长资料" * 15000), *media),
        "error",
        error="实际错误说明" * 2000,
    )
    history = (
        UserMessage("input", (TextPart("核对资料"),)),
        AssistantMessage("call", (ToolCallPart("read-source", "probe", {}),)),
        original,
    )
    client = from_test_turns([], options=ScriptedTurnOptions(context_window=8000))
    views = ToolResultViews(
        tmp_path, task_id=task.task_id, prepare=client.prepare_request
    )
    context = {"conversation_history": history, "tool_registry": ToolRegistry()}
    first = views.prepare("继续", context)
    result = next(
        message for message in first.messages if isinstance(message, ToolResultMessage)
    )
    assert result.content[1:] == media and result.status == "error" and result.error
    timestamps = {
        path: path.stat().st_mtime_ns
        for path in (tmp_path / "assets").rglob("*")
        if path.is_file()
    }
    assert timestamps
    again = views.prepare("继续", context)
    assert again.messages == first.messages
    assert {path: path.stat().st_mtime_ns for path in timestamps} == timestamps
