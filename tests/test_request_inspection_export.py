"""【阶段六】【请求验收】真实存储边界、分页、冻结原件与导出失败的可重复验证。

作者：xxx
时间：2026-09-30 14:00:00
"""

from __future__ import annotations

import json
import base64
import shutil
import hashlib
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import pytest

from artifacts.store import ArtifactStore
from llm.messages import TextPart, ToolResultMessage, agent_message_to_mapping
from llm.types import ModelAttemptEvent, ModelError
from runtime.lease import from_trigger
from runtime.model_evidence import ModelEvidenceWriter
from runtime.file_content import ContentFiles
from runtime.request_export import RequestExporter, freeze_export, write_export
from runtime.request_inspection import RequestInspection
from runtime.run_evidence import RunEvidenceStore
from runtime.run_facts import RunFactStore
from runtime.tool_operations import ToolOperationStore
from runtime.types import RunContext, Trigger


def _store(root: Path):
    """建立隔离运行；参数：临时根；返回：查询、写入器及运行。"""
    inspection = RequestInspection(root)
    facts = RunFactStore(root)
    context = RunContext(
        trigger=Trigger.USER,
        payload={},
        capability_lease=from_trigger("user"),
        session_id="s",
        run_id="r",
    )
    facts.append(
        {
            "session_id": "s",
            "run_id": "r",
            "event": "run:lifecycle",
            "lifecycle": "running",
        }
    )
    return inspection, ModelEvidenceWriter(RunEvidenceStore(root), facts), context


def _attempt(
    writer, context, number=1, *, request=None, sources=None, finished=True, error=None
):
    """提交一次实际尝试证据；参数：写入器、序号及发送内容；返回：开始记录。"""
    identity = f"request-{number}"
    writer.record_request(context, request_id=identity, request_index=number)
    attempt = ModelAttemptEvent(
        "started",
        identity,
        f"attempt-{number}",
        1,
        "fixture",
        "model",
        "2026-09-30T06:00:00Z",
        request=request or {"messages": [{"role": "user", "content": "hello"}]},
        sources=sources or {},
    )
    writer.record_attempt(context, attempt)
    if finished:
        writer.record_attempt(
            context,
            replace(
                attempt,
                phase="finished",
                error=error,
                response={
                    "text": f"answer-{number}",
                    "message_id": f"response-{number}",
                },
            ),
        )
    return attempt


def _detail(inspection, **selection):
    """按页取回完整选择以比较内容；参数：查询及身份；返回：全文与页长度。"""
    offset, parts = 0, []
    while True:
        page = inspection.query(
            {
                "action": "detail",
                "session_id": "s",
                "run_id": "r",
                "limit": 4096,
                **selection,
                "offset": offset,
            }
        )
        parts.append(page["text"])
        if not page["has_more"]:
            assert len("".join(parts)) == page["total_chars"]
            return "".join(parts)
        offset = page["next_offset"]


def _operation(root, number, *, artifact=None):
    """发布工具结果及原件引用；参数：序号和可选产物；返回：操作身份。"""
    identity = {"session_id": "s", "run_id": "r", "operation_id": f"op-{number}"}
    meta = {"operation_id": identity["operation_id"]}
    if artifact:
        meta["result_artifact_id"] = artifact.artifact_id
    payload = {
        "state": "completed",
        "call": {
            "call_id": f"call-{number}",
            "tool_name": "probe",
            "task_id": "task",
            "request_id": "request-1",
            "args": {"x": number},
        },
        "result": {"output": "完整的小结果", "status": "ok", "meta": meta},
    }
    ToolOperationStore(root).write(identity, payload)
    return identity


@pytest.mark.parametrize("level", ["off", "basic", "debug"])
def test_all_trace_levels_retain_exact_pages_and_failure(tmp_path, monkeypatch, level):
    """常规级别均保存原协议，失败尝试可见；参数：级别；返回：无。"""
    monkeypatch.setenv("REINS_TRACE_LEVEL", level)
    inspection, writer, context = _store(tmp_path)
    content = ('中文\n\u0000"末尾' * 1600) + "FINISH"
    request = {
        "instructions": content,
        "input": [{"role": "user", "content": content}],
        "tools": [],
    }
    error = ModelError.create(
        category="cancelled", summary="用户停止", stage="stream", retryable=False
    )
    _attempt(writer, context, request=request, error=error)
    assert (
        json.loads(_detail(inspection, request_id="request-1", attempt_id="attempt-1"))
        == request
    )
    assert (
        inspection.query(
            {
                "action": "attempts",
                "session_id": "s",
                "run_id": "r",
                "request_id": "request-1",
            }
        )["items"][0]["status"]
        == "cancelled"
    )
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert len(list(tmp_path.rglob(f"{digest}.txt"))) == 1


def test_catalog_snapshot_and_ownership_do_not_decode_bodies(tmp_path, monkeypatch):
    """目录不展开正文且游标固定成员与归属；参数：临时根；返回：无。"""
    inspection, writer, context = _store(tmp_path)
    for number in range(1, 4):
        _attempt(writer, context, number)
    page = inspection.query(
        {"action": "requests", "session_id": "s", "run_id": "r", "limit": 1}
    )
    _attempt(writer, context, 4)
    monkeypatch.setattr(
        ContentFiles, "unpack", lambda *args: pytest.fail("catalog decoded body")
    )
    second = inspection.query(
        {
            "action": "requests",
            "session_id": "s",
            "run_id": "r",
            "limit": 3,
            "cursor": page["next_cursor"],
        }
    )
    assert [row["request_id"] for row in second["items"]] == ["request-2", "request-3"]
    with pytest.raises(ValueError, match="another selection"):
        inspection.query(
            {"action": "runs", "session_id": "s", "cursor": page["next_cursor"]}
        )
    with pytest.raises(ValueError, match="does not belong"):
        inspection.query(
            {
                "action": "detail",
                "session_id": "other",
                "run_id": "r",
                "request_id": "request-1",
                "attempt_id": "attempt-1",
            }
        )


def test_tools_paginate_and_run_state_uses_lifecycle(tmp_path):
    """多页工具不会丢尾部，后续普通事件不覆盖终态；参数：临时根；返回：无。"""
    inspection, _, _ = _store(tmp_path)
    for number in range(5):
        _operation(tmp_path, number)
    query = {"action": "tools", "session_id": "s", "run_id": "r", "limit": 2}
    first = inspection.query(query)
    _operation(tmp_path, 6)
    second = inspection.query({**query, "cursor": first["next_cursor"]})
    third = inspection.query({**query, "cursor": second["next_cursor"]})
    assert [
        item["operation_id"]
        for page in (first, second, third)
        for item in page["items"]
    ] == [f"op-{i}" for i in range(5)]
    facts = RunFactStore(tmp_path)
    facts.append(
        {
            "session_id": "s",
            "run_id": "r",
            "event": "run:lifecycle",
            "lifecycle": "done",
        }
    )
    facts.append({"session_id": "s", "run_id": "r", "event": "notification:sent"})
    assert (
        inspection.query({"action": "runs", "session_id": "s"})["items"][0]["status"]
        == "done"
    )


def test_frozen_tool_original_and_all_feedback_versions(tmp_path):
    """百万字符原件与两份实际回喂分离，源文件改动不改变历史；参数：临时根；返回：无。"""
    inspection, writer, context = _store(tmp_path)
    source = tmp_path / "result.txt"
    original = "原始工具结果\n" * 100000 + "END"
    source.write_bytes(original.encode("utf-8"))
    artifact = ArtifactStore(tmp_path).create_artifact(
        "task", "output", "result.txt", "原件", source.stat().st_size
    )
    _operation(tmp_path, 1, artifact=artifact)
    for number, version in enumerate(("第一份完整回喂", "第二次选择的摘要"), 1):
        message = ToolResultMessage(
            "tool-message",
            "call-1",
            "probe",
            (
                TextPart(
                    json.dumps({"content": version, "meta": {"operation_id": "op-1"}})
                ),
            ),
            "success",
        )
        sources = {
            "messages": [
                {
                    "position": 0,
                    "message_id": message.message_id,
                    "kind": message.kind,
                    "call_id": message.call_id,
                    "fed_back": agent_message_to_mapping(message),
                }
            ]
        }
        _attempt(writer, context, number, sources=sources)
    source.write_text("用户已改动今天的文件", encoding="utf-8")
    page = inspection.query(
        {
            "action": "detail",
            "session_id": "s",
            "run_id": "r",
            "section": "tool_result",
            "operation_id": "op-1",
            "offset": len(original) - 10,
        }
    )
    assert page["text"] == original[-10:]
    assert page["total_chars"] == len(original)
    feedback = json.loads(
        _detail(inspection, section="tool_feedback", operation_id="op-1")
    )
    assert [item["request_id"] for item in feedback] == ["request-1", "request-2"]
    assert [
        json.loads(item["content"]["content"][0]["text"])["content"]
        for item in feedback
    ] == [
        "第一份完整回喂",
        "第二次选择的摘要",
    ]
    located = inspection.query(
        {"action": "locate", "session_id": "s", "run_id": "r", "call_id": "call-1"}
    )
    assert len(located["items"]) == 2


def test_export_freezes_committed_boundary_and_moves_with_assets(tmp_path):
    """导出不混入后到结果，移动后仍可读冻结附件；参数：临时根；返回：无。"""
    root = tmp_path / "data"
    inspection, writer, context = _store(root)
    source = root / "source.txt"
    source.write_text("冻结的工具字节", encoding="utf-8")
    artifact = ArtifactStore(root).create_artifact(
        "task", "output", "source.txt", "工具原件", source.stat().st_size
    )
    _operation(root, 1, artifact=artifact)
    attempt = _attempt(writer, context, finished=False)
    snapshot = freeze_export(inspection, {"session_id": "s", "run_id": "r"})
    writer.record_attempt(
        context, replace(attempt, phase="finished", response={"text": "迟到输出"})
    )
    _attempt(writer, context, 2)
    source.unlink()
    target = tmp_path / "export"
    target.mkdir()
    write_export(inspection, snapshot, target, Event())
    moved = tmp_path / "moved"
    shutil.move(str(target), str(moved))
    data = json.loads((moved / "requests.json").read_text(encoding="utf-8"))
    assert len(data["requests"]) == 1
    assert data["requests"][0]["attempts"][0]["status"] == "started"
    assert data["requests"][0]["attempts"][0]["response"] is None
    attachment = data["attachments"][0]
    assert (moved / attachment["package_path"]).read_text(
        encoding="utf-8"
    ) == "冻结的工具字节"
    markdown = (moved / "README.md").read_text(encoding="utf-8")
    assert "工具操作与完整结果" in markdown
    assert "op-1" in markdown
    assert "完整的小结果" in markdown
    assert attachment["package_path"] in markdown


def test_export_failure_and_close_waits_for_worker(tmp_path, monkeypatch):
    """取消释放线程，损坏和同名不发布成品；参数：临时根；返回：无。"""
    inspection, writer, context = _store(tmp_path / "data")
    _attempt(writer, context)
    started, released, closed = Event(), Event(), Event()

    def delayed_write(reader, manifest, target, cancel):
        """保持真实读取工作处于未退出状态；参数：导出上下文；返回：取消前不会完成。"""
        started.set()
        assert released.wait(5)
        write_export(reader, manifest, target, cancel)

    monkeypatch.setattr("runtime.request_export.write_export", delayed_write)
    exporter = RequestExporter(inspection)
    job = exporter.command(
        {"session_id": "s", "run_id": "r", "target_dir": str(tmp_path / "out")}
    )
    assert started.wait(5)

    def close():
        """等待服务真正关闭；参数：无；返回：释放后标记。"""
        exporter.close()
        closed.set()

    closer = Thread(target=close)
    closer.start()
    assert not closed.wait(0.05)
    released.set()
    closer.join(5)
    assert closed.is_set()
    assert (
        exporter.command({"action": "status", "job_id": job["job_id"]})["status"]
        == "cancelled"
    )
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob("*.partial"))
    existing = tmp_path / "exists"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        RequestExporter(inspection).command(
            {"session_id": "s", "run_id": "r", "target_dir": str(existing)}
        )


def test_corrupt_payload_export_is_failed_not_published(tmp_path):
    """正文损坏不能生成看似完整的包；参数：临时根；返回：无。"""
    inspection, writer, context = _store(tmp_path / "data")
    _attempt(writer, context, request={"messages": [{"content": "需核验" * 1000}]})
    with inspection.db.snapshot() as source:
        records = source.list_raw("run_evidence", session_id="s")
        reference = next(item for row in records for item in row.references)["content"]
    (inspection.db.data_root / reference["path"]).write_text(
        "corrupt", encoding="utf-8"
    )
    exporter = RequestExporter(inspection)
    job = exporter.command(
        {"session_id": "s", "run_id": "r", "target_dir": str(tmp_path / "bad")}
    )
    exporter._jobs[job["job_id"]].thread.join(5)
    status = exporter.command({"action": "status", "job_id": job["job_id"]})
    assert status["status"] == "failed"
    assert "corrupt" in status["error"]
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("shape", ["image_url", "anthropic_base64", "pdf_file_data"])
def test_sent_attachments_are_portable_without_original_files(tmp_path, shape):
    """三种实际发送附件结构均可携包读取；参数：协议结构；返回：无。"""
    inspection, writer, context = _store(tmp_path / "data")
    original = b"frozen-attachment-content" * 3000
    encoded = base64.b64encode(original).decode()
    if shape == "anthropic_base64":
        part = {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": encoded},
        }
    elif shape == "pdf_file_data":
        part = {
            "type": "input_file",
            "filename": "report.pdf",
            "file_data": "data:application/pdf;base64," + encoded,
        }
    else:
        part = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + encoded},
        }
    _attempt(
        writer, context, request={"messages": [{"role": "user", "content": [part]}]}
    )
    snapshot = freeze_export(inspection, {"session_id": "s", "run_id": "r"})
    target = tmp_path / "portable"
    target.mkdir()
    write_export(inspection, snapshot, target, Event())
    content = json.loads((target / "requests.json").read_text(encoding="utf-8"))
    assert len(content["attachments"]) == 1
    assert (target / content["attachments"][0]["package_path"]).read_bytes() == original
    assert content["requests"][0]["attempts"][0]["request"]["request"]["messages"][0][
        "content"
    ] == [part]


def test_tool_card_locates_origin_before_any_feedback(tmp_path):
    """最后一个工具尚未回喂也能找到发起请求；参数：临时根；返回：无。"""
    inspection, writer, context = _store(tmp_path)
    _attempt(writer, context)
    _operation(tmp_path, 1)
    items = inspection.query(
        {"action": "locate", "session_id": "s", "run_id": "r", "call_id": "call-1"}
    )["items"]
    assert [(item["request_id"], item["relation"]) for item in items] == [
        ("request-1", "tool_origin")
    ]


def test_export_reports_permission_failure_without_final_package(tmp_path, monkeypatch):
    """目标目录无权限时明确失败且不产生完成包；参数：临时根和注入边界；返回：无。"""
    inspection, writer, context = _store(tmp_path / "data")
    _attempt(writer, context)
    original = Path.mkdir

    def denied(path, *args, **kwargs):
        """模拟目标创建的真实权限错误；参数：目标路径；返回：其他目录沿原行为。"""
        if path.name.endswith(".partial"):
            raise PermissionError("permission denied for export")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", denied)
    exporter = RequestExporter(inspection)
    job = exporter.command(
        {"session_id": "s", "run_id": "r", "target_dir": str(tmp_path / "denied")}
    )
    exporter._jobs[job["job_id"]].thread.join(5)
    status = exporter.command({"action": "status", "job_id": job["job_id"]})
    assert status["status"] == "failed"
    assert "permission denied" in status["error"]
    assert not (tmp_path / "denied").exists()


def test_request_status_includes_attempts_beyond_one_page(tmp_path):
    """大量重试后请求摘要采用最后一次真实状态；参数：临时根；返回：无。"""
    inspection, writer, context = _store(tmp_path)
    error = ModelError.create(
        category="transport_error", summary="断流", stage="stream", retryable=True
    )
    with inspection.db.transaction():
        first = _attempt(writer, context, error=error)
        for number in range(2, 103):
            attempt = replace(first, attempt_id=f"retry-{number}", attempt_index=number)
            writer.record_attempt(context, attempt)
            writer.record_attempt(
                context,
                replace(
                    attempt, phase="finished", error=None if number == 102 else error
                ),
            )
    request = inspection.query(
        {"action": "requests", "session_id": "s", "run_id": "r"}
    )["items"][0]
    assert request["attempt_count"] == 102
    assert request["status"] == "completed"


@pytest.mark.parametrize("interruption", ["cancel", "collision"])
def test_export_publish_boundary_does_not_override_cancel_or_target(
    tmp_path, monkeypatch, interruption
):
    """生成结束与发布之间的取消或同名目录不能被覆盖；参数：竞争类型；返回：无。"""
    inspection, writer, context = _store(tmp_path / "data")
    _attempt(writer, context)
    prepared, release = Event(), Event()

    def hold_publish(reader, manifest, target, cancel):
        """在完整包写出后固定竞争窗口；参数：导出上下文；返回：等待测试释放。"""
        write_export(reader, manifest, target, cancel)
        prepared.set()
        assert release.wait(5)

    monkeypatch.setattr("runtime.request_export.write_export", hold_publish)
    exporter = RequestExporter(inspection)
    destination = tmp_path / "result"
    job = exporter.command(
        {"session_id": "s", "run_id": "r", "target_dir": str(destination)}
    )
    assert prepared.wait(5)
    if interruption == "cancel":
        exporter.command({"action": "cancel", "job_id": job["job_id"]})
    else:
        destination.mkdir()
        (destination / "existing.txt").write_text("用户原文件", encoding="utf-8")
    release.set()
    exporter._jobs[job["job_id"]].thread.join(5)
    status = exporter.command({"action": "status", "job_id": job["job_id"]})
    assert status["status"] == ("cancelled" if interruption == "cancel" else "failed")
    assert status["path"] is None
    assert not (destination / "requests.json").exists()
    assert not list(tmp_path.glob("*.partial"))
    if interruption == "collision":
        assert (destination / "existing.txt").read_text(
            encoding="utf-8"
        ) == "用户原文件"


def test_corrupt_retained_artifact_does_not_publish_export(tmp_path):
    """已冻结产物损坏时导出明确失败；参数：临时根；返回：无。"""
    inspection, writer, context = _store(tmp_path / "data")
    source = inspection.db.data_root / "source.txt"
    source.write_text("发布时原件", encoding="utf-8")
    artifact = ArtifactStore(inspection.db.data_root).create_artifact(
        "task", "output", "source.txt", "原件", source.stat().st_size
    )
    _operation(inspection.db.data_root, 1, artifact=artifact)
    _attempt(writer, context)
    (inspection.db.data_root / artifact.retained_path).write_text(
        "损坏", encoding="utf-8"
    )
    exporter = RequestExporter(inspection)
    job = exporter.command(
        {"session_id": "s", "run_id": "r", "target_dir": str(tmp_path / "damaged")}
    )
    exporter._jobs[job["job_id"]].thread.join(5)
    status = exporter.command({"action": "status", "job_id": job["job_id"]})
    assert status["status"] == "failed"
    assert "corrupt" in status["error"]
    assert not (tmp_path / "damaged").exists()


def test_observe_and_repl_read_same_error_source(tmp_path):
    """网页与命令行读取同一运行错误且不跨会话；参数：临时根；返回：无。"""
    from app.repl.status import read_latest_run_errors
    from frontends.observe.cross_run.failure_pivot import _read_errors
    from frontends.observe.readers.evidence_reader import EvidenceReader

    inspection, writer, context = _store(tmp_path)
    _attempt(writer, context)
    store = RunEvidenceStore(tmp_path)
    for number in range(3):
        store.append_error(
            session_id="s",
            run_id="r",
            error={"category": "transport_error", "message": f"失败{number}"},
        )
    expected = _read_errors(tmp_path, "s", "r")
    assert EvidenceReader(tmp_path).read_errors("s", "r") == expected
    assert (
        read_latest_run_errors(tmp_path, session_id="s", run_id="r", limit=2)
        == expected[-2:]
    )
    assert read_latest_run_errors(tmp_path, session_id="other", run_id="r") == []
    assert EvidenceReader(tmp_path).query(
        {"action": "requests", "session_id": "s", "run_id": "r"}
    ) == inspection.query({"action": "requests", "session_id": "s", "run_id": "r"})


@pytest.mark.parametrize(
    "text",
    [
        "data: this is a plain text label",
        "data:image/png;base64,dGV4dA==",
        "https://example.test/readme",
    ],
)
def test_plain_text_is_not_decoded_as_an_attachment(tmp_path, text):
    """普通正文即使像URL或data URI也保持文字；参数：正文；返回：无。"""
    inspection, writer, context = _store(tmp_path / "data")
    request = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]
    }
    _attempt(writer, context, request=request)
    target = tmp_path / "export"
    target.mkdir()
    write_export(
        inspection,
        freeze_export(inspection, {"session_id": "s", "run_id": "r"}),
        target,
        Event(),
    )
    exported = json.loads((target / "requests.json").read_text(encoding="utf-8"))
    assert exported["attachments"] == []
    assert exported["requests"][0]["attempts"][0]["request"]["request"] == request


def test_provider_file_reference_is_explicit_external_dependency(tmp_path):
    """只留供应商文件身份时不得伪造可移植文件；参数：临时根；返回：无。"""
    inspection, writer, context = _store(tmp_path / "data")
    _attempt(
        writer,
        context,
        request={"input": [{"type": "input_file", "file_id": "file-provider-123"}]},
    )
    target = tmp_path / "export"
    target.mkdir()
    write_export(
        inspection,
        freeze_export(inspection, {"session_id": "s", "run_id": "r"}),
        target,
        Event(),
    )
    exported = json.loads((target / "requests.json").read_text(encoding="utf-8"))
    assert exported["attachments"] == [
        {"source": "file-provider-123", "retention": "external_reference"}
    ]


def test_tool_arguments_and_schema_examples_are_not_attachments(tmp_path):
    """工具input与Schema内的任意JSON保持原样，不当作消息附件；参数：临时根；返回：无。"""
    inspection, writer, context = _store(tmp_path / "data")
    ordinary = {"type": "image_url", "image_url": {"url": "data: ordinary tool data"}}
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "echo",
                        "input": ordinary,
                    }
                ],
            }
        ],
        "tools": [
            {"name": "echo", "input_schema": {"type": "object", "examples": [ordinary]}}
        ],
    }
    _attempt(writer, context, request=request)
    target = tmp_path / "export"
    target.mkdir()
    write_export(
        inspection,
        freeze_export(inspection, {"session_id": "s", "run_id": "r"}),
        target,
        Event(),
    )
    exported = json.loads((target / "requests.json").read_text(encoding="utf-8"))
    assert exported["attachments"] == []
    assert exported["requests"][0]["attempts"][0]["request"]["request"] == request


def test_anthropic_tool_result_retains_actual_nested_image(tmp_path):
    """工具结果content中的真实图片仍可随包移动；参数：临时根；返回：无。"""
    inspection, writer, context = _store(tmp_path / "data")
    original = b"nested-tool-image"
    part = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(original).decode(),
        },
    }
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": [part]}
                ],
            }
        ]
    }
    _attempt(writer, context, request=request)
    target = tmp_path / "export"
    target.mkdir()
    write_export(
        inspection,
        freeze_export(inspection, {"session_id": "s", "run_id": "r"}),
        target,
        Event(),
    )
    exported = json.loads((target / "requests.json").read_text(encoding="utf-8"))
    assert len(exported["attachments"]) == 1
    assert (
        target / exported["attachments"][0]["package_path"]
    ).read_bytes() == original
