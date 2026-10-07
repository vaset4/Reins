from __future__ import annotations

from pathlib import Path

import pytest

from context.token_estimate import estimate_agent_messages_tokens
from llm.messages import (
    AgentMessage,
    AssistantMessage,
    TextPart,
    UserMessage,
)
from runtime.agent_loop import AgentLoop
from runtime.ledger import LedgerStore
from runtime.lease import from_trigger
from runtime.run_evidence_view import build_run_evidence_view
from runtime.run_facts import RunFactStore
from runtime.types import ReadOnlyInspectionRequest, RunContext, RunToolsResult, Trigger
from tools.readonly_inspection import ReadOnlyInspectionExecutor


def test_s3_pdf_offset_continuation_is_contiguous_for_multipage_pdf(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pypdf")
    pdf_path = tmp_path / "multipage.pdf"
    pdf_path.write_bytes(
        _multipage_pdf_bytes(
            [
                "PAGE_ONE_SENTINEL alpha beta gamma",
                "PAGE_TWO_SENTINEL delta epsilon zeta",
            ]
        )
    )
    full = _read_pdf(tmp_path, max_chars=10_000, offset=0)
    chunks: list[str] = []
    offset = 0
    for _ in range(20):
        chunk = _read_pdf(tmp_path, max_chars=17, offset=offset)
        chunks.append(chunk.output)
        next_offset = chunk.meta.get("next_offset")
        if next_offset is None:
            break
        offset = int(next_offset)

    assert "PAGE_ONE_SENTINEL" in full.output
    assert "PAGE_TWO_SENTINEL" in full.output
    assert "".join(chunks) == full.output


def test_s5_same_agent_loop_instance_resets_recovery_budget_between_runs(
    tmp_path: Path,
) -> None:
    loop = AgentLoop(tmp_path)
    first = _recovery_context("task-a")
    second = _recovery_context("task-b")
    result = RunToolsResult.error_result(
        action="file_read",
        tool_name="file_read",
        error="invalid_input: missing required parameters for tool: path",
    )

    first_observation = loop._handle_recoverable_tool_error(first, result)
    second_observation = loop._handle_recoverable_tool_error(second, result)

    assert first_observation.meta["budget_count"] == 1
    assert second_observation.meta["budget_count"] == 1


def test_s5_recovery_budget_still_increments_within_one_run(
    tmp_path: Path,
) -> None:
    loop = AgentLoop(tmp_path)
    context = _recovery_context("task-a")
    result = RunToolsResult.error_result(
        action="file_read",
        tool_name="file_read",
        error="invalid_input: missing required parameters for tool: path",
    )

    first_observation = loop._handle_recoverable_tool_error(context, result)
    second_observation = loop._handle_recoverable_tool_error(context, result)

    assert first_observation.meta["budget_count"] == 1
    assert second_observation.meta["budget_count"] == 2


def test_s9_context_segments_conversation_estimate_includes_message_overhead(
    tmp_path: Path,
) -> None:
    loop = AgentLoop(tmp_path)
    context = _recovery_context("task-segments")
    history: tuple[AgentMessage, ...] = tuple(
        UserMessage(f"m{index}", (TextPart("word"),))
        if index % 2 == 0
        else AssistantMessage(f"m{index}", (TextPart("word"),))
        for index in range(30)
    )

    segments = loop._production_context_builder().segment_evidence(
        {"conversation_history": history},
        model_task="word",
    )
    from scripts.testing.llm import from_test_stub

    loop.llm_client = from_test_stub("unused")
    loop._prepare_turn_core(context).store.close()
    loop.model_runner._emit_context_segments(context, segments)

    facts = RunFactStore(tmp_path).read_run(context.run_id)
    segment_fact = next(
        fact for fact in facts if fact.get("event") == "context:segments"
    )
    conversation = next(
        item
        for item in segment_fact["segments"]
        if item["name"] == "conversation_history"
    )
    message_estimate = estimate_agent_messages_tokens(history)

    assert conversation["message_count"] == len(history)
    assert conversation["token_estimate_kind"] == "message_with_overhead"
    assert conversation["tokens_est"] == message_estimate
    assert int(conversation["content_tokens_est"]) < message_estimate
    view = build_run_evidence_view(
        LedgerStore(tmp_path).read_run_events(context.run_id)
    )
    ledger_segments = view.context_segments[0]["payload"]["segments"]
    assert ledger_segments == segment_fact["segments"]


def _read_pdf(tmp_path: Path, *, max_chars: int, offset: int):
    executor = ReadOnlyInspectionExecutor(tmp_path, 50, max_chars, 50)
    return executor.execute(
        ReadOnlyInspectionRequest(
            action="read_file",
            target_path="multipage.pdf",
            offset=offset,
        )
    )


def _recovery_context(task_id: str) -> RunContext:
    return RunContext(
        task_id=task_id,
        trigger=Trigger.USER,
        payload={"message": "audit repro"},
        capability_lease=from_trigger("user", task_id=task_id),
    )


def _multipage_pdf_bytes(page_texts: list[str]) -> bytes:
    page_count = len(page_texts)
    font_object_id = 3 + page_count
    content_start_id = font_object_id + 1
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        _pages_object(page_count),
    ]
    for index in range(page_count):
        objects.append(
            _page_object(
                parent_id=2, font_id=font_object_id, content_id=content_start_id + index
            )
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.extend(_content_object(text) for text in page_texts)
    return _pdf_document(objects)


def _pages_object(page_count: int) -> bytes:
    kids = " ".join(f"{3 + index} 0 R" for index in range(page_count))
    return f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("ascii")


def _page_object(*, parent_id: int, font_id: int, content_id: int) -> bytes:
    return (
        f"<< /Type /Page /Parent {parent_id} 0 R /MediaBox [0 0 612 792] "
        f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
        f"/Contents {content_id} 0 R >>"
    ).encode("ascii")


def _content_object(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    return (
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream"
    )


def _pdf_document(objects: list[bytes]) -> bytes:
    body = b"%PDF-1.4\n"
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_start = len(body)
    xref = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    xref += b"".join(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets)
    trailer = (
        f"trailer\n<< /Root 1 0 R /Size {len(objects) + 1} >>\nstartxref\n"
    ).encode("ascii")
    return body + xref + trailer + str(xref_start).encode("ascii") + b"\n%%EOF\n"
