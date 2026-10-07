from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tools.readonly_file_tools import ReadOnlyFileToolExecutor
from tools.readonly_inspection import DEFAULT_READ_MAX_CHARS, ReadOnlyInspectionExecutor
from tools.readonly_web_tools import ReadOnlyWebToolExecutor


def test_readonly_file_executor_reads_lists_and_searches(tmp_path: Path) -> None:
    """验证正式文件执行器读取、列出和搜索真实内容；参数：隔离目录；返回：无。"""
    from runtime.types import RunToolsRequest

    (tmp_path / "runtime").mkdir()
    (tmp_path / "runtime" / "agent_loop.py").write_text(
        "class AgentLoop:\n    pass\n",
        encoding="utf-8",
    )

    file_tools = ReadOnlyFileToolExecutor(
        ReadOnlyInspectionExecutor(tmp_path, 50, DEFAULT_READ_MAX_CHARS, 50)
    )

    read_result = file_tools.execute(
        RunToolsRequest(
            action="file_read",
            payload="",
            tool_name="file_read",
            arguments={"path": "runtime/agent_loop.py"},
            target_scope="runtime/agent_loop.py",
        )
    )
    list_result = file_tools.execute(
        RunToolsRequest(
            action="list",
            payload="",
            tool_name="list",
            arguments={"path": "runtime"},
        )
    )
    grep_result = file_tools.execute(
        RunToolsRequest(
            action="grep",
            payload="",
            tool_name="grep",
            arguments={"path": "runtime", "query": "AgentLoop"},
        )
    )

    assert read_result.status == "ok"
    assert read_result.content is not None
    assert "class AgentLoop" in read_result.content
    assert read_result.target_scope == "runtime/agent_loop.py"
    assert list_result.status == "ok"
    assert list_result.content is not None
    assert "agent_loop.py" in list_result.content
    assert grep_result.status == "ok"
    assert grep_result.content is not None
    assert "AgentLoop" in grep_result.content


def test_file_read_rejects_legacy_character_offset_without_reinterpreting_it(
    tmp_path: Path,
) -> None:
    """新接口拒绝旧字符偏移，不能把其静默变成行号；参数：隔离目录；返回：无。"""
    from runtime.types import RunToolsRequest

    (tmp_path / "notes.txt").write_text("0123456789abcdef", encoding="utf-8")
    file_tools = ReadOnlyFileToolExecutor(
        ReadOnlyInspectionExecutor(tmp_path, 50, DEFAULT_READ_MAX_CHARS, 50)
    )
    result = file_tools.execute(
        RunToolsRequest(
            action="file_read",
            tool_name="file_read",
            arguments={"path": "notes.txt", "offset": 6},
        )
    )
    assert result.status == "error"
    assert "offset" in result.error


def test_file_read_extracts_pdf_text_when_parser_is_available(tmp_path: Path) -> None:
    """验证正式读取入口提取 PDF 正文与页数；参数：隔离目录；返回：无。"""
    if importlib.util.find_spec("pypdf") is None:
        pytest.skip("pypdf is not installed")

    from runtime.types import RunToolsRequest

    (tmp_path / "paper.pdf").write_bytes(_simple_pdf_bytes("Hello PDF paper"))
    file_tools = ReadOnlyFileToolExecutor(
        ReadOnlyInspectionExecutor(tmp_path, 50, DEFAULT_READ_MAX_CHARS, 50)
    )

    result = file_tools.execute(
        RunToolsRequest(
            action="file_read",
            payload="",
            tool_name="file_read",
            arguments={"path": "paper.pdf"},
        )
    )

    assert result.status == "ok"
    assert result.content is not None
    assert "Hello PDF paper" in result.content
    assert result.meta["file_type"] == "pdf"
    assert result.meta["page_count"] == 1


def test_readonly_web_executor_searches_fetches_and_scans() -> None:
    """验证正式网页执行器解析注入的网页内容；参数：无；返回：无。"""
    from runtime.types import RunToolsRequest

    html = """
    <html>
      <head><title>Example</title></head>
      <body>
        <h1>Example Result</h1>
        <p>Alpha Beta</p>
        <a href="https://example.com/page">Page Link</a>
      </body>
    </html>
    """

    def fetcher(url: str) -> str:
        return html if "duckduckgo" not in url else html

    web_tools = ReadOnlyWebToolExecutor(fetcher=fetcher)

    search_result = web_tools.execute(
        RunToolsRequest(
            action="web_search",
            payload="",
            tool_name="web_search",
            arguments={"query": "Example"},
        )
    )
    fetch_result = web_tools.execute(
        RunToolsRequest(
            action="web_fetch",
            payload="",
            tool_name="web_fetch",
            arguments={"url": "https://example.com"},
        )
    )
    scan_result = web_tools.execute(
        RunToolsRequest(
            action="web_scan",
            payload="",
            tool_name="web_scan",
            arguments={"url": "https://example.com"},
        )
    )

    assert search_result.status == "ok"
    assert search_result.content is not None
    assert "Example" in search_result.content
    assert "https://example.com/page" in search_result.content
    assert fetch_result.status == "ok"
    assert fetch_result.content is not None
    assert "Example Result" in fetch_result.content
    assert scan_result.status == "ok"
    assert scan_result.content is not None
    assert "Alpha Beta" in scan_result.content


def _simple_pdf_bytes(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\n"
        b"stream\n" + stream + b"\nendstream",
    ]
    body = b"%PDF-1.4\n"
    offsets: list[int] = []
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{idx} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"
    xref_start = len(body)
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    xref += b"".join(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets)
    trailer = (
        b"trailer\n<< /Root 1 0 R /Size 6 >>\nstartxref\n"
        + str(xref_start).encode("ascii")
        + b"\n%%EOF\n"
    )
    return body + xref + trailer
