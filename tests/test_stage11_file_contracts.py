"""【文件工具】【接口衔接】验证目录类型、行读取与版本绑定分页。

作者：xxx
时间：2026-10-02 14:10:00
"""

from contextlib import closing
import pytest

from runtime.types import RunToolsRequest
from tools.builtin_tools import build_tool_registry
from tools.readonly_file_tools import ReadOnlyFileToolExecutor
from tools.readonly_inspection import ReadOnlyInspectionExecutor


def file_tools(root, *, max_chars=40000):
    """装配真实只读工具并缩小测试页；参数：隔离目录、正文页长；返回：执行器。"""
    return ReadOnlyFileToolExecutor(ReadOnlyInspectionExecutor(root, 2, max_chars, 2))


def invoke(executor, name, **arguments):
    """经公开工具请求传递参数；参数：执行器、工具名和字段；返回：真实回执。"""
    return executor.execute(RunToolsRequest(name, tool_name=name, arguments=arguments))


def test_directory_pages_have_types_and_reject_changed_sources(tmp_path):
    """目录可逐页读齐，原目录变化后旧页不能混用；参数：隔离目录；返回：无。"""
    for name in ("一", "二", "三"):
        (tmp_path / name).mkdir()
    executor = file_tools(tmp_path)
    first = invoke(executor, "list", path=".")
    assert first.status == "ok" and len(first.meta["entries"]) == 2
    assert all(row["kind"] == "directory" for row in first.meta["entries"])
    second = invoke(executor, "list", path=".", cursor=first.meta["next_cursor"])
    assert second.status == "ok" and len(second.meta["entries"]) == 1
    assert second.meta["next_cursor"] is None
    assert (
        len({row["path"] for page in (first, second) for row in page.meta["entries"]})
        == 3
    )
    (tmp_path / "新增").mkdir()
    changed = invoke(executor, "list", path=".", cursor=first.meta["next_cursor"])
    assert (
        changed.status == "error" and changed.meta["error_category"] == "source_changed"
    )


def test_find_path_distinguishes_file_and_directory(tmp_path):
    """同名片段能按文件或目录查询，并返回可直接读取的路径；参数：隔离目录；返回：无。"""
    (tmp_path / "资料").mkdir()
    (tmp_path / "资料" / "资料说明.txt").write_text("说明", encoding="utf-8")
    executor = file_tools(tmp_path)
    directories = invoke(executor, "find_path", query="资料", kind="directory")
    files = invoke(executor, "find_path", query="资料", kind="file")
    assert directories.status == files.status == "ok"
    assert directories.meta["entries"][0]["path"] == "资料"
    assert files.meta["entries"][0]["path"] == "资料/资料说明.txt"
    assert (
        invoke(executor, "file_read", path=files.meta["entries"][0]["path"]).content
        == "说明"
    )


def test_line_selection_pages_a_long_line_without_omission(tmp_path):
    """超长单行仍有可续读出口，只读指定行且版本变化拒绝旧游标；参数：隔离目录；返回：无。"""
    text = "不选第一行\n" + "中文长行" * 7 + "\n最后一行\n"
    (tmp_path / "行.txt").write_text(text, encoding="utf-8", newline="\n")
    executor = file_tools(tmp_path, max_chars=9)
    first = invoke(executor, "file_read", path="行.txt", start_line=2, line_count=1)
    assert first.status == "ok" and first.meta["next_cursor"]
    chunks = [first.content]
    cursor = first.meta["next_cursor"]
    while cursor:
        page = invoke(executor, "file_read", path="行.txt", cursor=cursor)
        assert page.status == "ok"
        chunks.append(page.content)
        cursor = page.meta["next_cursor"]
    assert "".join(chunks) == "中文长行" * 7 + "\n"
    (tmp_path / "行.txt").write_text("已修改\n", encoding="utf-8")
    changed = invoke(
        executor, "file_read", path="行.txt", cursor=first.meta["next_cursor"]
    )
    assert changed.meta["error_category"] == "source_changed"


def test_missing_path_and_directory_have_distinct_errors(tmp_path):
    """不存在与目录误读分别返回真实类型，目录提供入口而不假装读成功；参数：隔离目录；返回：无。"""
    executor = file_tools(tmp_path)
    directory = invoke(executor, "file_read", path=".")
    missing = invoke(executor, "file_read", path="不存在")
    assert directory.status == missing.status == "error"
    assert directory.meta["actual_kind"] == "directory"
    assert missing.meta["actual_kind"] == "missing"
    assert directory.meta["read_action"] == {"tool": "list", "path": "."}
    with closing(
        build_tool_registry(repo_root=tmp_path, data_root=tmp_path / "data")
    ) as registry:
        assert "offset" not in registry.get("file_read").parameters["properties"]
        assert registry.get("find_path") is not None
        assert registry.get("find_file") is None


def test_grep_pages_bind_file_versions_and_keep_exact_line_readers(tmp_path):
    """同一查询分页面完整，任何被搜索文件变更均拒绝旧页；参数：隔离根；返回：无。"""
    target = tmp_path / "资料.txt"
    target.write_text("命中一\n命中二\n命中三\n", encoding="utf-8", newline="\n")
    executor = file_tools(tmp_path)
    first = invoke(executor, "grep", path=".", query="命中")
    second = invoke(
        executor, "grep", path=".", query="命中", cursor=first.meta["next_cursor"]
    )
    assert [
        row["line"] for page in (first, second) for row in page.meta["entries"]
    ] == [1, 2, 3]
    action = second.meta["entries"][0]["read_action"]
    assert (
        invoke(
            executor,
            action["tool"],
            **{key: value for key, value in action.items() if key != "tool"},
        ).content
        == "命中三\n"
    )
    target.write_text(
        "命中一\n命中二\n命中三\n无匹配的新增行\n", encoding="utf-8", newline="\n"
    )
    assert (
        invoke(
            executor, "grep", path=".", query="命中", cursor=first.meta["next_cursor"]
        ).meta["error_category"]
        == "source_changed"
    )


def test_pdf_pages_are_extracted_text_not_pdf_byte_positions(tmp_path):
    """PDF读取声明提取表示并可跨页读齐；参数：隔离目录；返回：无。"""
    from tests.test_readonly_tool_execution import _simple_pdf_bytes

    (tmp_path / "文章.pdf").write_bytes(_simple_pdf_bytes("Complete PDF text"))
    executor = file_tools(tmp_path, max_chars=7)
    result = invoke(executor, "file_read", path="文章.pdf")
    parts = []
    while True:
        assert result.status == "ok"
        assert result.meta["representation"].startswith("pdf_extracted_text:pypdf:")
        assert result.meta["unit"] == "characters"
        parts.append(result.content)
        if result.meta["next_cursor"] is None:
            break
        result = invoke(
            executor, "file_read", path="文章.pdf", cursor=result.meta["next_cursor"]
        )
    assert "Complete PDF text" in "".join(parts)


def test_line_read_of_redacted_config_does_not_claim_full_delivery(tmp_path):
    """只展示一行脱敏配置不授予整文件覆盖资格；参数：隔离目录；返回：无。"""
    from runtime.types import ReadOnlyInspectionRequest
    from tools.redacted_files import RedactedFiles

    path = tmp_path / ".env"
    path.write_text("PORT=3000\nTOKEN=PRIVATE\n", encoding="utf-8")
    files = RedactedFiles()
    page = files.read(
        path,
        session_id="session",
        max_chars=40000,
        selection=ReadOnlyInspectionRequest(
            "read_file", str(path), start_line=1, line_count=1
        ),
    )
    assert "PRIVATE" not in str(page) and not page["meta"]["output_complete"]
    with pytest.raises(ValueError, match="incomplete"):
        files.prepare(
            "file_write",
            {"view_id": page["meta"]["view_id"], "content": page["content"]},
            session_id="session",
            path=path,
        )
    assert "PRIVATE" in path.read_text(encoding="utf-8")
