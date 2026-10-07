"""【存储】【全文搜索】跨工作区检索、原件定位与只读正文分页。

作者：xxx
时间：2026-09-30 16:00:00
"""

from pathlib import Path

from runtime.file_content import ContentFiles
from runtime.persistence import RuntimeStore
from runtime.session_directory import directory_page, search_content, search_matches
from runtime.workspaces import WorkspaceStore


def test_chinese_body_and_tool_search_locate_frozen_originals(
    tmp_path: Path, monkeypatch
) -> None:
    """目录按中文全文命中而正文始终读冻结版本；参数：根和读取观测；返回：无。"""
    spaces = WorkspaceStore(tmp_path)
    a = spaces.bind_session("session-a", tmp_path / "project-a")
    spaces.bind_session("session-b", tmp_path / "project-b")
    spaces.messages.accept_input("session-a", "旧标题", input_id="title")
    original = spaces.messages.accept_input(
        "session-a", "关键业务：重建索引的验证报告", input_id="body"
    )
    spaces.messages.accept_input("session-b", "另一份重建索引报告")
    with spaces.database.transaction() as batch:
        batch.put(
            "tool_operation",
            "operation",
            {
                "session_id": "session-a",
                "run_id": "run-a",
                "result": {"output": "工具保存了重建索引结果"},
            },
            session_id="session-a",
        )

    def no_content(*args, **kwargs):
        """目录禁止读取大正文；参数：任意调用；返回：失败。"""
        raise AssertionError("directory read full content")

    with monkeypatch.context() as patch:
        patch.setattr(ContentFiles, "read", no_content)
        found = directory_page(tmp_path, "重建索引", workspace_id=a.workspace_id)
        assert [row["session_id"] for row in found["sessions"]] == ["session-a"]
        assert found["sessions"][0]["match"]["source_path"].startswith("workspaces/")
        assert "match" not in directory_page(tmp_path, "旧标题")["sessions"][0]
    matches = search_matches(tmp_path, "session-a", "重建索引")["matches"]
    assert {row["kind"] for row in matches} == {"session_entry", "tool_operation"}
    tool = next(row for row in matches if row["kind"] == "tool_operation")
    with spaces.database.transaction() as batch:
        batch.put(
            "tool_operation",
            "operation",
            {
                "session_id": "session-a",
                "run_id": "run-a",
                "result": {"output": "更新后的结果"},
            },
            session_id="session-a",
        )
    assert "工具保存了重建索引结果" in search_content(tmp_path, tool)["text"]
    assert spaces.messages.current_leaf("session-a") == original.entry_id


def test_prepared_binary_is_streamed_and_committed(tmp_path: Path, monkeypatch) -> None:
    """冻结大文件不需要read_bytes且源修改不污染已保存内容；参数：根和观测；返回：无。"""
    root = tmp_path / "data"
    source = tmp_path / "source.bin"
    source.write_bytes(bytes(range(256)) * 8192)
    store = RuntimeStore(root)
    read_bytes = Path.read_bytes

    def forbid_binary_read(path):
        """禁止二进制整文件物化；参数：文件；返回：仅允许其他小原件读取。"""
        if path.suffix == ".bin":
            raise AssertionError("binary read_bytes used")
        return read_bytes(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", forbid_binary_read)
        reference = store.prepare_file(source)
        with store.transaction() as batch:
            batch.reference_content(reference)
            batch.put(
                "artifact_records", "binary", {"reference": reference.to_mapping()}
            )
        assert (
            sum(len(chunk) for chunk in store.iter_content(reference)) == reference.size
        )
    source.write_bytes(b"changed")
    assert store.read_content(reference, offset=0, limit=256) == bytes(range(256))
    assert reference.path in (root / "commits.jsonl").read_text(encoding="utf-8")
