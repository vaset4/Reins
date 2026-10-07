"""验证真实文件并行、依赖顺序和操作变化查询。

作者：xxx
时间：2026-09-24 23:00:00
"""

from scripts.testing.llm import _from_scripted
import json
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from threading import Barrier

import pytest

from scripts.testing.llm import _ScriptedTurn
from llm.messages import ToolCallPart
from runtime.agent_loop import AgentLoop
from runtime.default_capabilities import build_local_agent_capabilities
from runtime.lease import from_trigger
from runtime.session_message_store import SessionMessageStore
from runtime.tool_executor import execution_groups
from runtime.tool_operations import ToolOperation, ToolOperationStore, file_changes
from runtime.types import RunContext, RunToolsRequest, Trigger
from runtime.workspaces import WorkspaceStore
from tasks.store import TaskStore
from tools.builtin_tools import build_tool_registry
from tools.file_persistence import content_sha256
from tools.file_resources import describe_resource
from tools.types import ToolError


def _runtime(root, calls):
    """组装写入后实际调用变化查询的主循环；传参：目录与调用；返回：循环、运行及注册表。"""
    data = root / "data"
    with closing(TaskStore(data)) as tasks:
        task = tasks.create_task("修改文件并核对本次变化")
    registry = build_tool_registry(repo_root=root, data_root=data)
    client = _from_scripted(
        [
            _ScriptedTurn(calls=tuple(calls)),
            _ScriptedTurn(
                calls=(
                    ToolCallPart(
                        "load-operations",
                        "capabilities",
                        {"action": "load", "domain": "operations"},
                    ),
                )
            ),
            _ScriptedTurn(
                calls=(
                    ToolCallPart("query", "operation_status", {"run_id": "current"}),
                )
            ),
            _ScriptedTurn(text="已核对文件变化"),
        ]
    )
    context = RunContext(
        task_id=task.task_id,
        trigger=Trigger.USER,
        payload={"message": task.goal},
        capability_lease=from_trigger(
            "user",
            task_id=task.task_id,
            capabilities=build_local_agent_capabilities(root, data),
        ),
    )
    # 1. 【文件操作】【执行归属】与正式入口一致绑定工作区并接纳输入，后台来源据此保持原权限
    WorkspaceStore(data).bind_session(context.session_id, root)
    context.payload["input_message_id"] = (
        SessionMessageStore(data)
        .accept_input(
            context.session_id,
            task.goal,
            run_id=context.run_id,
            task_id=task.task_id,
        )
        .entry_id
    )
    return AgentLoop(data, llm_client=client, tool_registry=registry), context, registry


def _query(loop, context):
    """读取真实查询工具保存的回执；传参：循环与运行；返回：文件变化投影。"""
    records = loop.operations.for_session(context.session_id)
    result = next(
        row["result"]
        for row in records
        if row["call"]["tool_name"] == "operation_status"
    )
    return json.loads(result["output"])["file_changes"]


def test_independent_writes_meet_at_publication_and_query_actual_versions(
    tmp_path, monkeypatch
):
    """两个独立文件的真实发布能会合，查询呈现新建的空前版本；传参：目录和替换器；返回：无。"""
    from tools import file_persistence

    barrier = Barrier(2, timeout=5)
    publish = file_persistence.publish_prepared_file

    def synchronized(path, temporary):
        """在实际发布处验证并行；传参：路径和字节；返回：无。"""
        if path.parent == tmp_path and path.name in {"a.txt", "b.txt"}:
            barrier.wait()
        publish(path, temporary)

    monkeypatch.setattr(file_persistence, "publish_prepared_file", synchronized)
    calls = [
        ToolCallPart(name, "file_write", {"path": name, "content": name})
        for name in ("a.txt", "b.txt")
    ]
    loop, context, _ = _runtime(tmp_path, calls)
    list(loop.run_stream(context))
    assert [(tmp_path / name).read_text() for name in ("a.txt", "b.txt")] == [
        "a.txt",
        "b.txt",
    ]
    view = _query(loop, context)
    assert (
        len(view["changes"]) == 2
        and view["scope"] == "captured_operations_and_legacy_file_changes"
    )
    assert all(
        row["state"] == "succeeded"
        and row["created"]
        and row["previous_sha256"] is None
        for row in view["changes"]
    )


def test_same_file_writes_keep_order_and_reject_stale_version(tmp_path):
    """第二次旧版本写不覆盖第一次结果，后续目录读取看见实际文件；传参：目录；返回：无。"""
    path = tmp_path / "a.txt"
    path.write_bytes(b"old")
    calls = [
        ToolCallPart(
            str(index),
            "file_write",
            {
                "path": "a.txt",
                "content": value,
                "expected_sha256": content_sha256(b"old"),
            },
        )
        for index, value in enumerate(("first", "second"))
    ]
    calls.append(ToolCallPart("read", "file_read", {"path": "a.txt"}))
    loop, context, _ = _runtime(tmp_path, calls)
    list(loop.run_stream(context))
    assert path.read_bytes() == b"first"
    changes = _query(loop, context)["changes"]
    assert [row["state"] for row in changes] == ["succeeded", "failed"]
    assert "content_sha256" not in changes[1]
    records = loop.operations.for_session(context.session_id)
    assert (
        next(
            row["result"]["output"]
            for row in records
            if row["call"]["tool_name"] == "file_read"
        )
        == "first"
    )


@pytest.mark.parametrize("alias", ["hardlink", "symlink", "case", "directory"])
def test_aliases_and_directory_reads_share_file_dependency(tmp_path, alias):
    """Windows真实别名与目录范围不能被当成独立资源；传参：目录与别名类型；返回：无。"""
    target = tmp_path / "a.txt"
    target.write_bytes(b"old")
    other = tmp_path / "other.txt"
    if alias == "hardlink":
        os.link(target, other)
    elif alias == "symlink":
        other.symlink_to(target)
    elif alias == "case":
        other = tmp_path / "A.TXT"
    else:
        other = tmp_path
    registry = build_tool_registry(repo_root=tmp_path)
    lease = from_trigger(
        "user",
        task_id="group",
        capabilities=build_local_agent_capabilities(tmp_path, tmp_path / "data"),
    )
    calls = []
    for index, (name, path) in enumerate(
        (
            ("file_write", target),
            ("list" if alias == "directory" else "file_read", other),
        )
    ):
        args = {"path": str(path)}
        request = RunToolsRequest(name, tool_name=name, arguments=args)
        calls.append(
            ToolOperation(
                request,
                str(index),
                name,
                args,
                resource=describe_resource(registry.get(name), args, lease),
            )
        )
    assert [len(group) for group in execution_groups(calls, registry)] == [1, 1]


def test_link_retargeted_after_preparation_is_not_written(tmp_path):
    """授权准备后链接换目标会显式冲突，两端文件都不改；传参：目录；返回：无。"""
    first, second, link = (
        tmp_path / name for name in ("first.txt", "second.txt", "link.txt")
    )
    first.write_bytes(b"old")
    second.write_bytes(b"old")
    link.symlink_to(first)
    loop, context, registry = _runtime(tmp_path, [])
    prepared = registry.prepare_tool_execution(
        "file_write",
        {
            "path": str(link),
            "content": "new",
            "expected_sha256": content_sha256(b"old"),
        },
        context.capability_lease,
    )
    assert not isinstance(prepared, ToolError)
    link.unlink()
    link.symlink_to(second)
    result = registry.execute_prepared_tool(prepared)
    assert isinstance(result, ToolError) and "resource changed" in result.message
    assert first.read_bytes() == second.read_bytes() == b"old"


def test_published_file_with_missing_result_remains_unknown_after_restart(
    tmp_path, monkeypatch
):
    """发布后记录失败不冒充未改文件，重读投影不会执行写入；传参：目录与替换器；返回：无。"""
    loop, context, _ = _runtime(
        tmp_path,
        [
            ToolCallPart(
                "write", "file_write", {"path": "a.txt", "content": "published"}
            )
        ],
    )
    write = loop.operations.write

    def fail_result(identity, payload):
        """只在保存结果窗口失败；传参：身份与记录；返回：无。"""
        if "result" in payload:
            raise OSError("result persistence interrupted")
        write(identity, payload)

    monkeypatch.setattr(loop.operations, "write", fail_result)
    with pytest.raises(OSError, match="result persistence"):
        list(loop.run_stream(context))
    assert (tmp_path / "a.txt").read_bytes() == b"published"
    records = ToolOperationStore(tmp_path / "data").for_session(context.session_id)
    view = file_changes(records)
    assert view["changes"][0]["state"] == "unknown"
    assert "content_sha256" not in view["changes"][0]
    assert file_changes(records) == view


@pytest.mark.parametrize("publication", ["unchanged", "failed"])
def test_query_preserves_no_change_and_failed_publication(
    tmp_path, monkeypatch, publication
):
    """无内容变化和发布失败分别保留真实状态，查询不会重复记账；传参：目录、替换器与窗口；返回：无。"""
    from tools import file_persistence

    path = tmp_path / "a.txt"
    path.write_bytes(b"original")
    if publication == "failed":

        def fail_publish(_path, _temporary, *, backup_path):
            """在实际发布入口制造IO失败；传参：路径和字节；返回：不返回。"""
            raise OSError("publication interrupted")

        monkeypatch.setattr(file_persistence, "replace_prepared_file", fail_publish)
    content = "original" if publication == "unchanged" else "candidate"
    loop, context, _ = _runtime(
        tmp_path,
        [
            ToolCallPart(
                "write",
                "file_write",
                {
                    "path": "a.txt",
                    "content": content,
                    "expected_sha256": content_sha256(b"original"),
                },
            )
        ],
    )
    list(loop.run_stream(context))
    assert path.read_bytes() == b"original"
    view = _query(loop, context)
    assert len(view["changes"]) == 1
    change = view["changes"][0]
    if publication == "unchanged":
        assert (
            change["state"] == "succeeded"
            and change["changed"] is False
            and change["created"] is False
        )
        assert (
            change["previous_sha256"]
            == change["content_sha256"]
            == content_sha256(b"original")
        )
    else:
        assert change["state"] == "failed" and "content_sha256" not in change
    assert file_changes(loop.operations.for_session(context.session_id)) == view


def test_independent_redacted_writes_do_not_share_a_publication_lock(
    tmp_path, monkeypatch
):
    """宿主保护映射不串行化两个独立文件的发布；传参：目录与替换器；返回：无。"""
    from tools import restore_protection

    registry = build_tool_registry(repo_root=tmp_path, data_root=tmp_path / "data")
    lease = from_trigger(
        "user",
        task_id="redacted",
        capabilities=build_local_agent_capabilities(tmp_path, tmp_path / "data"),
    )
    arguments = []
    for name in ("a", "b"):
        path = tmp_path / name / ".env"
        path.parent.mkdir()
        path.write_bytes(b"TOKEN=PRIVATE\nPORT=3000\n")
        view = registry.execute_tool("file_read", {"path": str(path)}, lease)
        arguments.append(
            {
                "path": str(path),
                "view_id": view["meta"]["view_id"],
                "expected_sha256": view["meta"]["content_sha256"],
                "old_text": "PORT=3000",
                "new_text": "PORT=4000",
            }
        )
    barrier = Barrier(2, timeout=5)
    publish = restore_protection.protected_replace_prepared_file

    def synchronized(path, temporary, *, backup_path):
        """两个敏感文件必须在发布处会合；传参：路径和字节；返回：无。"""
        barrier.wait()
        return publish(path, temporary, backup_path=backup_path)

    monkeypatch.setattr(
        restore_protection, "protected_replace_prepared_file", synchronized
    )
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(
            workers.map(
                lambda args: registry.execute_tool("file_patch", args, lease), arguments
            )
        )
    assert all(not isinstance(result, ToolError) for result in results), results
