"""阶段八以真实工作文件验证预览、选择、冲突、幂等和重启边界。

作者：xxx
时间：2026-09-30 22:00:00
"""

from dataclasses import replace
from pathlib import Path
import json

import pytest

from app.session_assembly import user_lease
from approval.session import ApprovalMode
from runtime.file_restore import FileRestoreService
from runtime.file_restore_records import RestoreAuthority
from runtime.file_snapshots import missing_state, new_point
from runtime.workspaces import WorkspaceStore


@pytest.fixture
def restore(tmp_path):
    """提供隔离数据与可变实时权限；参数：临时目录；返回：服务、根和权限。"""
    root, data = tmp_path / "work", tmp_path / "data"
    root.mkdir()
    workspace = WorkspaceStore(data).bind_session("session-test", root)
    authority = [
        RestoreAuthority(
            user_lease(task_id="task", project_root=root, data_root=data),
            ApprovalMode.WORKSPACE,
            "host-instance",
        )
    ]
    service = FileRestoreService(data, authority=lambda _session: authority[0])
    return service, root, workspace, authority


def capture(restore, changes, *, attribution="exact", after=True, input_id="input-one"):
    """保存真实修改前后原件；参数：环境、文件与新内容、归因；返回：恢复点。"""
    service, root, workspace, authority = restore
    point = new_point(
        {
            "session_id": "session-test",
            "workspace_id": workspace.workspace_id,
            "workspace_root": str(root),
            "run_id": "run-one",
            "input_id": input_id,
            "operation_id": "op-test",
        },
        scope="files",
        attribution=attribution,
    )
    for name, content in changes.items():
        path = root / name
        before = service.snapshots.capture_file(
            path, authority[0].lease, workspace.workspace_id
        )
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        current = (
            service.snapshots.capture_file(
                path, authority[0].lease, workspace.workspace_id
            )
            if after
            else None
        )
        point["entries"].append(
            {
                "path": str(path),
                "before": before,
                "after": current,
                "attribution": attribution,
            }
        )
    point["status"] = "complete" if after else "incomplete"
    service.snapshots.publish_point(point)
    return point


def preview(restore, point, *, choices=None):
    """先读条目再按具体选择生成计划；参数：环境、点及选择；返回：持久计划视图。"""
    service = restore[0]
    query = {"session_id": "session-test", "point_id": point["point_id"]}
    detail = service.query({**query, "action": "detail"})
    selection = [
        {
            "entry_id": entry["entry_id"],
            "choice": (choices or {}).get(Path(entry["path"]).name, "restore"),
        }
        for entry in detail["entries"]
    ]
    return service.query({**query, "action": "preview", "selection": selection})


def execute(service, plan, *, sensitive=False):
    """执行明确确认的计划；参数：服务、计划、敏感确认；返回：最终状态。"""
    operation, _created = service.accept(
        {
            "plan_id": plan["plan_id"],
            "confirmation": {"accepted": True, "sensitive": sensitive},
        }
    )
    return service.run(operation["operation_id"])


def test_restore_overwrite_delete_recreate_and_idempotence(restore):
    """混合动作恢复准确原字节，重复确认不再执行；参数：隔离环境；返回：无。"""
    service, root, _workspace, _authority = restore
    (root / "edited.txt").write_bytes(b"original\n")
    (root / "deleted.txt").write_bytes(b"gone\n")
    point = capture(
        restore,
        {"edited.txt": b"changed\n", "new.txt": b"added\n", "deleted.txt": None},
    )
    plan = preview(restore, point)
    result = execute(service, plan)
    assert result["status"] == "completed", result
    assert (root / "edited.txt").read_bytes() == b"original\n"
    assert (root / "deleted.txt").read_bytes() == b"gone\n"
    assert not (root / "new.txt").exists()
    same, created = service.accept(
        {"plan_id": plan["plan_id"], "confirmation": {"accepted": True}}
    )
    assert same["operation_id"] == result["operation_id"] and not created
    assert service.run(same["operation_id"]) == same
    undo_point = next(
        point
        for point in service.snapshots.list_points(_workspace.workspace_id)
        if point["operation_id"] == same["operation_id"]
        and point["entries"][0]["path"].endswith("edited.txt")
    )
    assert execute(service, preview(restore, undo_point))["status"] == "completed"
    assert (root / "edited.txt").read_bytes() == b"changed\n"


def test_unknown_source_and_missing_after_can_be_explicitly_restored(restore):
    """未知归因默认不选，但有效旧版本可明确确认；参数：隔离环境；返回：无。"""
    service, root, *_ = restore
    (root / "note.txt").write_bytes(b"saved")
    point = capture(
        restore, {"note.txt": b"after"}, attribution="observed", after=False
    )
    detail = service.query(
        {
            "action": "detail",
            "session_id": "session-test",
            "point_id": point["point_id"],
        }
    )
    assert detail["entries"][0]["state"] == "source_unknown"
    assert detail["entries"][0]["default_selected"] is False
    plan = preview(restore, point)
    assert "外部编辑" in plan["confirmation_text"]
    assert execute(service, plan)["status"] == "completed"
    assert (root / "note.txt").read_bytes() == b"saved"


def test_all_selected_versions_checked_before_any_write(restore):
    """后一个文件在预览后变化时，前一个也不能提前写；参数：隔离环境；返回：无。"""
    service, root, *_ = restore
    for name in ("a.txt", "b.txt"):
        (root / name).write_bytes(b"old")
    point = capture(restore, {"a.txt": b"new", "b.txt": b"new"})
    plan = preview(restore, point)
    (root / "b.txt").write_bytes(b"human")
    result = execute(service, plan)
    assert result["status"] == "failed"
    assert (root / "a.txt").read_bytes() == b"new"
    assert (root / "b.txt").read_bytes() == b"human"


def test_copy_preserves_later_edit_and_readonly_prevents_execution(restore):
    """另存只新建已预览副本，只读模式不授写权；参数：隔离环境；返回：无。"""
    service, root, _workspace, authority = restore
    (root / "a.txt").write_bytes(b"old")
    point = capture(restore, {"a.txt": b"new"})
    (root / "a.txt").write_bytes(b"human")
    plan = preview(restore, point, choices={"a.txt": "copy"})
    assert execute(service, plan)["status"] == "completed"
    assert (root / "a.txt").read_bytes() == b"human"
    assert Path(plan["entries"][0]["destination"]).read_bytes() == b"old"
    authority[0] = replace(authority[0], mode=ApprovalMode.READ_ONLY)
    denied = preview(restore, point)
    assert not denied["can_execute"]
    with pytest.raises(ValueError):
        execute(service, denied)


def test_turn_uses_first_affected_version_and_complete_diff_pages(restore):
    """逐轮取首次真实变化前态，分页差异不漏尾；参数：隔离环境；返回：无。"""
    service, root, *_ = restore
    (root / "a.txt").write_bytes(b"old\n")
    capture(restore, {"a.txt": b"middle\n"})
    point = capture(restore, {"a.txt": b"last\n"})
    detail = service.query(
        {
            "action": "detail",
            "session_id": "session-test",
            "input_id": point["input_id"],
        }
    )
    plan = service.query(
        {
            "action": "preview",
            "session_id": "session-test",
            "input_id": point["input_id"],
            "selection": [
                {"entry_id": detail["entries"][0]["entry_id"], "choice": "restore"}
            ],
        }
    )
    pieces, offset = [], 0
    while True:
        page = service.query(
            {
                "action": "diff",
                "plan_id": plan["plan_id"],
                "entry_id": plan["entries"][0]["entry_id"],
                "offset": offset,
                "limit": 7,
            }
        )
        pieces.append(page["text"])
        if not page["has_more"]:
            break
        offset = page["next_offset"]
    assert "+old" in "".join(pieces) and "-last" in "".join(pieces)
    assert execute(service, plan)["status"] == "completed"
    assert (root / "a.txt").read_bytes() == b"old\n"


def test_reconcile_does_not_replay_or_claim_matching_bytes(restore):
    """重启时字节匹配也不能冒充本操作已成功；参数：隔离环境；返回：无。"""
    service, root, workspace, authority = restore
    (root / "a.txt").write_bytes(b"old")
    point = capture(restore, {"a.txt": b"new"})
    plan = preview(restore, point)
    operation, _ = service.accept(
        {"plan_id": plan["plan_id"], "confirmation": {"accepted": True}}
    )
    entry = service.records.require(
        "file_restore_operation", operation["operation_id"]
    )["entries"][0]
    stage = root / ".reins-restore-synthetic"
    stage.mkdir()
    service.records.update(
        operation["operation_id"],
        {
            "status": "started",
            "before_restore": entry["current"],
            "temporary_path": str(stage / "target.tmp"),
            "backup_path": str(stage / "backup.tmp"),
        },
        entry_id=entry["entry_id"],
    )
    (root / "a.txt").write_bytes(b"old")
    restarted = FileRestoreService(
        service.snapshots.data_root, authority=lambda _session: authority[0]
    )
    restarted.reconcile()
    result = restarted.status({"operation_id": operation["operation_id"]})
    assert result["status"] == "needs_reconciliation"
    assert result["entries"][0]["status"] == "unknown"
    assert result["entries"][0]["reconciliation"]["disk_matches"] == "target"
    assert (root / "a.txt").read_bytes() == b"old"


def test_sensitive_restore_and_copy_never_return_secret(restore):
    """敏感正文只经过密文原件和私有暂存，副本仍保持敏感路径；参数：环境；返回：无。"""
    service, root, *_ = restore
    secret = b"API_KEY=old-restore-private-value\n"
    (root / ".env").write_bytes(secret)
    point = capture(restore, {".env": b"API_KEY=new-restore-private-value\n"})
    plan = preview(restore, point)
    diff = service.query(
        {
            "action": "diff",
            "plan_id": plan["plan_id"],
            "entry_id": plan["entries"][0]["entry_id"],
        }
    )
    with pytest.raises(PermissionError):
        execute(service, plan)
    result = execute(service, plan, sensitive=True)
    assert result["status"] == "completed", result
    assert (root / ".env").read_bytes() == secret
    assert "private-value" not in json.dumps([plan, diff, result])
    ordinary = list(service.snapshots.data_root.rglob("*.jsonl")) + list(
        service.snapshots.data_root.rglob("*.json")
    )
    assert all(secret not in path.read_bytes() for path in ordinary)
    copy_plan = preview(restore, point, choices={".env": "copy"})
    copy_result = execute(service, copy_plan, sensitive=True)
    assert copy_result["status"] == "completed", copy_result
    copy_path = Path(copy_plan["entries"][0]["destination"])
    assert copy_path.name == ".env" and copy_path.read_bytes() == secret


def test_binary_restore_streams_without_text_decode(restore, monkeypatch):
    """普通大二进制只用流式原件发布；参数：环境和读取探针；返回：无。"""
    service, root, *_ = restore
    raw = b"\x00\xff\xaa\x81" * (512 * 1024)
    (root / "big.bin").write_bytes(raw)
    point = capture(restore, {"big.bin": b"\x00changed"})

    def forbid_full_read(_state):
        """禁止服务把二进制整份读进内存；参数：状态；返回：失败。"""
        raise AssertionError("binary content must stream")

    monkeypatch.setattr(service.snapshots, "read_state", forbid_full_read)
    plan = preview(restore, point)
    assert plan["entries"][0]["binary"] is True
    assert execute(service, plan)["status"] == "completed"
    assert (root / "big.bin").read_bytes() == raw


def test_cancel_stops_later_items_without_undoing_completed_file(restore, monkeypatch):
    """取消仅停后项，已恢复文件不自动回滚；参数：环境、真实发布后的取消；返回：无。"""
    from runtime.file_restore_execution import RestoreExecution

    service, root, *_ = restore
    for name in ("a.txt", "b.txt"):
        (root / name).write_bytes(b"old")
    point = capture(restore, {"a.txt": b"new", "b.txt": b"new"})
    original = RestoreExecution._save_point

    def save_then_cancel(self, plan, identity, entry, before, **options):
        """真实归档第一项后发送取消；参数：原件；返回：实际点。"""
        result = original(self, plan, identity, entry, before, **options)
        service.cancel(identity)
        return result

    monkeypatch.setattr(RestoreExecution, "_save_point", save_then_cancel)
    result = execute(service, preview(restore, point))
    assert result["status"] == "partial"
    assert [entry["status"] for entry in result["entries"]] == ["restored", "cancelled"]
    assert (root / "a.txt").read_bytes() == b"old" and (
        root / "b.txt"
    ).read_bytes() == b"new"


def test_final_external_edit_is_archived_as_actual_displaced_version(
    restore, monkeypatch
):
    """最终替换前的用户稿实际移走后必须可恢复；参数：环境、竞争注入；返回：无。"""
    import tools.file_persistence as persistence

    service, root, workspace, *_ = restore
    (root / "a.txt").write_bytes(b"old")
    point = capture(restore, {"a.txt": b"new"})
    original = persistence.replace_prepared_file

    def race(path, temporary, *, backup_path):
        """在最终Replace前模拟外部覆盖；参数：发布路径；返回：真实发布。"""
        path.write_bytes(b"last human draft")
        return original(path, temporary, backup_path=backup_path)

    monkeypatch.setattr(persistence, "replace_prepared_file", race)
    result = execute(service, preview(restore, point))
    assert result["entries"][0]["status"] == "conflict", result
    saved = service.snapshots.get_point(result["entries"][0]["restore_point_id"])
    assert (
        service.snapshots.read_state(saved["entries"][0]["before"])
        == b"last human draft"
    )
    assert (root / "a.txt").read_bytes() == b"old"


def test_selected_valid_file_survives_unselected_missing_original(restore):
    """缺失原件只禁用对应项，保留它不阻止其他恢复；参数：环境；返回：无。"""
    service, root, *_ = restore
    for name in ("a.txt", "b.txt"):
        (root / name).write_bytes(name.encode())
    point = capture(restore, {"a.txt": b"new", "b.txt": b"new"})
    reference = point["entries"][1]["before"]["content"]
    (service.snapshots.data_root / reference["path"]).unlink()
    plan = preview(restore, point, choices={"b.txt": "keep"})
    assert plan["can_execute"]
    assert execute(service, plan)["status"] == "completed"
    assert (root / "a.txt").read_bytes() == b"a.txt"
    assert (root / "b.txt").read_bytes() == b"new"


def test_background_status_without_jobs_stays_empty_and_needs_no_model(restore):
    """无作业RPC仍是空结果，浏览恢复不建立模型客户端；参数：隔离环境；返回：无。"""
    from app.background.service import BackgroundService
    from app.background.server import BackgroundServer
    from app.background.sessions import SessionServices
    from tools.tool_registry import ToolRegistry

    service, root, *_ = restore

    def forbid_model(_options):
        """模拟模型配置缺失；参数：配置；返回：禁止建立客户端。"""
        raise AssertionError("restore must not build a model")

    owner = BackgroundService(
        SessionServices(root, service.snapshots.data_root, forbid_model, ToolRegistry)
    )
    server = BackgroundServer(owner, "isolated-test-token")
    try:
        reply = server.dispatch(
            "file_restore_query",
            {"payload": {"action": "status", "session_id": "session-test"}},
        )
        assert reply == {}
        listing = owner.file_restore.query(
            {"action": "list", "session_id": "session-test"}
        )
        assert listing["points"] == []
        (root / "no-model.txt").write_bytes(b"old")
        point = capture(restore, {"no-model.txt": b"new"})
        owner_restore = (owner.file_restore.service, *restore[1:])
        plan = preview(owner_restore, point)
        assert execute(owner.file_restore.service, plan)["status"] == "completed"
    finally:
        owner.file_restore.close()
        server.server_close()


def test_card_filter_binds_run_and_call_and_equal_times_keep_publication_order(restore):
    """同call跨run不能串点，同秒按首次提交取旧稿；参数：隔离环境；返回：无。"""
    from runtime.tool_operations import ToolOperationStore

    service, root, *_ = restore
    operations = ToolOperationStore(service.snapshots.data_root)
    (root / "a.txt").write_bytes(b"old")
    points = []
    for index, content in enumerate((b"first", b"second")):
        point = capture(restore, {"a.txt": content})
        point.update(
            operation_id=f"operation-{index}",
            run_id=f"run-{index}",
            started_at="2026-09-30T12:00:00+00:00",
        )
        service.snapshots.publish_point(point)
        operations.create(
            {
                "session_id": "session-test",
                "run_id": point["run_id"],
                "operation_id": point["operation_id"],
            },
            {
                "state": "completed",
                "call": {
                    "call_id": "same-call",
                    "tool_name": "file_write",
                    "args": {"path": str(root / "a.txt")},
                },
                "result": {
                    "status": "ok",
                    "meta": {"restore_point_ids": [point["point_id"]]},
                },
            },
        )
        points.append(point)
    listing = service.query(
        {
            "action": "list",
            "session_id": "session-test",
            "run_id": "run-1",
            "call_id": "same-call",
        }
    )
    assert [row["point_id"] for row in listing["points"]] == [points[1]["point_id"]]
    detail = service.query(
        {
            "action": "detail",
            "session_id": "session-test",
            "input_id": points[0]["input_id"],
        }
    )
    assert detail["entries"][0]["point_id"] == points[0]["point_id"]


def test_publish_receipt_failure_keeps_effect_and_reconciles_real_backup(
    restore, monkeypatch
):
    """发布后登记失败保留实际备份，重启只对账；参数：环境与一次写入故障；返回：无。"""
    service, root, *_ = restore
    (root / "a.txt").write_bytes(b"old")
    point = capture(restore, {"a.txt": b"new"})
    original = service.records.update
    failed = False

    def fail_receipt(identity, changes, *, entry_id=None):
        """仅模拟第一次成功回执写入失败；参数：实际结果；返回：其他写入照常。"""
        nonlocal failed
        if changes.get("status") == "restored" and not failed:
            failed = True
            raise OSError("receipt disk unavailable")
        return original(identity, changes, entry_id=entry_id)

    monkeypatch.setattr(service.records, "update", fail_receipt)
    result = execute(service, preview(restore, point))
    assert result["status"] == "needs_reconciliation"
    assert (root / "a.txt").read_bytes() == b"old"
    assert Path(result["entries"][0]["backup_path"]).read_bytes() == b"new"
    service.reconcile()
    reconciled = service.status({"operation_id": result["operation_id"]})
    assert reconciled["entries"][0]["reconciliation"]["backup_archived"]
    assert reconciled["entries"][0]["status"] == "unknown"


def test_sensitive_interruption_after_move_preserves_secret_and_does_not_replay(
    restore, monkeypatch
):
    """敏感原件已移走而新文件未发布时保留真前态；参数：环境、真实移动后中断；返回：无。"""
    import tools.file_persistence as persistence

    service, root, *_ = restore
    (root / ".env").write_bytes(b"KEY=old-private\n")
    point = capture(restore, {".env": b"KEY=new-private\n"})
    original = persistence.publish_prepared_file

    def interrupt_new_file(path, temporary):
        """敏感文件原件移动后阻止下一次发布；参数：实际路径；返回：模拟IO失败。"""
        if path == root / ".env":
            raise OSError("publication interrupted after sensitive rename")
        return original(path, temporary)

    monkeypatch.setattr(persistence, "publish_prepared_file", interrupt_new_file)
    result = execute(service, preview(restore, point), sensitive=True)
    assert result["status"] == "needs_reconciliation", result
    assert not (root / ".env").exists()
    backup = Path(result["entries"][0]["backup_path"])
    assert backup.read_bytes() == b"KEY=new-private\n"
    service.reconcile()
    assert not (root / ".env").exists()
    updated = service.status({"operation_id": result["operation_id"]})
    assert updated["entries"][0]["status"] == "unknown"
    assert "private" not in json.dumps(updated)


def test_deletion_interruption_keeps_real_moved_file_and_restarts_without_replay(
    restore, monkeypatch
):
    """删除后的进程中断保留真实移走版本，重启不重做动作；参数：环境与崩溃点；返回：无。"""
    from runtime.file_restore_execution import RestoreExecution

    service, root, *_ = restore
    point = capture(restore, {"created.txt": b"actual new file"})
    plan = preview(restore, point)
    operation, _ = service.accept(
        {"plan_id": plan["plan_id"], "confirmation": {"accepted": True}}
    )
    original = RestoreExecution._publish

    def crash_after_move(self, path, temporary, backup, **options):
        """真实移动后模拟进程直接离开而没有回执；参数：发布材料；返回：不返回。"""
        original(self, path, temporary, backup, **options)
        raise KeyboardInterrupt("simulated process exit")

    monkeypatch.setattr(RestoreExecution, "_publish", crash_after_move)
    with pytest.raises(KeyboardInterrupt):
        service.run(operation["operation_id"])
    assert not (root / "created.txt").exists()
    started = service.status({"operation_id": operation["operation_id"]})
    backup = Path(started["entries"][0]["backup_path"])
    assert backup.read_bytes() == b"actual new file"
    service.reconcile()
    reconciled = service.status({"operation_id": operation["operation_id"]})
    assert reconciled["status"] == "needs_reconciliation"
    assert not (root / "created.txt").exists()
    point = service.snapshots.get_point(reconciled["entries"][0]["restore_point_id"])
    assert (
        service.snapshots.read_state(point["entries"][0]["before"])
        == b"actual new file"
    )


def test_preview_current_version_is_a_committed_backup_dependency(restore):
    """当前用户稿只在预览捕获时也进入完整备份依赖；参数：环境；返回：无。"""
    service, root, *_ = restore
    (root / "a.txt").write_bytes(b"old")
    point = capture(restore, {"a.txt": b"new"})
    (root / "a.txt").write_bytes(b"human only captured by preview")
    plan = preview(restore, point)
    stored = service.records.require("file_restore_plan", plan["plan_id"])
    reference = stored["entries"][0]["current"]["content"]["path"]
    commits = [
        json.loads(line)
        for line in (service.snapshots.data_root / "commits.jsonl")
        .read_text()
        .splitlines()
    ]
    assert any(
        item["type"] == "immutable" and item["path"] == reference
        for commit in commits
        for item in commit["files"]
    )


def test_permissions_changed_after_preview_rejects_without_writes(restore):
    """实时撤销权限后旧预览无法执行；参数：环境；返回：无。"""
    service, root, _workspace, authority = restore
    (root / "a.txt").write_bytes(b"old")
    point = capture(restore, {"a.txt": b"new"})
    plan = preview(restore, point)
    operation, _ = service.accept(
        {"plan_id": plan["plan_id"], "confirmation": {"accepted": True}}
    )
    authority[0] = replace(authority[0], mode=ApprovalMode.READ_ONLY)
    result = service.run(operation["operation_id"])
    assert result["status"] == "failed"
    assert (root / "a.txt").read_bytes() == b"new"


def test_observed_new_sensitive_file_can_be_explicitly_removed(restore):
    """范围捕获的不存在前态未带敏感标记时仍按当前路径走保护通道；参数：环境；返回：无。"""
    service, root, workspace, authority = restore
    path = root / ".env"
    path.write_bytes(b"KEY=created-sensitive-value\n")
    after = service.snapshots.capture_file(
        path, authority[0].lease, workspace.workspace_id
    )
    point = new_point(
        {
            "session_id": "session-test",
            "workspace_id": workspace.workspace_id,
            "workspace_root": str(root),
            "run_id": "run",
            "input_id": "input",
            "operation_id": "exec",
        },
        scope="workspace",
        attribution="observed",
    )
    point.update(
        status="complete",
        entries=[
            {
                "path": str(path),
                "before": missing_state(str(path)),
                "after": after,
                "attribution": "observed",
            }
        ],
    )
    service.snapshots.publish_point(point)
    plan = preview(restore, point)
    assert plan["can_execute"] and plan["entries"][0]["sensitive"]
    result = execute(service, plan, sensitive=True)
    assert result["status"] == "completed", result
    assert not path.exists()


def test_reconcile_does_not_cancel_an_operation_owned_by_another_executor(restore):
    """对账遇到真实执行锁时保留活作业，不等待后误取消；参数：环境；返回：无。"""
    from threading import Event, Thread
    from tools.file_persistence import file_edit_lock

    service, root, *_ = restore
    point = capture(restore, {"created.txt": b"new"})
    plan = preview(restore, point)
    operation, _ = service.accept(
        {"plan_id": plan["plan_id"], "confirmation": {"accepted": True}}
    )
    acquired, release = Event(), Event()

    def active_executor():
        """代表持有原操作的另一个执行线程；参数：无；返回：无。"""
        with file_edit_lock(
            service.snapshots.data_root / (operation["operation_id"] + ".execution")
        ):
            acquired.set()
            assert release.wait(5)

    thread = Thread(target=active_executor)
    thread.start()
    try:
        assert acquired.wait(5)
        service.reconcile()
        actual = service.status({"operation_id": operation["operation_id"]})
        assert actual["status"] == "queued"
        assert actual["entries"][0]["status"] == "pending"
        assert (root / "created.txt").read_bytes() == b"new"
    finally:
        release.set()
        thread.join(5)
