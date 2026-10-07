"""【文件恢复】【跨层复核】恢复原件在索引丢失或损坏后仍可完整重建。

作者：xxx
时间：2026-09-30 23:00:00
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from tests.test_stage8_restore_service import capture, execute, preview

pytest_plugins = ("tests.test_stage8_restore_service",)


@pytest.mark.parametrize("broken_index", [False, True])
def test_restore_points_plans_and_operations_rebuild_from_sources(
    restore, tmp_path, broken_index
):
    """新进程只靠完整原件重建全部恢复身份；参数：隔离环境与索引损坏方式；返回：无。"""
    service, root, *_ = restore
    (root / "review.txt").write_bytes(b"original")
    point = capture(restore, {"review.txt": b"tool result"})
    plan = preview(restore, point)
    operation = execute(service, plan)
    copied = tmp_path / "source-copy"
    shutil.copytree(
        service.snapshots.data_root,
        copied,
        ignore=shutil.ignore_patterns("index.sqlite*"),
    )
    if broken_index:
        (copied / "index.sqlite").write_bytes(b"broken index")
    code = """
import json, sys
from runtime.persistence import RuntimeStore
from runtime.file_records import ContentReference
store = RuntimeStore(sys.argv[1])
store.rebuild_index(force=True)
with store.snapshot() as source:
    point = source.get('file_restore_point', sys.argv[2])
    plan = source.get('file_restore_plan', sys.argv[3])
    operation = source.get('file_restore_operation', sys.argv[4])
before = store.read_content(ContentReference.from_mapping(point['entries'][0]['before']['content']))
print(json.dumps({'before': before.decode(), 'plan_id': plan['plan_id'], 'operation_id': operation['operation_id'],
    'status': operation['status'], 'count': len(operation['entries'])}))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(copied),
            point["point_id"],
            plan["plan_id"],
            operation["operation_id"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "before": "original",
        "plan_id": plan["plan_id"],
        "operation_id": operation["operation_id"],
        "status": "completed",
        "count": 1,
    }
    assert (root / "review.txt").read_bytes() == b"original"
