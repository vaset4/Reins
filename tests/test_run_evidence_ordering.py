"""【运行证据】【提交顺序】验证近期诊断的原件顺序在重新打开和索引重建后不变。

作者：xxx
时间：2026-10-06 19:13:14
"""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

from runtime.run_evidence import RunEvidenceStore

REOPEN_TIMEOUT_SECONDS = 15


@pytest.mark.parametrize(
    "single_batch", [False, True], ids=["separate-commits", "single-commit"]
)
def test_evidence_order_survives_reopen_and_index_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_batch: bool,
) -> None:
    """修订后的诊断按提交位置排序；参数：隔离目录、替换器及是否合批；返回：无。"""
    monkeypatch.setattr(
        "runtime.run_evidence.utc_now", lambda: "2026-10-06T00:00:00+00:00"
    )
    store = RunEvidenceStore(tmp_path)
    # 1. 【运行证据】【提交顺序】末次修订复用最早身份，不能依赖记录首次插入字典的位置
    with store.store.transaction() if single_batch else nullcontext():
        for source_id, message in (
            ("diagnostic-0", "original"),
            ("diagnostic-1", "middle"),
            ("diagnostic-2", "later"),
            ("diagnostic-0", "revised last"),
        ):
            store.write_record(
                session_id="session-1",
                run_id="run-1",
                kind="error",
                source_id=source_id,
                payload={"message": message},
            )
    expected = ["middle", "later", "revised last"]
    assert [
        row["payload"]["message"]
        for row in store.list_records(
            session_id="session-1",
            run_id="run-1",
            kind="error",
        )
    ] == expected

    # 2. 【运行证据】【重开原件】独立进程排除共享内存缓存，随后强制重建派生索引
    code = """
import json
import sys
from app.repl.status import read_latest_run_errors
from runtime.run_evidence import RunEvidenceStore
store = RunEvidenceStore(sys.argv[1])
reopened = store.list_records(session_id="session-1", run_id="run-1", kind="error")
store.store.rebuild_index(force=True)
rebuilt = store.list_records(session_id="session-1", run_id="run-1", kind="error")
latest = read_latest_run_errors(sys.argv[1], session_id="session-1", run_id="run-1", limit=2)
print(json.dumps({
    "reopened": [row["payload"]["message"] for row in reopened],
    "rebuilt": [row["payload"]["message"] for row in rebuilt],
    "latest": [row["message"] for row in latest],
}))
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=REOPEN_TIMEOUT_SECONDS,
        check=True,
    )
    assert json.loads(result.stdout) == {
        "reopened": expected,
        "rebuilt": expected,
        "latest": expected[-2:],
    }
