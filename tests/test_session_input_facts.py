"""验证待处理输入的一次事实读取及实时更新边界。

作者：xxx
时间：2026-09-28 17:15:00
"""

from pathlib import Path

import pytest

from runtime.run_facts import RunFactStore
from runtime.file_records import SourceCorruptionError
from runtime.session_message_store import SessionMessageStore
from runtime.session_runtime import SessionRun, SessionRuntime
from scripts.benchmark_session_metrics import measure_reads


def unexpected_run(request: SessionRun) -> None:
    """读取检查不得启动模型；传参：运行请求；返回：不返回。"""
    raise AssertionError(f"unexpected dispatch {request.input_id}")


def test_handled_inputs_read_each_fact_file_once(tmp_path: Path) -> None:
    """同一查询只解析一次每运行文件；传参：隔离根；返回：无。"""
    facts = RunFactStore(tmp_path)
    opens = []
    for index in range(3):
        facts.append(
            {
                "session_id": "session-one",
                "run_id": f"run-{index}",
                "event": "input:handled",
                "input_ids": [f"input-{index}"],
            }
        )
        with measure_reads(tmp_path) as meter:
            handled = facts.handled_input_ids("session-one")
        opens.append(meter.snapshot().file_opens)
    assert handled == frozenset({"input-0", "input-1", "input-2"})
    assert opens[0] > 0
    assert opens == [opens[0]] * 3, "同一原件的打开次数不得随运行数量增长"


def test_pending_inputs_follow_new_facts_branch_and_restart(tmp_path: Path) -> None:
    """事实追加与分支变化无需失效缓存；传参：隔离根；返回：无。"""
    messages, facts = SessionMessageStore(tmp_path), RunFactStore(tmp_path)
    runtime = SessionRuntime(
        "session-one", messages=messages, facts=facts, run=unexpected_run
    )
    messages.accept_input("session-one", "第一条", input_id="input-a")
    messages.accept_input("session-one", "旧分支", input_id="input-b")
    assert [row.entry_id for row in runtime._unhandled_inputs()] == [
        "input-a",
        "input-b",
    ]
    facts.append(
        {
            "session_id": "session-one",
            "run_id": "run-first",
            "event": "input:handled",
            "input_ids": ["input-a"],
        }
    )
    assert [row.entry_id for row in runtime._unhandled_inputs()] == ["input-b"]
    messages.branch("session-one", "input-a")
    messages.accept_input("session-one", "新分支", input_id="input-c")
    assert [row.entry_id for row in runtime._unhandled_inputs()] == ["input-c"]
    restarted = SessionRuntime(
        "session-one",
        messages=SessionMessageStore(tmp_path),
        facts=RunFactStore(tmp_path),
        run=unexpected_run,
    )
    assert [row.entry_id for row in restarted._unhandled_inputs()] == ["input-c"]
    facts.append(
        {
            "session_id": "session-one",
            "run_id": "run-second",
            "event": "input:handled",
            "input_ids": ["input-c"],
        }
    )
    assert restarted._unhandled_inputs() == ()
    runtime.close()
    restarted.close()


def test_corrupt_fact_is_not_ignored(tmp_path: Path) -> None:
    """损坏事实不能让输入看似未处理而重放；传参：隔离根；返回：无。"""
    facts = RunFactStore(tmp_path)
    path = facts.append(
        {
            "session_id": "session-one",
            "run_id": "run-first",
            "event": "input:handled",
            "input_ids": ["input-a"],
        }
    )
    assert facts.handled_input_ids("session-one") == frozenset({"input-a"})
    original = path.read_bytes()
    corrupted = original.replace(b"input-a", b"input-b")
    assert corrupted != original
    path.write_bytes(corrupted)
    with pytest.raises(SourceCorruptionError, match="committed source corrupt"):
        facts.handled_input_ids("session-one")
    assert path.read_bytes() == corrupted
