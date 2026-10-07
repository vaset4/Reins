"""校验离线基准的数据可比性与失败记账。

作者：xxx
时间：2026-09-28 18:30:00
"""

from pathlib import Path
import json
import pytest
from pytest import MonkeyPatch

from runtime.run_facts import RunFactStore
from runtime.session_message_store import SessionMessageStore
from scripts.benchmark_session_runtime import (
    SESSION_ID,
    generate_fixture,
    measure_worker,
)
from scripts.benchmark_session_runtime import collect_read_measurements
from scripts import benchmark_session_runtime as benchmark


def test_unmatched_measurement_selection_fails_before_creating_output(
    tmp_path: Path,
) -> None:
    """错误过滤条件不能产生空成功证据；传参：隔离目录；返回：无。"""
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"inputs": {"messages-100": {}}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="matches no frozen input"):
        collect_read_measurements(
            source, tmp_path / "output", metric="reads", selection=("runs", 100)
        )
    assert not (tmp_path / "output").exists()


def test_timing_cli_reuses_frozen_input_without_regeneration(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """计时复测须读取原始字节而非另造相同规模数据；传参：目录及替换器；返回：无。"""
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    fixture = source / "messages-100"
    hashes = generate_fixture(fixture, "messages", 100)
    (source / "manifest.json").write_text(
        json.dumps({"inputs": {"messages-100": {"sha256": hashes}}}), encoding="utf-8"
    )
    seen: list[tuple[Path, str]] = []

    def collect(
        root: Path, dimension: str, size: int, *, metric: str
    ) -> list[dict[str, object]]:
        """捕获实际复测输入并执行一次真实读取；传参：原输入及指标；返回：观测。"""
        seen.append((root, metric))
        return measure_worker(root, dimension, size)

    monkeypatch.setattr(benchmark, "collect_samples", collect)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--metric",
            "timing",
            "--input",
            str(source),
            "--output",
            str(output),
        ],
    )
    assert benchmark.main() == 0
    assert seen == [(fixture, "timing")]
    assert not (output / "messages-100").exists()
    assert (
        json.loads((output / "manifest.json").read_text(encoding="utf-8"))["metric"]
        == "timing"
    )


def test_fixtures_are_identical_across_directories(tmp_path: Path) -> None:
    """同样规模生成相同字节；传参：隔离目录；返回：无，时间和绝对路径不污染输入。"""
    first = generate_fixture(tmp_path / "first", "messages", 100)
    second = generate_fixture(tmp_path / "second", "messages", 100)
    assert first == second
    assert all(
        row["error"] is None
        for row in measure_worker(tmp_path / "first", "messages", 100)
    )


def test_corrupt_operation_is_retained_as_failed_sample(tmp_path: Path) -> None:
    """损坏记录保留两次失败而非快速成功；传参：隔离目录；返回：无。"""
    root = tmp_path / "data"
    generate_fixture(root, "operations", 100)
    path = SessionMessageStore(root).database.source_path("tool_operation", "op-00000")
    path.write_text("{broken", encoding="utf-8")
    rows = measure_worker(root, "operations", 100)
    assert len(rows) == 2
    assert all("SourceCorruptionError" in str(row["error"]) for row in rows)


def test_new_input_and_handled_fact_change_read_result(tmp_path: Path) -> None:
    """跨次测量必须读取新输入与处理事实；传参：隔离目录；返回：无。"""
    root = tmp_path / "data"
    generate_fixture(root, "runs", 10)
    assert all(row["error"] is None for row in measure_worker(root, "runs", 10))
    messages = SessionMessageStore(root)
    messages.accept_input(SESSION_ID, "新增要求", input_id="input-new")
    assert all(
        row["result_count"] == 1 and row["error"]
        for row in measure_worker(root, "runs", 10)
    )
    RunFactStore(root).append(
        {
            "session_id": SESSION_ID,
            "run_id": "run-00000",
            "event": "input:handled",
            "input_ids": ["input-new"],
        }
    )
    assert all(row["error"] is None for row in measure_worker(root, "runs", 10))
