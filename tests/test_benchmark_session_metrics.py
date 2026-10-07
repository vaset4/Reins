"""确保文件计量没有改变生产读取语义。

作者：xxx
时间：2026-09-28 16:00:00
"""

import json
from pathlib import Path

import pytest

from scripts.benchmark_session_metrics import WorkingSetSampler, measure_reads


def test_read_bytes_include_utf8_and_windows_newlines(tmp_path: Path) -> None:
    """字节计数发生在换行转换前；传参：临时目录；返回：无。"""
    payload = '{"内容":"中文"}\r\n'.encode("utf-8")
    path = tmp_path / "facts.jsonl"
    path.write_bytes(payload)
    with measure_reads(tmp_path) as meter:
        with path.open(encoding="utf-8") as handle:
            assert [json.loads(line) for line in handle] == [{"内容": "中文"}]
        first = meter.snapshot()
        assert path.read_bytes() == payload
        second = meter.snapshot()
    assert first.file_opens == 1
    assert first.bytes_read == len(payload)
    assert first.json_values == 1
    assert second.file_opens == 2
    assert second.bytes_read == len(payload) * 2
    assert second.buffer_reads >= second.file_opens


def test_measurement_restores_io_after_parse_failure(tmp_path: Path) -> None:
    """解析失败不吞错且恢复插桩；传参：临时目录；返回：无。"""
    path = tmp_path / "broken.json"
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError), measure_reads(tmp_path) as meter:
        json.loads(path.read_text(encoding="utf-8"))
    before = meter.snapshot()
    assert before.bytes_read == len(b"{broken")
    assert before.json_values == 1
    path.write_text("{}", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == {}
    assert meter.snapshot() == before


def test_working_set_sampler_reports_native_process_bytes() -> None:
    """原生内存查询给出工作集及有序采样；传参：无；返回：无，系统失败不能当零内存。"""
    sampler = WorkingSetSampler()
    sampler.start()
    result = sampler.finish()
    assert isinstance(result["baseline_bytes"], int) and result["baseline_bytes"] > 0
    assert isinstance(result["sampled_peak_bytes"], int)
    assert result["sampled_peak_bytes"] >= result["baseline_bytes"]
    assert isinstance(result["samples"], list) and len(result["samples"]) >= 2
