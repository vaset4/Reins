"""验证目录诊断保留真实遍历行为；作者：xxx。"""

from pathlib import Path
import os
from unittest.mock import patch

import pytest

from scripts.benchmark_session_directories import DirectoryMeter


def test_directory_meter_counts_consumed_entries_and_preserves_direntry(
    tmp_path: Path,
) -> None:
    """只记录消费的真实项并保持文件元信息；传参：临时目录；返回：无。"""
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two", encoding="utf-8")
    meter = DirectoryMeter(tmp_path)
    with patch("os.scandir", meter.scan):
        with os.scandir(tmp_path) as entries:
            assert sorted(item.name for item in entries) == ["one.txt", "two.txt"]
        with os.scandir(tmp_path) as entries:
            assert next(entries).is_file()
    assert meter.counts == {".": {"calls": 2, "entries": 3}}


def test_directory_meter_preserves_missing_directory_error(tmp_path: Path) -> None:
    """诊断不得把不存在的目录伪造成空目录；传参：临时目录；返回：无。"""
    meter = DirectoryMeter(tmp_path)
    with patch("os.scandir", meter.scan), pytest.raises(FileNotFoundError):
        os.scandir(tmp_path / "missing")
