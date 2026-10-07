"""复现状态轮询与 Windows 原子替换之间的真实竞争。

作者：xxx
时间：2026-09-24 12:00:00
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from threading import Event

import pytest

from schedules.persistence import read_record, write_record


def test_polling_reader_does_not_break_record_publication(tmp_path, monkeypatch):
    """读者持有文件句柄时，发布等待读完而不失败；传参：目录和替换器；返回：无。"""
    path = tmp_path / "occurrence.json"
    write_record(path, {"state": "resume_queued"})
    opened, release = Event(), Event()
    original = Path.read_text

    def held_read(current, *args, **kwargs):
        """固定真实文件仍被打开的竞争窗口；传参：读取参数；返回：原文件正文。"""
        if current != path:
            return original(current, *args, **kwargs)
        with current.open("r", encoding="utf-8") as handle:
            opened.set()
            assert release.wait(10)
            return handle.read()

    monkeypatch.setattr(Path, "read_text", held_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        reader = executor.submit(read_record, path)
        try:
            assert opened.wait(10)
            writer = executor.submit(write_record, path, {"state": "running"})
            with pytest.raises(TimeoutError):
                writer.result(timeout=0.3)
        finally:
            release.set()
        assert reader.result(timeout=10) == {"state": "resume_queued"}
        writer.result(timeout=10)
    assert read_record(path) == {"state": "running"}
