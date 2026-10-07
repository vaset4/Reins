"""专项计量后台恢复的目录枚举，非物理磁盘IO；作者：xxx；时间：2026-09-28 19:00:00。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from unittest.mock import patch


class DirectoryMeter:
    """在独立诊断进程内统计scandir调用和实际枚举项。"""

    def __init__(self, root: Path) -> None:
        """绑定真实读取范围；传参：可写夹具根；返回：无。"""
        self.root = root.absolute()
        self.counts: dict[str, dict[str, int]] = {}
        self.original = os.scandir

    def scan(self, path: Any = ".") -> Any:
        """保留实际scandir行为；传参：目录；返回：原始或计数迭代器。"""
        iterator = self.original(path)
        if isinstance(path, int) or not Path(path).absolute().is_relative_to(self.root):
            return iterator
        name = Path(path).absolute().relative_to(self.root).as_posix()
        counts = self.counts.setdefault(name, {"calls": 0, "entries": 0})
        counts["calls"] += 1
        return CountedDirectory(iterator, counts)


class CountedDirectory:
    """不改变DirEntry与异常，只记录消费过的目录项。"""

    def __init__(self, iterator: Any, counts: dict[str, int]) -> None:
        """绑定本次目录枚举；传参：真实迭代器和计数器；返回：无。"""
        self.iterator, self.counts = iterator, counts

    def __iter__(self) -> CountedDirectory:
        """提供原迭代语义；传参：无；返回：本迭代器。"""
        return self

    def __next__(self) -> os.DirEntry[str]:
        """记录真实返回项；传参：无；返回：原DirEntry，结束和错误原样抛出。"""
        item = next(self.iterator)
        self.counts["entries"] += 1
        return cast(os.DirEntry[str], item)

    def __enter__(self) -> CountedDirectory:
        """进入真实句柄范围；传参：无；返回：计数迭代器。"""
        self.iterator.__enter__()
        return self

    def __exit__(self, *error: Any) -> None:
        """关闭实际枚举句柄；传参：异常信息；返回：无。"""
        self.iterator.__exit__(*error)

    def close(self) -> None:
        """保持显式关闭能力；传参：无；返回：无。"""
        self.iterator.close()


def forbid_model(_options: dict[str, object]) -> Any:
    """终态回执恢复不得重新请求模型；传参：模型选项；返回：无，意外执行立即失败。"""
    raise AssertionError("completed handoff attempted model execution")


def diagnose(source: Path, fixture: Path, output: Path) -> None:
    """单次诊断同一真实恢复路径；传参：冻结源码/固定输入/新证据文件；返回：无。"""
    from scripts.benchmark_session_extended import fingerprints, source_identity

    source = source.resolve()
    sys.path.insert(0, str(source))
    from app.background.service import BackgroundService
    from app.background.sessions import SessionServices
    from tools.tool_registry import ToolRegistry

    source_identity(source)

    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="directory-worker-", dir=fixture.parent))
    root = temporary / "data"
    shutil.copytree(fixture, root)
    hashes = fingerprints(fixture)
    if fingerprints(root) != hashes:
        raise ValueError("diagnostic copy differs from frozen fixture")
    service = BackgroundService(SessionServices(root, root, forbid_model, ToolRegistry))
    pending = tuple(
        item for item in service._sessions.values() if item.record.status == "running"
    )
    meter = DirectoryMeter(root)
    try:
        started = perf_counter()
        with patch("os.scandir", meter.scan):
            for session in service._sessions.values():
                session.recover()
        elapsed = perf_counter() - started
        assert all(
            item.record.status == "done" and not item.runtime.active for item in pending
        )
        if fingerprints(fixture) != hashes:
            raise ValueError("frozen diagnostic fixture changed")
        result = {
            "scope": "recovery only; constructor/copy/close excluded; directory enumeration is not physical IO",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source": str(source),
            "imports": source_identity(source),
            "inputs": hashes,
            "work_directory": str(temporary),
            "seconds_instrumented": elapsed,
            "sessions": len(service._sessions),
            "completed": sum(
                item.record.status == "done" for item in service._sessions.values()
            ),
            "scandir_calls": sum(item["calls"] for item in meter.counts.values()),
            "enumerated_entries": sum(
                item["entries"] for item in meter.counts.values()
            ),
            "directories": meter.counts,
        }
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "sessions",
                        "completed",
                        "scandir_calls",
                        "enumerated_entries",
                    )
                }
            ),
            flush=True,
        )
    finally:
        service.close()


def main() -> None:
    """运行单次目录诊断；传参：CLI；返回：无。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.source, args.input, args.output)


if __name__ == "__main__":
    main()
