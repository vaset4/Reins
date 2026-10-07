"""离线基准的文件读取计量；插桩结果与无插桩计时分开保存。

作者：xxx
时间：2026-09-28 16:00:00
"""

from __future__ import annotations

import io
import json
import ctypes
from ctypes import wintypes
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any, cast
from unittest.mock import patch

MEMORY_SAMPLE_INTERVAL_SECONDS = 0.01


@dataclass(frozen=True, slots=True)
class FileReadCounts:
    """一次读取范围的计量快照；单位为次数和字节，均不代表物理磁盘访问。"""

    file_opens: int
    buffer_reads: int
    bytes_read: int
    json_values: int


class ReadMeter:
    """仅记录指定合成数据根下的读取，保留真实文件和JSON解析行为。"""

    def __init__(self, root: Path, *, allow_writes: bool = False) -> None:
        """指定计量范围；传参：夹具根；返回：无，不修改生产Store。"""
        self.root = root.resolve()
        self.allow_writes = allow_writes
        self.file_opens = 0
        self.buffer_reads = 0
        self.bytes_read = 0
        self.json_values = 0
        self._open = io.open
        self._loads = json.loads

    def snapshot(self) -> FileReadCounts:
        """获取不可变读数；传参：无；返回：文件打开/缓冲读取/字节/JSON解析次数。"""
        return FileReadCounts(
            self.file_opens, self.buffer_reads, self.bytes_read, self.json_values
        )

    def open(
        self,
        file: Any,
        mode: str = "r",
        buffering: int = -1,
        *text_options: Any,
        **options: Any,
    ) -> Any:
        """在真实原始文件读取处计数；传参：io.open参数；返回：保持编码与换行语义的句柄。"""
        if isinstance(file, int) or not Path(file).absolute().is_relative_to(self.root):
            return self._open(file, mode, buffering, *text_options, **options)
        if mode not in {"r", "rb"}:
            if not self.allow_writes:
                raise ValueError(f"read measurement must not write fixture: {mode}")
            if "+" not in mode:
                return self._open(file, mode, buffering, *text_options, **options)
        if len(text_options) > 3:
            raise ValueError("benchmark reader only measures ordinary path-based IO")
        text_config = {
            **dict(zip(("encoding", "errors", "newline"), text_options)),
            **options,
        }
        raw = cast(
            io.FileIO,
            self._open(file, mode if "b" in mode else mode + "b", buffering=0),
        )
        self.file_opens += 1
        counted = _CountedFile(raw, self)
        if buffering == 0:
            if "b" not in mode:
                counted.close()
                raise ValueError("unbuffered text IO is not supported")
            return counted
        buffer_class = io.BufferedRandom if raw.writable() else io.BufferedReader
        buffered = buffer_class(
            counted, buffer_size=buffering if buffering > 1 else io.DEFAULT_BUFFER_SIZE
        )
        if "b" in mode:
            return buffered
        return io.TextIOWrapper(
            buffered,
            encoding=text_config.get("encoding"),
            errors=text_config.get("errors"),
            newline=text_config.get("newline"),
        )

    def loads(self, value: Any, **options: Any) -> Any:
        """计数真实JSON解析；传参：原始JSON及解码参数；返回：原解析结果，错误直抛。"""
        self.json_values += 1
        return self._loads(value, **options)


class _CountedFile(io.RawIOBase):
    """在文本解码和换行转换前计数实际读取字节。"""

    def __init__(self, source: io.FileIO, meter: ReadMeter) -> None:
        """绑定真实文件；传参：独占文件与计量器；返回：无。"""
        super().__init__()
        self._source = source
        self._meter = meter

    def readable(self) -> bool:
        """声明只读能力；传参：无；返回：真。"""
        return True

    def writable(self) -> bool:
        """保留恢复锁文件的读写能力；传参：无；返回：真实句柄是否可写。"""
        return self._source.writable()

    def write(self, data: Any) -> int:
        """透传真实写入且不冒充读取量；传参：字节；返回：实际写入数。"""
        count = self._source.write(data)
        if count is None:
            raise OSError("blocking benchmark file returned no write result")
        return int(count)

    def fileno(self) -> int:
        """保持操作系统文件锁使用的真实描述符；传参：无；返回：文件描述符。"""
        return self._source.fileno()

    def seekable(self) -> bool:
        """保持文件定位能力；传参：无；返回：原句柄能力。"""
        return self._source.seekable()

    def tell(self) -> int:
        """读取字节位置；传参：无；返回：原文件偏移。"""
        return self._source.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        """透传定位；传参：偏移和起点；返回：新位置。"""
        return self._source.seek(offset, whence)

    def readinto(self, buffer: Any) -> int:
        """读取真实字节并计数；传参：读取缓冲区；返回：读取量，EOF为零。"""
        count = self._source.readinto(buffer)
        if count is None:
            raise OSError("blocking benchmark file returned no read result")
        self._meter.buffer_reads += 1
        self._meter.bytes_read += count
        return int(count)

    def close(self) -> None:
        """关闭本计量句柄拥有的文件；传参：无；返回：无。"""
        try:
            self._source.close()
        finally:
            super().close()


@contextmanager
def measure_reads(root: Path, *, allow_writes: bool = False) -> Iterator[ReadMeter]:
    """对单线程离线读取插桩；传参：夹具根；返回：计量器，退出恢复原IO和解析器。"""
    meter = ReadMeter(root, allow_writes=allow_writes)
    # 【阶段四基准】【读取计量】1. 只在独立测量进程启用；JSON次数包含该范围全部解码，不冒充Store次数
    with patch.object(io, "open", meter.open), patch.object(json, "loads", meter.loads):
        yield meter


class _ProcessMemoryCounters(ctypes.Structure):
    """Windows PROCESS_MEMORY_COUNTERS布局，仅用于当前进程工作集采样。"""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


class WorkingSetSampler:
    """在独立内存测量进程中周期采样，错误在调用线程重新抛出。"""

    def __init__(
        self, interval_seconds: float = MEMORY_SAMPLE_INTERVAL_SECONDS
    ) -> None:
        """建立Windows原生查询；传参：采样间隔；返回：无。"""
        if interval_seconds <= 0:
            raise ValueError("sampling interval must be positive")
        self.interval_seconds = interval_seconds
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self._kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self._query = self._psapi.GetProcessMemoryInfo
        self._query.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        self._query.restype = wintypes.BOOL
        self._process = self._kernel.GetCurrentProcess()
        self._stop = Event()
        self._thread = Thread(
            target=self._sample_loop, name="benchmark-working-set", daemon=True
        )
        self._error: BaseException | None = None
        self.samples: list[tuple[float, int]] = []

    def _sample(self) -> None:
        """读取当前进程工作集；传参：无；返回：无，系统失败直接报错。"""
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not self._query(self._process, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        self.samples.append((perf_counter(), int(counters.WorkingSetSize)))

    def _sample_loop(self) -> None:
        """周期采样直到主体结束；传参：无；返回：无，保存异常供关闭时暴露。"""
        try:
            while not self._stop.wait(self.interval_seconds):
                self._sample()
        except BaseException as exc:
            self._error = exc

    def start(self) -> None:
        """先采基线再启动采样线程；传参：无；返回：无。"""
        self._sample()
        self._thread.start()

    def finish(self) -> dict[str, object]:
        """关闭采样并返回原始证据；传参：无；返回：基线/采样峰值/逐次工作集。"""
        self._stop.set()
        self._thread.join()
        if self._error is not None:
            raise self._error
        self._sample()
        started = self.samples[0][0]
        return {
            "baseline_bytes": self.samples[0][1],
            "sampled_peak_bytes": max(size for _, size in self.samples),
            "interval_seconds": self.interval_seconds,
            "samples": [
                {"elapsed_seconds": at - started, "working_set_bytes": size}
                for at, size in self.samples
            ],
            "scope": "current worker only; imports precede baseline; sampling thread included",
            "method": "Windows GetProcessMemoryInfo.WorkingSetSize; sampled peak, not instantaneous peak",
        }
