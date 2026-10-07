"""【代码执行】【进程树归属】在用户代码启动前将Windows子进程纳入可核验Job。

作者：xxx
时间：2026-09-30 21:00:00
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Any

CREATE_SUSPENDED = 0x00000004
_JOB_EXTENDED_LIMITS = 9
_JOB_ACCOUNTING = 1
_JOB_KILL_ON_CLOSE = 0x00002000
_THREAD_SNAPSHOT = 0x00000004
_THREAD_RESUME = 0x00000002
_INVALID_DWORD = 0xFFFFFFFF
_JOB_STOP_EXIT_CODE = 1


class _BasicLimits(ctypes.Structure):
    """Windows公开JOBOBJECT_BASIC_LIMIT_INFORMATION布局。"""

    _fields_ = [
        ("process_time", ctypes.c_longlong),
        ("job_time", ctypes.c_longlong),
        ("flags", wintypes.DWORD),
        ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t),
        ("active_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority", wintypes.DWORD),
        ("scheduling", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    """Windows公开IO_COUNTERS布局。"""

    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "read",
            "write",
            "other",
            "read_bytes",
            "write_bytes",
            "other_bytes",
        )
    ]


class _ExtendedLimits(ctypes.Structure):
    """Windows公开JOBOBJECT_EXTENDED_LIMIT_INFORMATION布局。"""

    _fields_ = [
        ("basic", _BasicLimits),
        ("io", _IoCounters),
        ("process_memory", ctypes.c_size_t),
        ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


class _Accounting(ctypes.Structure):
    """Windows公开JOBOBJECT_BASIC_ACCOUNTING_INFORMATION布局。"""

    _fields_ = [
        ("user_time", ctypes.c_longlong),
        ("kernel_time", ctypes.c_longlong),
        ("period_user_time", ctypes.c_longlong),
        ("period_kernel_time", ctypes.c_longlong),
        ("page_faults", wintypes.DWORD),
        ("total", wintypes.DWORD),
        ("active", wintypes.DWORD),
        ("terminated", wintypes.DWORD),
    ]


class _ThreadEntry(ctypes.Structure):
    """Windows公开THREADENTRY32布局。"""

    _fields_ = [
        ("size", wintypes.DWORD),
        ("usage", wintypes.DWORD),
        ("thread_id", wintypes.DWORD),
        ("process_id", wintypes.DWORD),
        ("base_priority", wintypes.LONG),
        ("delta_priority", wintypes.LONG),
        ("flags", wintypes.DWORD),
    ]


class ProcessTree:
    """由真实执行线程持有Job，根进程退出后仍能看到脱离stdio的后代。"""

    def __init__(self) -> None:
        """创建关闭即停止的Job；传参：无；返回：无，创建失败时不启动子进程。"""
        self.kernel = _kernel()
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = _ExtendedLimits()
        limits.basic.flags = _JOB_KILL_ON_CLOSE
        if not self.kernel.SetInformationJobObject(
            self.handle,
            _JOB_EXTENDED_LIMITS,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach(self, process_handle: int) -> None:
        """把尚未运行用户代码的进程纳入Job；传参：挂起进程句柄；返回：无，失败禁止恢复线程。"""
        if not self.kernel.AssignProcessToJobObject(self.handle, process_handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def resume(self, process_id: int) -> None:
        """定位唯一挂起主线程后启动；传参：新进程PID；返回：无，不接受已运行或身份不明确的线程。"""
        thread_ids = _process_threads(self.kernel, process_id)
        if len(thread_ids) != 1:
            raise RuntimeError(
                "suspended process does not have exactly one primary thread"
            )
        thread = self.kernel.OpenThread(_THREAD_RESUME, False, thread_ids[0])
        if not thread:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            suspended = self.kernel.ResumeThread(thread)
            if suspended == _INVALID_DWORD:
                raise ctypes.WinError(ctypes.get_last_error())
            if suspended != 1:
                raise RuntimeError(
                    "primary thread suspension state changed before Job attachment"
                )
        finally:
            self.kernel.CloseHandle(thread)

    def active_count(self) -> int:
        """查询Job内仍可能写入的全部进程；传参：无；返回：内核确认的活动数量。"""
        accounting = _Accounting()
        if not self.kernel.QueryInformationJobObject(
            self.handle,
            _JOB_ACCOUNTING,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            None,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(accounting.active)

    def terminate(self) -> None:
        """请求终止整个受控进程树；传参：无；返回：无，成功请求仍需查询活动数为零。"""
        if not self.kernel.TerminateJobObject(self.handle, _JOB_STOP_EXIT_CODE):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        """释放Job句柄；传参：无；返回：无，宿主崩溃时同样由内核停止遗留进程。"""
        if self.handle and not self.kernel.CloseHandle(self.handle):
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = None


def _process_threads(kernel: Any, process_id: int) -> list[int]:
    """枚举尚未启动进程的主线程身份；传参：内核API和PID；返回：该进程线程编号。"""
    snapshot = kernel.CreateToolhelp32Snapshot(_THREAD_SNAPSHOT, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = _ThreadEntry()
        entry.size = ctypes.sizeof(entry)
        found = kernel.Thread32First(snapshot, ctypes.byref(entry))
        threads = []
        while found:
            if entry.process_id == process_id:
                threads.append(int(entry.thread_id))
            entry.size = ctypes.sizeof(entry)
            found = kernel.Thread32Next(snapshot, ctypes.byref(entry))
        return threads
    finally:
        kernel.CloseHandle(snapshot)


def _kernel() -> Any:
    """绑定公开Windows API并保留64位句柄；传参：无；返回：带正确函数签名的内核入口。"""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        "SetInformationJobObject": (
            [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD],
            wintypes.BOOL,
        ),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "QueryInformationJobObject": (
            [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.c_void_p,
            ],
            wintypes.BOOL,
        ),
        "TerminateJobObject": ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        "CreateToolhelp32Snapshot": ([wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
        "Thread32First": (
            [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)],
            wintypes.BOOL,
        ),
        "Thread32Next": (
            [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)],
            wintypes.BOOL,
        ),
        "OpenThread": (
            [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD],
            wintypes.HANDLE,
        ),
        "ResumeThread": ([wintypes.HANDLE], wintypes.DWORD),
    }
    for name, (arguments, result) in signatures.items():
        getattr(kernel, name).argtypes = arguments
        getattr(kernel, name).restype = result
    return kernel
