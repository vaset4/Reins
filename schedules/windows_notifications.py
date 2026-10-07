"""通过 Windows Shell 交付通知，并分别保存提交与实际显示回执。

作者：xxx
时间：2026-09-14 22:00:00
"""

from __future__ import annotations

import importlib
import time
from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid4

from schedules.notifications import (
    DeliveryReceipt,
    NotificationRecord,
    NotificationStore,
)

TITLE_UNITS = 63
BODY_UNITS = 255
ICON_RETENTION_SECONDS = 120
BALLOON_SHOW = 0x402
BALLOON_HIDE = 0x403
BALLOON_TIMEOUT = 0x404
BALLOON_CLICK = 0x405


class WindowsNotifications:
    """由通知线程持有原生消息窗口，Shell 回调是可见证明，提交本身不是。"""

    def __init__(self, data_root: Path) -> None:
        """建立当前线程的原生通知窗口；传参：通知数据根；返回：无，设施不可用直接报错。"""
        importlib.import_module("pywintypes")
        self.gui: Any = importlib.import_module("win32gui")
        self.api: Any = importlib.import_module("win32api")
        self.con: Any = importlib.import_module("win32con")
        self.store = NotificationStore(data_root)
        self._icons: dict[int, tuple[str, float]] = {}
        self._observations: deque[tuple[int, int]] = deque()
        self._next_icon = 0
        self._callback = self.con.WM_USER + 20
        self._class_name = f"ReinsNotifications-{uuid4().hex}"
        window_class = self.gui.WNDCLASS()
        window_class.hInstance = self.api.GetModuleHandle(None)
        window_class.lpszClassName = self._class_name
        window_class.lpfnWndProc = self._window_message
        self.gui.RegisterClass(window_class)
        self._window = self.gui.CreateWindow(
            self._class_name, "Reins", 0, 0, 0, 0, 0, 0, 0, window_class.hInstance, None
        )

    def send(self, record: NotificationRecord) -> DeliveryReceipt:
        """向 Shell 提交完整记录的桌面预览；传参：持久通知；返回：API 的实际提交回执。"""
        self._next_icon += 1
        identity = self._next_icon
        icon = self.gui.LoadIcon(0, self.con.IDI_APPLICATION)
        base = (
            self._window,
            identity,
            self.gui.NIF_ICON | self.gui.NIF_MESSAGE | self.gui.NIF_TIP,
            self._callback,
            icon,
            "Reins",
        )
        try:
            self.gui.Shell_NotifyIcon(self.gui.NIM_ADD, base)
        except self.gui.error as exc:
            return DeliveryReceipt(
                "failed", f"Shell_NotifyIcon(NIM_ADD): {exc}", "windows_shell"
            )
        self._icons[identity] = (record.notification_id, time.monotonic())
        notice = (
            self._window,
            identity,
            self.gui.NIF_INFO,
            self._callback,
            icon,
            "Reins",
            _preview(record.message, BODY_UNITS),
            0,
            _preview(record.title, TITLE_UNITS),
            self.gui.NIIF_INFO,
        )
        try:
            self.gui.Shell_NotifyIcon(self.gui.NIM_MODIFY, notice)
        except self.gui.error as exc:
            self._remove_icon(identity)
            return DeliveryReceipt(
                "failed", f"Shell_NotifyIcon(NIM_MODIFY): {exc}", "windows_shell"
            )
        return DeliveryReceipt(
            "submitted",
            "Windows Shell accepted the preview; visibility awaits NIN_BALLOONSHOW",
            "windows_shell",
        )

    def pump(self) -> None:
        """处理 Shell 可见回调与预览资源释放；传参：无；返回：无，存储错误不吞掉。"""
        self.gui.PumpWaitingMessages()
        while self._observations:
            identity, event = self._observations.popleft()
            saved = self._icons.get(identity)
            if saved is None:
                continue
            if event == BALLOON_SHOW:
                self.store.acknowledge(saved[0])
            elif event in {BALLOON_HIDE, BALLOON_TIMEOUT, BALLOON_CLICK}:
                self._remove_icon(identity)
        for identity, (_notice, created) in tuple(self._icons.items()):
            if time.monotonic() - created >= ICON_RETENTION_SECONDS:
                self._remove_icon(identity)

    def close(self) -> None:
        """在拥有窗口的线程释放托盘和窗口资源；传参：无；返回：无。"""
        for identity in tuple(self._icons):
            self._remove_icon(identity)
        self.gui.DestroyWindow(self._window)
        self.gui.UnregisterClass(self._class_name, self.api.GetModuleHandle(None))

    def _remove_icon(self, identity: int) -> None:
        """移除已结束或超时的预览图标；传参：Shell 图标编号；返回：无，不改变交付状态。"""
        self.gui.Shell_NotifyIcon(self.gui.NIM_DELETE, (self._window, identity))
        self._icons.pop(identity, None)

    def _window_message(self, *parts: int) -> int:
        """将原生回调排队，避免回调内与发送锁互相等待；传参：Windows 消息；返回：处理结果。"""
        window, message, wparam, lparam = parts
        if message == self._callback:
            self._observations.append((wparam, lparam))
            return 0
        return int(self.gui.DefWindowProc(window, message, wparam, lparam))


def _preview(text: str, units: int) -> str:
    """按 Shell 的 UTF-16 字数限制生成预览，完整正文不裁剪；传参：正文和容量；返回：预览。"""
    return text.encode("utf-16-le")[: units * 2].decode("utf-16-le", errors="ignore")
