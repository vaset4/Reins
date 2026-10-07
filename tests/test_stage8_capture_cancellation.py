"""【文件恢复】【捕获取消】慢扫描和大文件读取响应真实停止请求。

作者：xxx
时间：2026-09-30 23:00:00
"""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from typing import Any, BinaryIO, cast
import subprocess
import sys

import pytest

from runtime.cancellation import CancellationToken, ExecutionCancelled
from runtime.file_content import CONTENT_CHUNK_BYTES
from tests.test_stage8_foundation import snapshot_environment


def test_cancel_before_inventory_does_not_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """扫描前取消不枚举目录；参数：隔离根与边界注入；返回：无。"""
    import runtime.file_snapshots as snapshots

    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    cancellation = CancellationToken()
    cancellation.cancel()

    def reject_scan(*args: Any) -> Any:
        """已取消时任何目录枚举均为错误；参数：扫描参数；返回：无。"""
        raise AssertionError("cancelled capture enumerated files")

    monkeypatch.setattr(snapshots.os, "scandir", reject_scan)
    with pytest.raises(ExecutionCancelled):
        store.capture_workspace(root, lease, workspace_id, cancellation=cancellation)


def test_cancel_during_inventory_stops_before_file_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """枚举期间取消不会开始读取原件；参数：隔离根与真实枚举边界；返回：无。"""
    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    target = root / "first.txt"
    target.write_bytes(b"before")
    cancellation = CancellationToken()
    original_resolve = Path.resolve

    def cancel_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        """首次处理用户文件时发送停止；参数：真实路径；返回：规范路径。"""
        resolved = original_resolve(path, *args, **kwargs)
        if path == target:
            cancellation.cancel()
        return resolved

    monkeypatch.setattr(Path, "resolve", cancel_resolve)
    with pytest.raises(ExecutionCancelled):
        store.capture_workspace(root, lease, workspace_id, cancellation=cancellation)
    assert not list(store.data_root.rglob("*.bin"))


def test_cancel_while_inspecting_large_file_stops_without_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """大文件第一块核验后取消不继续冻结；参数：隔离根与内容检查边界；返回：无。"""
    import runtime.file_snapshots as snapshots

    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    target = root / "large.bin"
    target.write_bytes(b"a" * CONTENT_CHUNK_BYTES * 4)
    cancellation = CancellationToken()
    checked = 0

    def cancel_inspection(raw: bytes) -> bool:
        """第一块即取消，禁止再次读取核验；参数：读取块；返回：不含私钥。"""
        nonlocal checked
        checked += 1
        cancellation.cancel()
        return False

    monkeypatch.setattr(snapshots, "is_private_key", cancel_inspection)
    with pytest.raises(ExecutionCancelled):
        store.capture_file(target, lease, workspace_id, cancellation=cancellation)
    assert checked == 1
    assert not list(store.data_root.rglob("*.bin"))


def test_cancel_while_freezing_stream_removes_incomplete_object(tmp_path: Path) -> None:
    """原件写入期间取消清理未完成临时文件；参数：隔离根；返回：无。"""
    root, store, _, workspace_id = snapshot_environment(tmp_path)
    target = root / "large.bin"
    target.write_bytes(b"a" * CONTENT_CHUNK_BYTES * 4)
    cancellation = CancellationToken()

    class CancellingStream:
        """保留实际文件句柄并在首块后发出取消。"""

        def __init__(self, source: BinaryIO) -> None:
            """绑定真实来源；参数：源流；返回：无。"""
            self.source = source
            self.reads = 0

        def __getattr__(self, name: str) -> Any:
            """透传文件句柄能力；参数：属性名；返回：原属性。"""
            return getattr(self.source, name)

        def read(self, size: int = -1) -> bytes:
            """首块后发出停止；参数：块大小；返回：真实内容。"""
            self.reads += 1
            cancellation.cancel()
            return self.source.read(size)

    with target.open("rb") as source:
        wrapper = CancellingStream(source)
        with pytest.raises(ExecutionCancelled):
            store.database.prepare_stream(
                cast(BinaryIO, wrapper),
                workspace_id=workspace_id,
                cancellation=cancellation,
            )
        assert wrapper.reads == 1
    objects = store.database.workspace_directory(workspace_id) / "objects"
    assert not [path for path in objects.rglob("*") if path.is_file()]


def test_real_inventory_cancellation_prevents_process_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实目录捕获期间取消及时退出且不启动后端命令；参数：隔离根和开始信号；返回：无。"""
    import runtime.file_snapshots as snapshots
    from runtime.file_capture import FileCapture

    root, store, lease, workspace_id = snapshot_environment(tmp_path)
    for index in range(200):
        (root / f"source-{index}.txt").write_bytes(b"unchanged source")
    marker = root / "command-started.txt"
    cancellation = CancellationToken()
    entered = Event()
    failures: list[BaseException] = []
    original_scan = snapshots.os.scandir

    def tracked_scan(path: Any) -> Any:
        """只观察真正扫描开始，不替代扫描；参数：目录；返回：真实迭代器。"""
        entered.set()
        return original_scan(path)

    def run_capture() -> None:
        """在工作线程保护真实命令；参数：无；返回：无，记录实际取消异常。"""
        capture = FileCapture(
            store,
            lease,
            {
                "workspace_id": workspace_id,
                "workspace_root": str(root),
                "session_id": "foundation-session",
                "run_id": "run",
                "input_id": "input",
                "operation_id": "cancel-check",
            },
            cancellation=cancellation,
        )
        try:
            capture.observe(
                lambda: subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('started')",
                        str(marker),
                    ],
                    check=True,
                )
            )
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(snapshots.os, "scandir", tracked_scan)
    worker = Thread(target=run_capture)
    worker.start()
    assert entered.wait(2), "real inventory did not start"
    cancellation.cancel()
    worker.join(2)
    assert not worker.is_alive(), "capture did not promptly observe cancellation"
    assert len(failures) == 1 and isinstance(failures[0], ExecutionCancelled)
    assert not marker.exists()
