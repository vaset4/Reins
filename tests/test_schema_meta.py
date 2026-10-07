"""【存储】【格式检查】空间原件与可重建索引分别处理。

作者：xxx
时间：2026-09-30 15:30:00
"""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from runtime.persistence import FORMAT_VERSION, RuntimeStore
from runtime.schema_meta import UnsupportedSchemaError, ensure_current_schema


def test_new_root_initializes_one_file_space(tmp_path: Path) -> None:
    """首次创建空间身份而索引可重建；参数：根；返回：无。"""
    root = tmp_path / "data"
    assert ensure_current_schema(root).status == "initialized"
    assert ensure_current_schema(root).status == "current"
    assert {path.name for path in root.iterdir()} == {
        "space.json",
        "commits.jsonl",
        "index.sqlite",
    }
    assert (
        json.loads((root / "space.json").read_bytes())["format_version"]
        == FORMAT_VERSION
    )


@pytest.mark.parametrize(
    "relative",
    [
        "reins.db",
        ".schema_meta.yaml",
        "sessions/one/entries.jsonl",
        "assets/existing.txt",
    ],
)
def test_old_data_is_rejected_without_mutation(tmp_path: Path, relative: str) -> None:
    """旧格式不能自动重置；参数：根和旧对象；返回：无。"""
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"original")
    with pytest.raises(UnsupportedSchemaError, match="old or incomplete runtime data"):
        ensure_current_schema(tmp_path)
    assert path.read_bytes() == b"original"
    assert not (tmp_path / "space.json").exists()


@pytest.mark.parametrize("mutation", ["missing", "corrupt", "future"])
def test_invalid_space_original_is_not_reinitialized(
    tmp_path: Path, mutation: str
) -> None:
    """空间身份损坏明确失败；参数：根和故障；返回：无。"""
    ensure_current_schema(tmp_path)
    marker = tmp_path / "space.json"
    if mutation == "missing":
        marker.unlink()
    elif mutation == "corrupt":
        marker.write_text("broken", encoding="utf-8")
    else:
        value = json.loads(marker.read_bytes())
        marker.write_text(
            json.dumps({**value, "format_version": FORMAT_VERSION + 1}),
            encoding="utf-8",
        )
    with pytest.raises(UnsupportedSchemaError):
        ensure_current_schema(tmp_path)


@pytest.mark.parametrize("mutation", ["missing", "corrupt"])
def test_index_loss_is_rebuildable(tmp_path: Path, mutation: str) -> None:
    """丢索引不丢空间身份；参数：根和故障；返回：无。"""
    ensure_current_schema(tmp_path)
    identity = RuntimeStore(tmp_path).data_space_id
    index = tmp_path / "index.sqlite"
    if mutation == "missing":
        index.unlink()
    else:
        index.write_bytes(b"not a database")
    assert ensure_current_schema(tmp_path).status == "current"
    assert RuntimeStore(tmp_path).data_space_id == identity


def test_missing_commit_original_is_not_a_missing_index(tmp_path: Path) -> None:
    """提交日志属于原件，缺失必须失败；参数：根；返回：无。"""
    ensure_current_schema(tmp_path)
    (tmp_path / "commits.jsonl").unlink()
    with pytest.raises(UnsupportedSchemaError, match="commits.jsonl is missing"):
        ensure_current_schema(tmp_path)


def _space_identity(root: Path) -> str:
    """从独立门面读取身份；参数：根；返回：持久编号。"""
    return RuntimeStore(root).data_space_id


def test_concurrent_initialization_has_one_identity(tmp_path: Path) -> None:
    """多个线程并发初始化不会覆盖身份；参数：根；返回：无。"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        identities = tuple(executor.map(_space_identity, [tmp_path] * 8))
    assert len(set(identities)) == 1


def test_separate_processes_initialize_same_space(tmp_path: Path) -> None:
    """多个进程使用同一系统互斥发布身份；参数：根；返回：无。"""
    code = "from runtime.persistence import RuntimeStore; import sys; print(RuntimeStore(sys.argv[1]).data_space_id)"
    processes = [
        subprocess.Popen(
            [sys.executable, "-X", "utf8", "-c", code, str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(3)
    ]
    try:
        outputs = [process.communicate(timeout=15) for process in processes]
        assert all(process.returncode == 0 for process in processes), outputs
        assert len({stdout.strip() for stdout, _stderr in outputs}) == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=15)
