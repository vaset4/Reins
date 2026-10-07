"""【维护】【重建索引】以文件原件及Markdown记忆重建可丢弃查询视图。

作者：xxx
时间：2026-09-30 18:00:00
"""

from __future__ import annotations

import argparse
from pathlib import Path

from memory.store import rebuild_memory_index
from runtime.persistence import SPACE_ID_FILE
from tasks.index_sync import rebuild_full_index


def main() -> int:
    """核对现存原件后重建派生视图；参数：命令行数据根；返回：退出状态，坏源明确抛错。"""
    parser = argparse.ArgumentParser(
        description="Rebuild derived search indexes from committed files and Markdown memory originals."
    )
    parser.add_argument("--data-dir", default=str(Path.home() / ".reins" / "data"))
    args = parser.parse_args()
    root = Path(args.data_dir)
    if not (root / SPACE_ID_FILE).is_file() or not (root / "commits.jsonl").is_file():
        raise FileNotFoundError(
            "runtime source identity or commit journal missing; cannot rebuild an absent space"
        )
    counts = rebuild_full_index(root)
    counts["memory"] = rebuild_memory_index(root)
    print(f"rebuild_index data_dir={args.data_dir} counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
