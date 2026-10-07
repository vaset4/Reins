"""【文件】【发布竞争】验证Windows替换备份与受控锁。

作者：xxx
时间：2026-09-30 15:30:00
"""

from pathlib import Path

import pytest

from tools import file_persistence


def test_replace_preserves_actual_external_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """最后检查后发生编辑仍保留实际字节并报告冲突；参数：根与注入；返回：无。"""
    path, backup = tmp_path / "current.md", tmp_path / "conflict.md"
    path.write_bytes(b"original")
    real_replace = file_persistence._KERNEL.ReplaceFileW

    def external_race(*args: object) -> object:
        """在系统替换前注入无协作编辑器；参数：系统调用参数；返回：真实替换结果。"""
        path.write_bytes(b"external new bytes")
        result = real_replace(*args)
        path.write_bytes(b"third edit")
        return result

    monkeypatch.setattr(file_persistence._KERNEL, "ReplaceFileW", external_race)
    with pytest.raises(
        file_persistence.FileEditConflict, match="actual replaced content retained"
    ) as error:
        file_persistence.publish_file(path, b"original", b"AI edit", backup_path=backup)
    assert error.value.backup_path == backup
    assert backup.read_bytes() == b"external new bytes"
    assert path.read_bytes() == b"third edit"


def test_successful_replace_preserves_expected_backup(tmp_path: Path) -> None:
    """正常替换保留前一修订且结果完整；参数：根；返回：无。"""
    path, backup = tmp_path / "current.md", tmp_path / "revision.md"
    path.write_bytes(b"original")
    file_persistence.publish_file(path, b"original", b"next", backup_path=backup)
    assert path.read_bytes() == b"next"
    assert backup.read_bytes() == b"original"
