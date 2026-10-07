"""启动目录变化后只读工具与授权范围仍按工作区判断。

作者：xxx
"""

from app.startup import resolve_startup_identity
from path_security import Decision, check_read
from runtime.lease import from_trigger
from runtime.types import ReadOnlyInspectionRequest
from tools.readonly_inspection import ReadOnlyInspectionExecutor


def test_current_directory_does_not_broaden_outside_file_access(tmp_path, monkeypatch):
    """独立数据根不扩大读取范围，越界路径不读出正文；参数：临时目录和替换器；返回：无。"""
    workspace = tmp_path / "papers"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-private-content", encoding="utf-8")
    monkeypatch.chdir(workspace)
    identity = resolve_startup_identity(environ={}, data_root=tmp_path / "runtime")
    lease = from_trigger(
        "user",
        capabilities={
            "fs": {
                "project_root": str(identity.project_root),
                "read": [str(identity.project_root)],
            }
        },
    )
    assert check_read(outside, lease) is Decision.CONFIRM
    executor = ReadOnlyInspectionExecutor(identity.project_root, 10, 100, 10)
    for target in ("../outside.txt", str(outside)):
        result = executor.execute(
            ReadOnlyInspectionRequest(action="read_file", target_path=target),
            lease=lease,
        )
        assert result.status == "rejected"
        assert "outside-private-content" not in result.output
