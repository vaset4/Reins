"""验证评测源码范围和连接身份记录，避免遗漏运行代码或保存凭证。

作者：xxx
时间：2026-09-28 11:15:37
"""

from scripts.testing.llm import from_test_turns
import json
from dataclasses import replace

from scripts.long_context_evidence import (
    evaluation_source_files,
    model_manifest,
    snapshot_sources,
)


def test_snapshot_follows_packages_and_keeps_existing_evidence_on_resume(tmp_path):
    """打包源码全部冻结而数据不进入，接续不覆盖；传参：临时项目；返回：无。"""
    (tmp_path / "pyproject.toml").write_text(
        '[tool.setuptools]\npy-modules = ["path_security"]\n'
        '[tool.setuptools.packages.find]\ninclude = ["app*"]\n',
        encoding="utf-8",
    )
    (tmp_path / "path_security.py").write_text("boundary", encoding="utf-8")
    package = tmp_path / "app"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = package / "new_runtime.py"
    source.write_text("before", encoding="utf-8")
    (package / "user-data.json").write_text('"private data"', encoding="utf-8")
    research = tmp_path / ".trellis"
    research.mkdir()
    (research / "old_source.py").write_text("historical", encoding="utf-8")
    names = evaluation_source_files(tmp_path)
    assert set(names) == {
        "pyproject.toml",
        "path_security.py",
        "app/__init__.py",
        "app/new_runtime.py",
    }
    destination = tmp_path / "frozen"
    first = snapshot_sources(tmp_path, names, destination, resume=False)
    source.write_text("after", encoding="utf-8")
    second = snapshot_sources(tmp_path, names, destination, resume=True)
    assert first["app/new_runtime.py"] != second["app/new_runtime.py"]
    assert (destination / "app/new_runtime.py").read_text(encoding="utf-8") == "before"


def test_model_manifest_tracks_endpoint_without_recording_secrets():
    """连接改变可识别但原始地址与密钥不落盘；传参：无；返回：无。"""
    target = replace(
        from_test_turns([]).resolved_target,
        profile_name="happy:deepseek-v4.1-flash",
        base_url="https://example.invalid/v1?token=private-marker",
        api_key="secret-value-marker",
        credential_name="happy_api_key",
    )
    first = model_manifest(target)
    second = model_manifest(replace(target, base_url="https://another.invalid/v1"))
    assert first["profile"] == target.profile_name
    assert first["connection_sha256"] != second["connection_sha256"]
    serialized = json.dumps(first)
    assert (
        "private-marker" not in serialized and "secret-value-marker" not in serialized
    )
    assert "base_url" not in first and "api_key" not in first
