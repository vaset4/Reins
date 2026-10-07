"""模型选择在凭据校验和落盘失败时不半应用。

作者：xxx
"""

from types import SimpleNamespace

import pytest
import yaml

from llm.profiles import load_model_profiles
from tests.frontends.tui.test_settings import configured_bridge


def test_missing_credential_keeps_file_and_current_client(tmp_path, monkeypatch):
    """缺少凭据时保存前失败，配置与输入客户端均不变；参数：目录和替换器；返回：无。"""
    bridge, host, events = configured_bridge(tmp_path, monkeypatch)
    path = load_model_profiles().path
    original = path.read_bytes()
    previous = host.config.llm_client
    monkeypatch.setattr(
        "app.cli.SecretsVault", lambda: SimpleNamespace(get=lambda _: None)
    )
    with pytest.raises(ValueError):
        bridge.submit("/model profile use new reasoning_effort=default")
    assert path.read_bytes() == original and bridge.llm_client is previous
    assert host.config.llm_client is previous
    assert not any(kind == "input-model" for kind, _ in events)


def test_write_failure_keeps_prepared_model_unapplied(tmp_path, monkeypatch):
    """候选已构建但原子替换失败时不发布新客户端；参数：目录和替换器；返回：无。"""
    bridge, host, events = configured_bridge(tmp_path, monkeypatch)
    path = load_model_profiles().path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["profiles"]["new"]["model"] = "glm-5.2"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    original = path.read_bytes()
    previous = host.config.llm_client

    def fail_replace(source, target):
        """模拟目标配置文件被占用；参数：临时文件与目标；返回：无。"""
        raise PermissionError("model config locked")

    monkeypatch.setattr("llm.profiles.os.replace", fail_replace)
    bridge.submit("/model profile use new reasoning_effort=max")
    assert path.read_bytes() == original and bridge.llm_client is previous
    assert host.config.llm_client is previous
    assert not any(kind == "input-model" for kind, _ in events)
    assert any("model config locked" in str(payload) for _, payload in events)
