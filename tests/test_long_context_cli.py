"""验证评测命令显式选模、保存真实边界结果，配置失败不会切换模型。

作者：xxx
时间：2026-09-28 11:15:37
"""

from scripts.testing.llm import _from_scripted, from_test_turns
import json

import pytest

from scripts.testing.llm import _ScriptedOptions, _ScriptedTurn
from llm.messages import ToolCallPart
from scripts import eval_long_context as evaluation
from scripts import eval_long_context_scenarios as scenarios
from scripts.long_context_cases import CASES


@pytest.mark.parametrize(
    "entrypoint", [evaluation, scenarios], ids=["comparison", "scenario"]
)
def test_profile_reaches_factory_and_real_evaluation_artifacts(
    tmp_path, monkeypatch, entrypoint
):
    """指定配置进入工厂且产物走真实请求链；传参：隔离根、替换器、入口；返回：无。"""
    if entrypoint is evaluation:
        client = from_test_turns([json.dumps(dict(CASES[0].expected))])
        selection = ["--strategy", "full", "--case", "conditions"]
    else:
        target = from_test_turns([]).resolved_target
        client = _from_scripted(
            [
                _ScriptedTurn(
                    calls=(
                        ToolCallPart(
                            "read-original",
                            "read_history",
                            {"view": "messages", "query": "UTF-16LE"},
                        ),
                    )
                ),
                _ScriptedTurn(
                    text=json.dumps(scenarios.NAVIGATION_EXPECTED, ensure_ascii=False)
                ),
            ],
            options=_ScriptedOptions(resolved_target=target),
        )
        selection = ["--scenario", "navigation"]
    received = []

    def build(overrides, *, project_root):
        """注入脚本Adapter构造的真实客户端；传参：CLI覆盖与根目录；返回：测试客户端。"""
        received.append(dict(overrides))
        assert project_root == tmp_path
        return client

    source = tmp_path / "source.py"
    source.write_text("evaluation source", encoding="utf-8")
    output = tmp_path / "evidence"
    monkeypatch.setattr(entrypoint, "build_llm_client", build)
    monkeypatch.setattr(entrypoint, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(entrypoint, "SOURCE_FILES", ("source.py",))
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluation",
            "--profile",
            "happy:deepseek-v4.1-flash",
            "--output",
            str(output),
            "--repeats",
            "1",
            *selection,
        ],
    )
    monkeypatch.setenv("REINS_TRACE_LEVEL", "off")
    assert entrypoint.main() == 0
    assert received == [
        {
            "profile_name": "happy:deepseek-v4.1-flash",
            "timeout_seconds": 90,
            "max_output_tokens": 16384,
        }
    ]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == client.resolved_target.model
    assert manifest["api_mode"] == client.resolved_target.api_mode
    assert manifest["connection_sha256"]
    assert (output / "source/source.py").read_bytes() == source.read_bytes()
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in output.glob("*/result.json")
    ]
    assert len(results) == 1
    if entrypoint is evaluation:
        assert results[0]["rounds"][0]["result"]["passed"]
    else:
        assert results[0]["passed"] and results[0]["history_reads"]


@pytest.mark.parametrize(
    "entrypoint", [evaluation, scenarios], ids=["comparison", "scenario"]
)
def test_invalid_profile_fails_before_creating_evidence(
    tmp_path, monkeypatch, entrypoint
):
    """未知配置明确失败且不尝试其他模型；传参：隔离根、替换器、入口；返回：无。"""
    requested = []

    def reject(overrides, *, project_root):
        """复现配置解析拒绝；传参：覆盖与项目目录；返回：不返回。"""
        requested.append(overrides["profile_name"])
        raise ValueError("profile not found: missing:model")

    output = tmp_path / "evidence"
    selection = (
        ["--strategy", "full"]
        if entrypoint is evaluation
        else ["--scenario", "navigation"]
    )
    monkeypatch.setattr(entrypoint, "build_llm_client", reject)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluation",
            "--profile",
            "missing:model",
            "--output",
            str(output),
            *selection,
        ],
    )
    with pytest.raises(ValueError, match="profile not found"):
        entrypoint.main()
    assert requested == ["missing:model"]
    assert not output.exists()
