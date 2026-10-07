"""正式入口与真实后台的请求查看、导出及跨工作区回归。

作者：xxx
时间：2026-09-30 21:00:00
"""

import json
import subprocess
import sys
from pathlib import Path

from app.background.client import ensure_running
from runtime.run_evidence import redact_value
from tests.test_background_process import REPOSITORY, process_setup, wait_until

__all__ = ["process_setup"]
DRIVER_TIMEOUT_SECONDS = 35


def test_formal_tui_inspects_exports_and_preserves_workspace_drafts(
    process_setup, tmp_path
):
    """真实SDK发送内容与TUI全文和导出一致，A/B切换仍保留原目录；参数：隔离进程和目录；返回：无。"""
    project, model = process_setup
    root = Path.home() / ".reins/data"
    client = ensure_running(project_root=project, data_root=root)
    session_a = client.call("attach", project_root=str(project))["session_id"]
    workspace_b = tmp_path / "另一个工作区"
    workspace_b.mkdir()
    session_b = client.call("create_session", project_root=str(workspace_b))[
        "session_id"
    ]
    model.first_release.set()
    model.final_release.set()
    client.call(
        "submit",
        session_id=session_a,
        input_id="stage6-real-input",
        text="写出隔离文件并核对请求",
        model_config={
            "provider": "openai_compatible",
            "model": "background-test-model",
            "base_url": f"http://127.0.0.1:{model.server_port}/v1",
            "api_mode": "chat_completions",
        },
        api_key="isolated-test-key",
    )
    wait_until(lambda: client.call("poll", session_id=session_a)["status"] == "done")
    config = {
        "session_a": session_a,
        "session_b": session_b,
        "workspace_b": str(workspace_b),
        "export_path": str(tmp_path / "导出的请求"),
    }
    script = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')"
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-c",
            script,
            str(REPOSITORY / "tests/scripts/tui_process_driver.py"),
            "inspect",
            json.dumps(config),
        ],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=DRIVER_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout.splitlines()[-1])
    expected = redact_value(model.packets[0])
    assert receipt["request"] == expected, differing_fields(
        receipt["request"], expected
    )
    exported = json.loads(
        (Path(config["export_path"]) / "requests.json").read_text(encoding="utf-8")
    )
    evidence = exported["requests"][0]["attempts"][0]["request"]
    assert evidence["request"] == expected and evidence["retention"] == "protected"
    assert len(exported["requests"]) == 1
    assert receipt["workspace_a"] == str(project) and receipt["workspace_b"] == str(
        workspace_b
    )
    assert (project / "background-result.txt").read_text(encoding="utf-8") == "只写一次"
    assert not (workspace_b / "background-result.txt").exists()
    assert len(model.packets) == 2


def differing_fields(actual, expected, path="request"):
    """定位合成发送内容的差异路径，不倾倒完整正文；参数：实际、期望与路径；返回：差异字段。"""
    if type(actual) is not type(expected):
        return [f"{path}: {type(actual).__name__} != {type(expected).__name__}"]
    if isinstance(actual, dict):
        result = []
        for key in actual.keys() | expected.keys():
            result.extend(
                differing_fields(actual.get(key), expected.get(key), f"{path}.{key}")
            )
        return result
    if isinstance(actual, list):
        result = [f"{path}.length"] if len(actual) != len(expected) else []
        for index, (left, right) in enumerate(zip(actual, expected)):
            result.extend(differing_fields(left, right, f"{path}[{index}]"))
        return result
    return (
        []
        if actual == expected
        else [f"{path}: {str(actual)[:100]!r} != {str(expected)[:100]!r}"]
    )
