"""【知识维护】【临时凭据】真实后台主请求与未接纳维护状态保持分离。

作者：xxx
时间：2026-10-01 19:00:00
"""

from pathlib import Path

from app.background.client import ensure_running
from runtime.knowledge_maintenance import KnowledgeMaintenance
from runtime.persistence import RuntimeStore
from tests import test_background_process

process_setup = test_background_process.process_setup


def test_ephemeral_main_request_finishes_and_exposes_unaccepted_maintenance(
    process_setup,
):
    """临时key真实主调用完成，自动维护拒绝接纳可查看且不消费来源；参数：隔离进程；返回：无。"""
    project, model = process_setup
    model.first_release.set()
    model.final_release.set()
    root = Path.home() / ".reins" / "data"
    client = ensure_running(project_root=project, data_root=root)
    session = client.call("attach")["session_id"]
    client.call(
        "submit",
        session_id=session,
        input_id="ephemeral-input",
        text="写出本地结果并记住约定",
        model_config={
            "base_url": f"http://127.0.0.1:{model.server_port}/v1",
            "model": "ephemeral-model",
            "api_mode": "chat_completions",
        },
        api_key="isolated-ephemeral-test-key",
    )
    final = test_background_process.wait_until(
        lambda: (
            view
            if (view := client.call("poll", session_id=session))["status"]
            in {"done", "failed"}
            else None
        )
    )
    assert final["status"] == "done", final
    assert (project / "background-result.txt").read_text(encoding="utf-8") == "只写一次"
    assert len(model.packets) == 2
    status = KnowledgeMaintenance(root).status(session_id=session)
    assert status["works"] == []
    admission = status["admissions"][0]
    assert admission["state"] == "not_accepted" and admission["source_count"] > 0
    assert "isolated-ephemeral-test-key" not in str(admission)
    owner = {"session_id": session, "data_space_id": RuntimeStore(root).data_space_id}
    overview = client.call(
        "context_management", payload={**owner, "action": "overview"}
    )
    assert overview["knowledge"]["admission"]["state"] == "not_accepted"
