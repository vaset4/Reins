from __future__ import annotations

from pathlib import Path

from app.run_task import run_task
from llm.types import LLMPlan
from runtime.run_facts import RunFactStore
from runtime.session_messages import materialize_messages
from runtime.types import RunToolsRequest, RunToolsResult


class StubLoopLLM:
    def __init__(self) -> None:
        self.seen_tool_results: list[RunToolsResult] = []

    def plan(self, task: str, context: object | None = None) -> LLMPlan:
        del task, context
        return LLMPlan(
            run_tools_request=RunToolsRequest(
                action="list",
                tool_name="list",
                arguments={"path": "tools"},
            )
        )

    def continue_from_run_tools(
        self,
        task: str,
        run_tools_result: RunToolsResult,
        context: object | None = None,
    ) -> LLMPlan:
        del task, context
        self.seen_tool_results.append(run_tools_result)
        assert run_tools_result.status == "ok"
        assert "file alpha.py" in run_tools_result.output
        assert "dir nested" in run_tools_result.output
        return LLMPlan(final_output="tools 目录包含 alpha.py 文件和 nested 子目录。")


def test_chat_real_loop_calls_tool_registry_then_final(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    tools_dir = project_root / "tools"
    (tools_dir / "nested").mkdir(parents=True)
    (tools_dir / "alpha.py").write_text("print('alpha')\n", encoding="utf-8")
    llm = StubLoopLLM()

    response = run_task(
        task="帮我总结当前项目 tools 目录的结构",
        project_root=project_root,
        llm_client=llm,
        data_root=project_root / ".reins" / "data",
        session_id="session-test-real-agent-loop",
    )

    assert response.status == "done"
    assert response.output == "tools 目录包含 alpha.py 文件和 nested 子目录。"
    assert llm.seen_tool_results

    data_root = project_root / ".reins" / "data"
    from tasks.store import TaskStore

    assert TaskStore(data_root).read_summary(response.task_id) == response.output

    # 一轮工具调用的完整对话由唯一消息 owner 按 session 记录：用户输入、
    # 带 tool_call 的 assistant、tool_result、最终 assistant 回答
    messages = materialize_messages(data_root, "session-test-real-agent-loop")
    assert [message.kind for message in messages] == [
        "user",
        "assistant",
        "tool_result",
        "assistant",
    ]

    facts = RunFactStore(project_root / ".reins" / "data").read_task_facts(
        response.task_id
    )
    assert any(row.get("event") == "checkpoint:saved" for row in facts)
    assert any(row.get("event") == "tool:request" for row in facts)
    assert any(row.get("event") == "tool:response" for row in facts)
    assert any(row.get("tool", {}).get("name") == "list" for row in facts)


def test_missing_model_config_is_not_reported_as_done(tmp_path: Path) -> None:
    response = run_task("inspect workspace", tmp_path)

    assert response.status == "failed"
    assert response.output.startswith(
        "MODEL_PROVIDER_ERROR: missing provider configuration"
    )
