"""验证专项评测确实经过生产核对和历史读取，脚本测试不充当语义证据。

作者：xxx
时间：2026-09-26 19:25:00
"""

from scripts.testing.llm import _from_scripted, from_test_turns
import json

import pytest

from scripts.testing.llm import ScriptedTurnOptions, _ScriptedTurn
from llm.messages import ToolCallPart, model_visible_text
from scripts import eval_long_context_scenarios as scenarios
from scripts.long_context_cases import CASES
from tests.test_semantic_compaction import _response, _summary_delta


@pytest.mark.parametrize("directory", [True, False])
def test_navigation_measures_real_history_reads(tmp_path, directory):
    """答案必须经过真实原文工具，目录可用性不改变原文；传参：隔离根与目录状态；返回：无。"""
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
        ]
    )
    result = scenarios.navigation(client, tmp_path, directory=directory)
    assert result["passed"] and len(result["history_reads"]) == 1
    assert "UTF-16LE" in result["history_reads"][0]["result"]["output"]
    assert result["history_reads"][0]["state"] == "completed"


def test_controlled_omission_keeps_candidate_and_audit_evidence(tmp_path, monkeypatch):
    """遗漏候选、核对结果和接续回答分别保存；传参：隔离根与替换器；返回：无。"""
    client = from_test_turns([])
    requests = []

    def stream(request, **_options):
        """用确定响应核对评测接线；传参：实际请求；返回：供应商事件。"""
        requests.append(request)
        body = "\n".join(model_visible_text(message) for message in request.messages)
        text = (
            _summary_delta(body)
            if "整理下面的会话资料" in body
            else json.dumps(dict(CASES[0].expected))
        )
        return _response(text, len(requests))

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    result = scenarios.controlled_omission(client, tmp_path)
    assert result["passed"] and result["calls"] == 2
    assert result["checks"]
    original = json.loads(
        (tmp_path / "controlled_candidate.json").read_text(encoding="utf-8")
    )
    corrected = json.loads(
        (tmp_path / "corrected_summary.json").read_text(encoding="utf-8")
    )
    assert "80" not in original["content"]["entries"][0]["text"]
    assert "80" in corrected["text"]


def test_working_fact_scenario_really_recovers_without_memory_write(
    tmp_path, monkeypatch
):
    """专项入口实际经历两次摘要与后台恢复，固定响应只验证接线；传参：隔离根和替换器；返回：无。"""
    client = from_test_turns([], options=ScriptedTurnOptions(context_window=16000))
    requests = []
    case = next(item for item in CASES if item.name == "working_fact")

    def stream(request, **_options):
        """保留用户原话和公开新事实，在恢复请求中核对两者；传参：真实请求；返回：测试事件。"""
        requests.append(request)
        body = "\n".join(model_visible_text(message) for message in request.messages)
        if "整理下面的会话资料" not in body:
            instructions = "\n".join(part.text for part in request.instructions)
            assert (
                "9000" in instructions
                and "8000" in instructions
                and "recovery_intent" in instructions
            )
            assert "skill_read" in instructions
            return _response(json.dumps(dict(case.expected)), len(requests))
        delta = json.loads(_summary_delta(body))
        payload = json.loads(body.rsplit("\n", 1)[-1])
        existing = {entry["entry_id"] for entry in payload["current_entries"]}
        for group in payload["original_groups"]:
            for message in group:
                text = "\n".join(part.get("text", "") for part in message["content"])
                if not text.startswith("同一服务的实际监听证据"):
                    continue
                if "port" not in existing:
                    delta["add"].append(
                        {
                            "entry_id": "port",
                            "kind": "conclusion",
                            "text": text,
                            "sources": [
                                {"message_id": message["message_id"], "quote": text}
                            ],
                        }
                    )
                # 【上下文评测】【事实接续】历史降档读取片段正文，四档均保留这条简短原话与来源
                for segment in delta["segments"]:
                    if message["message_id"] not in segment["message_ids"]:
                        continue
                    segment.update(
                        title="当前服务监听证据", p1=text, p2=text, p3=text, p4=text
                    )
        return _response(json.dumps(delta, ensure_ascii=False), len(requests))

    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    result = scenarios.working_fact_recovery(client, tmp_path)
    assert result["passed"] and result["memory_unchanged"] and result["no_fake_input"]
    assert len(result["summary_ids"]) == 2
