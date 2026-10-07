"""验证评测在传输中断后接续同一快照，不覆盖证据或污染原话。

作者：xxx
时间：2026-09-26 12:00:00
"""

from scripts.testing.llm import from_test_turns
import argparse
import json
from dataclasses import replace

import pytest

from llm.messages import model_visible_text
from llm.provider_result import ProviderError
from runtime.session_compaction import SessionCompactionStore, SummarySource
from runtime.session_message_store import SessionMessageStore
from scripts import eval_long_context as evaluation
from scripts.long_context_cases import CASES
from tests.test_model_attempts import _event
from tests.test_semantic_compaction import _response


def test_resume_keeps_published_summary_and_all_call_evidence(tmp_path, monkeypatch):
    """已发布摘要后的断流只重做接续，错误不入史；传参：临时根和替换器；返回：无。"""
    client = from_test_turns([])
    calls = []
    publications = []

    def compact(store, _model, _context, *, case_name, **_options):
        """用真实发布边界隔离接续故障；传参：存储和场景；返回：无。"""
        view = store.messages.materialize(case_name)
        publications.append(
            store.publish(
                SummarySource(view, len(view.messages) - 1),
                "保留用户限制",
                request_ids=("generation",),
            )
        )

    def stream(request, **_options):
        """首次断流，第二次返回预登记答案；传参：真实请求；返回：供应商事件。"""
        calls.append(request)
        if len(calls) == 1:
            error = ProviderError(
                category="provider_error",
                stage="transport",
                retryable=False,
                summary="connection lost",
                provider="scripted",
                model="stub",
                api_family="scripted",
            )
            return iter(
                [
                    _event("response_start", 0, message_id="lost"),
                    _event("response_error", 1, error=error),
                ]
            )
        return _response(json.dumps(dict(CASES[0].expected)), len(calls))

    monkeypatch.setattr(evaluation, "_compact_round", compact)
    monkeypatch.setattr(
        client._adapter_registry.require("scripted_test"), "stream", stream
    )
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match="continuation model failed"):
        evaluation.run_case(client, CASES[0], output, strategy="current", rounds=1)
    checkpoint = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    owner = SessionMessageStore(checkpoint["data_root"])
    original = owner.database.source_path("session_entry", CASES[0].name).read_bytes()
    first_evidence = (output / "call-001.json").read_bytes()
    assert evaluation.run_case(
        client, CASES[0], output, strategy="current", rounds=1, resume=True
    )
    assert len(publications) == 1
    assert (
        SessionCompactionStore(owner).current(owner.materialize(CASES[0].name))
        == publications[0]
    )
    assert (
        owner.database.source_path("session_entry", CASES[0].name)
        .read_bytes()
        .startswith(original)
    )
    assert (output / "call-001.json").read_bytes() == first_evidence
    assert (output / "call-002.json").exists()
    assert all(
        "MODEL_PROVIDER_ERROR" not in model_visible_text(message)
        for message in owner.materialize(CASES[0].name).messages
    )


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "model",
        "case",
        "rounds",
        "profile",
        "provider",
        "connection",
        "api_mode",
    ],
)
def test_resume_refuses_changed_experiment(tmp_path, monkeypatch, change):
    """接续必须使用同一模型、源码及样本参数；传参：变化类型；返回：无。"""
    source = tmp_path / "source.py"
    source.write_text("original", encoding="utf-8")
    monkeypatch.setattr(evaluation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "SOURCE_FILES", ("source.py",))
    args = argparse.Namespace(
        output=tmp_path / "evidence",
        strategy="current",
        baseline_source=None,
        resume=False,
        case=None,
        rounds=2,
        repeats=1,
    )
    target = from_test_turns([]).resolved_target
    evaluation._evaluation_manifest(args, target)
    original = (args.output / "manifest.json").read_bytes()
    args.resume = True
    if change == "source":
        source.write_text("changed", encoding="utf-8")
    elif change == "model":
        target = replace(target, model="changed")
    elif change == "case":
        args.case = CASES[0].name
    elif change == "profile":
        target = replace(target, profile_name="another:model")
    elif change == "provider":
        target = replace(target, provider="another")
    elif change == "connection":
        target = replace(target, base_url="https://another.invalid/v1")
    elif change == "api_mode":
        target = replace(target, api_mode="responses")
    else:
        args.rounds += 1
    with pytest.raises(ValueError, match="identical"):
        evaluation._evaluation_manifest(args, target)
    assert (args.output / "manifest.json").read_bytes() == original


def test_resume_after_publication_before_progress_write_reuses_saved_summary(
    tmp_path, monkeypatch
):
    """摘要已落盘但进度写入失败时，从真实覆盖范围恢复；传参：隔离根和替换器；返回：无。"""
    client = from_test_turns([json.dumps(dict(CASES[0].expected))])
    publications = []
    write = evaluation.write_json

    def compact(store, _model, _context, *, case_name, **_options):
        """发布真实摘要并记录次数；传参：存储和会话；返回：无。"""
        view = store.messages.materialize(case_name)
        previous = store.current(view)
        publications.append(
            store.publish(
                SummarySource(view, len(view.messages) - 1, previous),
                "保留用户限制",
                request_ids=(f"generation-{len(publications)}",),
            )
        )

    def fail_progress(path, value):
        """只在摘要发布后的首个进度写入处模拟磁盘故障；传参：路径和内容；返回：无。"""
        if path.name == "progress.json" and value["phase"] == "continuation":
            raise OSError("progress disk failed after summary publication")
        write(path, value)

    monkeypatch.setattr(evaluation, "_compact_round", compact)
    monkeypatch.setattr(evaluation, "write_json", fail_progress)
    output = tmp_path / "evidence"
    with pytest.raises(OSError, match="after summary publication"):
        evaluation.run_case(client, CASES[0], output, strategy="current", rounds=1)
    monkeypatch.setattr(evaluation, "write_json", write)
    assert evaluation.run_case(
        client, CASES[0], output, strategy="current", rounds=1, resume=True
    )
    assert len(publications) == 1
