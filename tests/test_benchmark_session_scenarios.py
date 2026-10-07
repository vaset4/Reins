"""验证补充基准使用真实材料与恢复边界。

作者：xxx
时间：2026-09-28 23:00:00
"""

from pathlib import Path

from runtime.session_message_store import SessionMessageStore
from scripts.benchmark_session_scenarios import (
    background_rows,
    fact_density_rows,
    generate_scenario,
    preparation_rows,
)
from scripts.benchmark_session_runtime import SESSION_ID
from scripts.benchmark_session_metrics import measure_reads


def test_request_fixture_contains_all_model_visible_messages(tmp_path: Path) -> None:
    """消息数必须表示模型实际历史而非尚未投影输入；传参：目录；返回：无。"""
    root = tmp_path / "fixture"
    generate_scenario(root, "prepare", 100)
    assert len(SessionMessageStore(root).materialize(SESSION_ID).messages) == 100
    rows = preparation_rows(root, 100)
    assert [row["phase"] for row in rows] == ["first_in_process", "repeat_in_process"]
    assert all(row["token_estimate"] for row in rows)


def test_background_handoff_does_not_dispatch_model(tmp_path: Path) -> None:
    """终态事实已有时恢复仅补宿主交付；传参：目录；返回：无，模型调用使测试失败。"""
    root = tmp_path / "fixture"
    generate_scenario(root, "background", 1)
    with measure_reads(root, allow_writes=True) as meter:
        rows = background_rows(root, 1)
    assert rows[-1]["step"] == "recover_completed_handoff"
    assert rows[-1]["result_count"] == 1
    assert meter.snapshot().file_opens > 0


def test_request_preparation_observes_branch_change_after_restart(
    tmp_path: Path,
) -> None:
    """回退后新构建者必须只准备当前分支；传参：目录；返回：无。"""
    root = tmp_path / "fixture"
    generate_scenario(root, "prepare", 100)
    before = preparation_rows(root, 100)
    SessionMessageStore(root).branch(SESSION_ID, "entry-00003")
    after = preparation_rows(root, 4)
    assert str(before[0]["token_estimate"]) != str(after[0]["token_estimate"])
    assert len(SessionMessageStore(root).materialize(SESSION_ID).messages) == 4


def test_fact_density_fixture_preserves_input_handling(tmp_path: Path) -> None:
    """增加不改变处理状态的事实后仍然没有待处理输入；传参：目录；返回：无。"""
    root = tmp_path / "fixture"
    generate_scenario(root, "facts", 30)
    assert all(row["result_count"] == 0 for row in fact_density_rows(root, 30))
