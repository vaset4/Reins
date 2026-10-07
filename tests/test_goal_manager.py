"""GoalManager 边界组件单测：new 建目标 + switch 三级瀑布匹配。

覆盖 PRD R1-R3 与 design §2.1：
- new：建目标 / 空 body 拒绝
- switch 三级瀑布：id 精确 / 完整名精确 / 子串唯一 / done 目标可切 / 精确优先于子串
- switch fail-closed：匹配 0 个 → GoalNotFoundError；匹配多个 → AmbiguousGoalError 带候选

作者：LKX
时间：2026-07-16 00:00:00
"""

from __future__ import annotations

from pathlib import Path

import pytest
from runtime.goal_manager import (
    AmbiguousGoalError,
    GoalDecision,
    GoalManager,
    GoalNotFoundError,
)
from tasks.records import TASK_STATUS_DONE
from tasks.store import TaskStore


def _manager(tmp_path: Path) -> tuple[GoalManager, TaskStore]:
    store = TaskStore(tmp_path)
    return GoalManager(store), store


def test_open_new_goal_creates_task(tmp_path: Path) -> None:
    """new：建目标并返回 kind=new/created=True 的决定，TaskStore 里查得到。"""
    manager, store = _manager(tmp_path)
    decision = manager.open_new_goal("买菜")
    assert isinstance(decision, GoalDecision)
    assert decision.kind == "new"
    assert decision.created is True
    assert decision.goal == "买菜"
    record = store.load_task(decision.target_task_id)
    assert record is not None
    assert record.goal == "买菜"


def test_open_new_goal_empty_body_rejected(tmp_path: Path) -> None:
    """new：空/纯空白 body 抛 ValueError，且不建任何任务（fail-closed）。"""
    manager, store = _manager(tmp_path)
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(ValueError):
            manager.open_new_goal(bad)
    assert store.list_tasks() == []


def test_switch_goal_by_exact_id(tmp_path: Path) -> None:
    """switch 瀑布①：goal_ref 是既有 task_id → 精确命中。"""
    manager, store = _manager(tmp_path)
    record = store.create_task("重构登录模块")
    decision = manager.switch_goal(record.task_id)
    assert decision.kind == "switch"
    assert decision.target_task_id == record.task_id
    assert decision.created is False


def test_switch_goal_by_exact_name(tmp_path: Path) -> None:
    """switch 瀑布②：goal_ref 是完整目标名 → 精确命中。"""
    manager, store = _manager(tmp_path)
    record = store.create_task("买菜清单")
    decision = manager.switch_goal("买菜清单")
    assert decision.target_task_id == record.task_id


def test_switch_goal_by_substring(tmp_path: Path) -> None:
    """switch 瀑布③：goal_ref 是目标名子串 → 唯一命中（大小写/空白归一）。"""
    manager, store = _manager(tmp_path)
    record = store.create_task("帮我规划本周买菜清单")
    decision = manager.switch_goal("买菜")
    assert decision.target_task_id == record.task_id


def test_switch_goal_substring_casefold(tmp_path: Path) -> None:
    """switch 瀑布③：子串匹配大小写归一。"""
    manager, store = _manager(tmp_path)
    record = store.create_task("Refactor Login Module")
    decision = manager.switch_goal("login")
    assert decision.target_task_id == record.task_id


def test_switch_goal_matches_done_task(tmp_path: Path) -> None:
    """switch 匹配集合含 done：已完成目标照切、不看 status（对齐 slash）。"""
    manager, store = _manager(tmp_path)
    record = store.create_task("上周那个买菜目标")
    store.update_task_status(record.task_id, TASK_STATUS_DONE)
    decision = manager.switch_goal("买菜")
    assert decision.target_task_id == record.task_id


def test_switch_goal_exact_beats_substring(tmp_path: Path) -> None:
    """switch 瀑布优先级：完整名精确命中不掉进子串歧义。

    目标 X 名恰为 '买菜'，目标 Y 名含 '买菜'；switch('买菜') 应走瀑布②
    精确命中 X，不进瀑布③把 X/Y 判成歧义。
    """
    manager, store = _manager(tmp_path)
    x = store.create_task("买菜")
    store.create_task("去超市买菜顺便买奶")
    decision = manager.switch_goal("买菜")
    assert decision.target_task_id == x.task_id


def test_switch_goal_ambiguous_raises_with_candidates(tmp_path: Path) -> None:
    """switch 歧义 fail-closed：子串匹配多个 → AmbiguousGoalError 带候选(id+名)。"""
    manager, store = _manager(tmp_path)
    a = store.create_task("写季度报告")
    b = store.create_task("审阅年度报告")
    with pytest.raises(AmbiguousGoalError) as exc_info:
        manager.switch_goal("报告")
    candidate_ids = {cid for cid, _ in exc_info.value.candidates}
    assert candidate_ids == {a.task_id, b.task_id}
    candidate_goals = {goal for _, goal in exc_info.value.candidates}
    assert candidate_goals == {"写季度报告", "审阅年度报告"}


def test_switch_goal_not_found_by_id(tmp_path: Path) -> None:
    """switch fail-closed：id 与名字都不匹配 → GoalNotFoundError，不建任务。"""
    manager, store = _manager(tmp_path)
    store.create_task("重构登录模块")
    with pytest.raises(GoalNotFoundError):
        manager.switch_goal("no-such-goal-xyz")
    assert len(store.list_tasks()) == 1
