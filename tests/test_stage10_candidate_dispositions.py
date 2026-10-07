"""【知识维护】【候选对账】保留失败原件并验证跨候选替代及明确放弃。

作者：xxx
时间：2026-10-02 11:05:00
"""

import copy

import pytest

from runtime.knowledge_worker import committed_changes


def candidate_operations():
    """复现一个错误候选与七个真实提交的回执形状；参数：无；返回：独立操作列表。"""
    failed = {
        "operation_id": "failed-candidate",
        "call": {
            "tool_name": "memory_manage",
            "args": {
                "memory_scope": "project",
                "subject": "repository",
                "fact_key": "article_and_commit_conventions",
            },
        },
        "result": {
            "status": "error",
            "error": "knowledge source must be a delivered user input",
        },
    }
    commits = [
        {
            "operation_id": f"saved-{index}",
            "call": {
                "tool_name": "memory_manage",
                "args": {
                    "memory_scope": "project",
                    "subject": "repository",
                    "fact_key": f"guideline-{index}",
                },
            },
            "result": {
                "status": "ok",
                "meta": {
                    "committed": True,
                    "record": {
                        "memory_id": f"memory-{index}",
                        "version": f"version-{index}",
                    },
                },
            },
        }
        for index in range(7)
    ]
    return [failed, *commits]


def test_seven_commits_can_resolve_replaced_candidate_without_deleting_failure():
    """换名候选通过实际提交身份替代旧失败，七条提交和原错误不变；参数：无；返回：无。"""
    operations = candidate_operations()
    before = copy.deepcopy(operations)
    commits, errors = committed_changes(
        operations,
        resolutions=[
            {
                "operation_id": "failed-candidate",
                "disposition": "superseded",
                "replacement_operation_id": "saved-0",
                "reason": "已按原用户输入保存为repository_guidelines",
            }
        ],
    )
    assert len(commits) == 7 and errors == []
    assert operations == before


def test_unresolved_failure_still_blocks_other_successful_candidates():
    """部分提交不能自动消除未处置候选；参数：无；返回：无。"""
    commits, errors = committed_changes(candidate_operations())
    assert len(commits) == 7 and errors == ["failed-candidate"]


@pytest.mark.parametrize("replacement", ["foreign-operation", "failed-candidate"])
def test_replacement_requires_actual_committed_operation(replacement):
    """不存在或失败的替代操作不能伪装已解决；参数：非法操作身份；返回：无。"""
    with pytest.raises(ValueError, match="committed"):
        committed_changes(
            candidate_operations(),
            resolutions=[
                {
                    "operation_id": "failed-candidate",
                    "disposition": "superseded",
                    "replacement_operation_id": replacement,
                    "reason": "声称替代",
                }
            ],
        )


def test_abandonment_is_explicit_and_requires_reason():
    """模型可解释无效候选并放弃，空理由不能处置失败；参数：无；返回：无。"""
    resolution = {
        "operation_id": "failed-candidate",
        "disposition": "abandoned",
        "reason": "没有可靠原始证据",
    }
    assert committed_changes(candidate_operations(), resolutions=[resolution])[1] == []
    with pytest.raises(ValueError, match="reason"):
        committed_changes(
            candidate_operations(), resolutions=[{**resolution, "reason": " "}]
        )
