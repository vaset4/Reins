"""GoalManager：goal_op 归属的 harness 侧边界组件。

模型经 goal_op 表达 new/switch，GoalManager 只做边界校验（目标存在性、
归属决定合法性）+ 建/认目标 + 产出结构化归属决定，不替模型决定"一句话
归哪个目标"，也不碰 loop 状态/ReplState/run fact——那些副作用在 agent_loop
侧完成，保持本组件可独立单测（给个临时 TaskStore 即可）。

作者：LKX
时间：2026-07-16 00:00:00
"""

from __future__ import annotations

import logging
import hashlib
import json
from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from runtime.goal_completion import resolve_completion_evidence
from runtime.ledger import LedgerStore
from runtime.tool_operations import ToolOperationStore
from runtime.session_message_store import SessionMessageStore
from tasks.records import TaskRecord
from tasks.store import TaskStore

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GoalDecision:
    """归属决定：AgentLoop 据此提交 durable focus 并记录审计事实。

    kind：new|switch
    target_task_id：new=新建 id；switch=命中的既有 id
    goal：new 的目标描述；switch 为命中目标的 goal
    created：new=True（真建了任务）；switch=False
    """

    kind: str
    target_task_id: str
    goal: str = ""
    created: bool = False


class GoalNotFoundError(Exception):
    """switch 的 goal_ref 三级瀑布都没匹到任何任务（fail-closed）。"""

    def __init__(self, goal_ref: str) -> None:
        super().__init__(f"no goal matches '{goal_ref}'")
        self.goal_ref = goal_ref


class AmbiguousGoalError(Exception):
    """switch 的 goal_ref 匹到多个任务，需模型用精确 id 消歧（fail-closed）。

    携带 candidates 供 caller 组回喂列候选 id+名。
    """

    def __init__(self, goal_ref: str, candidates: tuple[tuple[str, str], ...]) -> None:
        super().__init__(f"'{goal_ref}' matches {len(candidates)} goals")
        self.goal_ref = goal_ref
        self.candidates = candidates


class GoalManager:
    """目标归属边界：new 建目标、switch 三级瀑布认目标，产出归属决定。"""

    def __init__(
        self,
        task_store: TaskStore,
        *,
        messages: SessionMessageStore | None = None,
        operations: ToolOperationStore | None = None,
        ledger: LedgerStore | None = None,
    ) -> None:
        """依赖注入 TaskStore（不自己硬编码路径实例化），照 E5 先例由 caller 用
        data_root 构造后传入。
        传参：task_store 为目标存储边界"""
        self._tasks = task_store
        self._messages = messages
        self._operations = operations
        self._ledger = ledger

    def complete_goal(
        self,
        task_id: str,
        *,
        expected_revision: int | None,
        summary: str,
        evidence: Sequence[Mapping[str, object]],
        session_id: str,
        run_id: str,
        operation_id: str | None = None,
    ) -> GoalDecision:
        """核对真实引用后提交目标完成，开放性质量由模型依据目标判断。

        传参：task_id/revision 为所见目标；summary/evidence 为结论与依据；session/run 为来源
        返回：已持久提交的完成决定；错误或缺少依据时不写完成状态
        """
        if self._messages is None:
            raise RuntimeError("goal completion requires the session message store")
        if (
            type(expected_revision) is not int
            or expected_revision < 0
            or not summary.strip()
        ):
            raise ValueError("goal completion requires a revision and a result summary")
        intent = hashlib.sha256(
            json.dumps(
                {
                    "summary": summary,
                    "evidence": list(evidence),
                    "revision": expected_revision,
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        with self._messages.write_lock(session_id):
            target = self._tasks.load_task(task_id)
            if target is None or target.is_inbox:
                raise GoalNotFoundError(task_id)
            # 1. 【目标完成】【提交重传】目标已提交后回执失败，重传不能重复写目标或改变原意图
            if (
                operation_id
                and target.completion
                and target.completion.get("operation_id") == operation_id
            ):
                if target.completion.get("intent_digest") != intent:
                    raise ValueError(
                        "completion operation identity has different evidence"
                    )
                return GoalDecision(
                    kind="complete", target_task_id=target.task_id, goal=target.goal
                )
            if self._messages.pending_inputs(session_id):
                raise ValueError(
                    "new user input must be considered before goal completion"
                )
            confirmations = (
                self._ledger.read_session_events(session_id)
                if self._ledger is not None
                else ()
            )
            references = resolve_completion_evidence(
                self._messages.materialize(session_id),
                evidence,
                task_id=task_id,
                operation_refs=self._completion_operation_refs(session_id, task_id),
                confirmations=confirmations,
                expected_revision=expected_revision,
                run_id=run_id,
            )
            # 2. 【目标完成】【同窗核验】会话来源核验持锁到目标版本提交，期间不能插入更正或切分支
            result = self._tasks.complete_task(
                task_id,
                {
                    "summary": summary.strip(),
                    "evidence": references,
                    "operation_id": operation_id,
                    "intent_digest": intent,
                    "session_id": session_id,
                    "run_id": run_id,
                },
                expected_revision=expected_revision,
            )
        return GoalDecision(
            kind="complete", target_task_id=result.task_id, goal=result.goal
        )

    def _completion_operation_refs(
        self, session_id: str, task_id: str
    ) -> dict[str, str]:
        """把当前事项的成功操作映射到原调用，仍由Session验证分支与实际正文；传参：会话/事项；返回：操作引用。"""
        if self._operations is None:
            return {}
        return {
            str(row["operation_id"]): str(row["call"]["call_id"])
            for row in self._operations.for_session(session_id)
            if row.get("state") in {"completed", "late_completed"}
            and row["call"].get("task_id") == task_id
            and row.get("result", {}).get("status") == "ok"
        }

    def open_new_goal(self, goal_body: str) -> GoalDecision:
        """new：为模型判定的新目标建一条扁平任务，产出焦点搬到新目标的决定。

        传参：goal_body 为目标描述
        返回：kind=new 的 GoalDecision（created=True）
        异常：goal_body 去空白后为空 → ValueError（不建无名目标，fail-closed）"""
        goal = goal_body.strip()
        if not goal:
            raise ValueError("goal body required")
        record = self._tasks.create_task(goal)
        _LOG.info("【GoalManager】【新建目标】opened %s for: %s", record.task_id, goal)
        return GoalDecision(
            kind="new", target_task_id=record.task_id, goal=goal, created=True
        )

    def switch_goal(self, goal_ref: str) -> GoalDecision:
        """switch：三级瀑布把 goal_ref 解析到唯一既有目标，产出切换决定。

        瀑布逐级下探、上级命中即返回（精确优先于模糊）：
          ① id 精确 → ② 完整目标名精确 → ③ 名字子串（大小写/空白归一）
        匹配集合 = 全部任务（active+done），对齐 slash /task <id> 不看 status。

        传参：goal_ref 为目标 id 或目标名/子串
        返回：kind=switch 的 GoalDecision（created=False）
        异常：0 命中 → GoalNotFoundError；多命中 → AmbiguousGoalError（带候选）"""
        by_id = self._match_by_id(goal_ref)
        if by_id is not None:
            return self._switch_decision(by_id)
        tasks = self._tasks.list_tasks()
        exact = self._match_by_exact_name(goal_ref, tasks)
        if exact is not None:
            return self._switch_decision(exact)
        hit = self._match_by_substring(goal_ref, tasks)
        return self._switch_decision(hit)

    def _match_by_id(self, goal_ref: str) -> TaskRecord | None:
        """瀑布①：goal_ref 当 task_id 精确查，命中即用（走最快路径）。"""
        if any(char in goal_ref for char in "/\\:") or goal_ref in {
            "",
            ".",
            "..",
            "_inbox",
        }:
            return None
        return self._tasks.load_task(goal_ref)

    def _match_by_exact_name(
        self, goal_ref: str, tasks: list[TaskRecord]
    ) -> TaskRecord | None:
        """瀑布②：完整目标名精确匹配，唯一命中返回；多个真重名 → 歧义。"""
        ref = goal_ref.strip()
        matches = [task for task in tasks if task.goal.strip() == ref]
        if not matches:
            return None
        self._reject_if_ambiguous(goal_ref, matches)
        return matches[0]

    def _match_by_substring(self, goal_ref: str, tasks: list[TaskRecord]) -> TaskRecord:
        """瀑布③：名字子串匹配（大小写/空白归一），唯一命中返回。

        多个 → 歧义；0 个 → 未找到——两级 fail-closed 收口。"""
        ref = goal_ref.strip().casefold()
        matches = [task for task in tasks if ref in task.goal.casefold()]
        if not matches:
            raise GoalNotFoundError(goal_ref)
        self._reject_if_ambiguous(goal_ref, matches)
        return matches[0]

    def _reject_if_ambiguous(self, goal_ref: str, matches: list[TaskRecord]) -> None:
        """匹到多个目标时 fail-closed 抛歧义，携候选 id+名供回喂消歧。"""
        if len(matches) <= 1:
            return
        candidates = tuple((task.task_id, task.goal) for task in matches)
        _LOG.info(
            "【GoalManager】【切换歧义】'%s' matches %d goals", goal_ref, len(matches)
        )
        raise AmbiguousGoalError(goal_ref, candidates)

    def _switch_decision(self, record: TaskRecord) -> GoalDecision:
        """把命中的既有目标包成 kind=switch 的归属决定。"""
        _LOG.info(
            "【GoalManager】【切换目标】switch to %s: %s", record.task_id, record.goal
        )
        return GoalDecision(
            kind="switch", target_task_id=record.task_id, goal=record.goal
        )
