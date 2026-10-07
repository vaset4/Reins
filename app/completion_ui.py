"""前台完成确认菜单与现有存储服务装配。

作者：xxx
时间：2026-09-24 18:00:00
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from runtime.completion_confirmation import CompletionConfirmations
from runtime.ledger import LedgerStore
from runtime.session_message_store import SessionMessageStore
from runtime.tool_operations import ToolOperationStore
from tasks.store import TaskStore


@contextmanager
def completion_service(root: Path) -> Iterator[CompletionConfirmations]:
    """在当前前台或后台线程装配确认服务；传参：数据根；返回：生命周期受控的服务。"""
    with closing(TaskStore(root)) as tasks:
        yield CompletionConfirmations(
            tasks,
            SessionMessageStore(root),
            ToolOperationStore(root),
            LedgerStore(root),
        )


class CompletionMenu:
    """把界面已展示选项绑定到具体问题，普通正文不被猜作确认。"""

    def __init__(self) -> None:
        """初始化显示快照及重复提交身份；传参：无；返回：无。"""
        self._shown: list[dict[str, Any]] = []
        self._choices: list[tuple[str, bool]] = []
        self._actions: dict[tuple[str, bool], str] = {}

    def update(self, proposals: list[dict[str, Any]]) -> str:
        """更新明确选项，重连显示同一提案；传参：宿主视图；返回：变化后的展示文本。"""
        if proposals == self._shown:
            return ""
        self._shown = proposals
        self._choices = []
        lines: list[str] = []
        for proposal in proposals:
            lines.append(
                f"完成确认：{proposal['goal_title']}（版本 {proposal['expected_revision']}）"
            )
            lines.append(str(proposal["question"]))
            lines.extend(
                f"成果：{item}" for item in proposal.get("evidence_preview", [])
            )
            if not proposal["valid"]:
                lines.append(
                    f"此确认已失效：{proposal['error']}。请让助手按当前要求重新处理。"
                )
                continue
            for accepted, label in ((True, "确认完成"), (False, "仍需修改")):
                self._choices.append((proposal["question_id"], accepted))
                lines.append(f"[{len(self._choices)}] {label}")
        if self._choices:
            lines.append("输入对应数字作出明确选择；其他文字会作为普通要求交给助手。")
        return "\n".join(lines)

    def choose(self, text: str) -> dict[str, Any] | None:
        """只解析已显示选项，重试保持同一动作身份；传参：输入；返回：明确选择或普通输入。"""
        options = {str(index): choice for index, choice in enumerate(self._choices, 1)}
        key = options.get(text)
        if key is None:
            return None
        question_id, accepted = key
        if key not in self._actions:
            self._actions[key] = uuid4().hex
        return {
            "question_id": question_id,
            "action_id": self._actions[key],
            "accepted": accepted,
        }
