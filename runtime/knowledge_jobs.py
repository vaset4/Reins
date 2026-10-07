"""将主动经验提炼接到已有持久后台交付，原运行收尾不等待新模型调用。

作者：xxx
时间：2026-09-15 02:25:05
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from llm.base import LLMClient
from llm.messages import UserMessage
from runtime.history_reader import DEFAULT_HISTORY_PAGE_SIZE, read_history_page
from runtime.knowledge_sources import knowledge_source_view
from runtime.native_actions import NativeActionContext
from runtime.scheduled_actions import ScheduledActions
from runtime.tool_operations import ToolOperation
from runtime.types import RunToolsResult
from skills.store import SkillStore

KNOWLEDGE_ACTIONS = frozenset(
    {"knowledge_reflect", "knowledge_read", "knowledge_finish"}
)


class KnowledgeJobs:
    """只接纳明确提炼请求；发生、认领、恢复和通知均由已有后台 owner 负责。"""

    def __init__(
        self, context: NativeActionContext, *, data_root: Path, client: LLMClient
    ) -> None:
        """绑定当前运行与既有后台装配；传参：依赖、目录、客户端；返回：无。"""
        self.context, self.data_root, self.client = context, data_root, client

    def execute(self, call: ToolOperation) -> RunToolsResult:
        """接纳单次提炼或补读冻结来源；传参：原生动作；返回：工作身份或真实材料。"""
        try:
            if call.tool_name == "knowledge_reflect":
                return self._request(call)
            if call.tool_name == "knowledge_finish":
                return self._finish(call)
            else:
                payload = self._read(call)
        except (ValueError, FileNotFoundError) as exc:
            return RunToolsResult.error_result(action=call.tool_name, error=str(exc))
        return RunToolsResult.ok(
            action=call.tool_name,
            content=json.dumps(payload, ensure_ascii=False),
            meta=payload,
        )

    def _finish(self, call: ToolOperation) -> RunToolsResult:
        """候选处置先独立落盘，其他未完项不能撤回已经明确的放弃或替代；参数：完成动作；返回：真实完成或未完成回执。"""
        from runtime.knowledge_worker import (
            finish_knowledge,
            save_candidate_resolutions,
        )
        from runtime.persistence import RuntimeStore

        with RuntimeStore(self.data_root).transaction():
            saved = save_candidate_resolutions(
                self.context, call, data_root=self.data_root
            )
        try:
            with RuntimeStore(self.data_root).transaction():
                payload = finish_knowledge(self.context, call, data_root=self.data_root)
        except (ValueError, FileNotFoundError) as exc:
            return RunToolsResult.error_result(
                action=call.tool_name,
                error=str(exc),
                meta={
                    "candidate_resolutions": saved.get("candidate_resolutions", []),
                    "work_state": saved["state"],
                    "source_coverage_completed": False,
                },
            )
        return RunToolsResult.ok(
            action=call.tool_name,
            content=json.dumps(payload, ensure_ascii=False),
            meta=payload,
        )

    def _request(self, call: ToolOperation) -> RunToolsResult:
        """冻结来源范围后创建同身份的一次性工作；传参：提炼请求；返回：持久接纳结果，不调用模型。"""
        current = self.context.messages.materialize(self.context.run.session_id)
        anchor = current.entries[-1].parent_id
        if anchor is None:
            raise ValueError("knowledge reflection requires prior session evidence")
        if call.args.get("knowledge_only") is True:
            from runtime.knowledge_maintenance import (
                KnowledgeMaintenance,
                complete_source,
            )

            view = complete_source(
                self.context.messages.materialize(
                    current.session_id, at_entry_id=anchor
                )
            )
            selected_ids = cast(
                list[str],
                call.args.get(
                    "source_message_ids",
                    [message.message_id for message in view.messages],
                ),
            )
            work = KnowledgeMaintenance(self.data_root).admit_sources(
                self.context.run,
                client=self.client,
                view=view,
                message_ids=set(selected_ids),
                reason=f"explicit_selection: {call.args['objective']}",
            )
            payload: dict[str, object] = {
                "accepted": work is not None,
                "work": work,
                "reason": None
                if work is not None
                else "selected source already has a durable work record",
            }
            return RunToolsResult.ok(
                action=call.tool_name,
                content=json.dumps(payload, ensure_ascii=False),
                meta=payload,
            )
        skill_id = cast(str | None, call.args.get("skill_id"))
        expected = call.args.get("expected_version")
        if skill_id is not None and SkillStore(self.data_root).skill_exists(skill_id):
            selected = SkillStore(self.data_root).load_skill(
                skill_id, version=cast(str | None, expected)
            )
            expected = selected.version
        origin = {
            "request_id": call.operation_id,
            "source_session_id": self.context.run.session_id,
            "source_run_id": self.context.run.run_id,
            "source_entry_id": anchor,
            "objective": str(call.args["objective"]),
            "publish": call.args.get("publish") is True,
            "skill_id": skill_id,
            "expected_version": expected,
        }
        target = {
            key: origin[key] for key in ("skill_id", "expected_version", "publish")
        }
        prompt = (
            "处理一次已授权的知识提炼请求。原用户要求可通过knowledge_read(action=user_inputs)读取，"
            "完整过程可选messages或operations补读；来源范围已固定。\n"
            f"提炼目标：{origin['objective']}\n方法目标及发布意图：{json.dumps(target, ensure_ascii=False)}\n"
            "按材料选择是否记录长期事实或修订方法，保留反例、适用条件与来源。"
            "可用memory_manage/skill_manage，引用原分支最后一条真实用户输入用source_mode=origin_inputs，省略source_input_ids。"
            "若需指定其他原用户输入，先从knowledge_read取得其message_id；分支锚点不是用户输入编号。"
            "current_input属于本后处理任务，不能作为原用户来源；原操作证据用origin_results和knowledge_read取得的operation_id。"
            "方法只有publish=true时才发布，否则保留为可显式试用的draft；已有方法使用expected_version修订。"
            "没有独立案例的效果证据就保持not_evaluated。保存或脚本退出不能证明用户目标已实现。"
        )
        args: dict[str, object] = {
            "action": "create",
            "name": "知识提炼",
            "prompt": prompt,
            "kind": "work",
            "timezone": "UTC",
            "time": f"at:{datetime.now(timezone.utc).isoformat()}",
            "knowledge_origin": origin,
        }
        scheduled = ScheduledActions(
            self.context.run, data_root=self.data_root, client=self.client
        )
        result = scheduled.execute(replace(call, tool_name="schedule", args=args))
        return replace(result, action=call.tool_name, tool_name=call.tool_name)

    def _read(self, call: ToolOperation) -> dict[str, object]:
        """仅补读本后处理请求所引用的原分支，源会话后续变化不会扩展范围；传参：分页动作；返回：原文。"""
        from runtime.knowledge_maintenance import KnowledgeMaintenance, automatic_origin

        origin = automatic_origin(self.context.run)
        if origin is not None and KnowledgeMaintenance(self.data_root).load(
            origin["work_id"]
        )["state"] in {"cancelled", "cancelling"}:
            raise ValueError("knowledge work was cancelled before reading")
        view = knowledge_source_view(self.context, origin=True)
        action = call.args.get("action", "user_inputs")
        if action not in {"user_inputs", "messages", "operations", "operation"}:
            raise ValueError("unknown knowledge reading action")
        if action in {"operations", "operation"}:
            from runtime.knowledge_reading import read_source_operations

            return read_source_operations(self.context.messages, view, call.args)
        if action == "user_inputs":
            agent_ids = {
                entry.entry_id
                for entry in view.entries
                if entry.input_source == "agent"
            }
            messages = tuple(
                message
                for message in view.messages
                if isinstance(message, UserMessage)
                and message.message_id not in agent_ids
            )
            view = replace(view, messages=messages, pending_tool_calls=())
        payload = read_history_page(
            view,
            call_id=call.call_id,
            cursor=cast(str | None, call.args.get("cursor")),
            limit=cast(int, call.args.get("limit", DEFAULT_HISTORY_PAGE_SIZE)),
        )
        if origin is not None:
            store = KnowledgeMaintenance(self.data_root)
            row = store.load(origin["work_id"])
            returned = cast(list[dict[str, object]], payload["messages"])
            store.update(
                origin["work_id"],
                read_message_ids=sorted(
                    {
                        *row["read_message_ids"],
                        *(str(message["message_id"]) for message in returned),
                    }
                ),
            )
        return payload
