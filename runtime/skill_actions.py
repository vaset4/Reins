"""模型主动修订方法与记录独立案例反馈的原生动作。

作者：xxx
时间：2026-09-15 02:25:05
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from memory.safety_scan import scan
from runtime.knowledge_sources import knowledge_sources, read_stored_sources
from runtime.native_actions import NativeActionContext
from runtime.tool_operations import ToolOperation
from runtime.types import RunToolsResult
from skills.store import Skill, SkillStore, build_skill_markdown

SKILL_ACTIONS = frozenset({"skill_manage"})


class SkillActions:
    """存储、发布和效果各有明确动作，模型决定是否需要它们。"""

    def __init__(self, context: NativeActionContext, *, data_root: Path) -> None:
        """绑定当前证据来源和方法写者；传参：运行依赖及目录；返回：无。"""
        self.context, self.store = context, SkillStore(data_root)

    def execute(self, call: ToolOperation) -> RunToolsResult:
        """处理已校验的知识动作；传参：操作；返回：版本与实际状态。"""
        payload: dict[str, Any]
        try:
            if call.args["action"] == "versions":
                payload = {
                    "versions": self.store.list_versions(str(call.args["skill_id"]))
                }
            elif call.args["action"] == "sources":
                skill = self.store.load_skill(
                    str(call.args["skill_id"]),
                    version=cast(str | None, call.args.get("version")),
                )
                payload = {
                    "version": skill.version,
                    "sources": read_stored_sources(self.context, skill.sources),
                }
            else:
                payload = skill_view(self._change(call))
        except (ValueError, FileNotFoundError) as exc:
            return RunToolsResult.error_result(action=call.tool_name, error=str(exc))
        return RunToolsResult.ok(
            action=call.tool_name,
            content=json.dumps(payload, ensure_ascii=False),
            meta=payload,
        )

    def _change(self, call: ToolOperation) -> Skill:
        """保存或变更指定版本，任务效果须另有案例证据；传参：动作；返回：实际版本。"""
        args, identity = call.args, str(call.args["skill_id"])
        action, reason = str(args["action"]), str(args["reason"])
        origin = self.context.run.payload.get("knowledge_origin")
        publishes = action == "publish" or args.get("publish") is True
        if publishes and isinstance(origin, dict) and origin.get("publish") is not True:
            raise ValueError(
                "this reflection request allows drafts only; publication needs an explicit knowledge action"
            )
        if action in {"create", "revise"}:
            return self._write(call)
        version = str(args["version"])
        if action == "publish":
            return self.store.publish_version(
                identity, version, reason=reason, change_id=call.operation_id
            )
        if action == "withdraw":
            return self.store.withdraw_version(
                identity, version, reason=reason, change_id=call.operation_id
            )
        return self.store.record_outcome(
            identity,
            version,
            outcome=str(args["outcome"]),
            case_id=f"{self.context.run.session_id}:{self.context.run.run_id}",
            reason=reason,
            sources=knowledge_sources(self.context, call),
            event_id=call.operation_id,
        )

    def _write(self, call: ToolOperation) -> Skill:
        """保存指南和可选脚本，来源由运行解析；传参：创建或修订动作；返回：固定内容版本。"""
        args, identity = call.args, str(call.args["skill_id"])
        base = (
            self.store.load_skill(identity, version=str(args["expected_version"]))
            if args["action"] == "revise"
            else None
        )
        body = str(args["body"])
        script = cast(str | None, args.get("script"))
        if not scan(body + "\n" + (script or "")).is_safe:
            raise ValueError("skill content blocked by safety scan")
        markdown = build_skill_markdown(
            name=str(args.get("name", base.frontmatter.name if base else identity)),
            body=body,
            trigger_keywords=cast(
                list[str],
                args.get(
                    "trigger_keywords",
                    base.frontmatter.trigger_keywords if base else [],
                ),
            ),
            applicable_task_tags=cast(
                list[str],
                args.get("tags", base.frontmatter.applicable_task_tags if base else []),
            ),
            required_capabilities=cast(
                list[str],
                args.get(
                    "required_capabilities",
                    base.frontmatter.required_capabilities if base else [],
                ),
            ),
        )
        meta = dict(base.meta) if base else {}
        if "validation" in args:
            meta["author_validation"] = str(args["validation"])
        if script is not None:
            meta["script_entry"] = "script.py:main"
        options: dict[str, Any] = {
            "sources": knowledge_sources(self.context, call),
            "reason": str(args["reason"]),
            "script": script,
            "meta": meta,
            "publish": args.get("publish", False) is True,
            "change_id": call.operation_id,
        }
        if base is not None:
            return self.store.revise_skill(
                identity, markdown, expected_version=base.version, **options
            )
        return self.store.create_skill(identity, markdown, **options)


def skill_view(skill: Skill) -> dict[str, Any]:
    """输出可审阅方法状态，不把作者声明当作验证；传参：方法；返回：知识结果。"""
    return {
        "skill_id": skill.skill_id,
        "version": skill.version,
        "previous_version": skill.previous_version,
        "state": skill.frontmatter.state,
        "body": skill.body,
        "reason": skill.reason,
        "sources": [asdict(source) for source in skill.sources],
        "evaluation_status": skill.evaluation_status,
        "evidence_summary": skill.evidence_summary,
        "resource_ref": f"skill:{skill.skill_id}@{skill.version}",
    }
