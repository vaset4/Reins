from __future__ import annotations

import json
from typing import Protocol

from llm.types import LLMPlan
from memory.safety_scan import scan as safety_scan
from memory.sediment import ReflectionProposer, SedimentInput

# plan() 出错不抛异常而在 final_output 前置该标记（parser.py:90），proposer 须显式识别
_MODEL_ERROR_PREFIX = "MODEL_ERROR:"

# 反思 prompt 强约束模型只回严格 JSON，杜绝自然语言 final 混入解析（Q4 ①）
# 显式维护API的memory-only指令，只蒸馏一条记忆，不涉及Skill
_REFLECTION_INSTRUCTION = (
    "You are distilling one finished task into a single reusable memory. "
    "Reply with ONLY a strict JSON object, no prose, no markdown fence: "
    '{"content": "<one concise reusable lesson or fact>", '
    '"tags": ["<short-tag>", ...]}. '
    "The content must be a single actionable sentence; tags are lowercase keywords."
)

# skill-enabled 指令：额外让模型自主判断是否沉淀一个跨任务可复用的操作序列（skill），
# 默认省略 skill 字段——只有确实沉淀出可复用方法时才产，杜绝为凑字段而伪造技能
_SKILL_REFLECTION_INSTRUCTION = (
    "You are distilling one finished task into a reusable memory, and OPTIONALLY "
    "into one reusable skill. Reply with ONLY a strict JSON object, no prose, no "
    "markdown fence: "
    '{"memory": {"content": "<one concise reusable lesson or fact>", '
    '"tags": ["<short-tag>", ...]}, '
    '"skill": {"skill_id": "<kebab-case-id>", "name": "<short name>", '
    '"description": "<when to use it>", "body": "<the reusable steps>", '
    '"required_capabilities": ["<capability>", ...], '
    '"validation": "<how to verify it worked>", '
    '"applicable_task_tags": ["<tag>", ...]}}. '
    "Produce a skill ONLY when the task distilled a concrete, cross-task reusable "
    "procedure worth replaying; otherwise OMIT the skill field entirely. "
    "Do not invent a skill just to fill the field."
    " When revising a listed skill, include its exact expected_version and a change_reason explaining the new evidence."
)


class _ReflectionClient(Protocol):
    """显式反思所需的客户端契约，只依赖plan，默认运行收尾不调用它。"""

    def plan(self, task: str, context: object | None = None) -> LLMPlan: ...


def build_reflection_proposer(
    client: _ReflectionClient, propose_skill: bool = False
) -> ReflectionProposer:
    """构造显式反思 proposer：把任务轨迹经 LLM 反思成 memory（及可选 skill）沉淀 dict
    传参：
        client - 提供 plan(task, context) 的 LLM 客户端（结构化契约，非具体实现）
        propose_skill - 是否让模型在反思时额外产出可选的 skill 候选；
                        默认 False 仅提炼memory；生产后台提炼由KnowledgeJobs接纳
    返回：
        吃 SedimentInput、产出反思 dict（memory-only 扁平 或 memory+skill 结构化）的
        proposer 闭包；失败抛异常交 run_sediment 记失败
    作者：LKX
    时间：2026-07-22 00:00:00"""

    def _propose(draft: SedimentInput) -> dict[str, object]:
        prompt = _build_reflection_prompt(draft, propose_skill)
        # 发送给外部 provider 前先过输入侧隐私闸门（Q7），命中即中断本次反思
        _guard_input_privacy(prompt)
        plan = client.plan(prompt, context=None)
        return _parse_reflection_plan(plan)

    return _propose


def _build_reflection_prompt(draft: SedimentInput, propose_skill: bool) -> str:
    """只附上本次显式提炼所需材料；传参：原任务材料和方法开关；返回：反思提示词。"""
    # 按开关选指令：开启 skill 才用含 skill 可选字段的结构化指令，否则 memory-only
    instruction = (
        _SKILL_REFLECTION_INSTRUCTION if propose_skill else _REFLECTION_INSTRUCTION
    )
    # 把任务 summary/journal/trajectory 直接拼进 prompt 文本（不走 context bundle）
    trajectory = "\n".join(
        json.dumps(step, ensure_ascii=False, sort_keys=True)
        for step in draft.trajectory
    )
    versions = (
        f"# Existing skill versions\n{json.dumps(draft.skill_versions, ensure_ascii=False)}\n\n"
        if propose_skill
        else ""
    )
    return (
        f"{instruction}\n\n"
        f"# Task status\n{draft.status}\n\n"
        f"# Summary\n{draft.summary}\n\n"
        f"# Journal\n{draft.journal}\n\n"
        f"{versions}"
        f"# Trajectory\n{trajectory}\n"
    )


def _guard_input_privacy(prompt: str) -> None:
    # Q7 输入侧闸门：命中现有 safety_scan 规则即中断，不把敏感材料裸发给 provider
    result = safety_scan(prompt)
    if not result.is_safe:
        raise ValueError("reflection prompt blocked by safety scan")


def _parse_reflection_plan(plan: LLMPlan) -> dict[str, object]:
    # 1. 先显式识别模型/协议失败（Q4 ②），不靠 json.loads 撞死来间接发现
    _reject_model_failure(plan)
    # 2. 剥 markdown code fence 后再解析 final_output（Q4 ③）
    raw = _strip_code_fence(plan.final_output or "")
    # 3. 解析失败抛 JSONDecodeError、结构非法抛 ValueError，都走 _record_failure（Q4 ④）
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("reflection output must be a JSON object")
    # 4. 原样返回，交 normalize_reflection 分流：含 memory/skill 键走结构化、
    #    仅 content/tags 的扁平输出退回 memory-only（生产 memory-only 路径不受影响）
    return data


def _reject_model_failure(plan: LLMPlan) -> None:
    # 模型协议错误走 model_error 或 MODEL_ERROR: 前缀两条路（parser.py:90/405），都判失败
    if plan.model_error is not None:
        raise ValueError(f"reflection model error: {plan.model_error.category}")
    output = plan.final_output or ""
    if output.startswith(_MODEL_ERROR_PREFIX):
        raise ValueError(f"reflection model failure: {output}")


def _strip_code_fence(text: str) -> str:
    # 模型极常把 JSON 裹进 ```json ... ```，解析前剥掉首尾栅栏
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    body = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
    return "\n".join(body).strip()


__all__ = ["build_reflection_proposer"]
