from __future__ import annotations
from scripts.testing.llm import from_test_sequence

import json
from pathlib import Path

import pytest

from llm.client import RealLLMClient
from memory.reflection import normalize_reflection
from memory.sediment import SedimentConfig, SedimentInput, run_sediment
from memory.store import MEMORY_STATE_ACTIVE, MEMORY_STATE_DRAFT, MemoryStore
from runtime.agent_loop import AgentLoop, State
from runtime.lease import Lease
from runtime.run_facts import RunFactStore
from runtime.sediment_reflection import (
    _build_reflection_prompt,
    build_reflection_proposer,
)
from context.skill_recall import recall_skills
from skills.store import SkillStore
from runtime.types import RunContext, Trigger
from tasks.store import TaskStore


def test_run_sediment_produces_recallable_active_skill(tmp_path: Path) -> None:
    # AC9 端到端召回：真 LLM client（测试桩）→ 真 proposer → run_sediment(propose_skill=True)
    # → skill 默认上架 active → recall_skills 按 summary/tags 命中该 skill_id
    # 验到召回命中为止（不接真 LLM、不验产出质量，OOS）；memory 同落 active
    task_id = _seed_done_task(tmp_path, "e2e-skill")
    client = _structured_client(
        memory={"content": "Prefer focused pytest.", "tags": ["validation"]},
        skill={
            "skill_id": "focused-validation",
            "name": "Focused Validation",
            "description": "Run focused tests before commit.",
            "body": "Run focused pytest before commit to validate changes.",
            "required_capabilities": ["terminal"],
            "validation": "Focused pytest passes.",
            "applicable_task_tags": ["validation"],
        },
    )
    proposer = build_reflection_proposer(client, propose_skill=True)

    result = run_sediment(
        tmp_path, task_id, proposer, SedimentConfig(propose_skill=True)
    )

    assert result.status == "written"
    assert result.skill_id == "focused-validation"
    store = SkillStore(tmp_path)
    active = store.load_skill("focused-validation")
    assert active.frontmatter.state == "active"
    assert "draft_source" not in active.meta
    assert active.meta["source_task_id"] == task_id
    # 落 active 后能被召回命中
    recalled = recall_skills(
        tmp_path,
        task_summary="run focused pytest to validate before commit",
        task_tags=["validation"],
    )
    assert "focused-validation" in [item.skill.skill_id for item in recalled]
    memories = MemoryStore(tmp_path).list_memories(state=MEMORY_STATE_ACTIVE)
    assert "Prefer focused pytest." in [memory.content for memory in memories]


def test_terminal_reply_does_not_run_implicit_reflection(
    tmp_path: Path, monkeypatch
) -> None:
    """普通回复结束不再触发隐藏模型调用或写入记忆；传参：目录与替换器；返回：无。"""
    from tests.test_session_runtime import capture_requests

    task_id = _seed_done_task(tmp_path, "e2e-done")
    client = _plain_final_client("本轮回答结束")
    requests = capture_requests(client, monkeypatch)
    loop, context = _build_loop(tmp_path, client, task_id)

    assert loop.run(context) is State.DONE

    assert len(requests) == 1
    assert _task_payload(tmp_path, task_id)["sediment_done"] is False
    memories = MemoryStore(tmp_path).list_memories(state=MEMORY_STATE_ACTIVE)
    assert memories == []


def test_explicit_reflection_failure_preserves_original_task(tmp_path: Path) -> None:
    """显式提炼失败记录自身失败，不改写原事项终态；传参：目录；返回：无。"""
    task_id = _seed_done_task(tmp_path, "e2e-fail")
    client = _plain_final_client("I think the tests went fine overall.")
    result = run_sediment(tmp_path, task_id, build_reflection_proposer(client))
    assert result.status == "failed"

    payload = _task_payload(tmp_path, task_id)
    assert payload["sediment_done"] is False
    assert payload["sediment_attempts"] == 1
    assert payload["status"] == "done"
    assert MemoryStore(tmp_path).list_memories(state=MEMORY_STATE_ACTIVE) == []


def test_missing_llm_client_cannot_report_runtime_success(tmp_path: Path) -> None:
    """缺失执行模型时不能走旧模拟成功路径；传参：目录；返回：无。"""
    task_id = _seed_done_task(tmp_path, "e2e-noclient")
    loop = AgentLoop(tmp_path)
    context = _context(task_id)

    with pytest.raises(RuntimeError, match="llm_client"):
        loop.run(context)

    payload = _task_payload(tmp_path, task_id)
    assert payload["sediment_done"] is False
    assert "sediment_attempts" not in payload


def test_reflection_proposer_rejects_natural_language_final(tmp_path: Path) -> None:
    # Q4 ④：模型回自然语言而非严格 JSON → 诚实抛错，交 run_sediment 记失败，不伪成功
    client = _plain_final_client("Here is what I learned from the task.")
    proposer = build_reflection_proposer(client)

    with pytest.raises(json.JSONDecodeError):
        proposer(_draft_input())


def test_reflection_proposer_rejects_model_error_prefix(tmp_path: Path) -> None:
    # Q4 ②：plan() 出错不抛异常而回 MODEL_ERROR: 前缀 → proposer 显式识别为失败
    client = from_test_sequence([])
    proposer = build_reflection_proposer(client)

    with pytest.raises(ValueError, match="reflection model"):
        proposer(_draft_input())


def test_reflection_proposer_strips_markdown_code_fence(tmp_path: Path) -> None:
    # Q4 ③：模型把 JSON 裹进 ```json 栅栏 → 剥栅栏后正常解析
    fenced = (
        "```json\n"
        + json.dumps({"content": "Prefer focused pytest.", "tags": ["test"]})
        + "\n```"
    )
    client = _plain_final_client(fenced)
    proposer = build_reflection_proposer(client)

    result = proposer(_draft_input())

    assert result["content"] == "Prefer focused pytest."
    assert result["tags"] == ["test"]


def test_reflection_proposer_produces_skill_when_enabled(tmp_path: Path) -> None:
    # AC1：propose_skill=True 且模型回含 memory+skill 的结构化 JSON
    #      → 经 normalize_reflection 得 memory 与 skill 双载荷，skill 七字段就位
    client = _structured_client(
        memory={
            "content": "Record focused validation commands.",
            "tags": ["validation"],
        },
        skill={
            "skill_id": "focused-validation",
            "name": "Focused Validation",
            "description": "Use when a task needs focused validation before commit.",
            "body": "Run `python -m pytest tests/target -q` before commit.",
            "required_capabilities": ["terminal"],
            "validation": "Run focused pytest before commit.",
            "applicable_task_tags": ["validation"],
        },
    )
    proposer = build_reflection_proposer(client, propose_skill=True)

    proposal = normalize_reflection(proposer(_draft_input()), "fact")

    assert proposal.memory is not None
    assert proposal.memory.content == "Record focused validation commands."
    assert proposal.skill is not None
    assert proposal.skill.skill_id == "focused-validation"
    assert proposal.skill.required_capabilities == ["terminal"]
    assert proposal.skill.applicable_task_tags == ["validation"]


def test_reflection_proposer_omits_skill_by_default(tmp_path: Path) -> None:
    # AC2：propose_skill=True 但模型判定无可复用经验、只回 memory（省略 skill 键）
    #      → skill=None，沉淀退回 memory-only，不伪造 skill
    client = _structured_client(
        memory={"content": "Keep tests focused.", "tags": ["validation"]},
        skill=None,
    )
    proposer = build_reflection_proposer(client, propose_skill=True)

    proposal = normalize_reflection(proposer(_draft_input()), "fact")

    assert proposal.memory is not None
    assert proposal.skill is None


def test_reflection_prompt_gates_skill_instruction(tmp_path: Path) -> None:
    # AC3：prompt 门控——默认(False)只指导产 memory、不提 skill；
    #      开启(True)才追加 skill 可选指令，保证生产默认调用行为字节级不变
    draft = _draft_input()

    memory_only = _build_reflection_prompt(draft, False)
    with_skill = _build_reflection_prompt(draft, True)

    assert "skill" not in memory_only.lower()
    assert "skill" in with_skill.lower()


def test_reflection_input_gate_blocks_sensitive_material(tmp_path: Path) -> None:
    # Q7/R5b：拼好的反思材料含已知密钥常量名 → 发送前被输入侧闸门拦截、不调 plan
    client = _plain_final_client('{"content":"x","tags":[]}')
    proposer = build_reflection_proposer(client)
    leaking = _draft_input(journal="export API_KEY=leaked-value")

    with pytest.raises(ValueError, match="blocked by safety scan"):
        proposer(leaking)


def test_sensitive_memory_refused_from_active_in_auto_mode(tmp_path: Path) -> None:
    # AC3（auto）：沉淀内容命中隐私闸门 → auto 模式拒写、不裸落 active、记失败
    task_id = _seed_done_task(tmp_path, "sensitive-auto")

    result = run_sediment(tmp_path, task_id, _sensitive_proposer)

    assert result.status == "failed"
    assert MemoryStore(tmp_path).list_memories(state=MEMORY_STATE_ACTIVE) == []
    assert _task_payload(tmp_path, task_id)["sediment_attempts"] == 1


def test_sensitive_memory_is_never_saved_as_draft(tmp_path: Path) -> None:
    # AC3：沉淀内容命中隐私闸门 → 一律拒写不落库，manual 留证草稿这条路已退役
    #     显式配 auto（非默认档）以覆盖「两个合法模式都拒写」
    task_id = _seed_done_task(tmp_path, "sensitive-auto")
    (tmp_path / "config.yaml").write_text(
        "memory:\n  review_mode: auto\n", encoding="utf-8"
    )

    result = run_sediment(tmp_path, task_id, _sensitive_proposer)

    assert result.status == "failed"
    store = MemoryStore(tmp_path)
    assert store.list_memories(state=MEMORY_STATE_ACTIVE) == []
    assert store.list_memories(state=MEMORY_STATE_DRAFT) == []


def _seed_done_task(data_root: Path, task_id: str) -> str:
    # 造一个 done、轨迹≥5 步、带 summary/journal 的任务，满足 should_run_sediment 白名单
    store = TaskStore(data_root)
    store.create_task(task_id, task_id=task_id)
    store.update_task_status(task_id, "done")
    fact_store = RunFactStore(data_root)
    for index in range(5):
        fact_store.append(
            {
                "event": "step",
                "ts": f"2026-07-13T00:00:{index:02d}+00:00",
                "session_id": f"session-{task_id}",
                "run_id": f"run-{task_id}",
                "task_id": task_id,
                "index": index,
            }
        )
    store.update_summary(task_id, "finished the task")
    store.append_journal(task_id, "ran focused tests")
    return task_id


def _build_loop(
    data_root: Path, client: RealLLMClient, task_id: str
) -> tuple[AgentLoop, RunContext]:
    return AgentLoop(data_root, llm_client=client), _context(task_id)


def _context(task_id: str) -> RunContext:
    return RunContext(
        task_id=task_id,
        trigger=Trigger.USER,
        payload={"message": "本轮只回答当前进度"},
        capability_lease=Lease(),
        segment_id=f"user-{task_id}",
    )


def _plain_final_client(final_text: str) -> RealLLMClient:
    provider_text = json.dumps({"type": "final", "content": final_text})
    return from_test_sequence([provider_text])


def _structured_client(
    *, memory: dict[str, object], skill: dict[str, object] | None
) -> RealLLMClient:
    # 模型回结构化反思 JSON：memory 必产、skill 可选（省略即 LM 判定无可复用经验）
    payload: dict[str, object] = {"memory": memory}
    if skill is not None:
        payload["skill"] = skill
    reflection = json.dumps(payload, ensure_ascii=False)
    provider_text = json.dumps({"type": "final", "content": reflection})
    return from_test_sequence([provider_text])


def _draft_input(journal: str = "ran focused tests") -> SedimentInput:
    return SedimentInput(
        task_id="draft-task",
        status="done",
        task={"status": "done"},
        summary="finished the task",
        journal=journal,
        trajectory=[{"event": "step", "index": 0}],
        memory_type="fact",
    )


def _sensitive_proposer(draft: SedimentInput) -> dict[str, object]:
    # 沉淀内容里混入密钥常量名，用于验证输出侧隐私闸门（Q3/AC3）
    return {
        "type": draft.memory_type,
        "content": "Remember the API_KEY value used during this task.",
        "tags": ["secret"],
    }


def _task_payload(data_root: Path, task_id: str) -> dict[str, object]:
    """读取包含扩展字段的已提交任务；参数：数据根和目标身份；返回：任务原件内容。"""
    return TaskStore(data_root).load_task_payload(task_id)
