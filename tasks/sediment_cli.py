"""
沉淀补扫查漏（⑥ 记忆治理线）

提供三个 Python API 函数：
- list_missing_sediment: 查漏未沉淀任务
- backfill_sediment: 单任务补扫
- batch_backfill_sediment: 批量补扫

作者: LKX
时间: 2026-07-28 15:30:00
"""

import logging
from pathlib import Path
from typing import Any

from llm.client import RealLLMClient
from memory.sediment import SedimentConfig, run_sediment
from runtime.sediment_reflection import build_reflection_proposer
from runtime.run_facts import RunFactStore
from tasks.store import TaskStore

log = logging.getLogger(__name__)


def list_missing_sediment(
    data_root: Path | str,
    status: str = "all",
    limit: int | None = None,
    with_steps: bool = False,
) -> list[dict[str, Any]]:
    """列出尚未整理经验的已结束目标。

    参数：数据根、状态过滤、数量上限及是否读取运行事实数量
    返回：包含原任务状态、整理次数、原因及可选事实数量的清单
    """
    store = TaskStore(data_root)
    log.info("【沉淀】【补扫】开始查漏，status=%s, limit=%s", status, limit)
    # 1. 【沉淀】【补扫】只筛选已保存目标，索引缺失不丢掉待整理事项
    tasks = [
        task
        for task in store.list_tasks(None if status == "all" else status)
        if not task.sediment_done
        and (status != "all" or task.status in {"done", "failed"})
    ]
    tasks.sort(key=lambda task: task.done_at or "", reverse=True)
    results = []
    for task in tasks[:limit]:
        payload = store.load_task_payload(task.task_id)
        # 2. 【沉淀】【补扫】按需读取原运行事实，默认列表不读取执行正文
        count = (
            len(RunFactStore(data_root).read_task_facts(task.task_id))
            if with_steps
            else None
        )
        results.append(
            {
                "task_id": task.task_id,
                "status": task.status,
                "done_at": task.done_at,
                "sediment_done": task.sediment_done,
                "sediment_attempts": payload.get("sediment_attempts", 0),
                "sediment_failed": bool(payload.get("sediment_failed", False)),
                "steps": count,
                "reason": "未触发沉淀（可能为API中断/崩溃/旧任务）",
            }
        )
    log.info("【沉淀】【补扫】查漏完成，找到 %s 个未沉淀任务", len(results))
    return results


def backfill_sediment(
    data_root: Path | str,
    task_id: str,
    llm_client: RealLLMClient,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    单任务补扫

    参数:
        data_root: 数据根目录
        task_id: 任务 ID
        llm_client: LLM 客户端（v1 要求调用方传入）
        force: 是否覆盖已沉淀任务（会自动备份旧快照）
        dry_run: 是否只预览不真实执行

    返回:
        {"status": "written|failed|skipped|dry_run", ...}
        - written: {"status": "written", "memory_id": "...", "skill_id": "..."}
        - failed: {"status": "failed", "reason": "..."}
        - skipped: {"status": "skipped", "reason": "already_done (use --force)"}
        - dry_run: {"status": "dry_run", "would_run": True}
    """
    data_root = Path(data_root)
    store = TaskStore(data_root)

    log.info(f"【沉淀】【补扫】开始补扫 task {task_id}")

    # 1. 加载任务
    task = store.load_task(task_id)
    if task is None:
        raise ValueError(f"Task {task_id} not found")

    # 2. 安全边界：只允许终态任务补扫
    if task.status not in ["done", "failed"]:
        raise ValueError(
            f"Task {task_id} status={task.status}, "
            f"only done/failed tasks can be backfilled"
        )

    # 3. 幂等保护：已沉淀需 force
    if task.sediment_done and not force:
        log.info(f"【沉淀】【补扫】Task {task_id} 已沉淀，跳过（使用 force=True 覆盖）")
        return {"status": "skipped", "reason": "already_done (use --force)"}

    # 4. dry-run 入口拦截
    if dry_run:
        log.info(f"【沉淀】【补扫】Task {task_id} dry-run 预览")
        return {"status": "dry_run", "would_run": True}

    # 5. 调用 run_sediment（绕过 done 检查，允许 inbox，放宽 steps 限制）
    config = SedimentConfig(
        propose_skill=True,
        bypass_done_check=force,
        skip_inbox=False,
        min_trajectory_steps=0,  # 补扫时不限制最小步骤数
    )

    proposer = build_reflection_proposer(llm_client)

    try:
        result = run_sediment(data_root, task_id, proposer, config)

        if result.status == "written":
            log.info(
                f"【沉淀】【补扫】Task {task_id} 补扫成功，"
                f"memory={result.memory_id}, skill={result.skill_id}"
            )
            return {
                "status": "written",
                "memory_id": result.memory_id,
                "skill_id": result.skill_id,
            }
        else:
            log.error(f"【沉淀】【补扫】Task {task_id} 补扫失败：{result.reason}")
            return {"status": "failed", "reason": result.reason}

    except Exception as exc:
        log.error(f"【沉淀】【补扫】Task {task_id} 补扫异常：{exc}")
        return {"status": "failed", "reason": str(exc)}


def batch_backfill_sediment(
    data_root: Path | str,
    llm_client: RealLLMClient,
    status: str = "all",
    limit: int = 10,
    dry_run: bool = False,
) -> dict[str, int]:
    """
    批量补扫

    参数:
        data_root: 数据根目录
        llm_client: LLM 客户端
        status: 任务状态过滤 ("done" | "failed" | "all")
        limit: 最多尝试多少个任务（不是成功多少个）
        dry_run: 是否只预览不真实执行

    返回:
        {"success": N, "failed": M, "skipped": K}
    """
    log.info(f"【沉淀】【补扫】开始批量补扫，status={status}, limit={limit}")

    # 1. 查漏
    missing = list_missing_sediment(data_root, status=status, limit=limit)

    # 2. 逐任务补扫
    results = {"success": 0, "failed": 0, "skipped": 0}

    for item in missing:
        try:
            result = backfill_sediment(
                data_root, item["task_id"], llm_client, dry_run=dry_run
            )

            if result["status"] == "written":
                results["success"] += 1
            elif result["status"] == "failed":
                results["failed"] += 1
            else:
                results["skipped"] += 1

        except Exception as e:
            log.error(f"【沉淀】【补扫】Task {item['task_id']} error: {e}")
            results["failed"] += 1

    log.info(
        f"【沉淀】【补扫】批量补扫完成，"
        f"success={results['success']}, failed={results['failed']}, skipped={results['skipped']}"
    )
    return results
