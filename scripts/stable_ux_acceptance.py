"""Stable UX 七场景 deterministic acceptance 聚合入口

作者：xxx

本模块只负责编排场景、调用纯 oracle、写 v2 报告和返回最终退出码。
场景装配与副作用属于 ``stable_ux_scenarios``，行为判定属于
``stable_ux_oracles``。
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Final

from scripts.stable_ux_oracles import (
    OracleAssertion,
    OracleResult,
    OracleStatus,
    evaluate,
)
from scripts.stable_ux_scenarios import run_scenario

REPORT_SCHEMA: Final = "reins.stable_ux_acceptance.v2"
SCENARIO_EXECUTION_ERROR: Final = "SCENARIO_EXECUTION_ERROR"
SCENARIO_IDS: Final = tuple(f"scenario_{index}" for index in range(1, 8))
SCENARIO_TITLES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "scenario_1": "创建 Reins index.html",
        "scenario_2": "同 session 纠正个人介绍",
        "scenario_3": "模型协议错误后恢复",
        "scenario_4": "可恢复工具失败后换策略",
        "scenario_5": "pending resume",
        "scenario_6": "本地 echo MCP",
        "scenario_7": "本地页面 Chromium",
    }
)


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """保存单场景报告所需的固定字段"""

    scenario_id: str
    title: str
    status: OracleStatus
    user_inputs: tuple[str, ...]
    session_id: str
    run_ids: tuple[str, ...]
    actual_output: str
    assertions: tuple[OracleAssertion, ...]
    fact_files: tuple[str, ...]
    artifact_files: tuple[str, ...]
    error_code: str
    can_continue: bool
    notes: tuple[str, ...]


def build_report(
    project_root: Path,
    *,
    evidence_root: Path,
    scenario_ids: Sequence[str] = SCENARIO_IDS,
    report_root: Path | None = None,
) -> dict[str, object]:
    """运行七个隔离场景并生成完整 v2 报告

    参数：project_root 为夹具根，evidence_root 为证据根，scenario_ids 为执行场景
          report_root 为报告所在目录，省略时以项目目录为基准
    返回：包含七个 pass/fail ScenarioResult 的 JSON 可序列化映射
    """
    if not scenario_ids or any(name not in SCENARIO_IDS for name in scenario_ids):
        raise ValueError("select at least one known acceptance scenario")
    source_root = project_root.resolve()
    target_root = evidence_root.resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    results: list[ScenarioResult] = []
    # 【Stable UX】【七场景聚合】1. 单场景异常只转换当前结果，不能截断后续场景
    for scenario_id in scenario_ids:
        try:
            observed = run_scenario(
                scenario_id,
                evidence_root=target_root,
                source_root=source_root,
            )
            oracle = evaluate(scenario_id, observed)
            results.append(_scenario_result(scenario_id, observed, oracle))
        except Exception as exc:  # noqa: BLE001 - deterministic runner 的场景边界
            results.append(_exception_result(scenario_id, exc))
    # 【Stable UX】【七场景聚合】2. 全部场景运行后再计算总状态，fail 不会被部分 pass 覆盖
    counts = _status_counts(results)
    return {
        "schema": REPORT_SCHEMA,
        "generated_at": _utc_now(),
        "status": "pass" if counts["fail"] == 0 else "fail",
        "status_counts": counts,
        "evidence_root": _relative_path(
            source_root if report_root is None else report_root.resolve(), target_root
        ),
        "results": [_result_payload(result) for result in results],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """解析 CLI、写完整报告并按聚合状态返回 0 或 1

    参数：argv 为可注入命令参数；None 时使用当前进程参数
    返回：全部场景 pass 为 0，任一场景 fail 为 1
    """
    args = _parser().parse_args(argv)
    project_root = args.project_root.resolve()
    output_path = (
        args.output.resolve() if args.output else _default_output(project_root)
    )
    evidence_root = (
        args.evidence_root.resolve()
        if args.evidence_root
        else output_path.parent / "stable_ux_evidence"
    )
    report = build_report(
        project_root,
        evidence_root=evidence_root,
        scenario_ids=tuple(args.scenario or SCENARIO_IDS),
        report_root=output_path.parent,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(output_path))
    return 0 if report["status"] == "pass" else 1


def _scenario_result(
    scenario_id: str,
    observed: Mapping[str, object],
    oracle: OracleResult,
) -> ScenarioResult:
    """把 observed 身份字段与纯 oracle 结果组装为冻结 ScenarioResult"""
    return ScenarioResult(
        scenario_id=scenario_id,
        title=SCENARIO_TITLES[scenario_id],
        status=oracle.status,
        user_inputs=_strings(observed.get("user_inputs")),
        session_id=_text(observed.get("session_id")),
        run_ids=_strings(observed.get("run_ids")),
        actual_output=_text(observed.get("actual_output")),
        assertions=oracle.assertions,
        fact_files=_prefixed_paths(scenario_id, observed.get("fact_files")),
        artifact_files=_prefixed_paths(scenario_id, observed.get("artifact_files")),
        error_code=oracle.error_code,
        can_continue=True,
        notes=_strings(observed.get("notes")),
    )


def _exception_result(scenario_id: str, exc: Exception) -> ScenarioResult:
    """把场景边界异常转换为稳定 fail，且不携带绝对路径或潜在机密"""
    assertion = OracleAssertion(
        id="scenario_execution",
        passed=False,
        actual=type(exc).__name__,
    )
    return ScenarioResult(
        scenario_id=scenario_id,
        title=SCENARIO_TITLES[scenario_id],
        status="fail",
        user_inputs=(),
        session_id="",
        run_ids=(),
        actual_output="",
        assertions=(assertion,),
        fact_files=(),
        artifact_files=(),
        error_code=SCENARIO_EXECUTION_ERROR,
        can_continue=True,
        notes=(f"scenario boundary exception: {type(exc).__name__}",),
    )


def _result_payload(result: ScenarioResult) -> dict[str, object]:
    """把冻结结果显式转换为固定报告 schema，避免 dataclass 字段意外扩散"""
    return {
        "scenario_id": result.scenario_id,
        "title": result.title,
        "status": result.status,
        "user_inputs": list(result.user_inputs),
        "session_id": result.session_id,
        "run_ids": list(result.run_ids),
        "actual_output": result.actual_output,
        "assertions": [
            {"id": item.id, "passed": item.passed, "actual": item.actual}
            for item in result.assertions
        ],
        "fact_files": list(result.fact_files),
        "artifact_files": list(result.artifact_files),
        "error_code": result.error_code,
        "can_continue": result.can_continue,
        "notes": list(result.notes),
    }


def _status_counts(results: Sequence[ScenarioResult]) -> dict[str, int]:
    """统计只含 pass/fail 的结果数量"""
    return {
        "pass": sum(result.status == "pass" for result in results),
        "fail": sum(result.status == "fail" for result in results),
    }


def _prefixed_paths(scenario_id: str, value: object) -> tuple[str, ...]:
    """把场景根相对路径提升为 evidence root 相对路径并拒绝绝对路径"""
    paths = _strings(value)
    if any(Path(path).is_absolute() for path in paths):
        raise ValueError(f"absolute evidence path from {scenario_id}")
    return tuple((Path(scenario_id) / path).as_posix() for path in paths)


def _strings(value: object) -> tuple[str, ...]:
    """严格读取字符串序列；畸形值返回空 tuple 供 oracle/report 显式呈现"""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    if any(not isinstance(item, str) for item in value):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _text(value: object) -> str:
    """严格读取字符串，不把任意对象静默格式化成报告字段"""
    return value if isinstance(value, str) else ""


def _relative_path(base: Path, target: Path) -> str:
    """计算稳定 POSIX 相对路径，跨盘时显式失败而不泄露绝对路径"""
    try:
        return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()
    except ValueError as exc:
        raise ValueError("evidence root must share a filesystem volume") from exc


def _default_output(project_root: Path) -> Path:
    """返回公开项目验收报告路径

    传参：project_root 为项目根目录
    返回：ci-artifacts 下的 JSON 报告路径
    """
    return project_root / "ci-artifacts" / "stable_ux_acceptance.json"


def _parser() -> argparse.ArgumentParser:
    """创建不再包含 blocked/status dry-run 开关的 v2 CLI parser"""
    parser = argparse.ArgumentParser(
        description="Run seven deterministic Stable UX acceptance scenarios."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_IDS,
        action="append",
        help="Run selected scenarios; default runs all seven.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--evidence-root", type=Path, default=None)
    return parser


def _utc_now() -> str:
    """返回秒级 UTC 报告生成时间"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
