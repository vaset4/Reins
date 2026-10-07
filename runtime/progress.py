"""只依据实际动作、结果与可核对环境观察连续无进展。

作者：xxx
时间：2026-09-24 23:30:00
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from runtime.tool_operations import ToolOperation
from runtime.types import RunToolsResult
from tools.file_resources import FILE_WRITERS
from tools.tool_registry import ToolDefinition

DEFAULT_REMIND_AFTER = 3
_HISTORY_WINDOW = 4
_RUNTIME_META = frozenset(
    {
        "operation_id",
        "call_id",
        "execution_request",
        "artifact_refs",
        "result_artifact_id",
        "read_full_result",
        "elapsed_ms",
        "duration_ms",
        "duration_seconds",
        "diagnostics",
    }
)


@dataclass(frozen=True, slots=True)
class ProgressObservation:
    """一份实际观察可供模型和运行事实共用，正文只含指纹。"""

    operation_id: str
    tool: str
    definition_version: str
    action: str
    result: str
    environment: str
    files: tuple[dict[str, object], ...]
    known: bool
    reason: str
    semantics: tuple[str, ...] = ()
    exempt: bool = False
    repeats: int = 0
    decision: str = "observed"


def fingerprint(value: object) -> str:
    """计算规范JSON指纹，保留业务时间戳和nonce；传参：JSON数据；返回：摘要。"""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def observe(
    call: ToolOperation,
    result: RunToolsResult,
    definition: ToolDefinition | None,
    *,
    tracked_paths: set[str],
) -> ProgressObservation:
    """读取本运行已跟踪文件的当前版本，并核对工具真实返回；传参：操作、结果、定义及范围；返回：证据。"""
    execution = result.meta.get("execution_request")
    args = (
        dict(execution["arguments"]) if isinstance(execution, dict) else dict(call.args)
    )
    resource = call.resource or {}
    if resource.get("known") and "path" in args:
        args["path"] = os.path.normcase(str(resource["path"]))
    files, reason = _environment(tracked_paths)
    semantics = definition.effective_semantics if definition else ()
    if (
        definition is None
        or definition.exec_boundary
        or (not definition.readonly and definition.name not in FILE_WRITERS)
    ):
        reason = "unobserved_external_effects"
    if result.meta.get("execution_state") == "unknown":
        reason = "execution_unknown"
    if result.meta.get("execution_state") == "not_started":
        reason = "execution_not_started"
    if result.meta.get("truncated") and not (
        result.meta.get("content_sha256") or result.meta.get("full_content_sha256")
    ):
        reason = "truncated_result_without_full_hash"
    exempt = "polling" in semantics or (
        "retryable" in semantics
        and result.status != "ok"
        and result.meta.get("retryable") is True
    )
    result_data = {
        "status": result.status,
        "output": result.output,
        "error": result.error,
        "meta": {
            key: value for key, value in result.meta.items() if key not in _RUNTIME_META
        },
    }
    return ProgressObservation(
        call.operation_id,
        definition.name if definition else call.tool_name,
        call.definition_version,
        fingerprint(
            {
                "tool": definition.name if definition else call.tool_name,
                "version": call.definition_version,
                "arguments": args,
            }
        ),
        fingerprint(result_data),
        fingerprint(files) if not reason else "",
        files,
        not reason,
        reason or ("semantic_exemption" if exempt else "comparable"),
        semantics,
        exempt,
    )


def _environment(paths: set[str]) -> tuple[tuple[dict[str, object], ...], str]:
    """只核对操作记录中的文件，不全盘扫描；传参：路径集合；返回：存在性、哈希与未知原因。"""
    files: list[dict[str, object]] = []
    reason = ""
    for name in sorted(paths):
        path = Path(name)
        try:
            content = path.read_bytes()
            files.append(
                {
                    "path": name,
                    "exists": True,
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
        except FileNotFoundError:
            files.append({"path": name, "exists": False, "sha256": None})
        except OSError as exc:
            reason = "file_hash_unavailable"
            files.append({"path": name, "known": False, "error": type(exc).__name__})
    return tuple(files), reason


class ProgressGuard:
    """运行内连续单动作或稳定交替观察，未知、合法等待和新证据结束旧episode。"""

    def __init__(self, config: Mapping[str, object]) -> None:
        """校验重复证据的提醒阈值；传参：runtime配置；返回：无，退役的停止配置明确报错。"""
        raw = config.get("progress_guard", {})
        if not isinstance(raw, Mapping):
            raise ValueError("progress_guard must be an object")
        if "stop_after" in raw:
            raise ValueError(
                "progress_guard.stop_after is retired; repeated evidence no longer pauses the agent"
            )
        remind = raw.get("remind_after", DEFAULT_REMIND_AFTER)
        if type(remind) is not int or remind <= 0:
            raise ValueError("progress_guard.remind_after requires a positive integer")
        self.remind_after = remind
        self.run_id = ""
        self.tracked_paths: set[str] = set()
        self._history: list[ProgressObservation] = []
        self._pattern: tuple[str, ...] = ()
        self._repeats = 0
        self.notice: ProgressObservation | None = None

    @property
    def enabled(self) -> bool:
        """读取已有关闭开关，关闭时仍记录观察；传参：无；返回：是否向模型提示。"""
        return os.environ.get(
            "REINS_DISABLE_PROGRESS_GUARD", ""
        ).strip().lower() not in {"1", "true", "yes", "on"}

    def start_run(self, run_id: str) -> None:
        """新运行建立独立计数，同一运行接续保留；传参：真实运行身份；返回：无。"""
        if self.run_id != run_id:
            self.run_id = run_id
            self.tracked_paths.clear()
            self.reset()

    def reset(self) -> None:
        """结束旧episode，不清除本运行已经跟踪的文件范围；传参：无；返回：无。"""
        self._history.clear()
        self._pattern = ()
        self._repeats = 0
        self.notice = None

    def observe_batch(
        self, observations: Sequence[ProgressObservation]
    ) -> tuple[ProgressObservation, ...]:
        """并行批次任一新证据优先结束旧计数；传参：公告顺序的观察；返回：同一份带决策的证据。"""
        if not observations:
            return ()
        latest = {item.action: item for item in self._history}
        changed = len(observations) > 1 and any(
            not _same(item, latest.get(item.action)) for item in observations
        )
        if changed or any(not item.known or item.exempt for item in observations):
            self.reset()
            if all(item.known and not item.exempt for item in observations):
                self._history = list(observations[-_HISTORY_WINDOW:])
            return tuple(observations)
        return tuple(self._advance(item) for item in observations)

    def _advance(self, item: ProgressObservation) -> ProgressObservation:
        """单动作基线为0，四次稳定交替只建立模式，随后每次计一次；传参：已知观察；返回：决策证据。"""
        previous = self._history[-1] if self._history else None
        prior = next(
            (row for row in reversed(self._history) if row.action == item.action), None
        )
        if previous is not None and (
            previous.environment != item.environment
            or (prior is not None and not _same(item, prior))
        ):
            self.reset()
            previous = None
        if _same(item, previous):
            self._repeats = self._repeats + 1 if self._pattern == (item.action,) else 1
            self._pattern = (item.action,)
        elif (
            len(self._history) >= 3
            and _same(item, self._history[-2])
            and _same(self._history[-1], self._history[-3])
        ):
            pattern = tuple(sorted((item.action, self._history[-1].action)))
            self._repeats = self._repeats + 1 if self._pattern == pattern else 0
            self._pattern = pattern
        else:
            self._pattern, self._repeats, self.notice = (), 0, None
        if self._repeats < self.remind_after:
            self.notice = None
        decision = "observed"
        if self.enabled and self._repeats >= self.remind_after:
            # 【Agent运行】【重复证据】观察只提示模型，不以重复次数替模型决定结束或暂停
            decision = "suspected"
        observed = replace(item, repeats=self._repeats, decision=decision)
        self._history = [*self._history, observed][-_HISTORY_WINDOW:]
        if decision != "observed":
            self.notice = observed
        return observed

    def model_notice(self) -> str:
        """提供A1临时请求材料，不篡改原工具消息；传参：无；返回：观察或空串。"""
        if not self.enabled or self.notice is None:
            return ""
        return (
            "NO_PROGRESS_SUSPECTED: action, result and tracked files are unchanged. "
            "Use the current goal and this evidence to decide whether to continue, change approach, "
            "finish with supported results, or ask the user for information needed to proceed. "
            "Repetition alone does not establish completion.\n"
            + json.dumps(asdict(self.notice), ensure_ascii=False)
        )


def _same(left: ProgressObservation, right: ProgressObservation | None) -> bool:
    """只比较三维实际证据；传参：当前及基准；返回：可比较且相同。"""
    return (
        right is not None
        and left.known
        and right.known
        and not left.exempt
        and not right.exempt
        and (left.action, left.result, left.environment)
        == (right.action, right.result, right.environment)
    )
