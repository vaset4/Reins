from __future__ import annotations

import importlib
from pathlib import Path

from artifacts.store import ArtifactStore
from runtime.lease import Lease
from tasks.ids import new_ulid
from tools.types import ToolError, ToolErrorCategory


def screenshot(
    monitor: int = 0,
    *,
    data_root: Path | str,
    task_id: str,
    session_id: str | None = None,
) -> dict[str, object]:
    """捕获屏幕并保存到当前会话所属原件；传参：屏幕、数据根与身份；返回：产物编号和尺寸。"""
    mss = importlib.import_module("mss")
    tools = importlib.import_module("mss.tools")
    artifact_id = f"art-{new_ulid()}"
    ArtifactStore(data_root)
    relative_path = f"assets/{artifact_id}.png"
    path = Path(data_root) / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)

    with mss.mss() as capture:
        monitors = getattr(capture, "monitors", [])
        if monitor < 0 or monitor >= len(monitors):
            raise ValueError(f"invalid monitor: {monitor}")
        image = capture.grab(monitors[monitor])
        width = int(getattr(image, "width", 0) or image.size[0])
        height = int(getattr(image, "height", 0) or image.size[1])
        tools.to_png(image.rgb, image.size, output=str(path))

    ArtifactStore(data_root).create_artifact(
        task_id,
        "screenshot",
        relative_path,
        "screen screenshot",
        path.stat().st_size,
        artifact_id=artifact_id,
        session_id=session_id,
    )
    return {
        "artifact_id": artifact_id,
        "width": width,
        "height": height,
        "monitor": monitor,
    }


def screenshot_executor(args: dict[str, object]) -> object:
    """连接工具上下文与屏幕捕获；传参：工具参数和内部会话身份；返回：产物或明确错误。"""
    try:
        return screenshot(
            _int_arg(args.get("monitor"), default=0),
            data_root=_data_root_arg(args),
            task_id=_task_id_arg(args),
            session_id=str(args.get("__session_id__") or "") or None,
        )
    except ValueError as exc:
        return ToolError(ToolErrorCategory.INVALID_INPUT, str(exc), retryable=False)
    except Exception as exc:
        return ToolError(ToolErrorCategory.UNKNOWN, str(exc), retryable=False)


def _task_id_arg(args: dict[str, object]) -> str:
    task_id = str(args.get("__task_id__") or "").strip()
    lease = args.get("__lease__")
    if not task_id and isinstance(lease, Lease):
        task_id = lease.task_id
    if not task_id:
        raise ValueError("missing task_id")
    return task_id


def _data_root_arg(args: dict[str, object]) -> Path:
    value = args.get("__data_root__")
    if isinstance(value, str | Path):
        return Path(value)
    return Path.home() / ".reins" / "data"


def _int_arg(value: object, *, default: int) -> int:
    if value is None or value == "":
        return default
    return int(str(value).strip())


__all__ = ["screenshot", "screenshot_executor"]
