from __future__ import annotations

from pathlib import Path
from typing import Any

from artifacts.store import ArtifactStore
from tasks.ids import new_ulid
from tools.types import ToolError, ToolErrorCategory
from tools.vision.ocr import detect_sensitive_regions


def redact(
    artifact_id: str,
    regions: list[dict[str, object]] | None = None,
    *,
    data_root: Path | str,
    lang: str = "eng",
) -> dict[str, object]:
    store = ArtifactStore(data_root)
    source = store.load_artifact(artifact_id)
    if source is None:
        raise FileNotFoundError(artifact_id)
    image_path = store.read_path(artifact_id)
    regions_to_redact = (
        detect_sensitive_regions(artifact_id, data_root=data_root, lang=lang)
        if regions is None
        else [_normalize_region(item) for item in regions]
    )

    image_module = _image_module()
    draw_module = _draw_module()
    image = image_module.open(image_path).convert("RGB")
    draw = draw_module.Draw(image)
    for region in regions_to_redact:
        x = region["x"]
        y = region["y"]
        w = region["w"]
        h = region["h"]
        draw.rectangle((x, y, x + w, y + h), fill="black")

    redacted_artifact_id = f"art-{new_ulid()}"
    ArtifactStore(data_root)
    relative_path = f"assets/{redacted_artifact_id}.png"
    output_path = Path(data_root) / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    store.create_artifact(
        source.task_id,
        "screenshot",
        relative_path,
        f"redacted screenshot from {artifact_id}",
        output_path.stat().st_size,
        artifact_id=redacted_artifact_id,
        workspace_id=source.workspace_id,
        session_id=source.session_id,
    )
    return {
        "redacted_artifact_id": redacted_artifact_id,
        "regions_redacted": len(regions_to_redact),
    }


def redact_executor(args: dict[str, object]) -> object:
    try:
        return redact(
            str(args.get("artifact_id", "")).strip(),
            _regions_arg(args.get("regions")),
            data_root=_data_root_arg(args),
            lang=str(args.get("lang", "eng")).strip() or "eng",
        )
    except (FileNotFoundError, ValueError) as exc:
        return ToolError(ToolErrorCategory.INVALID_INPUT, str(exc), retryable=False)
    except Exception as exc:
        return ToolError(ToolErrorCategory.UNKNOWN, str(exc), retryable=False)


def _regions_arg(value: object) -> list[dict[str, object]] | None:
    if value is None or value == "":
        return None
    if not isinstance(value, list):
        raise ValueError("regions must be a list")
    return [dict(item) for item in value if isinstance(item, dict)]


def _normalize_region(region: dict[str, object]) -> dict[str, int]:
    values = {key: int(str(region[key])) for key in ("x", "y", "w", "h")}
    if values["w"] <= 0 or values["h"] <= 0:
        raise ValueError("region width and height must be positive")
    return values


def _data_root_arg(args: dict[str, object]) -> Path:
    value = args.get("__data_root__")
    if isinstance(value, str | Path):
        return Path(value)
    return Path.home() / ".reins" / "data"


def _image_module() -> Any:
    from PIL import Image

    return Image


def _draw_module() -> Any:
    from PIL import ImageDraw

    return ImageDraw


__all__ = ["redact", "redact_executor"]
