from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artifacts.store import ArtifactStore
from context.artifact_ref import store_large_output
from tools.types import ToolError, ToolErrorCategory

OCR_THRESHOLD = 4096

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_CN_PHONE_RE = re.compile(r"\b1[3-9]\d{9}\b")
_PASSWORD_WORDS = {"password", "pwd", "passcode", "密码", "口令"}


@dataclass(frozen=True, slots=True)
class OcrBox:
    text: str
    x: int
    y: int
    w: int
    h: int

    def region(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


def ocr(
    artifact_id: str,
    lang: str = "eng",
    *,
    data_root: Path | str,
) -> dict[str, object]:
    path = ArtifactStore(data_root).read_path(artifact_id)
    pytesseract = _pytesseract()
    text = str(pytesseract.image_to_string(str(path), lang=lang))
    ref = store_large_output(
        data_root,
        _task_id_for_artifact(data_root, artifact_id),
        text,
        type="file_dump",
        ext="txt",
        summary=text[:500].strip() or "ocr text",
    )
    if ref is None:
        return {"text": text, "lang": lang}
    return {
        "artifact_id": ref.artifact_id,
        "summary": ref.summary[:500],
        "key_excerpt": ref.key_excerpt,
        "lang": lang,
    }


def detect_sensitive_regions(
    artifact_id: str,
    *,
    data_root: Path | str,
    lang: str = "eng",
) -> list[dict[str, int]]:
    path = ArtifactStore(data_root).read_path(artifact_id)
    boxes = _ocr_boxes(path, lang=lang)
    return _sensitive_regions(boxes)


def ocr_executor(args: dict[str, object]) -> object:
    try:
        return ocr(
            str(args.get("artifact_id", "")).strip(),
            str(args.get("lang", "eng")).strip() or "eng",
            data_root=_data_root_arg(args),
        )
    except FileNotFoundError as exc:
        return ToolError(ToolErrorCategory.INVALID_INPUT, str(exc), retryable=False)
    except Exception as exc:
        return ToolError(ToolErrorCategory.UNKNOWN, str(exc), retryable=False)


def _ocr_boxes(path: Path, *, lang: str) -> list[OcrBox]:
    pytesseract = _pytesseract()
    output = getattr(pytesseract, "Output", None)
    dict_type = getattr(output, "DICT", "dict")
    data = pytesseract.image_to_data(str(path), lang=lang, output_type=dict_type)
    texts = [str(item) for item in data.get("text", [])]
    left = [int(item) for item in data.get("left", [])]
    top = [int(item) for item in data.get("top", [])]
    width = [int(item) for item in data.get("width", [])]
    height = [int(item) for item in data.get("height", [])]
    boxes: list[OcrBox] = []
    for idx, text in enumerate(texts):
        if not text.strip():
            continue
        boxes.append(
            OcrBox(
                text=text.strip(),
                x=left[idx],
                y=top[idx],
                w=width[idx],
                h=height[idx],
            )
        )
    return boxes


def _sensitive_regions(boxes: list[OcrBox]) -> list[dict[str, int]]:
    joined, spans = _joined_text(boxes)
    selected: list[OcrBox] = []
    for regex in (_EMAIL_RE, _CARD_RE, _CN_PHONE_RE):
        for match in regex.finditer(joined):
            selected.extend(_boxes_for_span(boxes, spans, match.start(), match.end()))
    for idx, box in enumerate(boxes):
        if box.text.strip(":：").lower() in _PASSWORD_WORDS:
            selected.append(box)
            if idx + 1 < len(boxes):
                selected.append(boxes[idx + 1])
    return [_merge_region(item.region()) for item in _dedupe_boxes(selected)]


def _joined_text(boxes: list[OcrBox]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for box in boxes:
        if parts:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(box.text)
        cursor += len(box.text)
        spans.append((start, cursor))
    return "".join(parts), spans


def _boxes_for_span(
    boxes: list[OcrBox],
    spans: list[tuple[int, int]],
    start: int,
    end: int,
) -> list[OcrBox]:
    return [
        box
        for box, (box_start, box_end) in zip(boxes, spans, strict=True)
        if box_start < end and box_end > start
    ]


def _dedupe_boxes(boxes: list[OcrBox]) -> list[OcrBox]:
    seen: set[tuple[int, int, int, int]] = set()
    deduped: list[OcrBox] = []
    for box in boxes:
        key = (box.x, box.y, box.w, box.h)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(box)
    return deduped


def _merge_region(region: dict[str, int]) -> dict[str, int]:
    pad = 3
    return {
        "x": max(0, region["x"] - pad),
        "y": max(0, region["y"] - pad),
        "w": region["w"] + pad * 2,
        "h": region["h"] + pad * 2,
    }


def _task_id_for_artifact(data_root: Path | str, artifact_id: str) -> str:
    record = ArtifactStore(data_root).load_artifact(artifact_id)
    if record is None:
        raise FileNotFoundError(artifact_id)
    return record.task_id


def _pytesseract() -> Any:
    import pytesseract  # type: ignore[import-untyped]

    return pytesseract


def _data_root_arg(args: dict[str, object]) -> Path:
    value = args.get("__data_root__")
    if isinstance(value, str | Path):
        return Path(value)
    return Path.home() / ".reins" / "data"


__all__ = ["OcrBox", "detect_sensitive_regions", "ocr", "ocr_executor"]
