"""【请求查看】【导出】冻结已提交范围，在短事务外生成可移动阅读包。

作者：xxx
时间：2026-09-30 12:00:00
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Mapping
from uuid import uuid4

from runtime.evidence_catalog import EvidenceCatalog
from runtime.evidence_content import evidence_node, json_chunks, member, text_chunks
from runtime.file_content import confined_path
from runtime.file_records import ContentReference
from runtime.persistence import RuntimeStore, SourceSnapshot
from runtime.request_inspection import RequestInspection
from tasks.ids import utc_now

_COPY_BYTES = 65536
_TERMINAL = frozenset({"completed", "failed", "cancelled"})


class ExportCancelled(Exception):
    """用户只取消本次导出，不修改模型取消信号。"""


@dataclass(slots=True)
class _ExportJob:
    """独立导出任务的内存进度，业务事实只读取已提交文件。"""

    job_id: str
    target: Path
    status: str = "pending"
    error: str | None = None
    cutoff: dict[str, Any] = field(default_factory=dict)
    cancel: Event = field(default_factory=Event)
    thread: Thread | None = None


class RequestExporter:
    """管理显式导出任务，状态查询不等待文件生成。"""

    def __init__(self, inspection: RequestInspection) -> None:
        """复用查询身份与文件原件；参数：领域查询服务；返回：无。"""
        self.inspection = inspection
        self._jobs: dict[str, _ExportJob] = {}
        self._lock = Lock()
        self._closed = False

    def command(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """开始、查看或取消导出；参数：动作、范围和目标路径；返回：任务真实状态。"""
        action = str(payload.get("action", "start"))
        if action == "start":
            target = Path(str(payload.get("target_dir", ""))).expanduser()
            if (
                not str(payload.get("target_dir", "")).strip()
                or not target.is_absolute()
            ):
                raise ValueError("export target must be an absolute new directory")
            if target.exists():
                raise FileExistsError(f"导出目标已存在：{target}")
            scope = {
                key: str(payload.get(key) or "")
                for key in ("session_id", "run_id", "request_id")
            }
            if not scope["session_id"] or not scope["run_id"]:
                raise ValueError("export requires session_id and run_id")
            job = _ExportJob(f"export-{uuid4().hex}", target.resolve())
            thread = Thread(target=self._run, args=(job, scope), daemon=True)
            job.thread = thread
            with self._lock:
                if self._closed:
                    raise RuntimeError("request exporter is closed")
                self._jobs[job.job_id] = job
                thread.start()
        else:
            with self._lock:
                job = self._jobs[str(payload.get("job_id", ""))]
                if action == "cancel" and job.status not in _TERMINAL:
                    job.cancel.set()
                elif action != "status" and action != "cancel":
                    raise ValueError("unknown export action")
        return self._status(job)

    def close(self) -> None:
        """取消并等待所有自有读取线程退出；参数：无；返回：无，不触碰模型运行。"""
        with self._lock:
            self._closed = True
            jobs = tuple(self._jobs.values())
            for job in self._jobs.values():
                if job.status not in _TERMINAL:
                    job.cancel.set()
        for job in jobs:
            if job.thread is not None:
                job.thread.join()

    def _status(self, job: _ExportJob) -> dict[str, Any]:
        """读取原子进度；参数：任务；返回：完成前不发布成品路径。"""
        with self._lock:
            return {
                "job_id": job.job_id,
                "status": job.status,
                "error": job.error,
                "path": str(job.target) if job.status == "completed" else None,
                "target_dir": str(job.target),
                "cutoff": dict(job.cutoff),
            }

    def _run(self, job: _ExportJob, scope: Mapping[str, str]) -> None:
        """冻结、生成并原子发布完整包；参数：任务与范围；返回：失败保留明确原因。"""
        temporary = job.target.with_name(f".{job.target.name}.{job.job_id}.partial")
        try:
            with self._lock:
                job.status = "running"
            manifest = freeze_export(self.inspection, scope)
            with self._lock:
                job.cutoff = manifest["cutoff"]
            _check_cancel(job.cancel)
            job.target.parent.mkdir(parents=True, exist_ok=True)
            temporary.mkdir()
            write_export(self.inspection, manifest, temporary, job.cancel)
            # 【请求查看】【发布导出】1. 取消与最终发布共用短锁，取消先到达时不能再报告完成
            with self._lock:
                _check_cancel(job.cancel)
                if job.target.exists():
                    raise FileExistsError(f"导出目标已存在：{job.target}")
                temporary.rename(job.target)
                job.status = "completed"
        except ExportCancelled:
            with self._lock:
                job.status = "cancelled"
        except Exception as exc:
            with self._lock:
                job.status, job.error = "failed", str(exc)
        finally:
            if temporary.exists():
                # 【请求查看】【取消导出】2. 只清理本任务生成且核验仍在目标父目录的临时包
                if temporary.resolve().parent != job.target.parent.resolve():
                    raise ValueError("export temporary directory escaped its parent")
                shutil.rmtree(temporary)


def freeze_export(
    inspection: RequestInspection, scope: Mapping[str, str]
) -> dict[str, Any]:
    """冻结一次提交边界的成员与原件引用；参数：服务和范围；返回：不展开大正文的读取清单。"""
    with inspection.db.snapshot() as source:
        catalog = EvidenceCatalog(source, scope["session_id"])
        catalog.validate(scope)
        requests: list[dict[str, Any]] = []
        for row in catalog.requests:
            if row["run_id"] != scope["run_id"] or (
                scope.get("request_id") and row["request_id"] != scope["request_id"]
            ):
                continue
            attempts = [
                {
                    "metadata": {
                        key: value for key, value in item.items() if key != "materials"
                    },
                    "request": catalog.body(item["request_ref"]),
                    "response": catalog.body(item["response_ref"]),
                }
                for item in catalog.attempts
                if item["request_id"] == row["request_id"]
            ]
            requests.append({"metadata": dict(row), "attempts": attempts})
        tools = _freeze_tools(catalog, scope)
        nodes = [item["payload"] for item in tools]
        nodes.extend(
            attempt["request"]
            for request in requests
            for attempt in request["attempts"]
        )
        artifacts = _freeze_artifacts(source, nodes)
        binding = source.get("session_workspace", scope["session_id"])
        workspace = (
            source.get("workspace", binding["workspace_id"])
            if binding
            else {"retention": "not_recorded"}
        )
        runs = {row["run_id"]: row for row in catalog.runs()}
        status = runs.get(scope["run_id"], {}).get("status", "running_or_unknown")
        return {
            "scope": dict(scope),
            "data_space_id": inspection.db.data_space_id,
            "workspace": workspace,
            "cutoff": {"captured_at": utc_now(), "commit": source.sequence},
            "status_at_cutoff": status,
            "requests": requests,
            "tools": tools,
            "artifacts": artifacts,
        }


def _freeze_tools(
    catalog: EvidenceCatalog, scope: Mapping[str, str]
) -> list[dict[str, Any]]:
    """选出本运行及请求引用的历史工具；参数：目录和范围；返回：保留文件引用的工具原件。"""
    request_ids = {
        row["request_id"]
        for row in catalog.requests
        if row["run_id"] == scope["run_id"]
        and (not scope.get("request_id") or row["request_id"] == scope["request_id"])
    }
    linked = {
        item["operation_id"]
        for attempt in catalog.attempts
        if attempt["request_id"] in request_ids
        for item in attempt["materials"]
        if item.get("operation_id")
    }
    result = []
    for raw in catalog.tools:
        row = raw.payload
        if row["run_id"] != scope["run_id"] and raw.record_id not in linked:
            continue
        if (
            scope.get("request_id")
            and row["call"].get("request_id") not in request_ids
            and raw.record_id not in linked
        ):
            continue
        result.append({"operation_id": raw.record_id, "payload": evidence_node(raw)})
    return result


def _freeze_artifacts(source: SourceSnapshot, nodes: list[Any]) -> list[dict[str, Any]]:
    """冻结显式产物身份的元数据；参数：源快照及协议节点；返回：实际原件或未保留声明。"""
    identities: set[str] = set()
    for node in nodes:
        _artifact_ids(node, identities)
    result: list[dict[str, Any]] = []
    for identity in sorted(identities):
        row = source.get("artifact_records", identity)
        if row is None:
            result.append({"artifact_id": identity, "retention": "not_retained"})
        else:
            result.append(
                {
                    key: row.get(key)
                    for key in (
                        "artifact_id",
                        "task_id",
                        "type",
                        "path",
                        "bytes",
                        "content_sha256",
                        "retained_path",
                    )
                }
            )
    return result


def _artifact_ids(node: Any, identities: set[str]) -> None:
    """读取协议声明的产物关系；参数：读取树与收集结果；返回：无，不猜测正文里的路径。"""
    if isinstance(node, dict):
        for key, child in node.items():
            if key == "result_artifact_id" and isinstance(child, str):
                identities.add(child)
            elif key == "artifact_refs" and isinstance(child, list):
                identities.update(item for item in child if isinstance(item, str))
            _artifact_ids(child, identities)
    elif isinstance(node, list):
        for child in node:
            _artifact_ids(child, identities)


def _manifest_node(
    manifest: Mapping[str, Any], attachments: list[dict[str, Any]]
) -> dict[str, Any]:
    """组装原协议阅读树；参数：冻结清单和已复制附件；返回：仅正文叶持有类型引用的JSON树。"""
    root = {
        key: manifest[key]
        for key in ("scope", "data_space_id", "workspace", "cutoff", "status_at_cutoff")
    }
    root["requests"] = [
        {
            **request["metadata"],
            "attempts": [
                {
                    **attempt["metadata"],
                    "request": attempt["request"],
                    "response": attempt["response"],
                }
                for attempt in request["attempts"]
            ],
        }
        for request in manifest["requests"]
    ]
    root["tools"] = [row["payload"] for row in manifest["tools"]]
    root["attachments"] = attachments
    return root


def write_export(
    inspection: RequestInspection,
    manifest: Mapping[str, Any],
    target: Path,
    cancel: Event,
) -> None:
    """在提交锁外生成完整阅读包；参数：服务、冻结清单、目录和取消信号；返回：无。"""
    attachments = _copy_artifacts(
        inspection.db.data_root, manifest["artifacts"], target, cancel
    )
    attachments.extend(_inline_attachments(inspection.db, manifest, target, cancel))
    node = _manifest_node(manifest, attachments)
    with (target / "requests.json").open("x", encoding="utf-8", newline="\n") as output:
        for chunk in json_chunks(inspection.db, node):
            _check_cancel(cancel)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    with (target / "README.md").open("x", encoding="utf-8", newline="\n") as output:
        output.write(
            "# 请求记录\n\n"
            + json.dumps(
                {
                    key: manifest[key]
                    for key in ("scope", "workspace", "cutoff", "status_at_cutoff")
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        output.write(
            "\n\n应用层发送内容沿原规则保护秘密；完整协议见 [requests.json](requests.json)。\n\n"
        )
        for attachment in attachments:
            path = attachment.get("package_path")
            output.write(
                f"- [{path}]({path})\n"
                if path
                else f"- {json.dumps(attachment, ensure_ascii=False)}\n"
            )
        for request in manifest["requests"]:
            output.write(f"\n## 请求 {request['metadata']['request_id']}\n")
            for attempt in request["attempts"]:
                output.write(
                    f"\n### 尝试 {attempt['metadata']['attempt_index']} · {attempt['metadata']['status']}\n"
                )
                for section in ("request", "response"):
                    output.write(f"\n{section}\n\n```json\n")
                    for chunk in json_chunks(inspection.db, attempt[section]):
                        _check_cancel(cancel)
                        output.write(chunk)
                    output.write("\n```\n")
        if manifest["tools"]:
            output.write("\n## 工具操作与完整结果\n")
        for operation in manifest["tools"]:
            output.write(f"\n### {operation['operation_id']}\n\n```json\n")
            for chunk in json_chunks(inspection.db, operation["payload"]):
                _check_cancel(cancel)
                output.write(chunk)
            output.write("\n```\n\n已冻结原件见上方附件清单，路径相对于本阅读包。\n")
        output.flush()
        os.fsync(output.fileno())


def _copy_artifacts(
    root: Path, artifacts: list[dict[str, Any]], target: Path, cancel: Event
) -> list[dict[str, Any]]:
    """复制已保存字节并核验摘要；参数：数据根、元数据和导出目录；返回：包内附件位置。"""
    result = []
    for record in artifacts:
        digest, retained = record.get("content_sha256"), record.get("retained_path")
        if not digest or not retained:
            result.append({**record, "retention": "not_retained"})
            continue
        source = confined_path(root, retained)
        suffix = Path(str(record.get("path") or "")).suffix
        relative = f"attachments/{digest}{suffix}"
        destination = target / relative
        destination.parent.mkdir(exist_ok=True)
        checksum = hashlib.sha256()
        with source.open("rb") as incoming, destination.open("wb") as outgoing:
            while chunk := incoming.read(_COPY_BYTES):
                _check_cancel(cancel)
                checksum.update(chunk)
                outgoing.write(chunk)
        if checksum.hexdigest() != digest:
            raise ValueError("retained artifact content is corrupt")
        result.append({**record, "retention": "captured", "package_path": relative})
    return result


def _inline_attachments(
    store: RuntimeStore, manifest: Mapping[str, Any], target: Path, cancel: Event
) -> list[dict[str, Any]]:
    """只从实际发送的附件区块取出base64；参数：服务、快照和目录；返回：附件与外部依赖。"""
    strings: dict[str, Any] = {}
    for request in manifest["requests"]:
        for attempt in request["attempts"]:
            _collect_strings(member(attempt["request"], "request"), strings)
    result = []
    for identity, (node, mime) in strings.items():
        _check_cancel(cancel)
        chunks = (
            text_chunks(store, node)
            if isinstance(node, ContentReference)
            else iter([node])
        )
        first = next(chunks)
        if mime is None and not first.startswith("data:"):
            result.append({"source": identity, "retention": "external_reference"})
            continue
        content = first
        if mime is None:
            header, content = first.split(",", 1)
            if ";base64" not in header:
                result.append({"source": identity, "retention": "unsupported_data_uri"})
                continue
            mime = header[5:].split(";")[0]
        suffix = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "application/pdf": ".pdf",
        }.get(mime, ".bin")
        relative = (
            f"attachments/{hashlib.sha256(identity.encode()).hexdigest()}{suffix}"
        )
        destination = target / relative
        destination.parent.mkdir(exist_ok=True)
        checksum = hashlib.sha256()
        with destination.open("xb") as output:
            pending = content
            for chunk in chunks:
                _check_cancel(cancel)
                pending += chunk
                boundary = len(pending) // 4 * 4
                decoded = base64.b64decode(pending[:boundary], validate=True)
                output.write(decoded)
                checksum.update(decoded)
                pending = pending[boundary:]
            decoded = base64.b64decode(pending, validate=True)
            output.write(decoded)
            checksum.update(decoded)
        result.append(
            {
                "source": identity,
                "retention": "captured",
                "mime_type": mime,
                "sha256": checksum.hexdigest(),
                "package_path": relative,
            }
        )
    return result


def _collect_strings(node: Any, found: dict[str, Any]) -> None:
    """枚举协议消息内容；参数：发送体及结果；返回：无，工具参数与Schema不参与附件发现。"""
    if not isinstance(node, dict):
        raise ValueError("captured request must be an object")
    for section in ("messages", "input"):
        messages = node.get(section)
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            _content_attachments(
                content if isinstance(content, list) else [message], found
            )


def _content_attachments(blocks: list[Any], found: dict[str, Any]) -> None:
    """枚举明确附件及工具结果内容块；参数：内容块和结果；返回：无，不解释工具调用参数。"""
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            content = block.get("content")
            if isinstance(content, list):
                _content_attachments(content, found)
            continue
        data, mime = _attachment_data(block)
        if data is not None:
            if not isinstance(data, (str, ContentReference)):
                raise ValueError("invalid captured attachment reference")
            identity = data.sha256 if isinstance(data, ContentReference) else data
            found[identity] = (data, mime)


def _attachment_data(node: Mapping[str, Any]) -> tuple[Any, str | None]:
    """按三类供应商的附件字段定位内容；参数：内容区块；返回：字符串/引用及base64媒体类型。"""
    block_type = node.get("type")
    data, mime = None, None
    if block_type == "image_url":
        data = member(node["image_url"], "url")
    elif block_type == "input_image":
        data = (
            node.get("image_url")
            if node.get("image_url") is not None
            else node.get("file_id")
        )
    elif block_type == "input_file":
        data = (
            node.get("file_data")
            if node.get("file_data") is not None
            else node.get("file_id")
        )
    elif block_type in {"image", "document"}:
        source = node.get("source")
        if isinstance(source, dict):
            if source.get("type") == "base64":
                data, mime = source.get("data"), source.get("media_type")
                if not isinstance(mime, str):
                    raise ValueError("invalid captured base64 attachment")
            elif source.get("type") == "url":
                data = source.get("url")
            elif source.get("type") == "file":
                data = source.get("file_id")
    return data, mime


def _check_cancel(cancel: Event) -> None:
    """在读写边界响应本次导出取消；参数：取消信号；返回：无，取消时抛专属异常。"""
    if cancel.is_set():
        raise ExportCancelled()
