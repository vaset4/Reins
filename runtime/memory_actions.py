"""有当前运行身份的记忆查询、更正和归档动作。

作者：xxx
时间：2026-09-15 05:40:00
"""

from __future__ import annotations

import json
from contextlib import closing, contextmanager
from collections.abc import Iterator
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from context.memory_recall import recall_memories_with_outcome
from memory.records import MemoryDetails, MemoryReplacement
from memory.files import MemoryPublicationError
from memory.safety_scan import scan
from memory.store import MemoryIndexUpdateError, MemoryStore, rebuild_memory_index
from memory.writer import MemoryWriter
from runtime.knowledge_sources import knowledge_sources, read_stored_sources
from runtime.native_actions import NativeActionContext
from runtime.tool_operations import ToolOperation
from runtime.types import RunToolsResult
from runtime.workspaces import WorkspaceStore
from tools.file_persistence import FileEditConflict

MEMORY_ACTIONS = frozenset({"memory_query", "memory_manage", "memory_note"})


class MemoryActions:
    """复用共同执行边界，仅在此绑定真实输入与事项范围。"""

    def __init__(self, context: NativeActionContext, *, data_root: Path) -> None:
        """注入运行和存储位置；传参：当前依赖与数据目录；返回：无。"""
        self.context, self.data_root = context, data_root

    def execute(self, call: ToolOperation) -> RunToolsResult:
        """执行已校验动作并区分原文提交与索引故障；传参：操作；返回：实际结果。"""
        try:
            if call.tool_name == "memory_query":
                payload = self._query(call)
            else:
                from runtime.knowledge_worker import validate_maintenance_change

                with _maintenance_publication(self.context, data_root=self.data_root):
                    validate_maintenance_change(
                        self.context, call, data_root=self.data_root
                    )
                    payload = self._change(call)
        except MemoryIndexUpdateError as exc:
            return RunToolsResult.error_result(
                action=call.tool_name,
                error=str(exc),
                meta={
                    "committed": True,
                    "memory_id": exc.memory_id,
                    "version": exc.version,
                    "index_state": "stale",
                },
            )
        except MemoryPublicationError as exc:
            return RunToolsResult.error_result(
                action=call.tool_name,
                error=str(exc),
                meta={
                    "committed": True,
                    "memory_id": exc.memory_id,
                    "version": exc.version,
                    "publication_state": f"{exc.phase}_failed",
                },
            )
        except FileEditConflict as exc:
            # 【记忆】【并发回执】存在备份时替换可能已经发生，先读实际原件，不能当作未写入重试
            conflict: dict[str, Any] = {
                "publication_state": "conflict",
                "current_file_requires_read": True,
            }
            if exc.backup_path is not None:
                conflict["conflict_backup_path"] = str(exc.backup_path)
            return RunToolsResult.error_result(
                action=call.tool_name, error=str(exc), meta=conflict
            )
        except (ValueError, FileNotFoundError) as exc:
            return RunToolsResult.error_result(action=call.tool_name, error=str(exc))
        return RunToolsResult.ok(
            action=call.tool_name,
            content=json.dumps(payload, ensure_ascii=False),
            meta=payload,
        )

    def _query(self, call: ToolOperation) -> dict[str, Any]:
        """精确查询始终读原文，相关性搜索须声明索引状态；传参：查询；返回：内容及版本来源。"""
        from runtime.knowledge_maintenance import automatic_origin
        from runtime.knowledge_validity import memory_source_validity

        args = call.args
        origin = automatic_origin(self.context.run)
        session_id = (
            str(origin["source_session_id"]) if origin else self.context.run.session_id
        )
        action = args["action"]
        if action == "search":
            scopes = self._scopes(args.get("memory_scope"))
            outcome = recall_memories_with_outcome(
                self.data_root,
                task_summary=str(args["query"]),
                task_tags=[],
                states=("active", "archived"),
                scopes=scopes,
                include_experience=True,
                include_global=origin is None,
                source_session_id=session_id,
            )
            with closing(MemoryStore(self.data_root)) as store:
                selected = [
                    item
                    for item in outcome.selected
                    if self._maintenance_visible(item.memory.details.scope)
                ]
                views = store.record_views([item.memory for item in selected])
                return {
                    "records": [
                        {**view, "score": item.score}
                        for view, item in zip(views, selected, strict=True)
                    ],
                    "index": asdict(outcome.index),
                    "skipped": [asdict(item) for item in outcome.skipped],
                }
        with closing(MemoryStore(self.data_root)) as store:
            if action == "index_status":
                return {"index": asdict(store.index_status())}
            if action in {"read", "sources"}:
                records = [
                    store.load_memory(
                        str(args["memory_id"]),
                        version=cast(str | None, args.get("version")),
                    )
                ]
                if not self._maintenance_visible(records[0].details.scope):
                    raise ValueError("memory is outside this maintenance scope")
                if action == "sources":
                    return {
                        "memory_id": records[0].memory_id,
                        "version": records[0].version,
                        "sources": read_stored_sources(
                            self.context, records[0].details.sources
                        ),
                    }
            else:
                records = store.list_memories(
                    state=cast(str | None, args.get("state")),
                    scope=self._query_scope(args.get("memory_scope")),
                    subject=cast(str | None, args.get("subject")),
                    fact_key=cast(str | None, args.get("fact_key")),
                    exact_fields=cast(dict[str, str] | None, args.get("exact_fields")),
                )
            records = [
                record
                for record in records
                if self._maintenance_visible(record.details.scope)
            ]
            views = store.record_views(records)
            # 【知识维护】【相关读取】当前版本使用前检查来源，历史查看保留当时原件和身份
            for record, view in zip(records, views, strict=True):
                if (
                    not view["historical_version"]
                    and view["effective_state"] == "active"
                ):
                    validity = memory_source_validity(
                        self.data_root, record, session_id=session_id
                    )
                    if validity:
                        view["source_validity"] = validity
            return {"records": views, "index": asdict(store.index_status())}

    def _change(self, call: ToolOperation) -> dict[str, Any]:
        """按模型动作提交版本，不固定知识生成流程；传参：动作；返回：已保存版本或明确拒写。"""
        args = call.args
        action = args.get("action", "create")
        if action == "rebuild_index":
            return {
                "rebuilt": rebuild_memory_index(self.data_root),
                "index_state": "current",
            }
        if action == "create":
            return self._create(call)
        sources = knowledge_sources(self.context, call)
        identity = str(args["memory_id"])
        with closing(MemoryStore(self.data_root)) as store:
            current = store.load_memory(identity)
            expected = str(args["expected_version"])
            if action == "revise":
                content = str(args["content"])
                if not scan(content).is_safe:
                    raise ValueError("memory correction blocked by safety scan")
                details = self._revised_details(current.details, args)
                record = store.revise_memory(
                    identity,
                    content,
                    expected_version=expected,
                    reason=str(args["reason"]),
                    sources=sources,
                    change_id=call.operation_id,
                    details=details,
                    tags=cast(list[str] | None, args.get("tags")),
                )
            elif action == "verify":
                record = store.verify_memory(
                    identity, evidence=sources, expected_version=expected
                )
            else:
                if action == "restore" and current.state != "archived":
                    raise ValueError("only an archived memory can be restored")
                states = {
                    "archive": "archived",
                    "restore": "active",
                    "withdraw": "withdrawn",
                }
                record = store.update_memory_state(
                    identity,
                    states[str(action)],
                    expected_version=expected,
                    reason=str(args.get("reason", action)),
                    sources=sources,
                    change_id=call.operation_id,
                )
            return {
                "committed": True,
                "record": store.record_view(record),
                "index_state": "current",
            }

    def _create(self, call: ToolOperation) -> dict[str, Any]:
        """区分工作便签、长期结论及原件引用；传参：创建动作；返回：持久身份和实际保存范围。"""
        args = call.args
        note = call.tool_name == "memory_note"
        scoped_call = (
            replace(call, args={**args, "source_mode": "inference"}) if note else call
        )
        sources = knowledge_sources(self.context, scoped_call)
        scope = (
            f"session:{self.context.run.session_id}"
            if note
            else self._write_scope(str(args["memory_scope"]))
        )
        kind = "note" if note else str(args.get("kind", "fact"))
        archive_ref = None
        if kind == "archive":
            if (
                not sources
                or sources[0].kind != "tool_result"
                or sources[0].result_status != "success"
            ):
                raise ValueError(
                    "archive references require a successful original-reading tool operation"
                )
            receipts = read_stored_sources(self.context, sources)
            if not all(
                item.get("available") and item.get("receipt") for item in receipts
            ):
                raise ValueError(
                    "archive references require a readable original receipt"
                )
            archive_ref = f"operation:{sources[0].reference}"
        details = MemoryDetails(
            kind=kind,
            scope=scope,
            subject=str(args.get("subject", "")),
            fact_key=str(args.get("fact_key", "")),
            sources=sources,
            archive_ref=archive_ref,
            observed_at=cast(str | None, args.get("observed_at")),
            expires_at=cast(str | None, args.get("expires_at")),
            exact_fields=cast(dict[str, str], args.get("exact_fields", {})),
            supersedes=_replacement_arguments(args.get("supersedes", [])),
        )
        with closing(
            MemoryWriter(self.data_root, config_path=self.data_root / "config.yaml")
        ) as writer:
            result = writer.write_memory(
                str(args.get("type", "fact")),
                str(args["note"] if note else args["content"]),
                cast(list[str], args.get("tags", [])),
                memory_id=f"memory-{call.operation_id}",
                details=details,
                change_id=call.operation_id,
            )
        if result.blocked_by_safety_scan:
            raise ValueError("memory blocked by safety scan")
        if result.blocked_by_conflict:
            return {
                "committed": False,
                "duplicate_of": result.conflict_with,
                "notice": result.notice,
            }
        with closing(MemoryStore(self.data_root)) as store:
            return {
                "committed": True,
                "record": store.record_view(store.load_memory(str(result.memory_id))),
                "notice": result.notice,
            }

    def _revised_details(
        self, current: MemoryDetails, args: dict[str, object]
    ) -> MemoryDetails:
        """更正元数据必须与正文一起形成修订，省略项保持原值；传参：原范围与显式变化；返回：新版范围。"""
        changes: dict[str, object] = {
            key: args[key]
            for key in (
                "subject",
                "fact_key",
                "observed_at",
                "expires_at",
                "exact_fields",
            )
            if key in args
        }
        if "memory_scope" in args:
            changes["scope"] = self._write_scope(str(args["memory_scope"]))
        if "supersedes" in args:
            changes["supersedes"] = _replacement_arguments(args["supersedes"])
        return replace(current, **cast(dict[str, Any], changes))

    def _write_scope(self, scope: str) -> str:
        """将适用范围绑定真实运行身份；传参：范围意图；返回：持久范围，拒绝项目名称冒充身份。"""
        from runtime.knowledge_maintenance import automatic_origin

        origin = automatic_origin(self.context.run)
        if origin is not None:
            if scope == "session":
                return f"session:{origin['source_session_id']}"
            if scope not in {"project", f"project:{origin['workspace_id']}"}:
                raise ValueError(
                    "automatic maintenance can only write its accepted project or original session scope"
                )
        if scope == "session":
            return f"session:{self.context.run.session_id}"
        if scope == "goal":
            if self.context.run.focus_task_id is None:
                raise ValueError("this run has no focused goal for memory")
            return f"goal:{self.context.run.focus_task_id}"
        if scope == "project" or scope.startswith("project:"):
            workspace = WorkspaceStore(self.data_root).for_session(
                self.context.run.session_id
            )
            canonical = f"project:{workspace.workspace_id}"
            if scope != "project" and scope != canonical:
                raise ValueError(
                    "project memory scope must match this session workspace; use memory_scope=project"
                )
            return canonical
        if scope != "global":
            raise ValueError("memory_scope must be global, session, goal, or project")
        return scope

    def _query_scope(self, scope: object) -> str | None:
        """解析当前范围简称并保留显式历史范围查询；参数：可选范围；返回：精确范围或不过滤。"""
        if scope is None:
            return None
        selected = str(scope)
        return (
            self._write_scope(selected)
            if selected in {"session", "goal", "project"}
            else selected
        )

    def _scopes(self, explicit: object) -> list[str]:
        """自动搜索只含当前范围，显式查询可点名其他已存资料；传参：可选范围；返回：范围列表。"""
        if explicit is not None:
            scope = str(self._query_scope(explicit))
            if not self._maintenance_visible(scope):
                raise ValueError("memory is outside this maintenance scope")
            return [scope]
        from runtime.knowledge_maintenance import automatic_origin

        origin = automatic_origin(self.context.run)
        if origin is not None:
            return [
                f"session:{origin['source_session_id']}",
                f"project:{origin['workspace_id']}",
            ]
        scopes = [f"session:{self.context.run.session_id}"]
        workspace = WorkspaceStore(self.data_root).find_for_session(
            self.context.run.session_id
        )
        if workspace is not None:
            scopes.append(f"project:{workspace.workspace_id}")
        if self.context.run.focus_task_id is not None:
            scopes.append(f"goal:{self.context.run.focus_task_id}")
        return scopes

    def _maintenance_visible(self, scope: str) -> bool:
        """自动任务只读取已选知识范围，普通显式查询保持原合同；参数：记忆范围；返回：是否可用。"""
        from runtime.knowledge_maintenance import automatic_origin

        origin = automatic_origin(self.context.run)
        return origin is None or scope in {
            f"session:{origin['source_session_id']}",
            f"project:{origin['workspace_id']}",
        }


def _replacement_arguments(value: object) -> tuple[MemoryReplacement, ...]:
    """解析模型明确指定的覆盖身份；传参：关系数组；返回：冻结引用，非法结构直接报错。"""
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(
            "supersedes must be an array of memory_id/version/reason objects"
        )
    return tuple(MemoryReplacement(**item) for item in value)


@contextmanager
def _maintenance_publication(
    context: NativeActionContext, *, data_root: Path
) -> Iterator[None]:
    """在短提交窗口内重核来源、取消与记忆版本，锁外完成模型核验；参数：运行/数据根；返回：提交边界。"""
    from runtime.knowledge_maintenance import automatic_origin
    from runtime.persistence import RuntimeStore

    if automatic_origin(context.run) is None:
        yield
        return
    # 1. 【知识维护】【发布核对】写者锁、发布锁先于事实批次，避免与原件对账的读者逆序等待
    with (
        closing(MemoryStore(data_root)) as store,
        store.locked(),
        store.publication(),
        RuntimeStore(data_root).transaction(),
    ):
        yield
