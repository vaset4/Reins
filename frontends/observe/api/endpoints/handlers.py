from __future__ import annotations

from pathlib import Path
from typing import Any

from frontends.observe.api.router import HttpError, Router
from frontends.observe.api.session_file_views import build_run_files
from frontends.observe.api.session_files import build_session_file_inventory
from frontends.observe.api.story import build_run_story, session_state_payload
from frontends.observe.panels.base import RunContext
from frontends.observe.readers.evidence_reader import EvidenceReader
from frontends.observe.readers.fact_reader import FactReader
from frontends.observe.readers.price_reader import PriceReader
from frontends.observe.registry import PanelRegistry

router = Router()


@router.get("/api/inspection")
def inspect_requests(
    *, _query: dict[str, list[str]] | None = None, **_: Any
) -> dict[str, Any]:
    """供观察网页读取统一请求目录与详情；参数：同RPC的查询字段；返回：相同分页与归属。"""
    payload: dict[str, Any] = {
        key: values[0] for key, values in (_query or {}).items() if values
    }
    for key in ("limit", "offset", "position"):
        if key in payload:
            payload[key] = int(payload[key])
    return _get_ctx().evidence_reader.query(payload)


class EndpointContext:
    """Shared state injected into endpoint handlers at server startup."""

    def __init__(
        self,
        *,
        data_root: Path,
        fact_reader: FactReader,
        evidence_reader: EvidenceReader,
        price_reader: PriceReader,
        registry: PanelRegistry,
    ) -> None:
        self.data_root = data_root
        self.fact_reader = fact_reader
        self.evidence_reader = evidence_reader
        self.price_reader = price_reader
        self.registry = registry

    def run_context(self, session_id: str, run_id: str) -> RunContext:
        return RunContext(
            data_root=self.data_root,
            session_id=session_id,
            run_id=run_id,
            fact_store=self.fact_reader.fact_store,
            session_store=self.fact_reader.session_store,
            evidence_reader=self.evidence_reader,
        )


_ctx: EndpointContext | None = None


def init_endpoints(ctx: EndpointContext) -> None:
    global _ctx
    _ctx = ctx


def _get_ctx() -> EndpointContext:
    if _ctx is None:
        raise HttpError(500, {"error": "server not initialized"})
    return _ctx


# --- /api/health ---


@router.get("/api/health")
def health(**_: Any) -> dict[str, Any]:
    ctx = _get_ctx()
    prices = ctx.price_reader.prices()
    return {
        "ok": True,
        "data_root": str(ctx.data_root),
        "prices_loaded": "_error" not in prices,
    }


# --- /api/panels ---


@router.get("/api/panels")
def list_panels(**_: Any) -> list[dict[str, Any]]:
    ctx = _get_ctx()
    return [
        {"id": d.id, "title": d.title, "section": d.section, "phase": d.phase}
        for d in ctx.registry.descriptors()
    ]


# --- /api/sessions ---


@router.get("/api/sessions")
def list_sessions(
    *, _query: dict[str, list[str]] | None = None, **_: Any
) -> dict[str, Any]:
    ctx = _get_ctx()
    query = _query or {}
    sessions = ctx.fact_reader.list_sessions(limit=50)
    q = (query.get("query") or [""])[0].strip().lower()
    status_filter = (query.get("status") or [""])[0].strip().lower()
    has_error_filter = (query.get("has_error") or [""])[0].strip().lower()
    items = []
    for s in sessions:
        if status_filter and (s.last_run_status or "").lower() != status_filter:
            continue
        if has_error_filter == "true" and s.last_checkpoint_state != "FAILED":
            continue
        if q and not _session_matches(s, q):
            continue
        items.append(
            {
                "session_id": s.session_id,
                "last_run_id": s.last_run_id,
                "status": s.last_run_status,
                "focus_task_id": s.focus_task_id,
                "summary": s.summary[:200],
                "updated_at": s.updated_at,
            }
        )
    return {"sessions": items, "count": len(items)}


def _session_matches(s: Any, q: str) -> bool:
    searchable = " ".join(
        [
            s.session_id,
            s.last_run_id or "",
            s.summary or "",
            s.focus_task_id or "",
            s.last_run_status or "",
        ]
    ).lower()
    return q in searchable


# --- /api/sessions/{session_id}/runs ---


@router.get("/api/sessions/{session_id}/runs")
def list_session_runs(*, session_id: str = "", **_: Any) -> dict[str, Any]:
    ctx = _get_ctx()
    runs = ctx.fact_reader.list_runs(session_id, limit=50)
    items = [
        {
            "run_id": r.run_id,
            "session_id": r.session_id,
            "status": r.status,
            "started_at": r.started_at,
            "updated_at": r.updated_at,
            "last_event": r.last_event,
            "task_id": r.task_id,
        }
        for r in runs
    ]
    return {"runs": items, "count": len(items)}


# --- /api/sessions/{session_id}/files ---


@router.get("/api/sessions/{session_id}/files")
def session_files(*, session_id: str = "", **_: Any) -> dict[str, Any]:
    ctx = _get_ctx()
    data = build_session_file_inventory(ctx.data_root, session_id)
    if data["status"] == "rejected":
        raise HttpError(400, {"error": "session path traversal rejected"})
    if data["status"] == "missing":
        raise HttpError(404, {"error": f"session {session_id} not found"})
    return data


# --- /api/runs/{run_id} ---


@router.get("/api/runs/{run_id}")
def get_run(*, run_id: str = "", **_: Any) -> dict[str, Any]:
    ctx = _get_ctx()
    facts = ctx.fact_reader.read_facts(run_id)
    if not facts:
        raise HttpError(404, {"error": f"run {run_id} not found"})
    session_id = str(facts[0].get("session_id", ""))
    run_ctx = ctx.run_context(session_id, run_id)
    summary = run_ctx.summary()
    return {
        "run_id": run_id,
        "session_id": session_id,
        "status": summary.status if summary else "",
        "started_at": summary.started_at if summary else "",
        "updated_at": summary.updated_at if summary else "",
        "last_event": summary.last_event if summary else "",
        "task_id": summary.task_id if summary else "",
        "total_facts": len(facts),
    }


# --- /api/runs/{run_id}/story ---


@router.get("/api/runs/{run_id}/story")
def get_run_story(*, run_id: str = "", **_: Any) -> dict[str, Any]:
    ctx = _get_ctx()
    facts = ctx.fact_reader.read_facts(run_id)
    if not facts:
        raise HttpError(404, {"error": f"run {run_id} not found"})
    session_id = str(facts[0].get("session_id", ""))
    run_ctx = ctx.run_context(session_id, run_id)
    return {
        "story": build_run_story(
            facts,
            run_ctx.summary(),
            run_ctx.state(),
            ctx.evidence_reader.read_raw_file,
            build_run_files(
                ctx.data_root,
                session_id,
                run_id,
            ),
        ),
        "session_state": session_state_payload(run_ctx.state()),
    }


# --- /api/runs/{run_id}/panels/{panel_id} ---


@router.get("/api/runs/{run_id}/panels/{panel_id}")
def get_panel(*, run_id: str = "", panel_id: str = "", **_: Any) -> dict[str, Any]:
    ctx = _get_ctx()
    panel = ctx.registry.get(panel_id)
    if panel is None:
        raise HttpError(404, {"error": f"panel {panel_id} not found"})
    facts = ctx.fact_reader.read_facts(run_id)
    if not facts:
        raise HttpError(404, {"error": f"run {run_id} not found"})
    session_id = str(facts[0].get("session_id", ""))
    run_ctx = ctx.run_context(session_id, run_id)
    return panel.build(run_ctx)


# --- /api/live ---


@router.get("/api/live")
def live(**_: Any) -> dict[str, Any]:
    ctx = _get_ctx()
    live_runs = ctx.fact_reader.find_live_runs()
    if not live_runs:
        return {"selected": None, "others": []}
    selected = live_runs[0]
    others = [
        {"run_id": r.run_id, "session_id": r.session_id, "status": r.status}
        for r in live_runs[1:]
    ]
    return {
        "selected": {
            "run_id": selected.run_id,
            "session_id": selected.session_id,
            "status": selected.status,
            "started_at": selected.started_at,
            "updated_at": selected.updated_at,
        },
        "others": others,
    }


# --- /api/raw ---


@router.get("/api/raw")
def raw_evidence(
    *, _query: dict[str, list[str]] | None = None, **_: Any
) -> dict[str, Any]:
    ctx = _get_ctx()
    query = _query or {}
    path_param = (query.get("path") or [""])[0]
    if not path_param:
        raise HttpError(400, {"error": "path parameter required"})
    result = ctx.evidence_reader.read_raw_file(path_param)
    if result["status"] == "traversal_rejected":
        raise HttpError(400, {"error": "path traversal rejected", "path": path_param})
    return result


# --- /api/config/prices ---


@router.get("/api/config/prices")
def config_prices(**_: Any) -> dict[str, Any]:
    ctx = _get_ctx()
    return ctx.price_reader.prices()


__all__ = ["EndpointContext", "init_endpoints", "router"]
