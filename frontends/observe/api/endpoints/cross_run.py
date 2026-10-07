from __future__ import annotations

from typing import Any

from frontends.observe.api.endpoints.handlers import _get_ctx, router
from frontends.observe.api.router import HttpError


@router.get("/api/cross/run-comparison")
def cross_run_comparison(
    *, _query: dict[str, list[str]] | None = None, **_: Any
) -> dict[str, Any]:
    ctx = _get_ctx()
    query = _query or {}
    run_a = (query.get("run_a") or [""])[0]
    run_b = (query.get("run_b") or [""])[0]
    if not run_a or not run_b:
        raise HttpError(400, {"error": "run_a and run_b parameters required"})
    from frontends.observe.cross_run.run_comparison import run_comparison

    return run_comparison(ctx.fact_reader, run_a, run_b)


@router.get("/api/cross/provider-model-distribution")
def cross_provider_model(**_: Any) -> dict[str, Any]:
    ctx = _get_ctx()
    from frontends.observe.cross_run.provider_model_distribution import (
        provider_model_distribution,
    )

    return provider_model_distribution(ctx.fact_reader)


@router.get("/api/cross/failure-pivot")
def cross_failure_pivot(**_: Any) -> dict[str, Any]:
    ctx = _get_ctx()
    from frontends.observe.cross_run.failure_pivot import failure_pivot

    return failure_pivot(ctx.fact_reader, ctx.data_root)


__all__ = [
    "cross_failure_pivot",
    "cross_provider_model",
    "cross_run_comparison",
]
