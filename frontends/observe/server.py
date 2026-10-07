from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from frontends.observe.api.endpoints import cross_run as cross_run_handlers
from frontends.observe.api.endpoints import handlers as endpoint_handlers
from frontends.observe.api.endpoints.handlers import EndpointContext, init_endpoints
from frontends.observe.api.router import HttpError, Router
from frontends.observe.path_security import resolve_relative_under
from frontends.observe.readers.evidence_reader import EvidenceReader
from frontends.observe.readers.fact_reader import FactReader
from frontends.observe.readers.price_reader import PriceReader
from frontends.observe.registry import build_default_registry

STATIC_ROOT = Path(__file__).parent / "static"
_REGISTERED_ENDPOINT_MODULES = (cross_run_handlers,)
STATIC_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


def build_handler(router: Router, data_root: Path) -> type[BaseHTTPRequestHandler]:
    class ObserveHandler(BaseHTTPRequestHandler):
        server_version = "ObserveDashboard/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(f"[observe] {self.address_string()} - {fmt % args}\n")

        def do_GET(self) -> None:  # noqa: N802
            try:
                self._dispatch_get()
            except HttpError as exc:
                self._write_json(exc.status, exc.body)
            except Exception as exc:
                self._write_json(500, {"error": "internal", "detail": str(exc)})

        def _dispatch_get(self) -> None:
            split = urlsplit(self.path)
            path = split.path
            query = parse_qs(split.query, keep_blank_values=True)

            if path == "/":
                self._serve_static("index.html")
                return
            if path.startswith("/static/"):
                self._serve_static(path[len("/static/") :])
                return

            resolved = router.resolve("GET", path)
            if resolved is None:
                self._write_json(404, {"error": "not_found", "path": path})
                return
            handler, params = resolved
            payload = handler(_query=query, **params)
            self._write_json(200, payload)

        def _serve_static(self, relative: str) -> None:
            safe = _safe_static(relative)
            if safe is None or not safe.is_file():
                self._write_text(404, "not found")
                return
            mime = STATIC_MIME.get(safe.suffix, "application/octet-stream")
            data = safe.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _write_json(self, status: int, payload: Any) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _write_text(self, status: int, message: str) -> None:
            data = message.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ObserveHandler


def _safe_static(relative: str) -> Path | None:
    return resolve_relative_under(STATIC_ROOT, relative)


def _resolve_data_root(raw: str) -> Path:
    path = Path(raw).resolve()
    if not path.is_dir():
        raise SystemExit(f"data-root not found: {path}")
    return path


def _resolve_config_root(data_root: Path, override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    candidate = data_root.parent / "config"
    if candidate.is_dir():
        return candidate
    return data_root.parent / "config"


def _build_endpoint_context(data_root: Path, config_root: Path) -> EndpointContext:
    fact_reader = FactReader(data_root)
    evidence_reader = EvidenceReader(data_root)
    price_reader = PriceReader(config_root)
    registry = build_default_registry()
    return EndpointContext(
        data_root=data_root,
        fact_reader=fact_reader,
        evidence_reader=evidence_reader,
        price_reader=price_reader,
        registry=registry,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="frontends.observe.server",
        description="Reins observability dashboard (Agent Flight Recorder).",
    )
    parser.add_argument("--data-root", required=True, help="Path to .reins/data")
    parser.add_argument("--config-root", default=None, help="Path to .reins/config")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        sys.stderr.write(
            f"[observe] refusing to bind non-loopback host {args.host!r}; "
            "MVP is loopback-only.\n"
        )
        return 2

    data_root = _resolve_data_root(args.data_root)
    config_root = _resolve_config_root(data_root, args.config_root)
    ctx = _build_endpoint_context(data_root, config_root)
    init_endpoints(ctx)

    handler_cls = build_handler(endpoint_handlers.router, data_root)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    sys.stderr.write(
        f"[observe] listening on http://{args.host}:{args.port}/ "
        f"data_root={data_root}\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("[observe] shutting down\n")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
