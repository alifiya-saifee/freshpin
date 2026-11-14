"""A small HTTP server (standard library only) for the web UI and JSON API."""
from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .app import FreshPin
from .config import ROOT, settings

WEB = ROOT / "web"


def make_handler(app: FreshPin):
    class Handler(BaseHTTPRequestHandler):
        server_version = "FreshPin/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
            print("  " + fmt % args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _body(self) -> dict[str, Any]:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 200_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._json(200, app.status())
            elif path == "/api/samples":
                self._json(200, app.samples())
            elif path == "/api/ledger":
                self._json(200, app.ledger.state())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            try:
                b = self._body()
                if path == "/api/check":
                    self._json(200, app.check(b.get("answer", ""), b.get("context", ""), bool(b.get("simulate_outage"))))
                elif path == "/api/load_sample":
                    self._json(200, app.load_sample(b["id"]))
                elif path == "/api/approve":
                    self._json(200, app.approve(b["id"], b.get("by", ""), b.get("note", "")))
                elif path == "/api/decisions":
                    self._json(200, app.pin_decision(b.get("title", ""), b.get("owner", ""), b.get("keys", []), b.get("by", "")))
                else:
                    self._json(404, {"error": "not found"})
            except (ValueError, KeyError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve() -> None:
    s = settings()
    app = FreshPin(s)
    httpd = ThreadingHTTPServer((s.host, s.port), make_handler(app))
    st = app.status()
    print(f"FreshPin running at http://{s.host}:{s.port}  (mode: {s.mode})")
    if s.mode == "live":
        print(f"  Bifrost: {s.bifrost_url}  reachable: {st['bifrost_reachable']}")
        print(f"  Tavily key set: {st['tavily_key_set']}")
    httpd.serve_forever()
