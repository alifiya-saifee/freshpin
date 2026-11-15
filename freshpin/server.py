"""A small HTTP server (standard library only) for the web UI and JSON API."""
from __future__ import annotations

import hmac
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .app import FreshPin, LimitReached
from .config import ROOT, settings

WEB = ROOT / "web"


RESULTS = ROOT / "bench" / "results"


def bench_files() -> list[dict[str, str]]:
    """Agent-benchmark result files, newest first."""
    import json as _json
    out = []
    for f in sorted(RESULTS.glob("agentbench-*.json"), reverse=True):
        try:
            meta = _json.loads(f.read_text(encoding="utf-8")).get("meta", {})
        except (OSError, ValueError):
            continue
        out.append({"name": f.name, "label": f"{f.stem[11:]} · {meta.get('model', '?')} · "
                                             f"{len(meta.get('agents', []))} agents"})
    return out


def bench_file(name: str):
    """Only plain agentbench-*.json names inside bench/results (no paths, no traversal)."""
    import re as _re
    if not _re.fullmatch(r"agentbench-[0-9-]+\.json", name or ""):
        return None
    f = RESULTS / name
    return f if f.is_file() else None


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

        def _authorised(self, path: str) -> bool:
            """With FRESHPIN_API_TOKEN set, every /api/ call except /api/status needs the token
            (header "Authorization: Bearer <token>" or "X-FreshPin-Token: <token>")."""
            token = app.s.api_token
            if not token or not path.startswith("/api/") or path == "/api/status":
                return True
            given = self.headers.get("X-FreshPin-Token") or ""
            auth = self.headers.get("Authorization") or ""
            if auth.lower().startswith("bearer "):
                given = given or auth[7:].strip()
            if given and hmac.compare_digest(given.encode(), token.encode()):
                return True
            self._json(401, {"error": "login required: send the FreshPin API token"})
            return False

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if not self._authorised(path):
                return
            if path in ("/", "/index.html"):
                self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path in ("/bench", "/bench.html"):
                self._send(200, (WEB / "bench.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/bench/list":
                self._json(200, {"files": bench_files()})
            elif path == "/api/bench/result":
                from urllib.parse import parse_qs, urlparse
                name = (parse_qs(urlparse(self.path).query).get("file") or [""])[0]
                f = bench_file(name)
                if f is None:
                    self._json(404, {"error": "no such result"})
                else:
                    self._send(200, f.read_bytes(), "application/json; charset=utf-8")
            elif path == "/api/status":
                self._json(200, app.status())
            elif path == "/api/samples":
                self._json(200, app.samples())
            elif path == "/api/ledger":
                self._json(200, app.ledger.state())
            else:
                self._json(404, {"error": "not found"})

        def _same_origin_json(self) -> bool:
            """Blocks cross-site requests from other web pages: a POST must be JSON (forms can't send that without
            a CORS preflight, which this server never approves) and, if the browser sends an Origin, it must be
            this server's own address."""
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                self._json(415, {"error": "send JSON (Content-Type: application/json)"})
                return False
            origin = self.headers.get("Origin")
            host = self.headers.get("Host") or ""
            if origin and origin not in (f"http://{host}", f"https://{host}"):
                self._json(403, {"error": "cross-site request refused"})
                return False
            return True

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?")[0]
            if not self._authorised(path) or not self._same_origin_json():
                return
            try:
                b = self._body()
                if path == "/api/check":
                    self._json(200, app.check(b.get("answer", ""), b.get("context", ""), bool(b.get("simulate_outage"))))
                elif path == "/api/load_sample":
                    self._json(200, app.load_sample(b["id"]))
                elif path == "/api/approve":
                    if b.get("ack") is not True:  # the person must confirm they read the new facts (checked here too)
                        raise ValueError("confirm you have read the new facts and their sources (ack)")
                    self._json(200, app.approve(b["id"], b.get("by", ""), b.get("note", "")))
                elif path == "/api/decisions":
                    self._json(200, app.pin_decision(b.get("title", ""), b.get("owner", ""), b.get("keys", []), b.get("by", "")))
                else:
                    self._json(404, {"error": "not found"})
            except LimitReached as exc:
                self._json(429, {"error": str(exc)})
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
