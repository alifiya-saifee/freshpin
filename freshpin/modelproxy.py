"""A pass-through in front of the model gateway (Bifrost), for gateway-side conditions.

The agent talks to this proxy exactly as it would talk to Bifrost (same paths, same headers). For a run set to use
gateway fallbacks, the proxy adds Bifrost's own per-request `"fallbacks": [...]` list to each JSON request body, so
the GATEWAY can rescue a rejected model even when the agent has no fallback logic of its own. Every other request
passes through unchanged. Runs are told apart by the `x-bf-session-id` header FreshPin gives each run.

Responses are returned whole (a streamed response arrives in one piece). Standard library only.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

HOP_HEADERS = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "accept-encoding",
               "proxy-connection", "te", "upgrade"}


class ModelProxy:
    def __init__(self, upstream: str, timeout_s: float = 180.0) -> None:
        """upstream: the gateway's root URL, e.g. http://localhost:8080 (paths such as /v1/... are kept)."""
        u = urlsplit(upstream)
        if u.scheme not in ("http", "https") or not u.netloc:
            raise ValueError(f"upstream must be an http(s) URL: {upstream!r}")
        self.upstream = f"{u.scheme}://{u.netloc}"
        self.timeout_s = timeout_s
        self._runs: dict[str, list[str]] = {}
        self._log: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self.url = ""

    def set_run(self, run_id: str, fallbacks: list[str] | None = None) -> None:
        """fallbacks: the list to add to this run's requests (None or empty: pass through unchanged)."""
        with self._lock:
            self._runs[run_id] = list(fallbacks or [])
            self._log.setdefault(run_id, [])

    def calls(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(c) for c in self._log.get(run_id, [])]

    def agent_base_url(self, gateway_base_url: str) -> str:
        """The base URL to hand the agent instead of the gateway's: same path, this proxy's host."""
        return self.url + (urlsplit(gateway_base_url).path or "")

    def _forward(self, method: str, path: str, headers: dict[str, str], body: bytes | None) -> tuple[int, str, bytes, dict]:
        run_id = headers.get("x-bf-session-id", "")
        with self._lock:
            fallbacks = list(self._runs.get(run_id) or [])
        note: dict[str, Any] = {"t": time.time(), "method": method, "path": path, "fallbacks_added": [], "model": None}
        if body and fallbacks:
            try:
                doc = json.loads(body)
            except ValueError:
                doc = None
            if isinstance(doc, dict) and "model" in doc:
                note["model"] = doc.get("model")
                if not doc.get("fallbacks"):                       # an agent's own list is left as it is
                    doc["fallbacks"] = fallbacks
                    note["fallbacks_added"] = fallbacks
                    body = json.dumps(doc).encode("utf-8")
        req = urllib.request.Request(self.upstream + path, data=body, method=method,
                                     headers={k: v for k, v in headers.items() if k.lower() not in HOP_HEADERS})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                status, ctype, data = r.status, r.headers.get("Content-Type", "application/json"), r.read()
        except urllib.error.HTTPError as e:
            status, ctype, data = e.code, e.headers.get("Content-Type", "application/json"), e.read()
        except (urllib.error.URLError, OSError) as e:
            status, ctype = 502, "application/json"
            data = json.dumps({"error": {"message": f"model proxy: gateway unreachable ({e})"}}).encode("utf-8")
        note["status"] = status
        return status, ctype, data, note

    def start(self, host: str = "127.0.0.1", port: int = 0) -> "ModelProxy":
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:  # quiet
                pass

            def _handle(self, method: str) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else None
                headers_lc = {k.lower(): v for k, v in self.headers.items()}   # one copy of each header
                status, ctype, data, note = proxy._forward(method, self.path, headers_lc, body)
                run_id = headers_lc.get("x-bf-session-id", "")
                with proxy._lock:
                    proxy._log.setdefault(run_id, []).append(note)
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:  # noqa: N802
                self._handle("POST")

            def do_GET(self) -> None:  # noqa: N802
                self._handle("GET")

        self._httpd = ThreadingHTTPServer((host, port), Handler)
        self._httpd.daemon_threads = True
        self.url = f"http://{host}:{self._httpd.server_address[1]}"
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
