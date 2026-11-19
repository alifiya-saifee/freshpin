"""Short-lived Bifrost virtual keys, for the request_cap condition: one key per run, allowed only the run's models and
capped at N requests (Bifrost governance `rate_limit.request_max_limit`). Created before the run, always deleted after.

Bifrost's management API (/api/governance/...) needs an admin Authorization header when auth is on. FreshPin reads
it from FRESHPIN_BIFROST_ADMIN, never from a file and never printed:
  "user:password"          sent as HTTP Basic (Bifrost's legacy admin login)
  "Basic ..." / "Bearer ..." sent as is (a session token or management API key)
Standard library only.
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any

# Words in a model call's error that mean Bifrost's per-key request cap answered (the injected fault), as opposed to
# a provider's own rate limit (Groq "Rate limit reached ... requests per minute"), which stays a provider limit.
CAP_WORDS = ("request_limited", "virtual key", "rate limit exceeded")


def auth_header(secret: str) -> str:
    s = (secret or "").strip()
    if not s:
        return ""
    if s.lower().startswith(("basic ", "bearer ")):
        return s
    if ":" in s:
        return "Basic " + base64.b64encode(s.encode("utf-8")).decode("ascii")
    return "Bearer " + s


def is_cap_error(error: str) -> bool:
    e = (error or "").lower()
    return any(w in e for w in CAP_WORDS)


def _find_key(doc: Any) -> dict[str, Any] | None:
    """The created key in Bifrost's reply, wherever it sits ({"virtual_key": {...}} or the object itself)."""
    if isinstance(doc, dict):
        if doc.get("id") and doc.get("value"):
            return doc
        for v in doc.values():
            found = _find_key(v)
            if found:
                return found
    return None


class BifrostKeys:
    def __init__(self, base_url: str, auth: str = "", timeout_s: float = 30.0) -> None:
        self.base = base_url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.auth = auth
        self.timeout_s = timeout_s

    @classmethod
    def from_env(cls, base_url: str) -> "BifrostKeys":
        return cls(base_url, auth_header(os.environ.get("FRESHPIN_BIFROST_ADMIN", "")))

    def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.auth:
            headers["Authorization"] = self.auth
        req = urllib.request.Request(self.base + path, method=method, headers=headers,
                                     data=json.dumps(body).encode("utf-8") if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            hint = " (set FRESHPIN_BIFROST_ADMIN to Bifrost's admin login)" if e.code in (401, 403) else ""
            raise RuntimeError(f"Bifrost {method} {path}: HTTP {e.code}{hint}: {detail}") from None
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"Bifrost {method} {path}: unreachable ({e})") from None
        return json.loads(raw) if raw.strip() else {}

    def check(self) -> None:
        """Raise RuntimeError with a clear reason when keys can't be managed (no login, Bifrost down)."""
        self._call("GET", "/api/governance/virtual-keys")

    def create_capped(self, name: str, models: list[str], max_requests: int) -> dict[str, str]:
        """A key allowed only `models` (provider/model ids) and at most `max_requests` requests. Returns {id, value}."""
        if max_requests < 1:
            raise ValueError("max_requests must be at least 1")
        by_provider: dict[str, list[str]] = {}
        for m in models:
            provider, _, model = m.partition("/")
            if not model:
                raise ValueError(f"model must be provider/model: {m!r}")
            by_provider.setdefault(provider, [])
            if model not in by_provider[provider]:
                by_provider[provider].append(model)
        body = {"name": name, "description": "FreshPin request_cap run (temporary)", "is_active": True,
                "provider_configs": [{"provider": p, "allowed_models": ms, "key_ids": ["*"]}
                                     for p, ms in by_provider.items()],
                "rate_limit": {"request_max_limit": int(max_requests), "request_reset_duration": "1h"}}
        key = _find_key(self._call("POST", "/api/governance/virtual-keys", body))
        if not key:
            raise RuntimeError("Bifrost created a key but its reply had no id/value")
        return {"id": str(key["id"]), "value": str(key["value"])}

    def delete(self, key_id: str) -> None:
        self._call("DELETE", f"/api/governance/virtual-keys/{key_id}")
