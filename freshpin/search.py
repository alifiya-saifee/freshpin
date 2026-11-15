"""Live web evidence through the Tavily Search API."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class SearchUnavailable(RuntimeError):
    pass


@dataclass
class Source:
    title: str
    url: str
    snippet: str
    published_date: str | None
    relevance: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class SearchBackend(Protocol):
    def search(self, query: str, claim_type: str, max_results: int, depth: str = "advanced") -> list[Source]: ...


# Fast-changing claim types search a recent window; news and markets use Tavily's news and finance topics.
# pricing and api_version pages are updated in place and keep old publish dates, so a time filter would drop them.
_TIME_RANGE = {"company": "year", "event": "week", "live_data": "day"}
_TOPIC = {"event": "news", "live_data": "finance"}


class TavilyClient:
    def __init__(self, api_key: str, url: str = "https://api.tavily.com/search", timeout_s: int = 30):
        self.api_key = api_key
        self.url = url
        self.timeout_s = timeout_s

    def search(self, query: str, claim_type: str, max_results: int = 5, depth: str = "advanced") -> list[Source]:
        """depth "basic" costs 1 Tavily credit, "advanced" 2."""
        if not self.api_key:
            raise SearchUnavailable("TAVILY_API_KEY is not set")
        body: dict[str, Any] = {
            "query": query,
            "search_depth": depth if depth in ("basic", "advanced") else "advanced",
            "max_results": max_results,
            "include_answer": False,
            "topic": _TOPIC.get(claim_type, "general"),
            "include_published_date": True,  # without it Tavily returns no dates except for news
        }
        if claim_type in _TIME_RANGE:
            body["time_range"] = _TIME_RANGE[claim_type]
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise SearchUnavailable(str(exc)) from exc
        out = []
        for r in data.get("results", []):
            out.append(Source(
                title=r.get("title") or r.get("url", ""),
                url=r.get("url", ""),
                snippet=(r.get("content") or "")[:700],
                published_date=r.get("published_date"),
                relevance=float(r.get("score") or 0.0),
            ))
        return out


# How long a search result may be reused, by claim type (seconds). Live values go stale fastest.
CACHE_TTL_S = {"live_data": 600, "event": 3600}
DEFAULT_CACHE_TTL_S = 86400


class CachedSearch:
    """Reuses identical searches for a while, so repeat checks and the bench don't spend Tavily credits twice.
    In memory, per process. Errors are never cached."""

    def __init__(self, inner: SearchBackend, clock=time.monotonic):
        self.inner = inner
        self.clock = clock
        self.hits = 0
        self.misses = 0
        self.credits = 0  # Tavily credits actually spent: basic 1, advanced 2
        self._store: dict[tuple[str, str, int], tuple[float, list[Source]]] = {}
        self._lock = threading.Lock()

    def search(self, query: str, claim_type: str, max_results: int = 5, depth: str = "advanced") -> list[Source]:
        key = (query.strip().lower(), claim_type, max_results, depth)
        ttl = CACHE_TTL_S.get(claim_type, DEFAULT_CACHE_TTL_S)
        now = self.clock()
        with self._lock:
            hit = self._store.get(key)
            if hit and now - hit[0] < ttl:
                self.hits += 1
                return list(hit[1])
        result = self.inner.search(query, claim_type, max_results, depth=depth)
        with self._lock:
            self.misses += 1
            self.credits += 1 if depth == "basic" else 2
            self._store[key] = (now, list(result))
        return result

    def clear(self) -> None:
        """Drop cached results (the arena clears it per answer, so no arm gets another arm's searches free)."""
        with self._lock:
            self._store.clear()
