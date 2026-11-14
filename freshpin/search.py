"""Live web evidence through the Tavily Search API."""
from __future__ import annotations

import json
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
    def search(self, query: str, claim_type: str, max_results: int) -> list[Source]: ...


# Fast-changing claim types search recent news; slower ones search the general web.
_TIME_RANGE = {"pricing": "year", "api_version": "year", "company": "year"}


class TavilyClient:
    def __init__(self, api_key: str, url: str = "https://api.tavily.com/search", timeout_s: int = 30):
        self.api_key = api_key
        self.url = url
        self.timeout_s = timeout_s

    def search(self, query: str, claim_type: str, max_results: int = 5) -> list[Source]:
        if not self.api_key:
            raise SearchUnavailable("TAVILY_API_KEY is not set")
        body: dict[str, Any] = {
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
            "topic": "general",
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
