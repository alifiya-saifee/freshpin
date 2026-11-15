"""Source checks with fixed rules (no AI): how official is a source, and how old is it?

Used by the recheck step to pick the best reference and to explain why an answer was out of date,
e.g. "the only sources agreeing with the old claim are undated blogs".
"""
from __future__ import annotations

from datetime import date
from typing import Any
from urllib.parse import urlparse

from .scoring import HALF_LIFE_DAYS, _parse_date

# Government and official bodies. Matched against the end of the host name.
_OFFICIAL_SUFFIXES = (
    ".gov", ".gov.ae", ".gov.uk", ".gov.au", ".gov.in", ".gov.sg", ".gc.ca", ".gouv.fr", ".go.jp",
    ".gov.sa", ".mil", ".int", ".europa.eu", ".un.org", ".who.int",
)
_OFFICIAL_HOSTS = ("u.ae", "dubai.ae", "dm.ae", "rta.ae", "dld.ae", "mediaoffice.ae", "wam.ae")
_NEWS_HOSTS = (
    "reuters.com", "apnews.com", "bloomberg.com", "ft.com", "bbc.com", "bbc.co.uk", "nytimes.com",
    "wsj.com", "cnbc.com", "theguardian.com", "gulfnews.com", "khaleejtimes.com", "thenationalnews.com",
    "wam.ae", "arabianbusiness.com", "zawya.com", "techcrunch.com", "theverge.com", "arstechnica.com",
)

AUTHORITY_RANK = {"official": 3, "vendor_docs": 2, "news": 1, "other": 0}


def _host(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def authority_of(url: str) -> str:
    host = _host(url)
    if not host:
        return "other"
    if host.endswith(_OFFICIAL_SUFFIXES) or host in _OFFICIAL_HOSTS or any(host.endswith("." + h) for h in _OFFICIAL_HOSTS):
        return "official"
    if host.startswith(("docs.", "developer.", "developers.", "learn.", "api.")) or "/docs" in urlparse(url).path:
        return "vendor_docs"
    if host in _NEWS_HOSTS or any(host.endswith("." + h) for h in _NEWS_HOSTS):
        return "news"
    return "other"


def grade(source: dict[str, Any], claim_type: str, today: date) -> dict[str, Any]:
    """Adds authority, age in days and an 'old' flag (older than the claim type's half-life)."""
    d = _parse_date(source.get("published_date"))
    age = max(0, (today - d).days) if d else None
    half_life = HALF_LIFE_DAYS.get(claim_type, HALF_LIFE_DAYS["other"])
    return {**source, "authority": authority_of(source.get("url", "")), "age_days": age,
            "is_old": age is not None and age > half_life}


def rank_key(g: dict[str, Any]) -> tuple[int, int, float]:
    """Best reference first: most official, then newest (undated counts as old), then relevance."""
    age = g["age_days"] if g.get("age_days") is not None else 10**6
    return (-AUTHORITY_RANK.get(g.get("authority", "other"), 0), age, -float(g.get("relevance") or 0))


def source_notes(graded: list[dict[str, Any]], backing_old_claim: list[str]) -> list[str]:
    """Plain-language notes on the evidence, written by rules, not by a model."""
    notes = []
    official = [g for g in graded if g["authority"] == "official"]
    if official:
        notes.append(f"{len(official)} official source(s) found.")
    else:
        notes.append("No official source found; treat the result with more care.")
    backers = [g for g in graded if g["url"] in set(backing_old_claim)]
    if backers:
        weak = [g for g in backers if g["is_old"] or g["age_days"] is None or g["authority"] == "other"]
        if len(weak) == len(backers):
            notes.append(f"The {len(backers)} source(s) agreeing with the original answer are old, undated "
                         "or unofficial, which is likely where the old information comes from.")
    undated = sum(g["age_days"] is None for g in graded)
    if undated:
        notes.append(f"{undated} of {len(graded)} source(s) have no publish date.")
    return notes
