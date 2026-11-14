"""Freshness score: a transparent 0-100 number per claim.

The AI decides *what the sources say* (verdict + which sources support or contradict).
The score itself is computed here with a fixed formula, so the same evidence always gives the
same number and every point can be explained to a user.

    score = 0.55 * verdict_points
          + 0.25 * recency_points   (how new the newest supporting source is, vs the claim type's half-life)
          + 0.20 * agreement_points (share of sources that agree with the verdict)

Hard caps keep it honest: a stale or contradicted claim can never score above 35, and a claim
with no usable sources can never score above 50.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from email.utils import parsedate_to_datetime
from typing import Any

# How quickly each kind of fact goes out of date (days until a source is "half as convincing").
HALF_LIFE_DAYS = {
    "pricing": 45,
    "api_version": 60,
    "company": 120,
    "regulation": 240,
    "statistic": 180,
    "other": 365,
}

VERDICT_POINTS = {
    "current": 100,
    "partly_current": 60,
    "unverifiable": 45,
    "stale": 15,
    "contradicted": 5,
}

BANDS = [(70, "fresh"), (40, "check")]


@dataclass
class ScoreBreakdown:
    score: int
    band: str
    verdict_points: float
    recency_points: float
    agreement_points: float
    newest_source_age_days: int | None
    half_life_days: int
    capped_by: str | None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _parse_date(value: str | None) -> date | None:
    """Accepts ISO dates ("2026-03-10", "2026-03-10T08:00:00Z") and RFC 2822 dates, as Tavily returns."""
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(value).date()
    except (TypeError, ValueError):
        return None


def band_for(score: int) -> str:
    for threshold, name in BANDS:
        if score >= threshold:
            return name
    return "stale"


def freshness_score(verdict: str, claim_type: str, sources: list[dict[str, Any]],
                    supporting_urls: list[str], contradicting_urls: list[str],
                    today: date | None = None) -> ScoreBreakdown:
    today = today or date.today()
    half_life = HALF_LIFE_DAYS.get(claim_type, HALF_LIFE_DAYS["other"])
    v_points = float(VERDICT_POINTS.get(verdict, VERDICT_POINTS["unverifiable"]))

    # Recency: the sources that back the verdict. For a "current" verdict that is the supporting
    # sources; for stale/contradicted it is the sources showing the newer fact.
    basis_urls = set(contradicting_urls if verdict in ("stale", "contradicted") else supporting_urls)
    ages = []
    for s in sources:
        if s.get("url") in basis_urls:
            d = _parse_date(s.get("published_date"))
            if d:
                ages.append(max(0, (today - d).days))
    newest_age = min(ages) if ages else None
    if newest_age is None:
        recency = 50.0 if basis_urls else 20.0  # undated evidence is weaker but not worthless
    else:
        recency = 100.0 * math.exp(-math.log(2) * newest_age / half_life)

    # For stale/contradicted claims, strong, recent evidence of the *newer* fact should push the
    # claim's own freshness DOWN, not up.
    if verdict in ("stale", "contradicted"):
        recency = 100.0 - recency

    n_sources = len(sources)
    if n_sources:
        agree = len(basis_urls & {s.get("url") for s in sources})
        agreement = 100.0 * agree / n_sources
        if verdict in ("stale", "contradicted"):
            agreement = 100.0 - agreement
    else:
        agreement = 0.0

    raw = 0.55 * v_points + 0.25 * recency + 0.20 * agreement
    capped_by = None
    if verdict in ("stale", "contradicted") and raw > 35:
        raw, capped_by = 35, "verdict is stale or contradicted"
    if n_sources == 0 and raw > 50:
        raw, capped_by = 50, "no usable sources"
    score = int(round(max(0, min(100, raw))))
    return ScoreBreakdown(score, band_for(score), v_points, round(recency, 1), round(agreement, 1),
                          newest_age, half_life, capped_by)
