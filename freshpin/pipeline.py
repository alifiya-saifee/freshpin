"""The check pipeline: answer -> claims -> live evidence -> verdicts -> scores -> ledger effects."""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

from .ledger import Ledger
from .llm import BudgetExceeded, ChatBackend, LLMUnavailable, RunBudget
from .scoring import freshness_score
from .search import SearchBackend, SearchUnavailable

CLAIM_TYPES = ["pricing", "api_version", "regulation", "company", "statistic", "other"]

EXTRACT_SYSTEM = """You split an AI-generated answer into atomic factual claims and judge which could go out of date.
Return ONLY JSON: {"claims": [{"text": str, "key": str, "type": str, "time_sensitive": bool, "search_query": str}]}
Rules:
- One claim = one checkable fact, rewritten to stand alone (no "it", "this").
- type is one of: pricing, api_version, regulation, company, statistic, other.
- time_sensitive is true when the fact could change over months or years (prices, versions, laws, who owns or runs a company, counts, limits).
- key is a stable snake_case id for the FACT, not the wording, e.g. "dubai.building_safety_certificate.required_for_old_buildings".
  If a claim is about the same fact as one in KNOWN_FACTS, reuse that key exactly.
- search_query is a short web query that would find the CURRENT value of this fact. Include the place and topic; do not include the claimed value.
- Skip opinions, advice and definitions."""

VERIFY_SYSTEM = """You check one claim against web sources found today. Use ONLY the sources given, never memory.
Return ONLY JSON:
{"verdict": "current"|"partly_current"|"stale"|"contradicted"|"unverifiable",
 "current_statement": str,  // the fact as the sources state it today, one sentence; repeat the claim if it is current
 "supporting": [int], "contradicting": [int],  // source numbers that agree / disagree with the CLAIM
 "explanation": str,  // two sentences max, cite sources as [1], [2]
 "changed_on": str|null}  // when the fact changed, if a source says (YYYY-MM-DD or YYYY-MM or YYYY)
Verdicts:
- current: sources confirm the claim as of today.
- partly_current: some of it still holds, some changed.
- stale: it was true once, but a newer source shows it changed.
- contradicted: sources say it is wrong, with no sign it was ever true.
- unverifiable: sources do not address it clearly. Prefer this over guessing."""


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60] or "claim"


def heuristic_claims(answer: str) -> list[dict[str, Any]]:
    """Used only when every model is down: keep sentences that look time-sensitive."""
    pattern = re.compile(r"\d|\b(price|cost|version|law|regulation|require|ceo|owned|acquired|latest|current)\b", re.I)
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", answer.strip()):
        if len(sent) > 20 and pattern.search(sent):
            out.append({"text": sent, "key": _slug(sent), "type": "other", "time_sensitive": True,
                        "search_query": sent[:200]})
    return out[:8]


def extract_claims(answer: str, context: str, chat: ChatBackend, models: list[str], ledger: Ledger,
                   budget: RunBudget, simulate_outage: bool, today: date) -> tuple[list[dict[str, Any]], bool]:
    known = ledger.known_facts()
    user = (f"TODAY: {today.isoformat()}\nCONTEXT: {context or 'none'}\n"
            f"KNOWN_FACTS: {json.dumps(known, ensure_ascii=False)}\n\nANSWER:\n{answer}")
    try:
        data = chat.chat_json("extract", models, EXTRACT_SYSTEM, user, budget, simulate_outage)
        claims = data.get("claims", [])
        degraded = False
    except (LLMUnavailable, BudgetExceeded):
        claims, degraded = heuristic_claims(answer), True
    clean = []
    for c in claims:
        if not c.get("text"):
            continue
        c["type"] = c.get("type") if c.get("type") in CLAIM_TYPES else "other"
        c["key"] = c.get("key") or _slug(c["text"])
        c["search_query"] = c.get("search_query") or c["text"]
        clean.append(c)
    return clean, degraded


def verify_claim(claim: dict[str, Any], sources: list[dict[str, Any]], chat: ChatBackend, models: list[str],
                 budget: RunBudget, simulate_outage: bool, today: date) -> dict[str, Any]:
    if not sources:
        return {"verdict": "unverifiable", "current_statement": claim["text"], "supporting": [], "contradicting": [],
                "explanation": "No live sources were found for this claim.", "changed_on": None, "ai_available": True}
    listing = "\n\n".join(
        f"[{i + 1}] {s['title']} ({s.get('published_date') or 'undated'})\n{s['url']}\n{s['snippet']}"
        for i, s in enumerate(sources))
    user = f"TODAY: {today.isoformat()}\nCLAIM: {claim['text']}\n\nSOURCES:\n{listing}"
    try:
        out = chat.chat_json(f"verify:{claim['key']}", models, VERIFY_SYSTEM, user, budget, simulate_outage)
        out["ai_available"] = True
        return out
    except (LLMUnavailable, BudgetExceeded) as exc:
        return {"verdict": "unverifiable", "current_statement": claim["text"], "supporting": [], "contradicting": [],
                "explanation": f"AI verifier unavailable ({type(exc).__name__}). Showing raw evidence only; "
                               "nothing in the ledger was changed.", "changed_on": None, "ai_available": False}


def run_check(answer: str, *, chat: ChatBackend, search: SearchBackend, ledger: Ledger,
              extractor_models: list[str], verifier_models: list[str], token_budget: int,
              max_sources: int = 5, context: str = "", simulate_outage: bool = False,
              apply_to_ledger: bool = True, today: date | None = None) -> dict[str, Any]:
    today = today or date.today()
    t0 = time.monotonic()
    budget = RunBudget(limit_tokens=token_budget)
    claims, degraded = extract_claims(answer, context, chat, extractor_models, ledger, budget, simulate_outage, today)
    checkable = [c for c in claims if c.get("time_sensitive", True)]
    skipped = [c for c in claims if not c.get("time_sensitive", True)]

    search_errors: list[str] = []

    def do_search(c: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return [s.as_dict() for s in search.search(c["search_query"], c["type"], max_sources)]
        except SearchUnavailable as exc:
            search_errors.append(f"{c['key']}: {exc}")
            return []

    with ThreadPoolExecutor(max_workers=6) as pool:
        all_sources = list(pool.map(do_search, checkable))

    def do_verify(pair: tuple[dict[str, Any], list[dict[str, Any]]]) -> dict[str, Any]:
        c, srcs = pair
        return verify_claim(c, srcs, chat, verifier_models, budget, simulate_outage, today)

    with ThreadPoolExecutor(max_workers=6) as pool:
        verdicts = list(pool.map(do_verify, zip(checkable, all_sources)))

    results = []
    for c, srcs, v in zip(checkable, all_sources, verdicts):
        def urls(idxs: list[Any]) -> list[str]:
            return [srcs[i - 1]["url"] for i in idxs if isinstance(i, int) and 0 < i <= len(srcs)]
        sup, con = urls(v.get("supporting", [])), urls(v.get("contradicting", []))
        verdict = v.get("verdict", "unverifiable")
        fs = freshness_score(verdict, c["type"], srcs, sup, con, today)
        claim = {
            **c, "verdict": verdict, "current_statement": v.get("current_statement") or c["text"],
            "explanation": v.get("explanation", ""), "changed_on": v.get("changed_on"),
            "supporting_urls": sup, "contradicting_urls": con, "sources": srcs,
            "freshness": fs.as_dict(), "ai_available": v.get("ai_available", True) and not degraded,
        }
        claim["ledger_effect"] = (ledger.record_check(claim)
                                  if apply_to_ledger and claim["ai_available"] else None)
        results.append(claim)

    traces = [t.as_dict() for t in budget.traces]
    return {
        "checked_on": today.isoformat(),
        "claims": results,
        "skipped": [{"text": c["text"], "reason": "not time-sensitive"} for c in skipped],
        "degraded": degraded,
        "search_errors": search_errors,
        "summary": {
            "claims": len(results),
            "fresh": sum(r["freshness"]["band"] == "fresh" for r in results),
            "check": sum(r["freshness"]["band"] == "check" for r in results),
            "stale": sum(r["freshness"]["band"] == "stale" for r in results),
            "decisions_reopened": sorted({d for r in results for d in ((r["ledger_effect"] or {}).get("reopened") or [])}),
            "decisions_at_risk": sorted({d for r in results for d in ((r["ledger_effect"] or {}).get("at_risk") or [])}),
        },
        "gateway": {
            "calls": len(traces),
            "failed_attempts": sum(not t["ok"] for t in traces),
            "fallbacks_used": sum(t["fallback_used"] or t["client_failover"] for t in traces if t["ok"]),
            "tokens": budget.used_tokens, "token_budget": token_budget,
            "traces": traces,
        },
        "tavily_searches": len(checkable),
        "elapsed_ms": int((time.monotonic() - t0) * 1000),
    }
