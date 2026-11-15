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
from .sources import grade, rank_key, source_notes

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
 "changed_on": str|null,  // when the fact changed, if a source says (YYYY-MM-DD or YYYY-MM or YYYY)
 "confirm_query": str}  // a short web query for a second, independent search (see below)
Verdicts:
- current: sources confirm the claim as of today.
- partly_current: some of it still holds, some changed.
- stale: it was true once, but a newer source shows it changed.
- contradicted: sources say it is wrong, with no sign it was ever true.
- unverifiable: sources do not address it clearly. Prefer this over guessing.
Silence is not proof: a source that simply does not mention a requirement, change, price or limit is NOT
evidence that none exists. Use "current" only when a source explicitly states the claim. For claims like
"X does not require Y", a source must explicitly say Y is not required; otherwise use "unverifiable".
confirm_query: if your verdict is current or unverifiable, write a query that would find any RECENT CHANGE to
this fact (for example add "new law", "amended", "price change", "latest version", "acquired" and the year).
Otherwise write a query for the official, newest source of the fact as it stands today."""


RECHECK_SYSTEM = """You are an independent second checker. Another model already judged a claim; your job is to
check that judgement against the sources, including new ones found by a second, targeted search.
Use ONLY the sources given, never memory. Each source shows its authority (official, vendor_docs, news,
other) and age. Prefer official and newer sources; an old or unofficial page can repeat an outdated fact.
Look hard for any source showing a RECENT CHANGE (new law, regulation, price, version, owner) that the first
judge missed. Silence is not proof: a page that does not mention a requirement or change is NOT evidence that
none exists. "current" needs a source that explicitly states the claim; otherwise say "unverifiable".
Return ONLY JSON:
{"agree": bool,  // do you agree with the FIRST verdict?
 "verdict": "current"|"partly_current"|"stale"|"contradicted"|"unverifiable",  // your own verdict
 "current_statement": str,  // the fact as the best source states it today, one sentence
 "best_source": int,  // number of the single best reference for the fact today
 "evidence_quote": str,  // one sentence COPIED WORD FOR WORD from the best source's text that backs current_statement
 "supporting": [int], "contradicting": [int],  // source numbers that agree / disagree with the CLAIM
 "why_outdated": str,  // if the claim is not current: why it is out of date or wrong (what changed, and when),
                       // and why a model or a source might still say it. Empty string if current.
 "explanation": str}  // two sentences max, cite sources as [1], [2]"""

# How verdicts compare: the two judges agree when they land in the same group.
_GROUP = {"current": "holds", "partly_current": "partial", "stale": "changed", "contradicted": "changed",
          "unverifiable": "unclear"}
_CORRECTIONS = ("stale", "contradicted", "partly_current")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'“”‘’]", "", text or "")).strip().lower().rstrip(".")


def find_quote(quote: str, sources: list[dict[str, Any]], prefer: int | None = None) -> int | None:
    """Index of the source whose text contains the quote word for word (case and spacing ignored).
    Rule: no quote, no correction. Short fragments don't count."""
    q = _norm(quote)
    if len(q) < 15:
        return None
    order = ([prefer] if prefer is not None and 0 <= prefer < len(sources) else []) + list(range(len(sources)))
    for i in order:
        s = sources[i]
        if q in _norm(f"{s.get('title', '')} {s.get('snippet', '')}"):
            return i
    return None


def decide(first: dict[str, Any], second: dict[str, Any] | None, quote_found: bool) -> tuple[str, str, str]:
    """Fixed rules (not AI) combining the two judges. Returns (final verdict, recheck status, reason)."""
    v1 = first.get("verdict", "unverifiable")
    if second is None:
        if v1 in ("stale", "contradicted"):
            return "disputed", "unavailable", "Second check could not run, so the change is not confirmed."
        return v1, "unavailable", "Second check could not run."
    v2 = second.get("verdict", "unverifiable")
    if not second.get("agree") or _GROUP.get(v1) != _GROUP.get(v2):
        return "disputed", "disputed", f"The judges disagree: first said '{v1}', second said '{v2}'."
    if v1 != "unverifiable" and not quote_found:
        # No quote, no verdict: both corrections and "still current" must be backed word for word.
        return "disputed", "disputed", "The verdict has no exact quote in its source, so it is not accepted."
    return v1, "confirmed", "Both judges agree" + (" and the quote was found in the source." if quote_found else ".")


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


def recheck_claim(claim: dict[str, Any], first: dict[str, Any], sources: list[dict[str, Any]],
                  chat: ChatBackend, models: list[str], budget: RunBudget, simulate_outage: bool,
                  today: date) -> dict[str, Any] | None:
    """The second judge: a different model reads the first verdict and every source (old and new).
    Returns None when it could not run."""
    if not sources:
        return None
    listing = "\n\n".join(
        f"[{i + 1}] {s['title']} | authority: {s['authority']} | "
        f"{('published ' + s['published_date']) if s.get('published_date') else 'undated'}\n{s['url']}\n{s['snippet']}"
        for i, s in enumerate(sources))
    user = (f"TODAY: {today.isoformat()}\nCLAIM: {claim['text']}\n"
            f"FIRST VERDICT: {first.get('verdict')}\nFIRST STATEMENT OF TODAY'S FACT: {first.get('current_statement')}\n"
            f"\nSOURCES:\n{listing}")
    try:
        return chat.chat_json(f"recheck:{claim['key']}", models, RECHECK_SYSTEM, user, budget, simulate_outage)
    except (LLMUnavailable, BudgetExceeded):
        return None


def run_check(answer: str, *, chat: ChatBackend, search: SearchBackend, ledger: Ledger,
              extractor_models: list[str], verifier_models: list[str], token_budget: int,
              max_sources: int = 5, context: str = "", simulate_outage: bool = False,
              apply_to_ledger: bool = True, today: date | None = None,
              checker_models: list[str] | None = None, recheck: bool = True) -> dict[str, Any]:
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

    # --- Recheck layer: for every claim the first judge could look at, search again (targeted at the
    # official, newest source for today's fact), then let a second model on a different provider judge
    # it with all the sources. Fixed rules then combine the two (see decide()).
    checker = checker_models or list(reversed(verifier_models))
    do_recheck = [recheck and v.get("ai_available", True) and not degraded for v in verdicts]

    def do_research(i: int) -> list[dict[str, Any]]:
        if not do_recheck[i]:
            return []
        c, v = checkable[i], verdicts[i]
        q = (v.get("confirm_query") or (v.get("current_statement") if v.get("verdict") in _CORRECTIONS else "")
             or c["text"]).strip()[:300]
        seen = {s["url"] for s in all_sources[i]}
        try:
            return [s.as_dict() for s in search.search(q, c["type"], max_sources) if s.url not in seen]
        except SearchUnavailable as exc:
            search_errors.append(f"{c['key']} (recheck): {exc}")
            return []

    with ThreadPoolExecutor(max_workers=6) as pool:
        extra_sources = list(pool.map(do_research, range(len(checkable))))

    graded_all = [[grade(s, c["type"], today) for s in srcs + extra]
                  for c, srcs, extra in zip(checkable, all_sources, extra_sources)]

    def do_second(i: int) -> dict[str, Any] | None:
        if not do_recheck[i]:
            return None
        return recheck_claim(checkable[i], verdicts[i], graded_all[i], chat, checker, budget, simulate_outage, today)

    with ThreadPoolExecutor(max_workers=6) as pool:
        seconds = list(pool.map(do_second, range(len(checkable))))

    results = []
    for i, (c, v) in enumerate(zip(checkable, verdicts)):
        srcs = graded_all[i]  # first-search sources come first, so the first judge's numbers still match

        def urls(idxs: list[Any]) -> list[str]:
            return [srcs[j - 1]["url"] for j in idxs if isinstance(j, int) and 0 < j <= len(srcs)]

        second = seconds[i]
        recheck_info: dict[str, Any] = {"status": "skipped", "reason": "AI unavailable or recheck off",
                                        "extra_sources": len(extra_sources[i])}
        verdict = v.get("verdict", "unverifiable")
        statement = v.get("current_statement") or c["text"]
        sup, con = urls(v.get("supporting", [])), urls(v.get("contradicting", []))
        if do_recheck[i]:
            best = second.get("best_source") if second else None
            best_idx = best - 1 if isinstance(best, int) and 0 < best <= len(srcs) else None
            quote = (second or {}).get("evidence_quote", "")
            q_idx = find_quote(quote, srcs, best_idx) if second else None
            verdict, status, reason = decide(v, second, q_idx is not None)
            if status == "confirmed":
                sup = sorted(set(sup) | set(urls(second.get("supporting", []))))
                con = sorted(set(con) | set(urls(second.get("contradicting", []))))
                if verdict in _CORRECTIONS and second.get("current_statement"):
                    statement = second["current_statement"]
            ref_idx = q_idx if q_idx is not None else (best_idx if best_idx is not None else
                                                        (min(range(len(srcs)), key=lambda j: rank_key(srcs[j]))
                                                         if srcs else None))
            ref = srcs[ref_idx] if ref_idx is not None else None
            recheck_info = {
                "status": status, "reason": reason, "checker_verdict": (second or {}).get("verdict"),
                "extra_sources": len(extra_sources[i]),
                "best_reference": ({k: ref.get(k) for k in ("title", "url", "published_date", "authority", "age_days")}
                                   if ref else None),
                "evidence_quote": quote if q_idx is not None else None,
                "quote_found": q_idx is not None,
                "why_outdated": (second or {}).get("why_outdated") or "",
                "second_explanation": (second or {}).get("explanation", ""),
                # Sources that agreed with the ORIGINAL claim matter only when the claim is not current.
                "source_notes": source_notes(srcs, urls(v.get("supporting", []))
                                             if verdict in ("stale", "contradicted", "partly_current", "disputed")
                                             else []),
            }
        fs = freshness_score(verdict, c["type"], srcs, sup, con, today)
        claim = {
            **c, "verdict": verdict, "first_verdict": v.get("verdict", "unverifiable"),
            "current_statement": statement,
            "explanation": v.get("explanation", ""), "changed_on": v.get("changed_on"),
            "supporting_urls": sup, "contradicting_urls": con, "sources": srcs,
            "freshness": fs.as_dict(), "ai_available": v.get("ai_available", True) and not degraded,
            "recheck": recheck_info,
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
            "rechecked": sum(r["recheck"]["status"] in ("confirmed", "disputed") for r in results),
            "confirmed": sum(r["recheck"]["status"] == "confirmed" for r in results),
            "disputed": sum(r["verdict"] == "disputed" for r in results),
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
        "tavily_searches": len(checkable) + sum(do_recheck),
        "elapsed_ms": int((time.monotonic() - t0) * 1000),
    }
