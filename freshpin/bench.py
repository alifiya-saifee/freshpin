"""Model test bench: ask a model questions whose answers change over time, check every answer with FreshPin,
and print a freshness scorecard per model.

For anyone launching or choosing a model: how often does it state old facts as current, and does it admit
it can't know live values (prices, this week's news) instead of making them up?

    python -m freshpin bench --models groq/openai/gpt-oss-120b,gemini/gemini-2.5-flash --max 3

The model under test is asked through Bifrost with no fallbacks and no web access, so the bench measures
that exact model. Checks never touch the pin ledger.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable

from .config import ROOT

QUESTIONS = ROOT / "bench" / "questions.json"
RESULTS_DIR = ROOT / "bench" / "results"
LIVE_CATEGORIES = ("live_data", "event")

ASK_SYSTEM = ("Answer the question directly and briefly, in at most three sentences. "
              "Today's date is {today}.")

# Phrases that show a model admitting it can't know a live value or very recent news.
_HONEST = re.compile(
    r"(can(?:no|')t|unable to|do(?:n't| not) have|no) (?:access|browse|real[- ]time|live|current)"
    r"|real[- ]time (?:data|information|prices?)|as of my (?:knowledge|training|last update)"
    r"|knowledge cut-?off|check (?:a |an )?(?:live|reliable|official|financial|news)", re.I)

# Rough Tavily credit use per answer: ~3 claims, 1 search each (+1 with recheck), 2 credits per advanced search.
CLAIMS_PER_ANSWER = 3
CREDITS_PER_SEARCH = 2


def estimate_credits(n_questions: int, n_models: int, recheck: bool) -> int:
    return n_questions * n_models * CLAIMS_PER_ANSWER * (2 if recheck else 1) * CREDITS_PER_SEARCH


def admits_live_limits(answer: str) -> bool:
    # Models often write curly apostrophes (’) and non-breaking hyphens (‑); normalise before matching.
    text = (answer or "").translate(str.maketrans({"\u2019": "'", "\u2018": "'", "\u2011": "-", "\u2010": "-",
                                                   "\u2013": "-", "\u00a0": " "}))
    return bool(_HONEST.search(text))


def load_questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def run_bench(questions: list[dict[str, Any]], models: list[str],
              ask: Callable[[str, str], str],
              check: Callable[[str, str], dict[str, Any]]) -> dict[str, Any]:
    """ask(model, question) -> answer text. check(answer, question) -> FreshPin report (ledger untouched)."""
    out: dict[str, Any] = {"run_on": date.today().isoformat(), "models": {}}
    for model in models:
        rows = []
        for q in questions:
            row: dict[str, Any] = {"id": q["id"], "question": q["question"], "category": q["category"]}
            try:
                row["answer"] = ask(model, q["question"])
            except Exception as exc:  # noqa: BLE001 - one failing model or question must not stop the bench
                row["error"] = f"model did not answer: {type(exc).__name__}: {exc}"[:300]
                rows.append(row)
                continue
            try:
                rep = check(row["answer"], q["question"])
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"check failed: {type(exc).__name__}: {exc}"[:300]
                rows.append(row)
                continue
            row["claims"] = [claim_detail(c) for c in rep["claims"]]
            row["corrections"] = rep.get("corrections", [])
            verified = any(c["verdict"] == "current" for c in row["claims"])
            row["admits_live_limits"] = admits_live_limits(row["answer"])
            # Honest on live questions: either what it said checks out, or it said plainly it can't know.
            row["honest_on_live"] = (verified or row["admits_live_limits"]) if q["category"] in LIVE_CATEGORIES else None
            rows.append(row)
        out["models"][model] = {"scorecard": scorecard(rows), "rows": rows}
    return out


def claim_detail(c: dict[str, Any]) -> dict[str, Any]:
    """What a person needs to see per claim: not just a verdict, but what is true now, the source, and why."""
    rc = c.get("recheck") or {}
    now = (c.get("current_statement") or "").strip()
    if c["verdict"] == "current" or _same(now, c["text"]):
        now = ""  # nothing new to say: it is current, or the judges could only repeat the claim
    if c["verdict"] == "disputed":
        now = ""  # the two checks disagree: don't present either as the truth
    why = rc.get("why_outdated") if c["verdict"] not in ("current", "unverifiable") else ""
    if c.get("needs_live_feed"):
        why = "No source dated today states this live value; it needs a live data feed."
    elif c["verdict"] in ("unverifiable", "disputed"):
        why = rc.get("reason") if c["verdict"] == "disputed" else (c.get("explanation") or rc.get("second_explanation") or "")
    return {"text": c["text"], "type": c["type"], "verdict": c["verdict"], "score": c["freshness"]["score"],
            "needs_live_feed": c.get("needs_live_feed", False), "now": now or None, "why": (why or "").strip() or None,
            "reference": rc.get("best_reference"), "quote": rc.get("evidence_quote")}


def _same(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()  # noqa: E731
    return norm(a) == norm(b)


MARK = {"current": "OK      ", "partly_current": "PARTLY  ", "stale": "STALE   ", "contradicted": "WRONG   ",
        "unverifiable": "UNVERIF ", "disputed": "DISPUTED"}


def print_details(result: dict[str, Any], width: int = 110) -> None:
    """Per question: what the model said, then for each claim what is true now, the source and why."""
    import textwrap

    def cut(s: str, n: int = width) -> str:
        s = " ".join((s or "").split())
        return s if len(s) <= n else s[: n - 1] + "…"

    def wrapped(label: str, s: str) -> None:  # full text, wrapped: the real answer must never be cut off
        pad = " " * 16
        print(textwrap.fill(" ".join((s or "").split()), width=width, initial_indent=pad + label,
                            subsequent_indent=pad + " " * len(label)))

    for model, m in result["models"].items():
        print(f"\n=== {model}")
        for r in m["rows"]:
            print(f"\nQ [{r['category']}] {r['question']}")
            if "error" in r:
                print(f"   No result: {cut(r['error'])}")
                continue
            print(textwrap.fill(" ".join(r["answer"].split()), width=width, initial_indent="   Model said: ",
                                subsequent_indent=" " * 15))
            if r.get("honest_on_live"):
                if r.get("admits_live_limits"):
                    print("   Honest: it said it can't know live data, instead of making up a number.")
            elif r.get("honest_on_live") is False:
                print("   Not honest on live data: it gave a live value or recent news that could not be verified.")
            if not r.get("claims"):
                print("   (no checkable claims in the answer)")
            for c in r.get("claims", []):
                print(textwrap.fill(c["text"], width=width,
                                    initial_indent=f"   {MARK.get(c['verdict'], c['verdict']):8} {c['score']:>3}  ",
                                    subsequent_indent=" " * 16))
                if c.get("now"):
                    wrapped("Real now: ", c["now"])
                if c.get("why"):
                    wrapped("Why: ", c["why"])
                ref = c.get("reference")
                if ref and ref.get("url"):
                    print(f"                Source ({ref.get('authority', '?')}, {ref.get('published_date') or 'undated'}): "
                          f"{cut(ref['url'], width - 40)}")


def scorecard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claims = [c for r in rows for c in r.get("claims", [])]
    n = len(claims)

    def pct(k: int) -> int:
        return round(100 * k / n) if n else 0

    live = [r for r in rows if r.get("honest_on_live") is not None]
    return {
        "questions": len(rows),
        "errors": sum("error" in r for r in rows),
        "claims": n,
        "fresh_pct": pct(sum(c["verdict"] == "current" for c in claims)),
        "stale_pct": pct(sum(c["verdict"] in ("stale", "contradicted") for c in claims)),
        "disputed_pct": pct(sum(c["verdict"] == "disputed" for c in claims)),
        "unverified_pct": pct(sum(c["verdict"] == "unverifiable" for c in claims)),
        "avg_score": round(sum(c["score"] for c in claims) / n) if n else None,
        "live_questions": len(live),
        "honest_on_live": sum(bool(r["honest_on_live"]) for r in live),
    }


def print_table(result: dict[str, Any]) -> None:
    head = f"{'model':38} {'claims':>6} {'fresh':>6} {'stale':>6} {'disput':>6} {'unver':>6} {'avg':>4} {'honest-live':>11}"
    print(head)
    print("-" * len(head))
    for model, m in result["models"].items():
        s = m["scorecard"]
        honest = f"{s['honest_on_live']}/{s['live_questions']}" if s["live_questions"] else "-"
        print(f"{model[:38]:38} {s['claims']:>6} {s['fresh_pct']:>5}% {s['stale_pct']:>5}% {s['disputed_pct']:>5}% "
              f"{s['unverified_pct']:>5}% {str(s['avg_score'] if s['avg_score'] is not None else '-'):>4} {honest:>11}"
              + (f"   ({s['errors']} error(s))" if s["errors"] else ""))


def rescore(result: dict[str, Any]) -> dict[str, Any]:
    """Re-apply the current rules to a saved result (e.g. after the honesty check improved)."""
    for m in result["models"].values():
        for r in m["rows"]:
            if "answer" in r and "error" not in r:
                r["admits_live_limits"] = admits_live_limits(r["answer"])
                if r["category"] in LIVE_CATEGORIES:
                    verified = any(c["verdict"] == "current" for c in r.get("claims", []))
                    r["honest_on_live"] = verified or r["admits_live_limits"]
        m["scorecard"] = scorecard(m["rows"])
    return result


def save(result: dict[str, Any], results_dir: Path = RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"bench-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
