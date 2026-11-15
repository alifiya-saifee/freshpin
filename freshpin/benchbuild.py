"""Build a fresh-events benchmark from this week's news, with a gold answer and an exact quote for each item.

Models fail most on things that happened after their training, like "what happened on Friday?". This builds
a dated question set from real, recent news, so any model (and FreshPin itself) can be scored against it:

    python -m freshpin bench-build --n 8            # writes bench/fresh-<date>.json (marked review: auto)
    python -m freshpin bench --questions bench/fresh-<date>.json --models ...

Every item keeps only a quote found word for word in the article text, and is marked "review: auto": a person
should read the set before it is used for anything that matters.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Callable

from .config import ROOT
from .llm import BudgetExceeded, ChatBackend, LLMUnavailable, RunBudget
from .pipeline import find_quote
from .scoring import _parse_date
from .search import SearchBackend, SearchUnavailable

BENCH_DIR = ROOT / "bench"
NEWS_QUERIES = ["major service outage", "product launch announced", "acquisition announced", "new law issued",
                "price change announced", "data breach disclosed", "central bank decision"]
CREDITS_PER_SEARCH = 2

BUILD_SYSTEM = """You turn ONE news article into ONE benchmark question about a recent, specific event.
Return ONLY JSON: {"usable": bool, "question": str, "gold_answer": str, "evidence_quote": str}
Rules:
- usable is false if the article is not about a specific, dated event (opinion, listicle, guide, ad).
- question: answerable only by someone who knows about this event. Say when (a date or "this week") and give
  just enough context to identify it, but do NOT put any fact from gold_answer in the question: ask for the
  who / what / how many / outcome instead. Never say "the article".
- gold_answer: one or two sentences, only facts stated in the article.
- evidence_quote: one sentence COPIED WORD FOR WORD from the article text that supports gold_answer."""

GOLD_SYSTEM = """You compare a model's answer with a gold answer taken from a dated news source.
Return ONLY JSON: {"result": "correct"|"partly"|"wrong"|"admitted_unknown", "reason": str}
- correct: the answer states the gold answer's key facts and nothing that contradicts them (hedging about
  OTHER possibilities is fine if the key fact is stated).
- partly: some key facts right, some missing or vague.
- wrong: it states facts that contradict the gold answer, or makes up an event.
- admitted_unknown: it says plainly it does not know or cannot access recent information.
reason: one sentence."""


def estimate_build_credits(queries: list[str] = NEWS_QUERIES) -> int:
    return len(queries) * CREDITS_PER_SEARCH


def build_fresh_set(search: SearchBackend, chat: ChatBackend, models: list[str], today: date,
                    n: int = 8, days: int = 7, queries: list[str] = NEWS_QUERIES,
                    per_query: int = 5, token_budget: int = 60000) -> dict[str, Any]:
    budget = RunBudget(token_budget)
    seen: set[str] = set()
    candidates = []
    for q in queries:
        try:
            results = search.search(q, "event", per_query)  # Tavily news topic, last week
        except SearchUnavailable:
            continue
        for s in results:
            d = _parse_date(s.published_date)
            if s.url in seen or (d is not None and (today - d).days > days):
                continue
            seen.add(s.url)
            candidates.append(s)

    items: list[dict[str, Any]] = []
    skipped = {"not_usable": 0, "quote_not_found": 0, "model_error": 0}
    for s in candidates:
        if len(items) >= n:
            break
        user = (f"TODAY: {today.isoformat()}\nARTICLE TITLE: {s.title}\nPUBLISHED: {s.published_date or 'unknown'}\n"
                f"URL: {s.url}\nARTICLE TEXT:\n{s.snippet}")
        try:
            out = chat.chat_json("build", models, BUILD_SYSTEM, user, budget)
        except (LLMUnavailable, BudgetExceeded):
            skipped["model_error"] += 1
            continue
        if not out.get("usable") or not out.get("question") or not out.get("gold_answer"):
            skipped["not_usable"] += 1
            continue
        src = {"title": s.title, "snippet": s.snippet}
        if find_quote(out.get("evidence_quote", ""), [src]) is None:  # no quote, no item
            skipped["quote_not_found"] += 1
            continue
        items.append({"id": f"fresh-{len(items) + 1}", "category": "event", "question": out["question"].strip(),
                      "gold_answer": out["gold_answer"].strip(), "evidence_quote": out["evidence_quote"].strip(),
                      "source_url": s.url, "source_title": s.title, "published_date": s.published_date,
                      "as_of": today.isoformat(), "review": "auto"})
    return {"_note": "Auto-built from news by FreshPin. Each gold answer has a quote found word for word in its source. "
                     "Read and correct before relying on it (review: auto).",
            "built_on": today.isoformat(), "candidates": len(candidates), "skipped": skipped, "questions": items}


def save_set(data: dict[str, Any], out: Path | None = None) -> Path:
    path = out or BENCH_DIR / f"fresh-{data['built_on']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def make_gold_judge(chat: ChatBackend, models: list[str], admits: Callable[[str], bool]):
    """Returns judge(answer, question_item) -> {"result", "reason"}. Always asks the judge model: an answer can
    say "I don't have confirmed information about X" and still give the right company, so a phrase rule is not
    enough here. Empty answers count as admitted unknown without a call."""
    def judge(answer: str, q: dict[str, Any]) -> dict[str, Any]:
        if not answer.strip():
            return {"result": "admitted_unknown", "reason": "Empty answer."}
        user = (f"QUESTION: {q['question']}\nGOLD ANSWER (from {q.get('source_url', 'a dated source')}, "
                f"{q.get('published_date') or 'recent'}): {q['gold_answer']}\nMODEL ANSWER: {answer}")
        try:
            out = chat.chat_json(f"gold:{q['id']}", models, GOLD_SYSTEM, user, RunBudget(8000))
        except (LLMUnavailable, BudgetExceeded) as exc:
            return {"result": "not_graded", "reason": f"judge unavailable: {type(exc).__name__}"}
        r = out.get("result")
        return {"result": r if r in ("correct", "partly", "wrong", "admitted_unknown") else "not_graded",
                "reason": str(out.get("reason", ""))[:300]}
    return judge


CORRUPT_SYSTEM = """Rewrite the statement so ONE key fact becomes wrong (a different number, date, name or outcome),
keeping it plausible and the same length. Return ONLY JSON: {"wrong": str}"""


def corrupt_rule(text: str) -> str | None:
    """Make a statement wrong without a model: change its first number (a year moves back two years)."""
    import re
    m = re.search(r"\d[\d,.]*", text)
    if not m:
        return None
    raw = m.group(0).rstrip(".,")
    digits = raw.replace(",", "")
    try:
        value = float(digits)
    except ValueError:
        return None
    if digits.isdigit() and 1900 <= value <= 2100:
        new = str(int(value) - 2)
    elif digits.isdigit():
        new = str(int(value) * 2 + 3)
    else:
        new = f"{value * 1.5:.2f}"
    return text[:m.start()] + new + text[m.start() + len(raw):]


def self_test(items: list[dict[str, Any]], check: Callable[[str, str], dict[str, Any]],
              corrupt: Callable[[str], str | None]) -> dict[str, Any]:
    """How often is FreshPin itself right? Each gold answer should come back current; a deliberately wrong
    version of it should come back stale, wrong, partly current or disputed (anything but 'current')."""
    rows, tally = [], {"gold_claims": 0, "gold_confirmed": 0, "gold_false_alarm": 0,
                       "wrong_claims": 0, "wrong_caught": 0, "wrong_missed": 0}
    for q in items:
        g = check(q["gold_answer"], q["question"])
        gv = [c["verdict"] for c in g["claims"]]
        tally["gold_claims"] += len(gv)
        tally["gold_confirmed"] += sum(v in ("current", "partly_current") for v in gv)
        tally["gold_false_alarm"] += sum(v in ("stale", "contradicted") for v in gv)
        bad = corrupt(q["gold_answer"])
        bv: list[str] = []
        if bad:
            b = check(bad, q["question"])
            bv = [c["verdict"] for c in b["claims"]]
            tally["wrong_claims"] += len(bv)
            tally["wrong_missed"] += sum(v == "current" for v in bv)
            tally["wrong_caught"] += sum(v in ("stale", "contradicted", "partly_current", "disputed") for v in bv)
        rows.append({"id": q["id"], "gold_verdicts": gv, "wrong_version": bad, "wrong_verdicts": bv})
    return {"tally": tally, "rows": rows}
