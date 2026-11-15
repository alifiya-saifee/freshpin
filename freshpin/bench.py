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
    # The model talking about ITSELF ("I can't access...", "my training..."), not about users who can't access email.
    r"\bI(?:'m| am)?(?: currently)? (?:can(?:no|')t|unable to|do(?:n't| not) have|have no|am not able to)"
    r" (?:a |an |any )?(?:access|browse|real[- ]time|live|current|up[- ]to[- ]date|any|information|details|data|reports?|confirmed)"
    r"|\bI(?:'m| am) not aware|as of my (?:knowledge|training|last update)|knowledge cut-?off"
    r"|(?:after|beyond) my (?:knowledge|training)|my (?:training )?data (?:only )?(?:goes|runs|extends)", re.I)

# Rough Tavily credit use per answer: ~3 claims; a basic first search (1 credit) plus, with recheck, an advanced
# recheck search (2 credits).
CLAIMS_PER_ANSWER = 3
CREDITS_PER_SEARCH = 2


def estimate_credits(n_questions: int, n_models: int, recheck: bool) -> int:
    return n_questions * n_models * CLAIMS_PER_ANSWER * (1 + (CREDITS_PER_SEARCH if recheck else 0))


def admits_live_limits(answer: str) -> bool:
    # Models often write curly apostrophes (’) and non-breaking hyphens (‑); normalise before matching.
    text = (answer or "").translate(str.maketrans({"\u2019": "'", "\u2018": "'", "\u2011": "-", "\u2010": "-",
                                                   "\u2013": "-", "\u00a0": " "}))
    return bool(_HONEST.search(text))


def load_questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


# Problems in an answer: out of date, wrong, partly wrong, or not supported by any source found.
ERROR_VERDICTS = ("stale", "contradicted", "partly_current", "unverifiable")


def feedback_message(question: str, answer: str, report: dict[str, Any]) -> str | None:
    """What we tell a model about its own answer: what is wrong (with the confirmed fact and source) and what
    could not be verified. None when there is nothing to fix."""
    wrong = report.get("corrections") or []
    unsure = report.get("unconfirmed") or []
    if not wrong and not unsure:
        return None
    lines = [f"Question: {question}", f"Your answer: {answer}", "",
             "A fact-check against dated web sources found problems."]
    if wrong:
        lines.append("These parts are out of date or wrong. Replace them with the confirmed fact:")
        for k in wrong:
            ref = (k.get("reference") or {}).get("url", "")
            lines.append(f"- Wrong: {k['claim']}\n  Confirmed now: {k['now']}" + (f"\n  Source: {ref}" if ref else ""))
    if unsure:
        lines.append("These could not be verified. Do not state them as fact; say they are unconfirmed or remove them:")
        for u in unsure:
            lines.append(f"- {u['claim']} ({u.get('reason', 'unverified')})")
            for ev in u.get("evidence") or []:
                lines.append(f"    Evidence found: {ev.get('title')} ({ev.get('published_date') or 'undated'}) "
                             f"{ev.get('url')}: {ev.get('snippet')}")
    lines += ["", "Rewrite your answer in at most three sentences. Use only the confirmed facts and the evidence above, "
              "and say plainly what you do not know. Do not repeat an unverified claim."]
    return "\n".join(lines)


def _errors(claims: list[dict[str, Any]]) -> int:
    return sum(c["verdict"] in ERROR_VERDICTS for c in claims)


def run_bench(questions: list[dict[str, Any]], models: list[str],
              ask: Callable[[str, str], str],
              check: Callable[[str, str], dict[str, Any]],
              gold_judge: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
              repair: Callable[[str, str], str] | None = None,
              repair_rounds: int = 1) -> dict[str, Any]:
    """ask(model, question) -> answer. check(answer, question) -> FreshPin report (ledger untouched).
    gold_judge(answer, item) grades against a gold answer when the item has one.
    repair(model, feedback) -> rewritten answer: the model is told what was wrong and tries again."""
    out: dict[str, Any] = {"run_on": date.today().isoformat(), "models": {}}
    for model in models:
        rows = []
        for q in questions:
            row: dict[str, Any] = {"id": q["id"], "question": q["question"], "category": q["category"]}
            if q.get("gold_answer"):
                row["gold_answer"] = q["gold_answer"]
                row["gold_source"] = q.get("source_url")
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
            if gold_judge and q.get("gold_answer"):
                row["gold"] = gold_judge(row["answer"], q)
            if repair:
                answer, report = row["answer"], rep
                for rnd in range(1, max(1, repair_rounds) + 1):
                    fb = feedback_message(q["question"], answer, report)
                    if fb is None:
                        break
                    try:
                        answer = repair(model, fb)
                        report = check(answer, q["question"])
                    except Exception as exc:  # noqa: BLE001
                        row["repair"] = {"error": f"{type(exc).__name__}: {exc}"[:200], "rounds": rnd}
                        break
                    after = [claim_detail(c) for c in report["claims"]]
                    row["repair"] = {"rounds": rnd, "answer": answer, "claims": after,
                                     "errors_before": _errors(row["claims"]), "errors_after": _errors(after),
                                     "fixed": _errors(row["claims"]) > 0 and _errors(after) == 0,
                                     "admits_unknown": admits_live_limits(answer)}
                    if gold_judge and q.get("gold_answer"):
                        row["repair"]["gold"] = gold_judge(answer, q)
            rows.append(row)
        out["models"][model] = {"scorecard": scorecard(rows), "rows": rows}
    return out


def claim_detail(c: dict[str, Any]) -> dict[str, Any]:
    """What a person needs to see per claim: not just a verdict, but what is true now, the source, and why."""
    rc = c.get("recheck") or {}
    now = (c.get("current_statement") or "").strip()
    if c["verdict"] in ("current", "unverifiable") or _same(now, c["text"]):
        now = ""  # nothing new to say: it is current, or the judges could only repeat the claim
    label = None
    if c["verdict"] == "disputed":
        now = ""  # the two checks disagree: don't present either as the truth...
        if rc.get("one_check") and rc["one_check"].get("statement"):
            now, label = rc["one_check"]["statement"], "one check only"  # ...unless one is quoted and the other unsure
    why = rc.get("why_outdated") if c["verdict"] not in ("current", "unverifiable") else ""
    if c.get("needs_live_feed"):
        why = "No source dated today states this live value; it needs a live data feed."
    elif c["verdict"] in ("unverifiable", "disputed"):
        why = rc.get("reason") if c["verdict"] == "disputed" else (c.get("explanation") or rc.get("second_explanation") or "")
    return {"text": c["text"], "type": c["type"], "verdict": c["verdict"], "score": c["freshness"]["score"],
            "needs_live_feed": c.get("needs_live_feed", False), "now": now or None, "now_label": label,
            "why": (why or "").strip() or None,
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
                    wrapped("Real now (one check only): " if c.get("now_label") else "Real now: ", c["now"])
                if c.get("why"):
                    wrapped("Why: ", c["why"])
                ref = c.get("reference")
                if ref and ref.get("url"):
                    print(f"                Source ({ref.get('authority', '?')}, {ref.get('published_date') or 'undated'}): "
                          f"{cut(ref['url'], width - 40)}")
            if r.get("gold_answer"):
                print(textwrap.fill(" ".join(r["gold_answer"].split()), width=width,
                                    initial_indent="   Gold answer: ", subsequent_indent=" " * 16))
                if r.get("gold_source"):
                    print(f"                Source: {cut(r['gold_source'], width - 24)}")
                if r.get("gold"):
                    print(f"   Against gold: {r['gold']['result'].upper()}. {cut(r['gold'].get('reason', ''), width - 30)}")
            rp = r.get("repair") or {}
            if rp.get("error"):
                print(f"   Told it was wrong, but the rewrite failed: {cut(rp['error'])}")
            elif "claims" in rp:
                verdict = "FIXED" if rp["fixed"] else ("still wrong" if rp["errors_after"] else "rewritten")
                print(f"   Told what was wrong -> rewrite ({verdict}; errors {rp['errors_before']} -> {rp['errors_after']}):")
                print(textwrap.fill(" ".join(rp["answer"].split()), width=width, initial_indent="     ",
                                    subsequent_indent="     "))
                if rp.get("gold"):
                    print(f"     Against gold after feedback: {rp['gold']['result'].upper()}")


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
        **_gold_counts([r.get("gold") for r in rows], "gold"),
        "repaired": sum("claims" in (r.get("repair") or {}) for r in rows),
        "fixed_after_feedback": sum(bool((r.get("repair") or {}).get("fixed")) for r in rows),
        # Against the answer key: wrong, partly or "don't know" before, correct after being told.
        "gold_fixed_after_feedback": sum((r.get("gold") or {}).get("result") in ("wrong", "partly", "admitted_unknown")
                                         and ((r.get("repair") or {}).get("gold") or {}).get("result") == "correct"
                                         for r in rows),
        "had_errors": sum(_errors(r.get("claims", [])) > 0 for r in rows),
        **_gold_counts([(r.get("repair") or {}).get("gold") for r in rows], "gold_after_feedback"),
    }


def _gold_counts(grades: list[dict[str, Any] | None], prefix: str) -> dict[str, int]:
    g = [x for x in grades if x]
    return {f"{prefix}_graded": len(g), **{f"{prefix}_{k}": sum(x.get("result") == k for x in g)
                                           for k in ("correct", "partly", "wrong", "admitted_unknown")}}


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
        extra = []
        if s.get("gold_graded"):
            extra.append(f"gold: {s['gold_correct']} correct, {s['gold_partly']} partly, {s['gold_wrong']} wrong, "
                         f"{s['gold_admitted_unknown']} said unknown (of {s['gold_graded']})")
        if s.get("repaired"):
            extra.append(f"told it was wrong: {s['fixed_after_feedback']} of {s['had_errors']} fixed")
            if s.get("gold_graded"):
                extra.append(f"gold fixed after feedback: {s['gold_fixed_after_feedback']}")
            if s.get("gold_after_feedback_graded"):
                extra.append(f"gold after feedback: {s['gold_after_feedback_correct']} correct")
        if extra:
            print(f"{'':38} " + " | ".join(extra))


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
