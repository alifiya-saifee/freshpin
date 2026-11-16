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
            # Both judges' raw verdicts, so we can later measure which claim types the recheck actually changes.
            "first_verdict": c.get("first_verdict"), "checker_verdict": rc.get("checker_verdict"),
            "recheck_status": rc.get("status"),
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


# ============================================================================================================
# Agent arena: benchmark AI AGENTS, not just models. The same model is run as different agent configurations
# ("arms") on questions whose answer changed recently, and graded against gold answers with quotes:
#   bare      the model alone (training data only)
#   grounded  FreshPin's ground_question runs first and the dated sources go into the prompt
#   agent     a tool loop: the model DECIDES whether to call search_web (Tavily via FreshPin), up to 2 times
#   repair    the bare answer, checked by FreshPin, then the model is told what was wrong and rewrites
# ============================================================================================================
import math  # noqa: E402

ARMS = ("bare", "grounded", "agent", "repair")
GROUNDED_SYSTEM = ("Answer the question directly and briefly, in at most three sentences, using ONLY the dated "
                   "sources below. Cite them as [1], [2]. If they don't answer it, say you could not find "
                   "recent information. Today's date is {today}.")
AGENT_SYSTEM = ("Answer the question directly and briefly, in at most three sentences. You can call search_web "
                "to look things up when the answer may have changed recently. Today's date is {today}.")
SEARCH_TOOL = [{"type": "function", "function": {
    "name": "search_web",
    "description": "Search the live web for recent, dated information (news, prices, laws, versions, who runs what).",
    "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "What to look up"}},
                   "required": ["query"]}}}]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def sign_test(b: int, c: int) -> float:
    """Exact two-sided McNemar (sign) test on discordant pairs: b = right only in A, c = right only in B."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


_GENERIC = {
    "the", "this", "that", "these", "those", "on", "in", "and", "for", "with", "after", "before", "its", "it", "more",
    "over", "under", "about", "around", "nearly", "almost", "some", "many", "most", "all", "both", "according", "also",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "yesterday", "week",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
    "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "users", "company", "service", "services", "reported", "report", "reports", "outage", "launch", "launched",
    "announced", "new", "news", "inc", "ltd", "llc", "corp",
}


def key_terms(gold: str, question: str) -> set[str]:
    """The distinguishing names and numbers in a gold answer (e.g. 'Workday', '11,500'): capitalised words and
    numbers the question doesn't already contain, minus filler ('More', 'Over'), dates (weekdays, months) and
    years. Used for a whole-word backstop next to the LLM grader."""
    qwords = set(re.findall(r"\w+", (question or "").casefold()))
    out = set()
    for t in re.findall(r"\b[A-Z][\w&.'-]{2,}\b|\b\d[\d,.]*\d\b|\b\d\b", gold or ""):
        t = t.rstrip(".,'")
        low = t.casefold()
        if low in _GENERIC or low in qwords:
            continue
        if re.fullmatch(r"(19|20)\d\d", t):  # a year is too common to identify an answer
            continue
        out.add(t)
    return out


def string_match(answer: str, terms: set[str]) -> bool | None:
    """Backstop next to the LLM grader: a 'correct' answer must contain at least one gold key term as a whole
    word or number ('25' does not match '2025'; '11,500' matches '11500')."""
    if not terms:
        return None
    a = (answer or "").replace(",", "")
    for t in terms:
        pat = re.escape(t.replace(",", ""))
        if re.search(rf"(?<![\w.]){pat}(?![\w])", a, re.I):
            return True
    return False


_QUAL = [(re.compile(r"(more than|over|above|at least|exceeding|upwards of)\s*$", re.I), ">="),
         (re.compile(r"(fewer than|less than|under|below|up to|at most)\s*$", re.I), "<="),
         (re.compile(r"(nearly|almost|about|around|roughly|approximately|some|close to|~)\s*$", re.I), "~")]


def _numbers(text: str) -> list[tuple[float, str]]:
    """Numbers with the qualifier written before them: 'more than 100,000' -> (100000, '>=')."""
    out = []
    # A space, no-break space or thin space between digit groups is a thousands separator, like a comma:
    # "130 000" is 130000 (models write it that way); "Nov 11 2026" stays two numbers.
    text = re.sub(r"(?<=\d)[ \u00a0\u202f\u2009](?=\d{3}(?!\d))", "", text or "")
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        raw = m.group(0).rstrip(",")
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if re.fullmatch(r"(19|20)\d\d", raw):
            continue  # years identify nothing
        before = text[max(0, m.start() - 25):m.start()]
        op = next((o for pat, o in _QUAL if pat.search(before)), "=")
        out.append((v, op))
    return out


def backstop(answer: str, gold: str, question: str) -> bool | None:
    """Deterministic check next to the LLM grader. True if a distinguishing NAME from the gold answer appears as a
    whole word, or a gold NUMBER is satisfied (130,000 satisfies 'more than 100,000'; 'about 25' accepts 20-30).
    None when the gold answer has nothing distinguishing to check."""
    names = {t for t in key_terms(gold, question) if not re.fullmatch(r"[\d,.]+", t)}
    if names and string_match(answer, names):
        return True
    q_nums = {v for v, _ in _numbers(question)}
    gold_nums = [(v, op) for v, op in _numbers(gold) if v not in q_nums]
    ans = [v for v, _ in _numbers(answer)]
    for v, op in gold_nums:
        for a in ans:
            if (op == ">=" and a >= v) or (op == "<=" and a <= v) or (op == "~" and abs(a - v) <= 0.25 * v) \
                    or (op == "=" and abs(a - v) <= 1e-9 * max(1.0, v)):
                return True
    if not names and not gold_nums:
        return None
    return False


def _pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(round(p * (len(s) - 1))))], 1)


def independent_models(agent_model: str, pool: list[str]) -> tuple[list[str], bool]:
    """Graders and checkers from a different model family than the agent under test (no self-preference).
    Returns (models, independent?)."""
    from .pipeline import model_family
    fam = model_family(agent_model)
    other = [m for m in dict.fromkeys(pool) if model_family(m) != fam]
    return (other, True) if other else (list(dict.fromkeys(pool)), False)


def run_arena(questions: list[dict[str, Any]], models: list[str], arms: list[str],
              runners: dict[str, Callable[[str, str], dict[str, Any]]],
              gold_judge_for: Callable[[str], Callable[[str, dict[str, Any]], dict[str, Any]]],
              check: Callable[[str, str], dict[str, Any]] | None = None, check_arms: tuple[str, ...] = (),
              repair: Callable[[str, str], dict[str, Any]] | None = None,
              set_model: Callable[[str], Any] | None = None,
              on_model_done: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
    """runners[arm](model, question) -> {"answer", "tool_calls": [...], "tokens", "searches", "latency_s"}.
    set_model(model) lets the caller point FreshPin's checkers at a different model family first."""
    out: dict[str, Any] = {"run_on": date.today().isoformat(), "arms": arms, "models": {}}
    for model in models:
        independent = set_model(model) if set_model else None
        judge = gold_judge_for(model)
        rows: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
        for q in questions:
            bare_row = None
            for arm in arms:
                row: dict[str, Any] = {"id": q["id"], "question": q["question"], "arm": arm}
                try:
                    if arm == "repair":
                        if bare_row is None or "answer" not in bare_row or not check or not repair:
                            raise RuntimeError("repair needs the bare arm and a checker")
                        rep = bare_row.get("_report") or check(bare_row["answer"], q["question"])
                        fb = feedback_message(q["question"], bare_row["answer"], rep)
                        res = repair(model, fb) if fb else {"answer": bare_row["answer"], "tool_calls": []}
                        res["told_what_was_wrong"] = bool(fb)
                        # The repair arm pays for the check that produced its feedback.
                        res["tokens"] = res.get("tokens", 0) + rep.get("gateway", {}).get("tokens", 0)
                        res["credits"] = res.get("credits", 0) + rep.get("_credits", 0)
                    else:
                        res = runners[arm](model, q["question"])
                    row.update({k: v for k, v in res.items() if k != "_report"})
                    row["chose_to_search"] = bool(res.get("tool_calls")) if arm == "agent" else None
                    if check and arm in check_arms and row.get("answer"):
                        rep = check(row["answer"], q["question"])
                        row["_report"] = rep
                        row["claims"] = [claim_detail(c) for c in rep["claims"]]
                except Exception as exc:  # noqa: BLE001 - one failure must not stop the arena
                    row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                if q.get("gold_answer") and "answer" in row:
                    row["gold"] = judge(row["answer"], q)
                    row["string_match"] = backstop(row["answer"], q["gold_answer"], q["question"])
                    g = row["gold"].get("result")
                    # LLM grader and string backstop must both agree for "correct".
                    row["correct"] = g == "correct" and row["string_match"] is not False
                    row["grader_disagrees"] = g == "correct" and row["string_match"] is False
                if arm == "bare":
                    bare_row = row
                rows[arm].append(row)
        for arm_rows in rows.values():
            for r in arm_rows:
                r.pop("_report", None)
        out["models"][model] = {"rows": rows, "scorecard": {a: arena_scorecard(rows[a]) for a in arms},
                                "paired": paired_tests(rows, arms), "graders_independent": independent}
        if on_model_done:
            on_model_done(out)  # save after each model, so a crash later doesn't lose finished work
    return out


def arena_scorecard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in rows if "gold" in r and r["gold"].get("result") != "not_graded"]
    n = len(graded)
    correct = sum(bool(r.get("correct")) for r in graded)
    abstain = sum(r["gold"].get("result") == "admitted_unknown" for r in graded)
    agent_rows = [r for r in rows if r.get("chose_to_search") is not None]
    lat = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
    return {
        "n": n, "errors": sum("error" in r for r in rows),
        "correct": correct, "accuracy": round(correct / n, 3) if n else None,
        "accuracy_ci95": [round(x, 3) for x in wilson(correct, n)],
        "partly": sum(r["gold"].get("result") == "partly" for r in graded),
        "wrong": sum(r["gold"].get("result") == "wrong" for r in graded),
        "abstain": abstain, "abstain_ci95": [round(x, 3) for x in wilson(abstain, n)],
        "grader_disagrees": sum(bool(r.get("grader_disagrees")) for r in graded),
        "chose_to_search": (sum(bool(r["chose_to_search"]) for r in agent_rows) if agent_rows else None),
        "tool_calls": sum(len(r.get("tool_calls") or []) for r in rows),
        "tool_errors": sum(r.get("tool_errors", 0) for r in rows),
        "tavily_credits": sum(r.get("credits", 0) for r in rows),
        "tokens_per_correct": (round(sum(r.get("tokens", 0) for r in rows) / correct) if correct else None),
        "tokens_mean": round(sum(r.get("tokens", 0) for r in rows) / len(rows)) if rows else 0,
        "tavily_searches": sum(r.get("searches", 0) for r in rows),
        "latency_p50_s": _pct(lat, 0.5), "latency_p95_s": _pct(lat, 0.95),
    }


def paired_tests(rows: dict[str, list[dict[str, Any]]], arms: list[str]) -> list[dict[str, Any]]:
    """Same items, two arms: who flipped from wrong to right and back, and an exact sign test."""
    out = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            ra = {r["id"]: r for r in rows[a] if "gold" in r}
            rb = {r["id"]: r for r in rows[b] if "gold" in r}
            both = [k for k in ra if k in rb]
            only_b = [k for k in both if not ra[k].get("correct") and rb[k].get("correct")]
            only_a = [k for k in both if ra[k].get("correct") and not rb[k].get("correct")]
            out.append({"a": a, "b": b, "items": len(both), "fixed_by_b": len(only_b), "broken_by_b": len(only_a),
                        "p_value": round(sign_test(len(only_a), len(only_b)), 4)})
    return out


def arena_markdown(result: dict[str, Any]) -> str:
    lines = [f"# FreshPin agent arena, {result['run_on']}", "",
             "Same questions, same model, different agent configurations. Correct = the LLM grader says correct "
             "AND a gold key term appears in the answer. 95% Wilson intervals; with n items an interval is wide, "
             "so read differences with the paired test below.", ""]
    for model, m in result["models"].items():
        lines += [f"## {model}", "", "| arm | n | correct | accuracy (95% CI) | partly | wrong | said unknown | "
                  "chose to search | tool calls (errors) | tokens/answer | tokens/correct | Tavily credits | p50 / p95 s |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm, s in m["scorecard"].items():
            acc = "-" if s["accuracy"] is None else f"{s['accuracy']:.0%} ({s['accuracy_ci95'][0]:.0%}–{s['accuracy_ci95'][1]:.0%})"
            cts = "-" if s["chose_to_search"] is None else f"{s['chose_to_search']}/{s['n']}"
            lines.append(f"| {arm} | {s['n']} | {s['correct']} | {acc} | {s['partly']} | {s['wrong']} | {s['abstain']} | "
                         f"{cts} | {s['tool_calls']} ({s.get('tool_errors', 0)}) | {s['tokens_mean']} | "
                         f"{s.get('tokens_per_correct') or '-'} | {s.get('tavily_credits', 0)} | "
                         f"{s['latency_p50_s']} / {s['latency_p95_s']} |")
        lines += ["", "| A vs B | items | B fixed | B broke | sign test p |", "|---|---|---|---|---|"]
        for p in m["paired"]:
            lines.append(f"| {p['a']} vs {p['b']} | {p['items']} | {p['fixed_by_b']} | {p['broken_by_b']} | {p['p_value']} |")
        lines.append("")
    return "\n".join(lines)
