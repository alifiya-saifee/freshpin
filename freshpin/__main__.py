"""Command line:  python -m freshpin serve | mcp | check "answer text" | sample <id>"""
from __future__ import annotations

import json
import sys

from .app import FreshPin
from .config import settings
from .bench import claim_detail as bench_detail
from .server import serve

USAGE = """usage:
  python -m freshpin serve                 start the web app (default http://127.0.0.1:8787)
  python -m freshpin mcp                   run as an MCP server (stdio) for AI assistants
  python -m freshpin bench --models a,b [--max N] [--questions file] [--no-recheck] [--yes]
                                           test models on facts that change; print a freshness scorecard
  python -m freshpin bench-build [--n 8]   build a fresh-events question set (gold answers + quotes) from this week's news
  python -m freshpin watch [--all]         re-check pinned facts past their half-life; alert on reopened decisions
  python -m freshpin check "answer text"   check an answer and print the JSON report
  python -m freshpin sample <id>           load a sample ledger, check its answer, print a summary
"""


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    cmd = argv[0]
    if cmd == "serve":
        serve()
        return 0
    if cmd == "mcp":
        from .mcp import run
        run()
        return 0
    if cmd == "bench":
        return bench_cmd(argv[1:])
    if cmd == "bench-build":
        return bench_build_cmd(argv[1:])
    if cmd == "watch":
        return watch_cmd(argv[1:])
    app = FreshPin(settings())
    if cmd == "check" and len(argv) > 1:
        print(json.dumps(app.check(" ".join(argv[1:])), indent=2, ensure_ascii=False))
        return 0
    if cmd == "sample" and len(argv) > 1:
        app.load_sample(argv[1])
        rep = app.check(app.samples()[argv[1]]["answer"])
        for c in rep["claims"]:
            f = c["freshness"]
            label = {"unverifiable": "UNVER", "disputed": "DISPU"}.get(c["verdict"], f["band"].upper())
            print(f"[{label:5}] {f['score']:3}  {c['text']}")
            d = bench_detail(c)
            if d["now"]:
                print(f"         now: {d['now']}")
            r = c.get("recheck") or {}
            if r.get("status") not in (None, "skipped"):
                ref = r.get("best_reference") or {}
                print(f"         recheck: {r['status']} | best ref: {ref.get('authority', '-')} {ref.get('url', '-')}")
            if d["why"]:
                print(f"         why: {d['why']}")
        print("reopened:", rep["summary"]["decisions_reopened"] or "none")
        g = rep["gateway"]
        s = rep["summary"]
        print(f"recheck: {s['confirmed']} confirmed, {s['disputed']} disputed, {rep['tavily_searches']} Tavily searches")
        print(f"gateway: {g['calls']} calls, {g['failed_attempts']} failed, {g['fallbacks_used']} fallbacks, {g['tokens']} tokens")
        return 0
    print(USAGE)
    return 2



def watch_cmd(args: list[str]) -> int:
    import argparse
    from datetime import date

    from .watch import alert_text, post_webhook, run_watch

    ap = argparse.ArgumentParser(prog="python -m freshpin watch")
    ap.add_argument("--all", action="store_true", help="re-check every pinned fact, not only those past their half-life")
    a = ap.parse_args(args)
    s = settings()
    app = FreshPin(s)

    def check(statement: str, claim_type: str) -> dict:
        return app.check(statement, context=f"Re-check of a pinned {claim_type} fact")

    res = run_watch(app.ledger, check, date.today(), force=a.all)
    print(f"{res['due']} pinned fact(s) due; checked {len(res['checked'])}; errors {len(res['errors'])}.")
    for e in res["errors"]:
        print("  error:", e)
    if not res["alerts"]:
        print("No decisions changed.")
        return 0
    text = alert_text(res["alerts"])
    print(text)
    if s.webhook_url:
        print("Webhook:", post_webhook(s.webhook_url, text))
    return 0


def bench_build_cmd(args: list[str]) -> int:
    import argparse
    from datetime import date
    from pathlib import Path

    from .benchbuild import NEWS_QUERIES, build_fresh_set, estimate_build_credits, save_set

    ap = argparse.ArgumentParser(prog="python -m freshpin bench-build")
    ap.add_argument("--n", type=int, default=8, help="how many questions to keep")
    ap.add_argument("--days", type=int, default=7, help="only news from the last N days")
    ap.add_argument("--out", help="output file (default bench/fresh-<date>.json)")
    a = ap.parse_args(args)
    s = settings()
    if s.mode != "live":
        print("bench-build searches real news, so it needs FRESHPIN_MODE=live (Bifrost + Tavily).")
        return 2
    print(f"Searching this week's news ({len(NEWS_QUERIES)} searches, about {estimate_build_credits()} Tavily credits).")
    app = FreshPin(s)
    data = build_fresh_set(app.search, app.chat, s.extractor_models, date.today(), n=a.n, days=a.days)
    path = save_set(data, Path(a.out) if a.out else None)
    print(f"Kept {len(data['questions'])} of {data['candidates']} articles; skipped {data['skipped']}.")
    for q in data["questions"]:
        print(f"- {q['question']}\n    gold: {q['gold_answer']}\n    quote: \"{q['evidence_quote']}\"\n"
              f"    source: {q['source_url']} ({q.get('published_date') or 'undated'})")
    print(f"\nSaved: {path}  (review: auto - read it before relying on it)")
    return 0


def bench_cmd(args: list[str]) -> int:
    import argparse
    from datetime import date
    from pathlib import Path

    from . import bench
    from .llm import BifrostClient, RunBudget

    ap = argparse.ArgumentParser(prog="python -m freshpin bench")
    ap.add_argument("--models", help="comma-separated Bifrost provider/model names to test")
    ap.add_argument("--show", metavar="FILE", help="print a saved result (bench/results/...) again; no credits used")
    ap.add_argument("--questions", default=str(bench.QUESTIONS))
    ap.add_argument("--max", type=int, default=6, help="use only the first N questions")
    ap.add_argument("--no-recheck", action="store_true", help="skip the recheck layer (half the Tavily credits)")
    ap.add_argument("--yes", action="store_true", help="run even if the estimate is above 40 Tavily credits")
    ap.add_argument("--repair", action="store_true", help="tell the model what was wrong and let it rewrite, then re-check")
    ap.add_argument("--repair-rounds", type=int, default=1)
    ap.add_argument("--self-test", action="store_true",
                    help="measure FreshPin itself: gold answers should pass, deliberately wrong versions should fail")
    a = ap.parse_args(args)
    if a.show:
        import json as _json
        saved = bench.rescore(_json.loads(Path(a.show).read_text(encoding="utf-8")))
        bench.print_table(saved)
        bench.print_details(saved)
        return 0
    if not a.models:
        ap.error("--models is required (or use --show FILE)")

    s = settings()
    if s.mode != "live":
        print("The bench asks real models, so it needs FRESHPIN_MODE=live (Bifrost + Tavily).")
        return 2
    if a.no_recheck:
        s.recheck = False
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    questions = bench.load_questions(Path(a.questions))[: max(1, a.max)]
    est = bench.estimate_credits(len(questions), len(models), s.recheck)
    print(f"{len(questions)} question(s) x {len(models)} model(s), recheck {'on' if s.recheck else 'off'}: "
          f"about {est} Tavily credits at most.")
    if est > 40 and not a.yes:
        print("Above 40 credits: run again with --yes, or use --max / --no-recheck.")
        return 2

    app = FreshPin(s)
    client = BifrostClient(s.bifrost_url, s.bifrost_virtual_key, s.timeout_s)
    system = bench.ASK_SYSTEM.format(today=date.today().isoformat())

    def ask(model: str, question: str) -> str:
        print(f"  asking {model}: {question[:70]}")
        return client.chat_text(model, system, question, RunBudget(8000))

    def check(answer: str, question: str) -> dict:
        return app.check(answer, context=question, apply_to_ledger=False)

    if a.self_test:
        from .benchbuild import corrupt_rule, self_test
        items = [q for q in questions if q.get("gold_answer")]
        if not items:
            print("--self-test needs a question set with gold answers (python -m freshpin bench-build).")
            return 2
        st = self_test(items, lambda ans, q: app.check(ans, context=q, apply_to_ledger=False), corrupt_rule)
        t = st["tally"]
        print(f"Gold answers: {t['gold_confirmed']} of {t['gold_claims']} claims confirmed, "
              f"{t['gold_false_alarm']} wrongly called stale or wrong.")
        print(f"Deliberately wrong versions: {t['wrong_caught']} of {t['wrong_claims']} claims caught, "
              f"{t['wrong_missed']} wrongly called current.")
        print(f"Saved: {bench.save({'self_test': st})}")
        return 0

    gold_judge = None
    if any(q.get("gold_answer") for q in questions):
        from .benchbuild import make_gold_judge
        gold_judge = make_gold_judge(app.chat, s.checker_models, bench.admits_live_limits)

    def repair(model: str, feedback: str) -> str:
        print(f"  telling {model} what was wrong and asking it to rewrite")
        return client.chat_text(model, system, feedback, RunBudget(8000), step="repair")

    result = bench.run_bench(questions, models, ask, check, gold_judge=gold_judge,
                             repair=repair if a.repair else None, repair_rounds=a.repair_rounds)
    print()
    bench.print_table(result)
    bench.print_details(result)
    print(f"\nSaved: {bench.save(result)}")
    if hasattr(app.search, "hits"):
        print(f"Tavily searches: {app.search.misses} made, {app.search.hits} reused from cache")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
