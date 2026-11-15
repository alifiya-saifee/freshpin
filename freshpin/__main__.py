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

    result = bench.run_bench(questions, models, ask, check)
    print()
    bench.print_table(result)
    bench.print_details(result)
    print(f"\nSaved: {bench.save(result)}")
    if hasattr(app.search, "hits"):
        print(f"Tavily searches: {app.search.misses} made, {app.search.hits} reused from cache")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
