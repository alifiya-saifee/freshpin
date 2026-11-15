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
  python -m freshpin agentbench [...]      benchmark AGENTS (any framework) on a task suite: repeats, failure
                                           conditions, steps/tools/tokens/$/time, step-by-step traces (see --help)
  python -m freshpin prices --models a,b   write bench/list-prices.json from Bifrost's pricing table (list-price $)
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
    if cmd == "agentbench":
        return agentbench_cmd(argv[1:])
    if cmd == "watch":
        return watch_cmd(argv[1:])
    if cmd == "prices":
        return prices_cmd(argv[1:])
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
        print("Webhook:", post_webhook(s.webhook_url, text, allow_private=s.allow_private_webhook))
    return 0


def arena_cmd(a, s, app, client, questions, models, check) -> int:
    """Agent arena: the same model as several agent configurations, graded against gold answers."""
    import functools
    import json as _json
    import time as _time
    from datetime import date

    from . import bench
    from .benchbuild import make_gold_judge
    from .llm import RunBudget
    from .pipeline import model_family

    log = functools.partial(print, flush=True)  # progress shows up live even when output goes to a file
    arms = [x.strip() for x in a.arms.split(",") if x.strip() in bench.ARMS]
    check_arms = tuple(x.strip() for x in a.check_arms.split(",") if x.strip())
    if "repair" in arms and "bare" not in arms:
        arms.insert(0, "bare")
    today = date.today().isoformat()
    gold_url = {q["question"]: q.get("source_url") for q in questions}
    search = app.search
    log(f"Arena: {len(questions)} question(s) x {len(models)} model(s) x arms {arms}; FreshPin checks on {check_arms}.")

    def credits() -> int:
        return getattr(search, "credits", 0)

    def fresh_cache() -> None:  # no arm gets another arm's searches for free
        if hasattr(search, "clear"):
            search.clear()

    def sources_text(sources: list, question: str, advice: str = "") -> tuple[str, int]:
        # The gold answer's own article is left out, so grounding can't simply copy the answer key.
        leak = gold_url.get(question)
        kept = [x for x in sources if not leak or x.get("url") != leak]
        if not kept:
            return "(no recent sources found)", len(sources) - len(kept)
        text = "\n".join(f"[{i}] {x.get('title')} ({x.get('published_date') or 'undated'}) {x.get('url')}\n"
                         f"    {(x.get('snippet') or '')[:400]}" for i, x in enumerate(kept, 1))
        return text + (f"\nAdvice: {advice}" if advice else ""), len(sources) - len(kept)

    def timed(fn):
        def run(model: str, question: str) -> dict:
            fresh_cache()
            budget, c0, t0 = RunBudget(40000), credits(), _time.monotonic()
            res = fn(model, question, budget)
            res.update(tokens=budget.used_tokens, credits=credits() - c0, latency_s=round(_time.monotonic() - t0, 1))
            log(f"  {model.split('/')[-1]} [{fn.__name__}] {question[:60]} -> "
                f"{len(res.get('tool_calls') or [])} tool call(s), {res.get('tool_errors', 0)} tool error(s)")
            return res
        run.__name__ = fn.__name__
        return run

    @timed
    def bare(model, question, budget):
        return {"answer": client.chat_text(model, bench.ASK_SYSTEM.format(today=today), question, budget), "tool_calls": []}

    @timed
    def grounded(model, question, budget):
        g = app.ground(question, depth="basic")
        budget.charge(g.get("tokens", 0))  # the grounding classifier's tokens belong to this arm
        text, leaked = sources_text(g["sources"], question, g.get("advice", ""))
        user = f"QUESTION: {question}\n\nSOURCES:\n{text}"
        return {"answer": client.chat_text(model, bench.GROUNDED_SYSTEM.format(today=today), user, budget),
                "tool_calls": [{"name": "ground_question", "args": {"question": question}}], "gold_source_removed": leaked}

    @timed
    def agent(model, question, budget):
        def run_tool(name: str, args: dict) -> str:
            # The agent's own search goes straight to Tavily: no classifier can veto a search the agent chose.
            query = str(args.get("query") or question)
            found = [x.as_dict() for x in search.search(query, "other", 5, depth="basic")]
            return sources_text(found, question)[0]
        return client.chat_tools(model, bench.AGENT_SYSTEM.format(today=today), question, bench.SEARCH_TOOL,
                                 run_tool, budget)

    def repair_fn(model: str, feedback: str) -> dict:
        budget, t0 = RunBudget(30000), _time.monotonic()
        ans = client.chat_text(model, bench.ASK_SYSTEM.format(today=today), feedback, budget, step="repair")
        return {"answer": ans, "tool_calls": [], "tokens": budget.used_tokens, "latency_s": round(_time.monotonic() - t0, 1)}

    def check_counted(answer: str, question: str) -> dict:
        fresh_cache()
        c0 = credits()
        rep = check(answer, question)
        rep["_credits"] = credits() - c0
        return rep

    base = {"extractor": list(s.extractor_models), "verifier": list(s.verifier_models), "checker": list(s.checker_models)}
    pool = list(dict.fromkeys(base["checker"] + base["verifier"] + base["extractor"]))
    # ONE gold grader for every model in this run, from a family none of them belongs to (a fair comparison).
    contestant_families = {model_family(m) for m in models}
    shared = [m for m in pool if model_family(m) not in contestant_families]
    if shared:
        log(f"Gold grader for all models: {shared[0]} (family {model_family(shared[0])}).")
    else:
        log("  warning: no grader family outside all contestants; each model gets its own independent grader.")

    def set_model(model: str) -> bool:
        # FreshPin's own judges must not share the agent's model family (self-preference).
        ok = True
        for role, attr in (("extractor", "extractor_models"), ("verifier", "verifier_models"), ("checker", "checker_models")):
            chosen, indep = bench.independent_models(model, base[role])
            if not indep:
                chosen, indep = bench.independent_models(model, pool)
            setattr(app.s, attr, chosen)
            ok = ok and indep
        if not ok:
            log(f"  warning: no checker from a different model family than {model}")
        return ok

    def judge_for(model: str):
        judges = shared or bench.independent_models(model, pool)[0]
        return make_gold_judge(app.chat, judges, bench.admits_live_limits)

    path = bench.RESULTS_DIR / f"bench-{_time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    def save_partial(res: dict) -> None:
        res["gold_grader"] = shared[0] if shared else "per-model"
        path.write_text(_json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        path.with_suffix(".md").write_text(bench.arena_markdown(res), encoding="utf-8")
        log(f"  saved after {len(res['models'])} model(s): {path}")

    result = bench.run_arena(questions, models, arms, {"bare": bare, "grounded": grounded, "agent": agent},
                             judge_for, check=check_counted, check_arms=check_arms, repair=repair_fn,
                             set_model=set_model, on_model_done=save_partial)
    for role, attr in (("extractor", "extractor_models"), ("verifier", "verifier_models"), ("checker", "checker_models")):
        setattr(app.s, attr, base[role])
    save_partial(result)
    log()
    log(bench.arena_markdown(result))
    return 0


def tool_cache_path(questions: "Path", explicit: str, bench_dir: "Path") -> "Path":
    """Where the recorded searches for a task suite live: --tool-cache if given, else next to the suite (so a
    committed suite and its recording travel together), else the old bench/ location."""
    from pathlib import Path
    if explicit:
        return Path(explicit)
    beside = questions.parent / f"toolcache-{questions.stem}.json"
    old = bench_dir / f"toolcache-{questions.stem}.json"
    return beside if beside.exists() or not old.exists() else old


def agentbench_cmd(args: list[str]) -> int:
    import argparse
    import json as _json
    import shlex
    import time as _time
    from pathlib import Path

    from . import bench
    from .agentbench import Agent, BifrostLog, builtin_agents, leaderboard_markdown, run_suite, summarize
    from .benchbuild import make_gold_judge
    from .toolserver import FAULTS, ToolServer

    ap = argparse.ArgumentParser(prog="python -m freshpin agentbench",
                                 description="Benchmark AI agents (any framework) on research tasks that changed recently.")
    ap.add_argument("--questions", required=True, help="task suite JSON with gold answers (bench-build makes one)")
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2, help="runs per task, to measure reliability")
    ap.add_argument("--conditions", default=",".join(FAULTS), help="normal,model_outage,tool_error,misleading")
    ap.add_argument("--agents", default="all", help="built-in names (bare,search_first,tool_agent,reflect) or all")
    ap.add_argument("--agent", action="append", default=[], metavar="NAME=COMMAND",
                    help="add an outside agent run as a command, e.g. --agent 'mine=python3 my_agent.py'")
    ap.add_argument("--agent-url", action="append", default=[], metavar="NAME=URL", help="add an outside agent over HTTP")
    ap.add_argument("--model", required=True, help="model under test, Bifrost provider/model")
    ap.add_argument("--fallbacks", default="", help="fallback models offered to agents (comma-separated)")
    ap.add_argument("--search-type", default="event", help="claim type for live searches (event = news, last week)")
    ap.add_argument("--pace", type=float, default=0.0, help="seconds to wait between runs (free-tier rate limits)")
    ap.add_argument("--tool-cache", default="", help="recorded searches to replay (default: toolcache-<suite>.json "
                    "next to the suite file, else in bench/)")
    ap.add_argument("--replay-only", action="store_true",
                    help="never search live: replay the recorded searches only (no Tavily credits; unrecorded "
                    "queries return no results)")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args(args)

    s = settings()
    if s.mode != "live":
        print("agentbench runs real agents, so it needs FRESHPIN_MODE=live (Bifrost + Tavily).")
        return 2
    conds = [c for c in a.conditions.split(",") if c in FAULTS]
    pool = builtin_agents()
    wanted = {x.strip() for x in a.agents.split(",")} if a.agents != "all" else None
    agents = [ag for ag in pool if wanted is None or ag.name.split(" ")[0] in wanted]
    for spec in a.agent:
        name, _, cmd = spec.partition("=")
        agents.append(Agent(name.strip(), shlex.split(cmd), description="outside agent (command)"))
    for spec in a.agent_url:
        name, _, url = spec.partition("=")
        agents.append(Agent(name.strip(), url=url.strip(), description="outside agent (HTTP)"))
    tasks = [q for q in bench.load_questions(Path(a.questions)) if q.get("gold_answer")][: max(1, a.max)]
    n_runs = len(agents) * len(conds) * len(tasks) * a.repeats
    print(f"{len(agents)} agent(s) x {len(conds)} condition(s) x {len(tasks)} task(s) x {a.repeats} repeat(s) = "
          f"{n_runs} runs. Searches are recorded once and replayed, so Tavily credits are spent only on new queries.",
          flush=True)
    if n_runs > 60 and not a.yes:
        print("More than 60 runs: add --yes to go ahead.")
        return 2

    app = FreshPin(s)
    fallbacks = [m.strip() for m in a.fallbacks.split(",") if m.strip()]
    judges, independent = bench.independent_models(a.model, list(dict.fromkeys(
        s.checker_models + s.verifier_models + s.extractor_models)))
    grade = make_gold_judge(app.chat, judges, bench.admits_live_limits)
    if not s.bifrost_logs_db:
        print("warning: FRESHPIN_BIFROST_LOGS_DB is not set, so model calls, tokens and $ can't be read from Bifrost.")
    bifrost = BifrostLog(s.bifrost_logs_db) if s.bifrost_logs_db else None

    def search(q: str) -> list:
        return [x.as_dict() for x in app.search.search(q, a.search_type, 5, depth="basic")]

    cache = tool_cache_path(Path(a.questions), a.tool_cache, bench.RESULTS_DIR.parent)
    if a.replay_only and not cache.exists():
        print(f"--replay-only needs recorded searches, and {cache} does not exist.")
        return 2
    print(f"Tool recording: {cache}{' (replay only, no live searches)' if a.replay_only else ''}", flush=True)
    tools = ToolServer(search, cache_path=cache, live=not a.replay_only).start()
    headers = {"x-bf-vk": s.bifrost_virtual_key} if s.bifrost_virtual_key else {}
    llm = {"base_url": s.bifrost_url.rstrip("/") + "/v1", "headers": headers}
    path = bench.RESULTS_DIR / f"agentbench-{_time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"run_on": date_today(), "model": a.model, "fallbacks": fallbacks, "tasks": len(tasks),
            "repeats": a.repeats, "conditions": conds, "grader": judges[0], "grader_independent": independent,
            "agents": [{"name": ag.name, "description": ag.description,
                        "how": "command" if ag.command else "http"} for ag in agents],
            "questions_file": Path(a.questions).name, "tool_cache": cache.name, "replay_only": a.replay_only,
            "list_prices": "list-prices.json"}
    done: list = []

    def on_progress(rec: dict) -> None:
        done.append(rec)
        print(f"  ran {len(done)}/{n_runs}: {rec['agent'][:22]:22} {rec['condition']:13} {rec['task_id']:9} "
              f"r{rec['repeat']} {rec['seconds']}s{' ERROR ' + str(rec['agent_error'])[:60] if rec.get('agent_error') else ''}",
              flush=True)
        if len(done) % 20 == 0:  # save raw answers as we go, so a crash doesn't lose them
            path.with_suffix(".partial.json").write_text(_json.dumps({"meta": meta, "runs": [
                {k: v for k, v in r.items() if k != "_task"} for r in done]}, indent=1, ensure_ascii=False), encoding="utf-8")

    def on_run(rec: dict) -> None:
        mark = "ok " if rec["success"] else "-- "
        print(f"  {mark}{rec['agent'][:22]:22} {rec['condition']:13} {rec['task_id']:9} r{rec['repeat']} "
              f"steps={rec['steps']} tools={rec['tool_calls']} tokens={rec['tokens']} ${rec['cost_usd']:.4f} "
              f"{rec['seconds']}s {rec['grade']}", flush=True)

    try:
        result = run_suite(tasks, agents, conds, a.repeats, a.model, fallbacks, tools, llm, grade, bifrost, on_run,
                           on_progress=on_progress, pace_s=a.pace)
        print("All agents ran; reading Bifrost's log and grading (4 at a time)...", flush=True)
    finally:
        tools.stop()
    result["meta"] = dict(meta, live_searches=tools.live_searches)
    path.write_text(_json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    path.with_suffix(".md").write_text(leaderboard_markdown(result), encoding="utf-8")
    path.with_suffix(".partial.json").unlink(missing_ok=True)
    print()
    print(leaderboard_markdown(result))
    print(f"Saved: {path}\nOpen the comparison page: python -m freshpin serve, then http://127.0.0.1:{s.port}/bench")
    return 0


def prices_cmd(args: list[str]) -> int:
    """Pick one list price per model from Bifrost's own pricing table and save bench/list-prices.json."""
    import argparse
    import os
    from datetime import date
    from pathlib import Path

    from .prices import PRICES_FILE, build_from_bifrost, load_prices

    ap = argparse.ArgumentParser(prog="python -m freshpin prices")
    ap.add_argument("--models", required=True, help="comma-separated Bifrost provider/model names")
    ap.add_argument("--bifrost-config", default=os.environ.get("FRESHPIN_BIFROST_CONFIG_DB", ""),
                    help="Bifrost's config.db (default: next to FRESHPIN_BIFROST_LOGS_DB)")
    ap.add_argument("--out", default=str(PRICES_FILE))
    a = ap.parse_args(args)
    cfg = a.bifrost_config or (str(Path(settings().bifrost_logs_db).with_name("config.db"))
                               if settings().bifrost_logs_db else "")
    if not cfg or not Path(cfg).exists():
        print("Needs Bifrost's config.db: pass --bifrost-config PATH (or set FRESHPIN_BIFROST_LOGS_DB).")
        return 2
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    found = build_from_bifrost(cfg, models)
    out = Path(a.out)
    prices = dict(load_prices(out)) if out.exists() else {}
    prices.update({m: p for m, p in found.items() if p})
    out.write_text(json.dumps({"built_on": date.today().isoformat(), "unit": "USD per token",
                               "from": "Bifrost governance_model_pricing (python -m freshpin prices)",
                               "prices": dict(sorted(prices.items()))}, indent=1) + "\n", encoding="utf-8")
    for m, p in found.items():
        print(f"{m:55} " + (f"${p['input'] * 1e6:.3f} in / ${p['output'] * 1e6:.3f} out per 1M tokens  ({p['source']})"
                            if p else "no paid list price in Bifrost's table"))
    print(f"Saved: {out}")
    return 0


def date_today() -> str:
    from datetime import date
    return date.today().isoformat()


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
    ap.add_argument("--arms", help="agent arena: comma-separated arms from bare,grounded,agent,repair "
                                   "(benchmarks agent configurations, not just models)")
    ap.add_argument("--check-arms", default="bare",
                    help="arms whose answers FreshPin also claim-checks (Tavily credits!); repair needs bare")
    ap.add_argument("--self-test", action="store_true",
                    help="measure FreshPin itself: gold answers should pass, deliberately wrong versions should fail")
    a = ap.parse_args(args)
    if a.show:
        import json as _json
        saved = _json.loads(Path(a.show).read_text(encoding="utf-8"))
        if "runs" in saved and "summary" in saved:          # an agentbench result
            from .agentbench import add_list_cost, leaderboard_markdown, summarize
            add_list_cost(saved["runs"])
            saved["summary"] = summarize(saved["runs"], saved.get("meta", {}).get("repeats", 1))
            print(leaderboard_markdown(saved))
            return 0
        if "arms" in saved:                                 # an agent-arena result: re-score with the current rules
            for m in saved["models"].values():
                m["scorecard"] = {arm: bench.arena_scorecard(m["rows"][arm]) for arm in saved["arms"]}
                m["paired"] = bench.paired_tests(m["rows"], saved["arms"])
            print(bench.arena_markdown(saved))
            return 0
        saved = bench.rescore(saved)
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

    if a.arms:
        return arena_cmd(a, s, app, client, questions, models, check)

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
