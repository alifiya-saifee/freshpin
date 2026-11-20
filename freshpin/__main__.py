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
  python -m freshpin regrade FILE --graders a,b   grade a saved agentbench result again with another grader
                                           (keeps every trace; reports how often the two graders agree)
  python -m freshpin otel-export FILE [--out F]  write an agentbench result as OpenTelemetry traces (OTLP/JSON):
                                           one trace per run, gen_ai.* spans for the agent, model calls and tool calls
  python -m freshpin gate FILE --min-success 0.8 [--max-answered-anyway 0] [--agent NAME]
                                           pass/fail an agentbench result for CI (exit 1 when a rule fails)
  python -m freshpin significance FILE     is each agent-vs-agent difference real or chance? (paired by task)
  python -m freshpin bench-static DIR      write a read-only snapshot of /bench (committed samples only) for
                                           static hosting such as GitHub Pages
  python -m freshpin prices --models a,b   write bench/list-prices.json from Bifrost's pricing table (list-price $)
  python -m freshpin bench-build [--n 8]   build a fresh-events question set (gold answers + quotes) from this week's news
  python -m freshpin watch [--all]         re-check pinned facts past their half-life; alert on reopened decisions
  python -m freshpin check "answer text"   check an answer and print the JSON report
  python -m freshpin sample <id>           load a sample ledger, check its answer, print a summary
"""


def reserve_result_path(results_dir, stamp: str | None = None):
    """A new agentbench-<date>-<time>[-N].json name nobody else is using. Runs started in the same second (e.g. two
    models benchmarked in parallel) used to get the same name and overwrite each other's results. The name is
    reserved with an atomic create of <name>.reserved, removed once the result is written."""
    import os
    import time as _time
    from pathlib import Path
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or _time.strftime("%Y%m%d-%H%M%S")
    n = 1
    while True:
        path = results_dir / (f"agentbench-{stamp}.json" if n == 1 else f"agentbench-{stamp}-{n}.json")
        n += 1
        if path.exists():
            continue
        try:
            os.close(os.open(path.with_suffix(".reserved"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        if path.exists():  # finished between the check and the reservation
            path.with_suffix(".reserved").unlink(missing_ok=True)
            continue
        return path


def _paced(fn, seconds: float):
    """Calls fn at most once every `seconds` (shared across threads): stays under a provider's per-minute limit."""
    import threading
    import time as _time
    lock, last = threading.Lock(), [0.0]

    def wrapped(*args, **kw):
        with lock:
            wait = last[0] + seconds - _time.time()
            if wait > 0:
                _time.sleep(wait)
            last[0] = _time.time()
        return fn(*args, **kw)
    return wrapped


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
    if cmd == "regrade":
        return regrade_cmd(argv[1:])
    if cmd == "otel-export":
        return otel_cmd(argv[1:])
    if cmd == "gate":
        return gate_cmd(argv[1:])
    if cmd == "significance":
        return significance_cmd(argv[1:])
    if cmd == "bench-static":
        if len(argv) != 2 or argv[1].startswith("-"):
            print("usage: python -m freshpin bench-static DIR   (a new or empty folder)", file=sys.stderr)
            return 2
        from pathlib import Path

        from .benchstatic import build
        try:
            n = build(Path(argv[1]))
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        print(f"{n['results']} results -> {argv[1]} (open index.html through a web server, e.g. "
              f"python -m http.server -d {argv[1]})")
        return 0
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
    from .toolserver import ALL_FAULTS, FAULTS, ToolServer

    ap = argparse.ArgumentParser(prog="python -m freshpin agentbench",
                                 description="Benchmark AI agents (any framework) on research tasks that changed recently.")
    ap.add_argument("--questions", required=True, help="task suite JSON with gold answers (bench-build makes one)")
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2, help="runs per task, to measure reliability")
    ap.add_argument("--conditions", default=",".join(FAULTS), help="normal,model_outage,tool_error,misleading,misleading_prompt "
                    "(default; misleading_prompt = a false hint in the task); also model_outage_gateway (rejected model, no "
                    "fallback list for the agent; Bifrost's own fallbacks added by FreshPin), request_cap (a Bifrost key per run "
                    "capped at --request-cap requests; needs FRESHPIN_BIFROST_ADMIN), slow_tool (search answers after "
                    "FRESHPIN_SLOW_TOOL_S seconds), wrong_results (search returns pages about other stories), "
                    "flaky_tool (the first FRESHPIN_FLAKY_FAILS searches of a run fail, then search works: does the "
                    "agent retry?), model_outage_midrun (the model answers the first call, then goes down; needs "
                    "--fallbacks) and stale_memory (task.memory holds an old note with a wrong answer: does the agent "
                    "re-check?)")
    ap.add_argument("--agents", default="all",
                    help="built-in names (bare,search_first,search_retry,tool_agent,reflect) or all")
    ap.add_argument("--agent", action="append", default=[], metavar="NAME=COMMAND",
                    help="add an outside agent run as a command, e.g. --agent 'mine=python3 my_agent.py'")
    ap.add_argument("--agent-url", action="append", default=[], metavar="NAME=URL", help="add an outside agent over HTTP")
    ap.add_argument("--model", required=True, help="model under test, Bifrost provider/model")
    ap.add_argument("--fallbacks", default="", help="fallback models offered to agents (comma-separated)")
    ap.add_argument("--search-type", default="event", help="claim type for live searches (event = news, last week)")
    ap.add_argument("--pace", type=float, default=0.0, help="seconds to wait between runs (free-tier rate limits)")
    ap.add_argument("--request-cap", type=int, default=2,
                    help="request_cap condition: requests each run's Bifrost key allows (default 2)")
    ap.add_argument("--step-budget", type=int, default=None, metavar="N",
                    help="tell every agent it has N steps (model + tool calls) in task.budget.max_steps, and count "
                    "the runs that used more (over_budget); the answer is still graded as usual")
    ap.add_argument("--tool-cache", default="", help="recorded searches to replay (default: toolcache-<suite>.json "
                    "next to the suite file, else in bench/)")
    ap.add_argument("--replay-only", action="store_true",
                    help="never search live: replay the recorded searches only (no Tavily credits). A query that "
                    "was never recorded gets the closest recorded pages, so every model searches the same pages")
    ap.add_argument("--graders", default="", help="grader models in order (comma-separated); the first one from a "
                    "different family than --model grades. Fix this to compare models under the same grader")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--fail-under", type=float, default=None, metavar="RATE",
                    help="exit 1 if any agent's normal success rate is below RATE (0.8 or 80): a CI gate. "
                         "For more rules run python -m freshpin gate on the saved result")
    a = ap.parse_args(args)
    unknown = [c for c in a.conditions.split(",") if c and c not in ALL_FAULTS]
    if unknown:
        print(f"Unknown condition(s): {', '.join(unknown)}. Choose from {', '.join(ALL_FAULTS)}.")
        return 2

    s = settings()
    if s.mode != "live":
        print("agentbench runs real agents, so it needs FRESHPIN_MODE=live (Bifrost + Tavily).")
        return 2
    conds = [c for c in a.conditions.split(",") if c in ALL_FAULTS]
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
    grader_pool = ([m.strip() for m in a.graders.split(",") if m.strip()] or
                   list(dict.fromkeys(s.checker_models + s.verifier_models + s.extractor_models)))
    judges, independent = bench.independent_models(a.model, grader_pool)
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
    keys = None
    if "request_cap" in conds:
        from .bifrostkeys import BifrostKeys
        keys = BifrostKeys.from_env(s.bifrost_url)
        try:
            keys.check()
        except RuntimeError as e:
            print(f"request_cap can't run: {e}")
            return 2
    tools = ToolServer(search, cache_path=cache, live=not a.replay_only).start()
    headers = {"x-bf-vk": s.bifrost_virtual_key} if s.bifrost_virtual_key else {}
    llm = {"base_url": s.bifrost_url.rstrip("/") + "/v1", "headers": headers}
    path = reserve_result_path(bench.RESULTS_DIR)
    meta = {"run_on": date_today(), "model": a.model, "fallbacks": fallbacks, "tasks": len(tasks),
            "repeats": a.repeats, "conditions": conds, "grader": judges[0], "grader_independent": independent,
            **({"request_cap": a.request_cap} if "request_cap" in conds else {}),
            **({"step_budget": a.step_budget} if a.step_budget else {}),
            "agents": [{"name": ag.name, "description": ag.description,
                        "how": "command" if ag.command else "http"} for ag in agents],
            "questions_file": Path(a.questions).name, "tool_cache": cache.name, "replay_only": a.replay_only,
            "list_prices": "list-prices.json",
            "replay": ("recorded pages only: exact query, else the closest recorded pages" if a.replay_only
                       else "recorded once, then replayed; new queries searched live")}
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
                           on_progress=on_progress, pace_s=a.pace, keys=keys, request_cap=a.request_cap,
                           step_budget=a.step_budget)
        print("All agents ran; reading Bifrost's log and grading (4 at a time)...", flush=True)
    finally:
        tools.stop()
    result["meta"] = dict(meta, live_searches=tools.live_searches)
    path.write_text(_json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    path.with_suffix(".md").write_text(leaderboard_markdown(result), encoding="utf-8")
    path.with_suffix(".partial.json").unlink(missing_ok=True)
    path.with_suffix(".reserved").unlink(missing_ok=True)
    print()
    print(leaderboard_markdown(result))
    print(f"Saved: {path}\nOpen the comparison page: python -m freshpin serve, then http://127.0.0.1:{s.port}/bench")
    if a.fail_under is not None:
        from .gate import check, parse_min_success, report_markdown
        checks = check(result, parse_min_success([str(a.fail_under)]))
        print(report_markdown(checks, path.name))
        return 0 if all(c["ok"] for c in checks) else 1
    return 0


def _load_result(name: str):
    from pathlib import Path
    src = Path(name)
    try:
        result = json.loads(src.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"Can't read {src}: {e}", file=sys.stderr)
        return None
    if not isinstance(result, dict) or not result.get("runs"):
        print(f"Not an agentbench result (no runs): {src}", file=sys.stderr)
        return None
    return result


def gate_cmd(args: list[str]) -> int:
    """Pass/fail a saved agentbench result (exit 0 pass, 1 a rule failed, 2 bad input), for CI."""
    import argparse
    import os

    from .gate import check, parse_min_success, report_markdown
    ap = argparse.ArgumentParser(prog="python -m freshpin gate",
                                 description="Fail CI when agents fall below your bar. Rules apply per agent and "
                                             "condition, to counted runs only (provider limits are left out).")
    ap.add_argument("result", help="an agentbench result file")
    ap.add_argument("--min-success", action="append", default=[], metavar="[COND=]RATE",
                    help="success rate to reach, e.g. 0.8 (normal) or tool_error=0.5; repeatable")
    ap.add_argument("--lower-bound", action="store_true",
                    help="compare the low end of the 95%% interval, not the rate (stricter with few runs)")
    ap.add_argument("--max-answered-anyway", type=int, default=None, metavar="N",
                    help="at most N made-up answers when search is broken or off-topic")
    ap.add_argument("--max-planted", type=int, default=None, metavar="N",
                    help="at most N answers that went along with the planted false source")
    ap.add_argument("--max-over-budget", type=int, default=None, metavar="N",
                    help="at most N runs over the step budget (results run with --step-budget)")
    ap.add_argument("--agent", action="append", default=[], help="check only these agents (default: all)")
    a = ap.parse_args(args)
    try:
        rules = parse_min_success(a.min_success)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if not rules and a.max_answered_anyway is None and a.max_planted is None and a.max_over_budget is None:
        print("Give at least one rule: --min-success, --max-answered-anyway, --max-planted or --max-over-budget",
              file=sys.stderr)
        return 2
    result = _load_result(a.result)
    if result is None:
        return 2
    checks = check(result, rules, a.agent or None, a.max_answered_anyway, a.max_planted, a.lower_bound,
                   a.max_over_budget)
    if not checks:
        print("No rule applied to this result (wrong condition names?)", file=sys.stderr)
        return 2
    md = report_markdown(checks, os.path.basename(a.result))
    print(md)
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:  # shown on the GitHub Actions run page
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    return 0 if all(c["ok"] for c in checks) else 1


def significance_cmd(args: list[str]) -> int:
    """Print every agent-vs-agent comparison of a saved result: is the gap real, or could it be chance?"""
    import argparse

    from .agentbench import rescore
    ap = argparse.ArgumentParser(prog="python -m freshpin significance",
                                 description="Paired by task: a sign test and a 95%% bootstrap interval. A difference "
                                             "is called real only when both agree.")
    ap.add_argument("result", help="an agentbench result file")
    ap.add_argument("--condition", default="", help="only this condition (default: all)")
    a = ap.parse_args(args)
    result = _load_result(a.result)
    if result is None:
        return 2
    pw = rescore(result)["pairwise"]
    print("| condition | A | B | tasks | A better / B better / tie | A − B (95% interval) | p | verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for cond, rows in pw.items():
        if a.condition and cond != a.condition:
            continue
        for x in rows:
            verdict = {"real": "real difference", "cant_tell": "can't tell (could be chance)",
                       "no_shared_tasks": "no shared tasks", "same": "same on every task",
                       "too_few_tasks": "too few tasks to tell (fewer than 6 differ)"}[x["verdict"]]
            print(f"| {cond} | {x['a']} | {x['b']} | {x['tasks']} | {x['a_better']} / {x['b_better']} / {x['ties']} "
                  f"| {x['diff'] * 100:+.0f} pts ({x['ci95'][0] * 100:+.0f} to {x['ci95'][1] * 100:+.0f}) | "
                  f"{x['p']:.3f} | {verdict} |")
    return 0


def otel_cmd(args: list[str]) -> int:
    """Write a saved agentbench result as OpenTelemetry traces (OTLP/JSON), for Jaeger, Tempo, Langfuse, Phoenix..."""
    import argparse
    from pathlib import Path

    from .otel import to_otlp
    ap = argparse.ArgumentParser(prog="python -m freshpin otel-export",
                                 description="Write an agentbench result as OpenTelemetry traces (OTLP/JSON). "
                                             "POST the file to a collector's /v1/traces, or import it into a tracing tool.")
    ap.add_argument("result", help="an agentbench result file (bench/samples/... or bench/results/...)")
    ap.add_argument("--out", default="", help="output file (default: print to stdout)")
    a = ap.parse_args(args)
    src = Path(a.result)
    if not src.is_file():
        print(f"No such file: {src}", file=sys.stderr)
        return 2
    try:
        result = json.loads(src.read_text(encoding="utf-8"))
    except ValueError as e:
        print(f"Not a JSON file: {src} ({e})", file=sys.stderr)
        return 2
    if not isinstance(result, dict) or not isinstance(result.get("runs"), list):
        print(f"Not an agentbench result (no runs): {src}", file=sys.stderr)
        return 2
    out = json.dumps(to_otlp(result, source=src.name), ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(out + "\n", encoding="utf-8")
        n = sum(len(ss["spans"]) for rs in json.loads(out)["resourceSpans"] for ss in rs["scopeSpans"])
        print(f"{len(result['runs'])} runs, {n} spans -> {a.out}")
    else:
        print(out)
    return 0


def regrade_cmd(args: list[str]) -> int:
    """Grade a saved agentbench result again (e.g. when its grader was down), and measure grader agreement."""
    import argparse
    import time as _time
    from pathlib import Path

    from . import bench
    from .agentbench import leaderboard_markdown, regrade
    from .benchbuild import make_gold_judge

    ap = argparse.ArgumentParser(prog="python -m freshpin regrade")
    ap.add_argument("file", help="a saved agentbench-*.json result")
    ap.add_argument("--graders", required=True, help="grader models in order; the first from another family grades")
    ap.add_argument("--questions", default="", help="the task suite (default: the one named in the result)")
    ap.add_argument("--workers", type=int, default=4, help="grader calls at once (1 for free tiers with per-minute "
                    "limits)")
    ap.add_argument("--pace", type=float, default=0.0, help="seconds between grader calls (e.g. 4.5 for a free tier "
                    "of 15 requests a minute; use with --workers 1)")
    ap.add_argument("--consensus", action="store_true", help="keep BOTH grades: a run counts only when the old and "
                    "the new grader agree on correct-or-not (disagreements are reported and excluded)")
    a = ap.parse_args(args)
    src = Path(a.file)
    if not src.exists():
        print(f"No such file: {src}")
        return 2
    result = json.loads(src.read_text(encoding="utf-8"))
    if "runs" not in result:
        print("Not an agentbench result (no runs).")
        return 2
    meta = result.get("meta", {})
    qfile = Path(a.questions) if a.questions else next(
        (p for p in (bench.RESULTS_DIR.parent / "suites" / meta.get("questions_file", ""),
                     bench.RESULTS_DIR.parent / meta.get("questions_file", "")) if p.is_file()), None)
    if not qfile or not qfile.exists():
        print("Can't find the task suite: pass --questions FILE.")
        return 2
    tasks = {q["id"]: q for q in bench.load_questions(qfile)}
    missing = sorted({r["task_id"] for r in result["runs"]} - set(tasks))
    if missing:
        print(f"The suite has no gold answers for: {', '.join(missing)}")
        return 2
    s = settings()
    if s.mode != "live":
        print("regrade calls grader models, so it needs FRESHPIN_MODE=live (Bifrost).")
        return 2
    judges, independent = bench.independent_models(meta.get("model", ""), [m.strip() for m in a.graders.split(",") if m.strip()])
    if not independent:
        print(f"warning: no grader outside the model's family ({meta.get('model')}); grading anyway")
    app = FreshPin(s)
    print(f"Regrading {len(result['runs'])} runs with {judges[0]} (was {meta.get('grader')})...", flush=True)
    judge = make_gold_judge(app.chat, judges, bench.admits_live_limits)
    if a.pace > 0:
        judge = _paced(judge, a.pace)
    regrade(result, tasks, judge, judges[0], workers=max(1, a.workers), consensus=a.consensus)
    # agentbench-<digits>.json, so the /bench page lists it (meta.regraded says what changed)
    out = reserve_result_path(bench.RESULTS_DIR)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    out.with_suffix(".md").write_text(leaderboard_markdown(result), encoding="utf-8")
    out.with_suffix(".reserved").unlink(missing_ok=True)
    print(leaderboard_markdown(result))
    print(f"Grader agreement: {result['meta']['regraded']}")
    if a.consensus:
        print(f"Consensus: {result['meta']['consensus']['disagreements']} runs where the graders disagree (excluded)")
    print(f"Saved: {out}")
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
            from .agentbench import add_list_cost, leaderboard_markdown, rescore
            add_list_cost(saved["runs"])
            rescore(saved)  # the current rules (quota detection, planted source), no model calls
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
