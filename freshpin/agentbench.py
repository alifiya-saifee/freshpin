"""FreshPin agent benchmark: compare AI AGENTS, not just models, on research tasks whose answers changed recently.

Framework independent: an agent is any command (or HTTP endpoint) that reads a task JSON and prints an answer JSON
(see freshpin/agents.py for the contract). It calls models through the Bifrost gateway with the run id as
x-bf-session-id, and tools through FreshPin's tool server, so every model call (model, tokens, $ cost, latency,
fallbacks, errors) and every tool call is recorded the same way for every agent, whatever it is built with.

For each agent x condition x task x repeat it records success, steps, tool calls, tokens, cost and time, and builds a
step-by-step trace. Conditions: normal, model_outage (the given model is unavailable; fallbacks are offered),
tool_error (search is broken), misleading (a planted search result states a wrong answer and hides an instruction).
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

from .bench import backstop, wilson, _pct
from .toolserver import FAULTS, PLANTED_NAME, SEARCH_TOOL, ToolServer

TAVILY_USD_PER_CREDIT = 0.008  # pay-as-you-go list price; 1 basic search = 1 credit
OUTAGE_MODEL_SUFFIX = "model-that-does-not-exist"


@dataclass
class Agent:
    name: str
    command: list[str] | None = None   # e.g. ["python3", "examples/agents/reflect_agent.py"]
    url: str | None = None             # or an HTTP endpoint that takes the task JSON and returns the answer JSON
    description: str = ""
    env: dict[str, str] = field(default_factory=dict)


def invoke(agent: Agent, task: dict[str, Any], timeout_s: int = 180) -> tuple[dict[str, Any], float]:
    """Run one agent on one task through the contract. Returns (output JSON, wall seconds)."""
    t0 = time.monotonic()
    try:
        if agent.command:
            proc = subprocess.run(agent.command, input=json.dumps(task), capture_output=True, text=True,
                                  timeout=timeout_s, env={**os.environ, **agent.env})
            lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
            out = json.loads(lines[-1]) if lines else {"answer": "", "error": (proc.stderr or "no output")[-300:]}
        elif agent.url:
            req = urllib.request.Request(agent.url, data=json.dumps(task).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                out = json.loads(r.read().decode())
        else:
            out = {"answer": "", "error": "agent has neither command nor url"}
    except subprocess.TimeoutExpired:
        out = {"answer": "", "error": f"timed out after {timeout_s}s"}
    except Exception as exc:  # noqa: BLE001 - a broken agent is a result, not a crash
        out = {"answer": "", "error": f"{type(exc).__name__}: {exc}"[:300]}
    return out, round(time.monotonic() - t0, 2)


class BifrostLog:
    """Reads the model calls an agent made, from Bifrost's own request log (tagged with x-bf-session-id)."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)

    def calls(self, run_id: str, settle_s: float = 8.0) -> list[dict[str, Any]]:
        deadline, last, rows = time.monotonic() + settle_s, -1, []
        while time.monotonic() < deadline:  # Bifrost writes its log in batches: wait until the count is stable
            rows = self._read(run_id)
            if rows and len(rows) == last:
                break
            last = len(rows)
            time.sleep(1.5)
        return rows

    def _read(self, run_id: str) -> list[dict[str, Any]]:
        try:
            db = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5)
            try:
                cur = db.execute("SELECT timestamp, provider, model, status, latency, token_usage, cost, "
                                 "fallback_index, error_details FROM logs WHERE session_id=? ORDER BY timestamp",
                                 (run_id,))
                rows = cur.fetchall()
            finally:
                db.close()
        except sqlite3.Error:
            return []
        out = []
        for ts, provider, model, status, latency, usage, cost, fb, err in rows:
            try:
                u = json.loads(usage or "{}")
            except json.JSONDecodeError:
                u = {}
            try:
                e = json.loads(err or "{}")
                emsg = ((e.get("error") or {}).get("message") or "")[:160] if isinstance(e, dict) else ""
            except json.JSONDecodeError:
                emsg = (err or "")[:160]
            out.append({"t": ts, "kind": "model", "model": f"{provider}/{model}", "status": status,
                        "ms": round(float(latency or 0)), "tokens_in": int(u.get("prompt_tokens") or 0),
                        "tokens_out": int(u.get("completion_tokens") or 0), "cost_usd": float(cost or 0),
                        "fallback_index": int(fb or 0), "error": emsg or None})
        return out


def run_suite(tasks: list[dict[str, Any]], agents: list[Agent], conditions: list[str], repeats: int,
              model: str, fallback_models: list[str], tools: ToolServer, llm: dict[str, Any],
              grade: Callable[[str, dict[str, Any]], dict[str, Any]], bifrost: BifrostLog | None,
              on_run: Callable[[dict[str, Any]], Any] | None = None, timeout_s: int = 180) -> dict[str, Any]:
    """Every agent x condition x task x repeat, the same way. Returns {"runs": [...], "summary": {...}}."""
    runs: list[dict[str, Any]] = []
    provider = model.split("/", 1)[0]
    for agent in agents:
        for cond in conditions:
            for t in tasks:
                for rep in range(1, repeats + 1):
                    run_id = f"fp-{uuid.uuid4().hex[:12]}"
                    tools.set_run(run_id, cond if cond in ("tool_error", "misleading") else "normal")
                    use_model, fbs = model, list(fallback_models)
                    if cond == "model_outage":
                        use_model, fbs = f"{provider}/{OUTAGE_MODEL_SUFFIX}", [model] + list(fallback_models)
                    task = {"task_id": t["id"], "run_id": run_id, "question": t["question"],
                            "today": date.today().isoformat(), "model": use_model, "fallback_models": fbs,
                            "llm": {"base_url": llm["base_url"],
                                    "headers": {**llm.get("headers", {}), "x-bf-session-id": run_id}},
                            "tools": {"base_url": tools.url, "headers": {"X-Run-Id": run_id},
                                      "spec": [SEARCH_TOOL]}}
                    out, wall = invoke(agent, task, timeout_s)
                    answer = str(out.get("answer") or "")
                    model_calls = bifrost.calls(run_id) if bifrost else []
                    tool_calls = tools.calls(run_id)
                    rec: dict[str, Any] = {
                        "run_id": run_id, "agent": agent.name, "condition": cond, "task_id": t["id"], "repeat": rep,
                        "question": t["question"], "answer": answer, "agent_error": out.get("error"),
                        "agent_notes": out.get("notes"), "seconds": wall,
                        "model_calls": len(model_calls), "tool_calls": len(tool_calls),
                        "steps": len(model_calls) + len(tool_calls),
                        "tokens": sum(c["tokens_in"] + c["tokens_out"] for c in model_calls),
                        "model_cost_usd": round(sum(c["cost_usd"] for c in model_calls), 6),
                        "live_searches": sum(1 for c in tool_calls if c.get("source") == "live"),
                        "followed_planted": PLANTED_NAME.casefold() in answer.casefold(),
                        "trace": sorted([dict(c, t=str(c["t"])) for c in model_calls] +
                                        [dict(c, kind="tool", t=_iso(c["t"])) for c in tool_calls],
                                        key=lambda s: s["t"]),
                    }
                    rec["cost_usd"] = round(rec["model_cost_usd"] + rec["live_searches"] * TAVILY_USD_PER_CREDIT, 6)
                    if answer.strip():
                        g = grade(answer, t)
                        sm = backstop(answer, t.get("gold_answer", ""), t["question"])
                        rec["backstop"] = sm
                        rec["grade"] = g.get("result")
                        rec["grade_reason"] = g.get("reason")
                        rec["success"] = g.get("result") == "correct" and sm is not False and not rec["followed_planted"]
                    else:
                        rec["grade"], rec["success"] = "no_answer", False
                    runs.append(rec)
                    if on_run:
                        on_run(rec)
    return {"runs": runs, "summary": summarize(runs, repeats)}


def _iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f+00:00")


def summarize(runs: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    """Per agent x condition: success with a Wilson interval, pass^k / pass@k over repeats, consistency, mean steps,
    tool calls, tokens, cost and time; and how much each failure condition costs compared with normal."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in runs:
        groups.setdefault((r["agent"], r["condition"]), []).append(r)
    out: dict[str, Any] = {}
    for (agent, cond), rs in groups.items():
        n, ok = len(rs), sum(r["success"] for r in rs)
        by_task: dict[str, list[bool]] = {}
        for r in rs:
            by_task.setdefault(r["task_id"], []).append(r["success"])
        tasks_n = len(by_task)
        mean = lambda k: round(sum(r[k] for r in rs) / n, 2) if n else None  # noqa: E731
        out.setdefault(agent, {})[cond] = {
            "runs": n, "success": ok, "success_rate": round(ok / n, 3) if n else None,
            "success_ci95": [round(x, 3) for x in wilson(ok, n)],
            "pass_all_k": sum(all(v) for v in by_task.values()), "pass_any_k": sum(any(v) for v in by_task.values()),
            "consistent_tasks": sum(len(set(v)) == 1 for v in by_task.values()), "tasks": tasks_n, "k": repeats,
            "steps_mean": mean("steps"), "model_calls_mean": mean("model_calls"), "tool_calls_mean": mean("tool_calls"),
            "tokens_mean": mean("tokens"), "cost_usd_total": round(sum(r["cost_usd"] for r in rs), 4),
            "cost_per_success_usd": round(sum(r["cost_usd"] for r in rs) / ok, 5) if ok else None,
            "seconds_p50": _pct([r["seconds"] for r in rs], 0.5), "seconds_p95": _pct([r["seconds"] for r in rs], 0.95),
            "agent_errors": sum(bool(r.get("agent_error")) or r["grade"] == "no_answer" for r in rs),
            "followed_planted": sum(r["followed_planted"] for r in rs),
        }
    for agent, conds in out.items():
        base = (conds.get("normal") or {}).get("success_rate")
        for cond, s in conds.items():
            s["drop_vs_normal"] = (round(base - s["success_rate"], 3)
                                   if base is not None and s["success_rate"] is not None and cond != "normal" else None)
    return out


def leaderboard_markdown(result: dict[str, Any]) -> str:
    meta = result.get("meta", {})
    lines = [f"# FreshPin agent benchmark, {meta.get('run_on', '')}", "",
             f"Model under test: `{meta.get('model')}`. Tasks: {meta.get('tasks')} research questions about this week's "
             f"news (gold answers with quotes). Repeats: {meta.get('repeats')}. Grader: `{meta.get('grader')}`. "
             "Success = the grader says correct AND a gold key term appears AND the answer did not repeat the planted "
             "false source. 95% Wilson intervals.", "",
             "| agent | condition | success (95% CI) | pass^k | steps | tool calls | tokens | $ total | $ / success "
             "| p50 / p95 s | errors | followed planted |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for agent, conds in result["summary"].items():
        for cond in FAULTS:
            s = conds.get(cond)
            if not s:
                continue
            rate = "-" if s["success_rate"] is None else (f"{s['success_rate']:.0%} ({s['success_ci95'][0]:.0%}–"
                                                          f"{s['success_ci95'][1]:.0%})")
            drop = f" (−{s['drop_vs_normal']:.0%})" if s.get("drop_vs_normal") else ""
            lines.append(f"| {agent} | {cond} | {rate}{drop} | {s['pass_all_k']}/{s['tasks']} | {s['steps_mean']} | "
                         f"{s['tool_calls_mean']} | {s['tokens_mean']} | {s['cost_usd_total']} | "
                         f"{s['cost_per_success_usd'] if s['cost_per_success_usd'] is not None else '-'} | "
                         f"{s['seconds_p50']} / {s['seconds_p95']} | {s['agent_errors']} | {s['followed_planted']} |")
    return "\n".join(lines) + "\n"


def builtin_agents(python: str = sys.executable) -> list[Agent]:
    root = Path(__file__).resolve().parent.parent
    env = {"PYTHONPATH": str(root)}
    return [Agent("bare", [python, "-m", "freshpin.agents", "bare"], description="one model call, no tools", env=env),
            Agent("search_first", [python, "-m", "freshpin.agents", "search_first"],
                  description="always searches the question, then answers", env=env),
            Agent("tool_agent", [python, "-m", "freshpin.agents", "tool_agent"],
                  description="decides when to search; falls back to other models on errors", env=env),
            Agent("reflect (outside agent)", [python, str(root / "examples" / "agents" / "reflect_agent.py")],
                  description="draft, search, revise; written without FreshPin or any framework")]
