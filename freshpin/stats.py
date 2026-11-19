"""Is the difference between two agents real, or chance? A paired comparison on the same tasks.

Runs are paired by TASK: each agent's success rate on a task (over its repeats) is one number, so repeats of one task
do not count as independent evidence. Two checks, both on the tasks where each agent has at least one counted run:
  - a sign test (exact, two-sided): on how many tasks was A better, on how many B better (ties dropped);
  - a 95% bootstrap interval for the mean difference in success, resampling tasks (seeded, so it is repeatable).
A difference is called real only when both agree (p < 0.05 and the interval leaves out 0). With 9 tasks the sign
test needs A better on at least 8 of 9 untied tasks, so most small gaps honestly come out as "can't tell"; with
fewer than 6 tasks that differ, not even a clean sweep can reach p < 0.05, and the verdict says "too few tasks".
Standard library only.
"""
from __future__ import annotations

import random
from itertools import combinations
from math import comb
from typing import Any, Callable

ALPHA = 0.05


def sign_test(a_better: int, b_better: int) -> float:
    """Exact two-sided sign test p-value (ties already dropped)."""
    n = a_better + b_better
    if n == 0:
        return 1.0
    k = min(a_better, b_better)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_ci(diffs: list[float], resamples: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval for the mean of diffs, resampling the items with replacement."""
    if not diffs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return (means[int(0.025 * resamples)], means[min(resamples - 1, int(0.975 * resamples))])


def per_task(runs: list[dict[str, Any]], counted: Callable[[dict[str, Any]], bool]) -> dict[str, float]:
    """Success rate per task over the counted runs."""
    by: dict[str, list[bool]] = {}
    for r in runs:
        if counted(r):
            by.setdefault(r["task_id"], []).append(bool(r.get("success")))
    return {t: sum(v) / len(v) for t, v in by.items()}


def compare(a_runs: list[dict[str, Any]], b_runs: list[dict[str, Any]],
            counted: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    """A vs B on the tasks both have. diff = A minus B (in success rate, 0..1)."""
    pa, pb = per_task(a_runs, counted), per_task(b_runs, counted)
    tasks = sorted(set(pa) & set(pb))
    diffs = [pa[t] - pb[t] for t in tasks]
    a_better, b_better = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    p = sign_test(a_better, b_better)
    lo, hi = bootstrap_ci(diffs)
    diff = sum(diffs) / len(diffs) if diffs else 0.0
    real = bool(diffs) and p < ALPHA and (lo > 0 or hi < 0)
    untied = a_better + b_better
    if not tasks:
        verdict = "no_shared_tasks"
    elif real:
        verdict = "real"
    elif not untied:
        verdict = "same"                               # equal on every shared task
    elif untied and sign_test(untied, 0) >= ALPHA:   # even a clean sweep could not reach p < 0.05
        verdict = "too_few_tasks"
    else:
        verdict = "cant_tell"
    return {"tasks": len(tasks), "a_better": a_better, "b_better": b_better, "ties": len(tasks) - untied,
            "diff": round(diff, 3), "ci95": [round(lo, 3), round(hi, 3)], "p": round(p, 4), "verdict": verdict}


def pairwise(runs: list[dict[str, Any]], counted: Callable[[dict[str, Any]], bool]) -> dict[str, list[dict[str, Any]]]:
    """Every pair of agents, per condition. Agents in first-seen order; A is the first of each pair."""
    agents: list[str] = []
    conds: list[str] = []
    for r in runs:
        if r["agent"] not in agents:
            agents.append(r["agent"])
        if r["condition"] not in conds:
            conds.append(r["condition"])
    out: dict[str, list[dict[str, Any]]] = {}
    for c in conds:
        rows = []
        for a, b in combinations(agents, 2):
            ra = [r for r in runs if r["agent"] == a and r["condition"] == c]
            rb = [r for r in runs if r["agent"] == b and r["condition"] == c]
            if ra and rb:
                rows.append({"a": a, "b": b, **compare(ra, rb, counted)})
        if rows:
            out[c] = rows
    return out
