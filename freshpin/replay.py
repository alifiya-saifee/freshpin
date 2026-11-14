"""Offline backends that replay recorded responses (fixtures/replay.json).

They exist so the test suite and UI can run without API keys or internet. They follow the same
interfaces as the live Bifrost and Tavily clients, including outage simulation, and every trace is
marked "replay" so nobody mistakes it for a live check.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ROOT
from .llm import CallTrace, LLMUnavailable, RunBudget
from .search import Source

FIXTURES = ROOT / "fixtures" / "replay.json"
SAMPLES = ROOT / "samples" / "samples.json"


def load_samples() -> dict[str, Any]:
    return json.loads(SAMPLES.read_text(encoding="utf-8"))


class ReplayData:
    def __init__(self, path: Path = FIXTURES):
        self.data = json.loads(path.read_text(encoding="utf-8"))
        self.samples = load_samples()
        self.query_to_key = {c["search_query"]: c["key"]
                             for ex in self.data["extract"].values() for c in ex["claims"]}


class ReplayChat:
    def __init__(self, data: ReplayData, outage_model: str = "openai/model-that-does-not-exist"):
        self.d = data
        self.outage_model = outage_model

    def chat_json(self, step: str, models: list[str], system: str, user: str, budget: RunBudget,
                  simulate_outage: bool = False) -> dict[str, Any]:
        budget.check()
        if simulate_outage:
            budget.log(CallTrace(step=step, requested_model=self.outage_model, ok=False, latency_ms=40,
                                 served_by=None, error="replay: simulated provider outage"))
        if step == "extract":
            result = next((self.d.data["extract"][sid] for sid, s in self.d.samples.items()
                           if s["answer"].strip() in user), None)
        else:
            result = self.d.data["verify"].get(step.split(":", 1)[1])
        if result is None:
            budget.log(CallTrace(step=step, requested_model=models[0], ok=False,
                                 error="replay: no recording for this input"))
            raise LLMUnavailable("no replay recording for this input; use live mode for custom answers")
        tokens_in, tokens_out = len(system + user) // 4, len(json.dumps(result)) // 4
        served = models[0]
        budget.charge(tokens_in + tokens_out)
        budget.log(CallTrace(step=step, requested_model=self.outage_model if simulate_outage else served,
                             served_by=f"{served} (replay)", fallback_used=simulate_outage,
                             prompt_tokens=tokens_in, completion_tokens=tokens_out, latency_ms=5))
        return json.loads(json.dumps(result))


class ReplaySearch:
    def __init__(self, data: ReplayData):
        self.d = data

    def search(self, query: str, claim_type: str, max_results: int = 5) -> list[Source]:
        key = self.d.query_to_key.get(query)
        rows = self.d.data["search"].get(key, []) if key else []
        return [Source(**r) for r in rows[:max_results]]
