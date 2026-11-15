"""Wiring: builds the right backends for the mode and exposes the operations the UI and CLI use."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Any

from .config import Settings
from .ledger import Ledger
from .llm import BifrostClient
from .pipeline import run_check
from .replay import ReplayChat, ReplayData, ReplaySearch, load_samples
from .search import CachedSearch, TavilyClient


class FreshPin:
    def __init__(self, s: Settings):
        self.s = s
        self.ledger = Ledger(s.db_path)
        self.lock = threading.Lock()
        if s.mode == "replay":
            data = ReplayData()
            self.chat = ReplayChat(data, s.outage_model)
            self.search = ReplaySearch(data)
        else:
            self.chat = BifrostClient(s.bifrost_url, s.bifrost_virtual_key, s.timeout_s, s.outage_model,
                                      tuple(s.no_thinking_prefixes))
            self.search = CachedSearch(TavilyClient(s.tavily_api_key, s.tavily_url, s.timeout_s))

    def status(self) -> dict[str, Any]:
        bifrost_ok = None
        if self.s.mode == "live":
            try:
                with urllib.request.urlopen(f"{self.s.bifrost_url}/health", timeout=3) as r:
                    bifrost_ok = r.status < 500
            except urllib.error.HTTPError as exc:  # the server answered, so the gateway is up
                bifrost_ok = exc.code < 500
            except Exception:  # noqa: BLE001
                bifrost_ok = False
        return {"mode": self.s.mode, "bifrost_url": self.s.bifrost_url, "bifrost_reachable": bifrost_ok,
                "tavily_key_set": bool(self.s.tavily_api_key) or self.s.mode == "replay",
                "extractor_models": self.s.extractor_models, "verifier_models": self.s.verifier_models,
                "checker_models": self.s.checker_models, "recheck": self.s.recheck,
                "run_token_budget": self.s.run_token_budget, "virtual_key_set": bool(self.s.bifrost_virtual_key)}

    def samples(self) -> dict[str, Any]:
        return {k: {"title": v["title"], "context": v["context"], "answer": v["answer"]}
                for k, v in load_samples().items()}

    def load_sample(self, sample_id: str) -> dict[str, Any]:
        sample = load_samples()[sample_id]
        with self.lock:
            self.ledger.reset()
            versions = {f["key"]: self.ledger.seed_fact(f["key"], f["statement"], f["type"])
                        for f in sample["seed"]["facts"]}
            for d in sample["seed"]["decisions"]:
                self.ledger.add_decision(d["id"], d["title"], d["kind"], d["owner"],
                                         {k: versions[k] for k in d["pins"]}, d["approved_by"], d["approved_at"])
            return self.ledger.state()

    def check(self, answer: str, context: str = "", simulate_outage: bool = False,
              apply_to_ledger: bool = True) -> dict[str, Any]:
        if not answer.strip():
            raise ValueError("answer text is empty")
        if len(answer) > 12000:
            raise ValueError("answer is too long (max 12,000 characters)")
        with self.lock:
            report = run_check(answer, chat=self.chat, search=self.search, ledger=self.ledger,
                               extractor_models=self.s.extractor_models, verifier_models=self.s.verifier_models,
                               token_budget=self.s.run_token_budget, max_sources=self.s.max_sources_per_claim,
                               context=context, simulate_outage=simulate_outage,
                               checker_models=self.s.checker_models, recheck=self.s.recheck,
                               apply_to_ledger=apply_to_ledger, model_concurrency=self.s.model_concurrency)
            report["mode"] = self.s.mode
            report["ledger"] = self.ledger.state()
            return report

    def ground(self, question: str) -> dict[str, Any]:
        """Dated sources for a question BEFORE it is answered (no ledger changes)."""
        from .pipeline import ground_question
        if not question.strip():
            raise ValueError("question is empty")
        return ground_question(question, chat=self.chat, search=self.search, models=self.s.extractor_models,
                               max_sources=self.s.max_sources_per_claim)

    def approve(self, decision_id: str, by: str, note: str = "") -> dict[str, Any]:
        with self.lock:
            self.ledger.approve(decision_id, by, note)
            return self.ledger.state()

    def pin_decision(self, title: str, owner: str, keys: list[str], by: str) -> dict[str, Any]:
        """Save a new decision that relies on the facts just checked."""
        if not title.strip() or not by.strip() or not keys:
            raise ValueError("title, approver and at least one fact are required")
        with self.lock:
            pins = {}
            for k in keys:
                cur = self.ledger.current_fact(k)
                if not cur:
                    raise ValueError(f"unknown fact {k}")
                pins[k] = cur["version"]
            did = f"D-{abs(hash(title + json.dumps(pins))) % 10000:04d}"
            self.ledger.add_decision(did, title, "decision", owner or "Me", pins, by)
            return self.ledger.state()
