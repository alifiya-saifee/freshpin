"""List prices for models, so a benchmark run on free tiers still shows what it WOULD cost on a paid plan.

Free tiers make every model call $0 in Bifrost's log, which hides the cost difference between agents. FreshPin keeps a
small committed price file (bench/list-prices.json): USD per input and output token for each Bifrost "provider/model"
name, with the source of each price. It is built from Bifrost's own pricing table (config.db, governance_model_pricing)
by `python -m freshpin prices`, so the numbers are Bifrost's, not FreshPin's guesses.

Choosing one price per model, in this order:
    1. the model's own maker sells it (gemini-* from "gemini", mistral-* from "mistral"),
    2. OpenRouter's paid listing (a common reference price),
    3. otherwise the median over every paid provider that lists it.
Free (0) and ":free" / ":batch" listings are ignored. A model with no listing gets no price, and costs that use it are
reported as unknown, never as $0.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from statistics import median
from typing import Any

PRICES_FILE = Path(__file__).resolve().parent.parent / "bench" / "list-prices.json"
MAKERS = {"gemini": "gemini", "gemma": "gemini", "mistral": "mistral", "gpt-oss": None}


def base_name(model: str) -> str:
    """'openai/nvidia/nemotron-3.5-lightning-30b-a3b' -> 'nemotron-3.5-lightning-30b-a3b'."""
    return model.rsplit("/", 1)[-1].strip().lower()


# Only size / precision suffixes may differ ("-30b-a3b", "-1t-a52b", "-bf16"), never a variant like "-lite" or "-mini".
_SUFFIX = re.compile(r"(-(\d+(\.\d+)?[bmt]|a\d+(\.\d+)?b|e\d+b|bf16|fp8|fp4|nvfp4|int4|int8))+")


def _same_model(a: str, b: str) -> bool:
    """True when one name is the other plus size/precision suffixes only."""
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return long_.startswith(short + "-") and bool(_SUFFIX.fullmatch(long_[len(short):]))


def _maker(name: str) -> str | None:
    for prefix, maker in MAKERS.items():
        if name.startswith(prefix):
            return maker
    return None


def pick_price(model: str, rows: list[tuple[str, str, float, float]]) -> dict[str, Any] | None:
    """rows: (provider, model, input $/token, output $/token) from Bifrost's pricing table. Returns the chosen price
    with its source, or None when no paid listing matches."""
    name = base_name(model)
    exact, close = [], []
    for prov, m, pin, pout in rows:
        mm = (m or "").lower()
        if ":free" in mm or ":batch" in mm or "-free" in mm or not (pin or pout):
            continue
        tail = mm.rsplit("/", 1)[-1]
        tail = tail[len("nvidia-"):] if tail.startswith("nvidia-") else tail
        row = (prov, m, float(pin or 0), float(pout or 0))
        if tail == name:
            exact.append(row)
        elif _same_model(name, tail):
            close.append(row)
    paid = exact or close
    if not paid:
        return None
    maker = _maker(name)
    for prov, m, pin, pout in paid:
        if maker and prov == maker:
            return {"input": pin, "output": pout, "source": f"bifrost pricing: {prov} {m} (maker's price)"}
    for prov, m, pin, pout in paid:
        if prov == "openrouter":
            return {"input": pin, "output": pout, "source": f"bifrost pricing: openrouter {m}"}
    return {"input": median(p[2] for p in paid), "output": median(p[3] for p in paid),
            "source": f"bifrost pricing: median of {len(paid)} paid providers"}


def build_from_bifrost(config_db: str | Path, models: list[str]) -> dict[str, Any]:
    """Read Bifrost's pricing table (read-only) and pick a list price for each model."""
    db = sqlite3.connect(f"file:{config_db}?mode=ro", uri=True, timeout=5)
    try:
        out: dict[str, Any] = {}
        for model in models:
            name = re.sub(_SUFFIX.pattern + "$", "", base_name(model)) or base_name(model)  # search without sizes
            rows = db.execute("SELECT provider, model, input_cost_per_token, output_cost_per_token "
                              "FROM governance_model_pricing WHERE mode='chat' AND lower(model) LIKE ?",
                              (f"%{name}%",)).fetchall()
            out[model] = pick_price(model, rows)
        return out
    finally:
        db.close()


def load_prices(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else PRICES_FILE
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("prices", data)


def call_list_cost(call: dict[str, Any], prices: dict[str, Any]) -> float | None:
    """List-price $ of one model call from the trace. A call with no tokens costs 0 (e.g. a refused outage model);
    a call with tokens but no known price is None (unknown)."""
    tin, tout = int(call.get("tokens_in") or 0), int(call.get("tokens_out") or 0)
    if not tin and not tout:
        return 0.0
    p = prices.get(call.get("model") or "")
    if not p:
        return None
    return tin * p["input"] + tout * p["output"]


def apply_list_prices(runs: list[dict[str, Any]], prices: dict[str, Any], tavily_usd_per_credit: float) -> None:
    """Add list_cost_usd to every model call in each run's trace and to the run itself: model calls at list price
    plus one Tavily credit per search that returned results. Replayed searches count too, because in production each
    one would be a real search; that keeps the cost the same whichever agent happened to search first. If any call's
    price is unknown, the run's list cost is None and list_price_missing names the models."""
    for r in runs:
        total, missing, searches = 0.0, set(), 0
        for step in r.get("trace") or []:
            if step.get("kind") == "tool":
                searches += int(step.get("status") == 200)
                continue
            if step.get("kind") != "model":
                continue
            c = call_list_cost(step, prices)
            step["list_cost_usd"] = None if c is None else round(c, 8)
            if c is None:
                missing.add(step.get("model"))
            else:
                total += c
        r["list_price_missing"] = sorted(m for m in missing if m)
        r["searches"] = searches
        r["list_cost_usd"] = (None if missing else
                              round(total + searches * tavily_usd_per_credit, 8))
