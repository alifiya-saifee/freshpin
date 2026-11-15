"""Personal data never leaves FreshPin: it is masked before any web search or model call.

Free and trial model tiers may use what you send them to improve their products (Gemini unpaid, NVIDIA API
trial), so customer data must not reach them. Masking is the default; FRESHPIN_CUSTOMER_DATA=1 also refuses to
run on routes listed in FRESHPIN_TRIAL_PREFIXES.
"""
from __future__ import annotations

import re

_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("emirates_id", re.compile(r"\b784[- ]?\d{4}[- ]?\d{7}[- ]?\d\b")),
    ("iban", re.compile(r"\bAE\d{2}\s?(?:\d{4}\s?){4}\d{3}\b", re.I)),
    ("card", re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")),
    ("phone", re.compile(r"(?<![\w.])(?:\+971|00971|0)[\s-]?(?:5\d|[2-79])[\s-]?\d{3}[\s-]?\d{4}\b")),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Replace personal data with [EMAIL], [PHONE], ... Returns the masked text and counts by kind."""
    counts: dict[str, int] = {}
    for kind, pat in _PATTERNS:
        text, n = pat.subn(f"[{kind.upper()}]", text or "")
        if n:
            counts[kind] = n
    return text, counts


def trial_routes(models: list[str], prefixes: list[str]) -> list[str]:
    return [m for m in models if any(p and m.startswith(p) for p in prefixes)]
