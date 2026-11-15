"""FreshPin as an MCP server, so AI assistants (Claude Code, Claude Desktop, Cursor...) can check
an answer for stale facts before they give it to a person.

Standard library only: JSON-RPC 2.0 over stdin/stdout, one JSON message per line.
Run with:  python -m freshpin mcp

Tools:
  check_answer  checks an answer (Tavily evidence + models via Bifrost), updates the pin ledger and
                reports stale claims and any decisions that reopened.
  get_ledger    read-only view of facts, decisions and recent events.

There is deliberately no approve tool: an AI may check and reopen, but only a named person can
re-approve a decision, in the web app.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Callable, TextIO

from . import __version__
from .app import FreshPin
from .config import settings

DEFAULT_PROTOCOL = "2025-06-18"

TOOLS = [
    {
        "name": "check_answer",
        "description": (
            "Check an AI answer for stale or wrong facts before relying on it. Splits the answer into "
            "time-sensitive claims (prices, API versions, laws, company facts, statistics), searches the "
            "live web with Tavily, has a second model judge each claim using only those sources, then rechecks "
            "every claim (a second search for recent changes, an independent judge on another provider, and an "
            "exact quote from the best source; disagreements come back DISPUTED), and gives "
            "each claim a 0-100 freshness score. Also reopens any approved decision that relied on a fact "
            "that has changed. Use it before stating facts that may have changed since your training data."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The answer text to check (max 12,000 characters)."},
                "context": {"type": "string", "description": "Optional: the question or topic, to sharpen the search."},
            },
            "required": ["answer"],
        },
        "annotations": {"title": "Check an answer for stale facts", "readOnlyHint": False, "openWorldHint": True},
    },
    {
        "name": "ground_question",
        "description": (
            "Call BEFORE answering a question whose answer may have changed recently (news, this week's events, "
            "live prices, laws, versions, who runs a company). Returns dated web sources ranked by authority and "
            "advice on how to answer, so you answer from current facts instead of training data. For live values "
            "with no source dated today it tells you not to state a number."
        ),
        "inputSchema": {"type": "object", "properties": {
            "question": {"type": "string", "description": "The user's question."}}, "required": ["question"]},
        "annotations": {"title": "Ground a question before answering", "readOnlyHint": True, "openWorldHint": True},
    },
    {
        "name": "get_ledger",
        "description": "Show the pin ledger: facts and their versions, decisions and their status "
                       "(approved, reopened, at risk), and recent audit events. Read-only.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"title": "Show the pin ledger", "readOnlyHint": True},
    },
]


def summarize(report: dict[str, Any]) -> str:
    """A short, readable summary of a check report for the assistant (and the person) to read."""
    s = report["summary"]
    claims = report["claims"]
    # Label by verdict, not only by score band: a claim nobody could verify is "unverified", never "stale".
    stale = sum(c["verdict"] in ("stale", "contradicted") for c in claims)
    unverified = sum(c["verdict"] == "unverifiable" for c in claims)
    disputed = sum(c["verdict"] == "disputed" for c in claims)
    lines = [f"FreshPin checked {len(claims)} claim(s): {s['fresh']} fresh, {stale} stale or contradicted, "
             f"{disputed} disputed, {unverified} could not be verified. "
             f"{s.get('confirmed', 0)} of {len(claims)} confirmed by an independent second check."]
    if report.get("degraded") or any(not c.get("ai_available", True) for c in claims):
        lines.append("Note: the AI models were unavailable for some claims, so those are unverified "
                     "and the ledger was not changed for them.")
    for i, c in enumerate(claims, 1):
        f = c["freshness"]
        label = {"unverifiable": "UNVERIFIED", "disputed": "DISPUTED"}.get(c["verdict"], f["band"].upper())
        lines.append(f"\n{i}. [{label} {f['score']}/100] {c['text']}")
        if c["verdict"] in ("stale", "contradicted", "partly_current") and c.get("current_statement"):
            lines.append(f"   Now: {c['current_statement']}")
        if c.get("needs_live_feed"):
            lines.append("   Needs a live data feed: no source dated today states this live value.")
        r = c.get("recheck") or {}
        if r.get("status") in ("confirmed", "disputed", "unavailable"):
            lines.append(f"   Second check: {r['status']}. {r.get('reason', '')}")
        if c["verdict"] == "disputed" and (r.get("one_check") or {}).get("statement"):
            lines.append(f"   One check only (not confirmed): {r['one_check']['statement']}")
        if r.get("why_outdated") and c["verdict"] != "current":
            lines.append(f"   Why it was out of date: {r['why_outdated']}")
        ref = r.get("best_reference")
        if ref:
            when = ref.get("published_date") or "undated"
            lines.append(f"   Best reference ({ref.get('authority')}, {when}): {ref.get('url')}")
            if r.get("evidence_quote"):
                lines.append(f"   Quote: \"{r['evidence_quote']}\"")
        else:
            urls = c.get("supporting_urls") or c.get("contradicting_urls") or [x.get("url") for x in c.get("sources", [])[:2]]
            for u in [u for u in urls if u][:2]:
                lines.append(f"   Source: {u}")
    if s["decisions_reopened"]:
        lines.append("\nDecisions reopened (a person must re-approve them in the FreshPin web app): "
                     + ", ".join(s["decisions_reopened"]))
    if s["decisions_at_risk"]:
        lines.append("Decisions at risk (evidence unclear, not reopened): " + ", ".join(s["decisions_at_risk"]))
    if stale:
        lines.append("\nDo not repeat the stale claims as current. Use the 'Now' statements and cite the sources.")
    if unverified:
        lines.append("Treat unverified claims as unconfirmed; say so rather than stating them as fact.")
    if disputed:
        lines.append("Disputed claims: the two checks disagree. Show both views and let a person decide.")
    if report.get("corrections"):
        lines.append("\nUse these corrections (confirmed by two checks, each with its source):")
        for k in report["corrections"]:
            ref = (k.get("reference") or {}).get("url", "")
            lines.append(f"- Replace: {k['claim']}\n  With: {k['now']}\n  Source: {ref}")
    return "\n".join(lines)


def _text_result(text: str, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class Server:
    def __init__(self, app_factory: Callable[[], FreshPin]):
        self._factory = app_factory
        self._app: FreshPin | None = None

    @property
    def app(self) -> FreshPin:
        if self._app is None:  # built on first use, so "initialize" stays fast
            self._app = self._factory()
        return self._app

    def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "check_answer":
                report = self.app.check(str(args.get("answer", "")), str(args.get("context", "")))
                compact = {k: v for k, v in report.items() if k != "ledger"}
                return _text_result(summarize(report) + "\n\nFull report (JSON):\n"
                                    + json.dumps(compact, ensure_ascii=False))
            if name == "ground_question":
                g = self.app.ground(str(args.get("question", "")))
                lines = [f"Question type: {g['type']} ({'time-sensitive' if g['time_sensitive'] else 'not time-sensitive'})",
                         f"Advice: {g['advice']}"]
                for i, s in enumerate(g["sources"], 1):
                    lines.append(f"[{i}] {s['title']} ({s.get('authority')}, {s.get('published_date') or 'undated'})\n"
                                 f"    {s['url']}\n    {s['snippet']}")
                return _text_result("\n".join(lines))
            if name == "get_ledger":
                return _text_result(json.dumps(self.app.ledger.state(), ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001 - reported to the assistant instead of crashing the server
            return _text_result(f"FreshPin could not finish: {type(exc).__name__}: {exc}", is_error=True)
        raise KeyError(name)

    def handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Answer one JSON-RPC message. Returns None for notifications."""
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:  # notification, e.g. notifications/initialized
            return None
        try:
            if method == "initialize":
                params = msg.get("params") or {}
                result: Any = {
                    "protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "freshpin", "version": __version__},
                    "instructions": "Use check_answer before stating facts that may have changed "
                                    "(prices, versions, laws, company facts, statistics).",
                }
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": TOOLS}
            elif method == "tools/call":
                params = msg.get("params") or {}
                try:
                    result = self.call_tool(params.get("name", ""), params.get("arguments") or {})
                except KeyError:
                    return _error(mid, -32602, f"unknown tool: {params.get('name')}")
            else:
                return _error(mid, -32601, f"method not found: {method}")
        except Exception as exc:  # noqa: BLE001
            return _error(mid, -32603, f"{type(exc).__name__}: {exc}")
        return {"jsonrpc": "2.0", "id": mid, "result": result}


def _error(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def run(stdin: TextIO | None = None, stdout: TextIO | None = None,
        app_factory: Callable[[], FreshPin] | None = None) -> None:
    out = stdout or sys.stdout
    if stdout is None:
        sys.stdout = sys.stderr  # stdout carries protocol messages only; stray prints go to stderr
    server = Server(app_factory or (lambda: FreshPin(settings())))
    for line in (stdin or sys.stdin):
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            reply: dict[str, Any] | None = _error(None, -32700, "parse error")
        else:
            reply = server.handle(msg) if isinstance(msg, dict) else _error(None, -32600, "invalid request")
        if reply is not None:
            out.write(json.dumps(reply, ensure_ascii=False) + "\n")
            out.flush()
