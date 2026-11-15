"""Scheduled re-checks and alerts: re-check pinned facts whose half-life has passed, and tell the decision
owner when a decision reopens or goes at risk.

    python -m freshpin watch            # re-check due facts once; run it from cron (e.g. every 6 hours)
    python -m freshpin watch --all      # re-check every pinned fact now

Alerts print to the terminal and, if FRESHPIN_WEBHOOK_URL is set, are posted as {"text": ...} (Slack-style
incoming webhook). The message holds only decision titles, statuses and facts: never keys or settings.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import date
from typing import Any, Callable

from .ledger import Ledger
from .scoring import HALF_LIFE_DAYS

ALERT_STATUSES = ("reopened", "at_risk")


def run_watch(ledger: Ledger, check: Callable[[str, str], dict[str, Any]], today: date,
              force: bool = False) -> dict[str, Any]:
    """check(statement, claim_type) runs a full FreshPin check that applies to the ledger."""
    before = ledger.decision_statuses()
    due = ledger.facts_due(today, HALF_LIFE_DAYS, force=force)
    checked, errors = [], []
    for f in due:
        try:
            rep = check(f["statement"], f["claim_type"])
            checked.append({"key": f["key"], "claims": [(c["key"], c["verdict"]) for c in rep.get("claims", [])]})
        except Exception as exc:  # noqa: BLE001 - one failing fact must not stop the others
            errors.append(f"{f['key']}: {type(exc).__name__}: {exc}"[:200])
    after = ledger.decision_statuses()
    alerts = [{"id": did, "title": d["title"], "owner": d["owner"], "from": before.get(did, {}).get("status"),
               "to": d["status"]}
              for did, d in after.items()
              if d["status"] in ALERT_STATUSES and before.get(did, {}).get("status") != d["status"]]
    return {"due": len(due), "checked": checked, "errors": errors, "alerts": alerts}


def alert_text(alerts: list[dict[str, Any]]) -> str:
    lines = ["FreshPin: decisions need a person"]
    for a in alerts:
        what = "REOPENED (a fact it relied on changed)" if a["to"] == "reopened" else "AT RISK (evidence unclear)"
        lines.append(f"- {a['id']} {a['title']} (owner: {a['owner']}): {what}. Review and re-approve in FreshPin.")
    return "\n".join(lines)


def post_webhook(url: str, text: str, timeout_s: int = 10) -> str:
    """Returns "sent" or a short error. Never raises: alerts must not crash a scheduled job."""
    req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return "sent" if resp.status < 300 else f"HTTP {resp.status}"
    except (urllib.error.URLError, TimeoutError) as exc:
        return f"not sent: {type(exc).__name__}"
