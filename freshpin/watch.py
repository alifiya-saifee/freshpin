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
    alerts.sort(key=lambda a: (a["to"] != "reopened", a["id"]))  # reopened first: those need a person most
    return {"due": len(due), "checked": checked, "errors": errors, "alerts": alerts}


def alert_text(alerts: list[dict[str, Any]]) -> str:
    lines = ["FreshPin: decisions need a person"]
    for a in alerts:
        what = "REOPENED (a fact it relied on changed)" if a["to"] == "reopened" else "AT RISK (evidence unclear)"
        lines.append(f"- {a['id']} {a['title']} (owner: {a['owner']}): {what}. Review and re-approve in FreshPin.")
    return "\n".join(lines)


def webhook_allowed(url: str, allow_private: bool = False) -> str | None:
    """SSRF guard: only http(s), and never to private, loopback, link-local or reserved addresses (a webhook
    setting must not become a way to reach internal services). Returns a reason when blocked."""
    import ipaddress
    import socket
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return "only http(s) webhook URLs are allowed"
    if allow_private:
        return None
    try:
        addrs = {ai[4][0] for ai in socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80))}
    except OSError:
        return "webhook host does not resolve"
    for a in addrs:
        ip = ipaddress.ip_address(a.split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return f"webhook points to a private or local address ({a}); set FRESHPIN_ALLOW_PRIVATE_WEBHOOK=1 to allow"
    return None


def post_webhook(url: str, text: str, timeout_s: int = 10, allow_private: bool = False) -> str:
    """Returns "sent" or a short error. Never raises: alerts must not crash a scheduled job."""
    blocked = webhook_allowed(url, allow_private)
    if blocked:
        return f"not sent: {blocked}"
    req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return "sent" if resp.status < 300 else f"HTTP {resp.status}"
    except (urllib.error.URLError, TimeoutError) as exc:
        return f"not sent: {type(exc).__name__}"
