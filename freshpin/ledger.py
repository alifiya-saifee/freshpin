"""The pin ledger: which decisions relied on which version of which fact.

This is what turns "this answer is stale" into "these three approvals are now unsafe".

  facts     - versioned facts (key + version). A stale verdict creates a new version.
  decisions - things people approved (a quote, a report, a policy, a forecast).
  pins      - the exact fact versions a decision was approved on.
  events    - an append-only log of every check, change, reopen and approval.

Rules:
  * A decision pinned to a superseded fact version is REOPENED and must be re-approved by a person.
  * A fact that scores in the "check" band puts its decisions AT RISK (warned, not reopened).
  * Re-approving pins the decision to the current versions and records the evidence hash it saw.
  * Nothing is ever approved automatically.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
  key TEXT NOT NULL, version INTEGER NOT NULL, statement TEXT NOT NULL, claim_type TEXT NOT NULL,
  band TEXT NOT NULL, score INTEGER, evidence_json TEXT NOT NULL DEFAULT '[]',
  evidence_hash TEXT NOT NULL DEFAULT '', checked_at TEXT, superseded INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (key, version));
CREATE TABLE IF NOT EXISTS decisions (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, kind TEXT NOT NULL, owner TEXT NOT NULL,
  status TEXT NOT NULL, approved_by TEXT, approved_at TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS pins (
  decision_id TEXT NOT NULL, fact_key TEXT NOT NULL, fact_version INTEGER NOT NULL,
  PRIMARY KEY (decision_id, fact_key));
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, kind TEXT NOT NULL,
  decision_id TEXT, fact_key TEXT, detail_json TEXT NOT NULL DEFAULT '{}');
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def evidence_hash(evidence: list[dict[str, Any]]) -> str:
    canon = json.dumps(sorted((e.get("url", ""), e.get("snippet", "")[:200]) for e in evidence))
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


class Ledger:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ---------- facts ----------
    def current_fact(self, key: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM facts WHERE key=? ORDER BY version DESC LIMIT 1", (key,)).fetchone()

    def known_facts(self) -> list[dict[str, str]]:
        rows = self.db.execute(
            "SELECT f.key, f.statement, f.claim_type FROM facts f JOIN "
            "(SELECT key, MAX(version) v FROM facts GROUP BY key) m ON f.key=m.key AND f.version=m.v").fetchall()
        return [dict(r) for r in rows]

    def _insert_fact(self, key: str, statement: str, claim_type: str, band: str, score: int | None,
                     evidence: list[dict[str, Any]]) -> int:
        cur = self.current_fact(key)
        version = (cur["version"] + 1) if cur else 1
        if cur:
            self.db.execute("UPDATE facts SET superseded=1 WHERE key=? AND version<?", (key, version))
        self.db.execute(
            "INSERT INTO facts (key, version, statement, claim_type, band, score, evidence_json, evidence_hash, checked_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (key, version, statement, claim_type, band, score, json.dumps(evidence), evidence_hash(evidence), _now()))
        return version

    def record_check(self, claim: dict[str, Any]) -> dict[str, Any]:
        """Apply one verified claim to the ledger. Returns what changed."""
        key, verdict, band = claim["key"], claim["verdict"], claim["freshness"]["band"]
        evidence = claim.get("sources", [])
        cur = self.current_fact(key)
        effect: dict[str, Any] = {"key": key, "reopened": [], "at_risk": [], "new_version": None}

        if verdict in ("stale", "contradicted"):
            corrected = claim.get("current_statement") or claim["text"]
            new_version = self._insert_fact(key, corrected, claim["type"], "fresh", None, evidence)
            effect["new_version"] = new_version
            self._event("fact_changed", None, key, {
                "from_version": cur["version"] if cur else None, "to_version": new_version,
                "old_statement": cur["statement"] if cur else claim["text"], "new_statement": corrected,
                "checked_claim": claim["text"], "score": claim["freshness"]["score"],
                "evidence_hash": evidence_hash(evidence)})
            effect["reopened"] = self._reopen_pinned_below(key, new_version, claim)
        elif cur is None:
            self._insert_fact(key, claim["text"], claim["type"], band, claim["freshness"]["score"], evidence)
            self._event("fact_added", None, key, {"statement": claim["text"], "band": band})
        else:
            self.db.execute("UPDATE facts SET band=?, score=?, evidence_json=?, evidence_hash=?, checked_at=? "
                            "WHERE key=? AND version=?",
                            (band, claim["freshness"]["score"], json.dumps(evidence), evidence_hash(evidence),
                             _now(), key, cur["version"]))
            self._event("fact_rechecked", None, key, {"band": band, "score": claim["freshness"]["score"]})
            if band == "check":
                effect["at_risk"] = self._flag_at_risk(key, claim)
        self.db.commit()
        return effect

    def _reopen_pinned_below(self, key: str, version: int, claim: dict[str, Any]) -> list[str]:
        rows = self.db.execute(
            "SELECT d.id FROM decisions d JOIN pins p ON p.decision_id=d.id "
            "WHERE p.fact_key=? AND p.fact_version<? AND d.status!='reopened'", (key, version)).fetchall()
        ids = [r["id"] for r in rows]
        for did in ids:
            self.db.execute("UPDATE decisions SET status='reopened' WHERE id=?", (did,))
            self._event("decision_reopened", did, key, {
                "reason": f"Fact '{key}' changed after this was approved",
                "stale_claim": claim["text"], "now": claim.get("current_statement")})
        return ids

    def _flag_at_risk(self, key: str, claim: dict[str, Any]) -> list[str]:
        rows = self.db.execute(
            "SELECT d.id FROM decisions d JOIN pins p ON p.decision_id=d.id "
            "WHERE p.fact_key=? AND d.status='approved'", (key,)).fetchall()
        ids = [r["id"] for r in rows]
        for did in ids:
            self.db.execute("UPDATE decisions SET status='at_risk' WHERE id=?", (did,))
            self._event("decision_at_risk", did, key, {"reason": "Evidence is unclear; a person should look",
                                                       "score": claim["freshness"]["score"]})
        return ids

    def seed_fact(self, key: str, statement: str, claim_type: str) -> int:
        version = self._insert_fact(key, statement, claim_type, "fresh", None, [])
        self._event("fact_added", None, key, {"statement": statement, "seeded": True})
        self.db.commit()
        return version

    # ---------- decisions ----------
    def add_decision(self, did: str, title: str, kind: str, owner: str, pins: dict[str, int],
                     approved_by: str, approved_at: str | None = None, note: str = "") -> None:
        self.db.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?,?,?,?)",
                        (did, title, kind, owner, "approved", approved_by, approved_at or _now(), note))
        self.db.execute("DELETE FROM pins WHERE decision_id=?", (did,))
        for key, version in pins.items():
            self.db.execute("INSERT INTO pins VALUES (?,?,?)", (did, key, version))
        self._event("decision_approved", did, None, {"by": approved_by, "pins": pins, "seeded": True})
        self.db.commit()

    def approve(self, did: str, by: str, note: str = "") -> dict[str, Any]:
        """A person re-approves a decision on the CURRENT fact versions."""
        d = self.db.execute("SELECT * FROM decisions WHERE id=?", (did,)).fetchone()
        if not d:
            raise KeyError(did)
        if not by.strip():
            raise ValueError("an approver name is required")
        pins = {}
        hashes = {}
        for p in self.db.execute("SELECT fact_key FROM pins WHERE decision_id=?", (did,)).fetchall():
            cur = self.current_fact(p["fact_key"])
            pins[p["fact_key"]] = cur["version"]
            hashes[p["fact_key"]] = cur["evidence_hash"]
            self.db.execute("UPDATE pins SET fact_version=? WHERE decision_id=? AND fact_key=?",
                            (cur["version"], did, p["fact_key"]))
        self.db.execute("UPDATE decisions SET status='approved', approved_by=?, approved_at=?, note=? WHERE id=?",
                        (by, _now(), note, did))
        self._event("decision_approved", did, None, {"by": by, "note": note, "pins": pins, "evidence_hashes": hashes})
        self.db.commit()
        return {"id": did, "pins": pins}

    # ---------- read model ----------
    def state(self) -> dict[str, Any]:
        decisions = []
        for d in self.db.execute("SELECT * FROM decisions ORDER BY id").fetchall():
            pins = []
            for p in self.db.execute("SELECT * FROM pins WHERE decision_id=?", (d["id"],)).fetchall():
                pinned = self.db.execute("SELECT * FROM facts WHERE key=? AND version=?",
                                         (p["fact_key"], p["fact_version"])).fetchone()
                cur = self.current_fact(p["fact_key"])
                pins.append({
                    "key": p["fact_key"], "pinned_version": p["fact_version"],
                    "pinned_statement": pinned["statement"] if pinned else None,
                    "current_version": cur["version"] if cur else None,
                    "current_statement": cur["statement"] if cur else None,
                    "current_evidence": json.loads(cur["evidence_json"]) if cur else [],
                    "outdated": bool(cur and cur["version"] != p["fact_version"]),
                })
            decisions.append({**dict(d), "pins": pins})
        events = [dict(e) | {"detail": json.loads(e["detail_json"])}
                  for e in self.db.execute("SELECT * FROM events ORDER BY id DESC LIMIT 60").fetchall()]
        for e in events:
            e.pop("detail_json", None)
        return {"decisions": decisions, "facts": self.known_facts(), "events": events}

    def _event(self, kind: str, decision_id: str | None, fact_key: str | None, detail: dict[str, Any]) -> None:
        self.db.execute("INSERT INTO events (ts, kind, decision_id, fact_key, detail_json) VALUES (?,?,?,?,?)",
                        (_now(), kind, decision_id, fact_key, json.dumps(detail)))

    def reset(self) -> None:
        self.db.executescript("DELETE FROM facts; DELETE FROM decisions; DELETE FROM pins; DELETE FROM events;")
        self.db.commit()
