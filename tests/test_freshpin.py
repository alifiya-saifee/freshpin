"""Run with:  python -m unittest discover -s tests -v   (no keys or internet needed)"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from freshpin.app import FreshPin  # noqa: E402
from freshpin.config import Settings  # noqa: E402
from freshpin.ledger import Ledger  # noqa: E402
from freshpin.llm import BifrostClient, LLMUnavailable, RunBudget, parse_json_reply  # noqa: E402
from freshpin.pipeline import heuristic_claims  # noqa: E402
from freshpin.scoring import freshness_score  # noqa: E402

TODAY = date(2026, 11, 14)


def replay_app(tmp: str) -> FreshPin:
    s = Settings()
    s.mode, s.db_path = "replay", Path(tmp) / "t.db"
    return FreshPin(s)


class ScoringTests(unittest.TestCase):
    def test_current_with_recent_sources_is_fresh(self):
        srcs = [{"url": "a", "published_date": "2026-11-01"}, {"url": "b", "published_date": "2026-10-20"}]
        s = freshness_score("current", "pricing", srcs, ["a", "b"], [], TODAY)
        self.assertEqual(s.band, "fresh")
        self.assertGreaterEqual(s.score, 90)

    def test_stale_is_capped(self):
        srcs = [{"url": "a", "published_date": "2020-01-01"}]
        s = freshness_score("stale", "regulation", srcs, [], ["a"], TODAY)
        self.assertLessEqual(s.score, 35)
        self.assertEqual(s.band, "stale")

    def test_no_sources_is_capped_at_50(self):
        s = freshness_score("current", "other", [], [], [], TODAY)
        self.assertLessEqual(s.score, 50)

    def test_old_support_scores_lower_than_new_support(self):
        new = freshness_score("current", "pricing", [{"url": "a", "published_date": "2026-11-10"}], ["a"], [], TODAY)
        old = freshness_score("current", "pricing", [{"url": "a", "published_date": "2025-06-01"}], ["a"], [], TODAY)
        self.assertGreater(new.score, old.score)

    def test_rfc_dates_parse(self):
        s = freshness_score("current", "pricing", [{"url": "a", "published_date": "Tue, 10 Nov 2026 08:00:00 GMT"}],
                            ["a"], [], TODAY)
        self.assertEqual(s.newest_source_age_days, 4)


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.l = Ledger(Path(self.tmp.name) / "l.db")
        v = self.l.seed_fact("k", "old value", "pricing")
        self.l.add_decision("D1", "Quote", "quote", "ops", {"k": v}, "Ann")

    def tearDown(self):
        self.l.db.close()
        self.tmp.cleanup()

    def _claim(self, verdict, band, score):
        return {"key": "k", "text": "old value", "type": "pricing", "verdict": verdict,
                "current_statement": "new value", "sources": [{"url": "u", "snippet": "s"}],
                "freshness": {"band": band, "score": score}}

    def test_stale_reopens_pinned_decision(self):
        eff = self.l.record_check(self._claim("stale", "stale", 10))
        self.assertEqual(eff["reopened"], ["D1"])
        d = self.l.state()["decisions"][0]
        self.assertEqual(d["status"], "reopened")
        self.assertTrue(d["pins"][0]["outdated"])

    def test_uncertain_marks_at_risk_but_does_not_reopen(self):
        eff = self.l.record_check(self._claim("unverifiable", "check", 45))
        self.assertEqual(eff["at_risk"], ["D1"])
        self.assertEqual(self.l.state()["decisions"][0]["status"], "at_risk")

    def test_current_leaves_decision_alone(self):
        self.l.record_check(self._claim("current", "fresh", 90))
        self.assertEqual(self.l.state()["decisions"][0]["status"], "approved")

    def test_reapproval_pins_new_version_and_needs_a_name(self):
        self.l.record_check(self._claim("stale", "stale", 10))
        with self.assertRaises(ValueError):
            self.l.approve("D1", "  ")
        self.l.approve("D1", "Bob", "updated price")
        d = self.l.state()["decisions"][0]
        self.assertEqual(d["status"], "approved")
        self.assertEqual(d["pins"][0]["pinned_version"], 2)
        self.assertFalse(d["pins"][0]["outdated"])


class PipelineReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = replay_app(self.tmp.name)

    def tearDown(self):
        self.app.ledger.db.close()
        self.tmp.cleanup()

    def test_dubai_law_scenario_end_to_end(self):
        self.app.load_sample("dubai_building_law")
        rep = self.app.check(self.app.samples()["dubai_building_law"]["answer"])
        self.assertEqual(rep["summary"]["stale"], 2)
        self.assertEqual(rep["summary"]["fresh"], 1)
        self.assertEqual(rep["summary"]["decisions_reopened"], ["PB-LEASE-3", "Q-112"])
        guide = next(d for d in rep["ledger"]["decisions"] if d["id"] == "GUIDE-7")
        self.assertEqual(guide["status"], "approved")

    def test_outage_simulation_records_fallback(self):
        self.app.load_sample("ai_vendor_register")
        rep = self.app.check(self.app.samples()["ai_vendor_register"]["answer"], simulate_outage=True)
        self.assertGreater(rep["gateway"]["failed_attempts"], 0)
        self.assertGreater(rep["gateway"]["fallbacks_used"], 0)
        self.assertEqual(rep["summary"]["decisions_reopened"], ["VR-07"])

    def test_unknown_answer_degrades_without_touching_ledger(self):
        self.app.load_sample("ai_vendor_register")
        rep = self.app.check("The price of the Pro plan is $20 per month as of 2024.")
        self.assertTrue(rep["degraded"])
        self.assertTrue(all(c["ledger_effect"] is None for c in rep["claims"]))
        self.assertTrue(all(d["status"] == "approved" for d in rep["ledger"]["decisions"]))


class FakeBifrost(BaseHTTPRequestHandler):
    """Mimics Bifrost: fails for one model name, otherwise answers with JSON content."""
    calls: list[dict] = []

    def log_message(self, *a):
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeBifrost.calls.append(body)
        if body["model"] == "bad/model" and not body.get("fallbacks"):
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"down"}')
            return
        served = body["fallbacks"][0] if body["model"] == "bad/model" else body["model"]
        provider, _, model = served.partition("/")
        out = {"model": model, "extra_fields": {"provider": provider},
               "usage": {"prompt_tokens": 10, "completion_tokens": 5},
               "choices": [{"message": {"content": "```json\n{\"ok\": true}\n```"}}]}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class BifrostClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeBifrost)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.httpd.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def test_sends_fallbacks_and_detects_gateway_fallback(self):
        c = BifrostClient(self.url)
        b = RunBudget(1000)
        out = c.chat_json("x", ["bad/model", "good/model"], "s", "u", b)
        self.assertEqual(out, {"ok": True})
        self.assertEqual(FakeBifrost.calls[-1]["fallbacks"], ["good/model"])
        self.assertTrue(b.traces[-1].fallback_used)
        self.assertEqual(b.used_tokens, 15)

    def test_model_name_with_slash_is_not_a_fallback(self):
        # Groq serves "openai/gpt-oss-120b"; that is the model we asked for, not a fallback.
        b = RunBudget(1000)
        BifrostClient(self.url).chat_json("x", ["groq/openai/gpt-oss-120b"], "s", "u", b)
        self.assertEqual(b.traces[-1].served_by, "groq/openai/gpt-oss-120b")
        self.assertFalse(b.traces[-1].fallback_used)

    def test_raises_when_everything_is_down(self):
        c = BifrostClient(self.url)
        with self.assertRaises(LLMUnavailable):
            c.chat_json("x", ["bad/model"], "s", "u", RunBudget(1000))

    def test_budget_blocks_calls(self):
        from freshpin.llm import BudgetExceeded
        b = RunBudget(10)
        b.charge(10)
        with self.assertRaises(BudgetExceeded):
            BifrostClient(self.url).chat_json("x", ["good/model"], "s", "u", b)


class MiscTests(unittest.TestCase):
    def test_parse_json_reply_handles_prose(self):
        self.assertEqual(parse_json_reply('Sure! {"a": 1} hope that helps'), {"a": 1})

    def test_heuristic_keeps_time_sensitive_sentences(self):
        out = heuristic_claims("Hello there friend. The API costs $5 per 1,000 calls in 2025.")
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    os.environ.setdefault("FRESHPIN_MODE", "replay")
    unittest.main()
