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
        if body["model"] == "busy/model" and sum(c["model"] == "busy/model" for c in FakeBifrost.calls) == 1:
            self.send_response(429)
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(b'{"error":"rate limit"}')
            return
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
        cls.httpd.server_close()

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

    def test_rate_limit_is_retried_once(self):
        b = RunBudget(1000)
        out = BifrostClient(self.url).chat_json("x", ["busy/model"], "s", "u", b)
        self.assertEqual(out, {"ok": True})
        self.assertEqual(sum(c["model"] == "busy/model" for c in FakeBifrost.calls), 2)
        self.assertTrue(b.traces[-1].ok)

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


class RecheckTests(unittest.TestCase):
    """The recheck layer: second search, second judge, exact-quote rule and fixed decision rules."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = replay_app(self.tmp.name)
        self.app.load_sample("dubai_building_law")
        self.answer = self.app.samples()["dubai_building_law"]["answer"]
        self.recheck = self.app.chat.d.data["recheck"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_claim_is_rechecked_and_confirmed(self):
        rep = self.app.check(self.answer)
        self.assertEqual(rep["summary"]["rechecked"], 3)
        self.assertEqual(rep["summary"]["confirmed"], 3)
        self.assertEqual(rep["tavily_searches"], 6)  # one first search + one recheck search per claim
        cert = next(c for c in rep["claims"] if c["key"] == "dubai.old_buildings.safety_certificate_required")
        r = cert["recheck"]
        self.assertTrue(r["quote_found"])
        self.assertEqual(r["best_reference"]["authority"], "official")
        self.assertIn("dlp.dubai.gov.ae", r["best_reference"]["url"])
        self.assertIn("before Law No. 3 of 2026", r["why_outdated"])
        steps = [t["step"] for t in rep["gateway"]["traces"]]
        self.assertEqual(sum(s.startswith("recheck:") for s in steps), 3)

    def test_disagreement_marks_at_risk_and_never_reopens(self):
        for key in ("dubai.old_buildings.safety_certificate_required",
                    "dubai.ejari.attestation_needs_building_certificate"):
            self.recheck[key] = {**self.recheck[key], "agree": False, "verdict": "current"}
        rep = self.app.check(self.answer)
        self.assertEqual(rep["summary"]["decisions_reopened"], [])
        self.assertIn("Q-112", rep["summary"]["decisions_at_risk"])
        cert = next(c for c in rep["claims"] if c["key"] == "dubai.old_buildings.safety_certificate_required")
        self.assertEqual(cert["verdict"], "disputed")
        self.assertEqual(cert["first_verdict"], "stale")
        self.assertEqual(cert["freshness"]["band"], "check")
        self.assertIn("disagree", cert["recheck"]["reason"])

    def test_correction_without_exact_quote_is_not_accepted(self):
        key = "dubai.old_buildings.safety_certificate_required"
        self.recheck[key] = {**self.recheck[key], "evidence_quote": "Owners of old buildings now need a safety certificate"}
        rep = self.app.check(self.answer)
        cert = next(c for c in rep["claims"] if c["key"] == key)
        self.assertEqual(cert["verdict"], "disputed")
        self.assertFalse(cert["recheck"]["quote_found"])
        self.assertIn("no exact quote", cert["recheck"]["reason"])

    def test_current_verdict_also_needs_a_quote(self):
        key = "dubai.building_permits.issuing_authority"
        self.recheck[key] = {**self.recheck[key], "evidence_quote": "The municipality handles all permits everywhere"}
        rep = self.app.check(self.answer)
        claim = next(c for c in rep["claims"] if c["key"] == key)
        self.assertEqual(claim["verdict"], "disputed")

    def test_wrong_current_verdict_is_caught(self):
        # The first judge says "current"; the second judge disagrees. Rechecking every claim catches it.
        key = "dubai.building_permits.issuing_authority"
        self.recheck[key] = {**self.recheck[key], "agree": False, "verdict": "stale"}
        rep = self.app.check(self.answer)
        claim = next(c for c in rep["claims"] if c["key"] == key)
        self.assertEqual(claim["verdict"], "disputed")

    def test_find_quote_and_decide_rules(self):
        from freshpin.pipeline import decide, find_quote
        srcs = [{"title": "Law", "snippet": "Owners must obtain a certificate once a building is 20 years old."}]
        self.assertEqual(find_quote("owners must  obtain a certificate", srcs), 0)  # case and spacing ignored
        self.assertIsNone(find_quote("owners need a permit", srcs))
        self.assertIsNone(find_quote("Owners", srcs))  # too short to count
        first = {"verdict": "stale"}
        self.assertEqual(decide(first, {"agree": True, "verdict": "contradicted"}, True)[1], "confirmed")
        self.assertEqual(decide(first, {"agree": True, "verdict": "stale"}, False)[0], "disputed")
        self.assertEqual(decide(first, {"agree": False, "verdict": "stale"}, True)[0], "disputed")
        self.assertEqual(decide(first, None, False), ("disputed", "unavailable",
                                                      "Second check could not run, so the change is not confirmed."))
        self.assertEqual(decide({"verdict": "current"}, None, False)[0], "current")
        self.assertEqual(decide({"verdict": "unverifiable"}, {"agree": True, "verdict": "unverifiable"}, False)[1],
                         "confirmed")  # agreeing that nothing can be verified needs no quote

    def test_source_grading_and_notes(self):
        from freshpin.sources import authority_of, grade, source_notes
        self.assertEqual(authority_of("https://dlp.dubai.gov.ae/x"), "official")
        self.assertEqual(authority_of("https://www.mediaoffice.ae/en/news"), "official")
        self.assertEqual(authority_of("https://docs.getbifrost.ai/x"), "vendor_docs")
        self.assertEqual(authority_of("https://www.thenationalnews.com/a"), "news")
        self.assertEqual(authority_of("https://someblog.example.com/a"), "other")
        g = [grade({"url": "https://blog.example.com/a", "published_date": "2023-01-01"}, "regulation", TODAY)]
        self.assertTrue(g[0]["is_old"])
        notes = source_notes(g, ["https://blog.example.com/a"])
        self.assertTrue(any("old, undated or unofficial" in n for n in notes))
        self.assertTrue(any("No official source" in n for n in notes))

    def test_disputed_score_stays_in_check_band(self):
        srcs = [{"url": "a", "published_date": "2026-11-01"}]
        for sup in (["a"], []):
            s = freshness_score("disputed", "regulation", srcs, sup, [], TODAY)
            self.assertEqual(s.band, "check")


class StubChat:
    """Answers by step name: extract -> claims, verify:<key> / recheck:<key> -> canned JSON. Records prompts."""

    def __init__(self, claims, verify, recheck):
        self.claims, self.verify, self.recheck, self.prompts = claims, verify, recheck, {}

    def chat_json(self, step, models, system, user, budget, simulate_outage=False):
        self.prompts[step] = user
        kind, _, key = step.partition(":")
        if kind == "extract":
            return {"claims": self.claims}
        return json.loads(json.dumps({"verify": self.verify, "recheck": self.recheck}[kind][key]))


class StubSearch:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    def search(self, query, claim_type, max_results=5):
        from freshpin.search import Source
        self.calls.append((query, claim_type))
        return [Source(**r) for r in self.rows.get(query, [])][:max_results]


def stub_run(claims, verify, recheck, rows, tmp):
    from freshpin.pipeline import run_check
    chat, search = StubChat(claims, verify, recheck), StubSearch(rows)
    rep = run_check("answer", chat=chat, search=search, ledger=Ledger(Path(tmp) / "s.db"),
                    extractor_models=["a/x"], verifier_models=["b/y"], token_budget=10**6, today=TODAY)
    return rep, chat, search


class NewsLiveDataTests(unittest.TestCase):
    GOLD = {"text": "Gold trades at $2,000 an ounce today.", "key": "gold.spot", "type": "live_data",
            "time_sensitive": True, "search_query": "gold spot price today"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def gold_run(self, published):
        src = {"title": "Gold price", "url": "https://markets.example.com/gold", "relevance": 0.9,
               "snippet": "Gold trades at $2,000 an ounce in early trading.", "published_date": published}
        verify = {"gold.spot": {"verdict": "current", "current_statement": "Gold trades at $2,000 an ounce.",
                                "supporting": [1], "contradicting": [], "explanation": "[1]",
                                "confirm_query": "gold price news"}}
        recheck = {"gold.spot": {"agree": True, "verdict": "current", "current_statement": "Gold trades at $2,000.",
                                 "best_source": 1, "evidence_quote": "Gold trades at $2,000 an ounce in early trading",
                                 "supporting": [1], "contradicting": [], "why_outdated": "", "explanation": "[1]"}}
        rep, _, _ = stub_run([self.GOLD], verify, recheck, {"gold spot price today": [src]}, self.tmp.name)
        return rep["claims"][0], rep

    def test_live_value_needs_a_source_dated_today(self):
        old, rep = self.gold_run("2026-10-01")
        self.assertEqual(old["verdict"], "unverifiable")
        self.assertTrue(old["needs_live_feed"])
        self.assertEqual(rep["unconfirmed"][0]["reason"], "needs a live data feed")
        today, _ = self.gold_run(TODAY.isoformat())
        self.assertEqual(today["verdict"], "current")
        self.assertFalse(today["needs_live_feed"])

    def test_tavily_gets_news_and_finance_topics(self):
        from freshpin.search import TavilyClient
        bodies = []

        class FakeTavily(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):  # noqa: N802
                bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                data = b'{"results": []}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeTavily)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            c = TavilyClient("k", f"http://127.0.0.1:{httpd.server_port}")
            for t in ("event", "live_data", "regulation"):
                c.search("q", t)
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual([(b["topic"], b.get("time_range")) for b in bodies],
                         [("news", "week"), ("finance", "day"), ("general", None)])


class EfficiencyTests(unittest.TestCase):
    def test_cache_reuses_until_it_expires(self):
        from freshpin.search import CachedSearch
        now = [0.0]
        inner = StubSearch({"q": [{"title": "t", "url": "u", "snippet": "s", "published_date": None, "relevance": 1}]})
        c = CachedSearch(inner, clock=lambda: now[0])
        c.search("q", "live_data")
        c.search("Q ", "live_data")  # same query, different case and spacing
        self.assertEqual((len(inner.calls), c.hits), (1, 1))
        now[0] = 601  # live data is reused for 10 minutes only
        c.search("q", "live_data")
        self.assertEqual(len(inner.calls), 2)
        c.search("q", "regulation")
        now[0] = 601 + 3600
        c.search("q", "regulation")  # regulations are reused for a day
        self.assertEqual(len(inner.calls), 3)

    def test_second_judge_sees_top_six_trimmed_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            claim = {"text": "X costs $5.", "key": "x.price", "type": "pricing", "time_sensitive": True,
                     "search_query": "x price"}
            rows = {"x price": [{"title": f"S{i}", "url": f"https://s{i}.example.com", "snippet": "y" * 900,
                                 "published_date": f"2026-11-{i + 1:02d}", "relevance": 0.5} for i in range(5)],
                    "x price change": [{"title": "Gov", "url": "https://x.gov/p", "snippet": "z" * 900,
                                        "published_date": None, "relevance": 0.5},
                                       {"title": "S9", "url": "https://s9.example.com", "snippet": "w",
                                        "published_date": "2026-01-01", "relevance": 0.5}]}
            verify = {"x.price": {"verdict": "current", "current_statement": "X costs $5.", "supporting": [1],
                                  "contradicting": [], "explanation": "", "confirm_query": "x price change"}}
            recheck = {"x.price": {"agree": True, "verdict": "current", "current_statement": "X costs $5.",
                                   "best_source": 1, "evidence_quote": "", "supporting": [1], "contradicting": [],
                                   "why_outdated": "", "explanation": ""}}
            rep, chat, _ = stub_run([claim], verify, recheck, rows, tmp)
            prompt = chat.prompts["recheck:x.price"]
            self.assertEqual(prompt.count("\n[") + prompt.startswith("["), 6)  # 7 sources found, 6 sent
            self.assertNotIn("y" * 501, prompt)
            # The checker's "1" is the official source (ranked first), mapped back to the full list.
            self.assertEqual(rep["claims"][0]["recheck"]["best_reference"]["url"], "https://x.gov/p")
            self.assertGreater(rep["gateway"]["tokens_by_step"]["recheck"], -1)


class CorrectionsAndBenchTests(unittest.TestCase):
    def test_corrections_hold_confirmed_changes_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = replay_app(tmp)
            app.load_sample("dubai_building_law")
            app.chat.d.data["recheck"]["dubai.ejari.attestation_needs_building_certificate"]["agree"] = False
            rep = app.check(app.samples()["dubai_building_law"]["answer"])
            self.assertEqual([k["claim"][:20] for k in rep["corrections"]], ["Dubai does not requi"])
            self.assertEqual(rep["corrections"][0]["reference"]["authority"], "official")
            self.assertTrue(rep["corrections"][0]["quote"])
            self.assertEqual(rep["unconfirmed"][0]["verdict"], "disputed")

    def test_bench_scorecard_and_honesty(self):
        from freshpin import bench
        qs = [{"id": "gold", "category": "live_data", "question": "Gold price today?"},
              {"id": "py", "category": "api_version", "question": "Latest Python?"}]
        answers = {("m1", "Gold price today?"): "Gold is $2,000 an ounce.",
                   ("m2", "Gold price today?"): "I don't have access to real-time prices; check a live source.",
                   ("m1", "Latest Python?"): "Python 3.11.", ("m2", "Latest Python?"): "Python 3.14."}

        def ask(model, q):
            if model == "m3":
                raise RuntimeError("down")
            return answers[(model, q)]

        def check(answer, q):
            verdict = {"Gold is $2,000 an ounce.": "unverifiable", "Python 3.11.": "stale",
                       "Python 3.14.": "current"}.get(answer)
            claims = [] if verdict is None else [{"text": answer, "type": "x", "verdict": verdict,
                                                  "freshness": {"score": {"stale": 10, "current": 90}.get(verdict, 45)},
                                                  "current_statement": "Python 3.14."}]
            return {"claims": claims, "corrections": []}

        res = bench.run_bench(qs, ["m1", "m2", "m3"], ask, check)
        m1, m2, m3 = (res["models"][m]["scorecard"] for m in ("m1", "m2", "m3"))
        self.assertEqual((m1["claims"], m1["stale_pct"], m1["unverified_pct"], m1["honest_on_live"]), (2, 50, 50, 0))
        self.assertEqual((m2["fresh_pct"], m2["honest_on_live"], m2["live_questions"]), (100, 1, 1))
        self.assertEqual(m3["errors"], 2)
        self.assertEqual(bench.estimate_credits(3, 2, True), 72)
        # A real answer from the live bench: curly apostrophe and non-breaking hyphen.
        self.assertTrue(bench.admits_live_limits("I\u2019m sorry, but I don\u2019t have real\u2011time data access."))
        self.assertFalse(bench.admits_live_limits("Gold is $2,650 an ounce today."))
        self.assertEqual(bench.estimate_credits(3, 2, False), 36)

    def test_bench_check_leaves_ledger_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = replay_app(tmp)
            app.load_sample("dubai_building_law")
            before = app.ledger.state()["decisions"]
            app.check(app.samples()["dubai_building_law"]["answer"], apply_to_ledger=False)
            self.assertEqual(app.ledger.state()["decisions"], before)

    def test_bench_shows_what_is_true_now_and_why(self):
        from freshpin.bench import claim_detail, rescore
        with tempfile.TemporaryDirectory() as tmp:
            app = replay_app(tmp)
            app.load_sample("dubai_building_law")
            rep = app.check(app.samples()["dubai_building_law"]["answer"], apply_to_ledger=False)
        d = {c["key"]: claim_detail(c) for c in rep["claims"]}
        stale = d["dubai.old_buildings.safety_certificate_required"]
        self.assertIn("20 years past its completion", stale["now"])  # the real answer, not just "false"
        self.assertIn("before Law No. 3 of 2026", stale["why"])
        self.assertEqual(stale["reference"]["authority"], "official")
        current = d["dubai.building_permits.issuing_authority"]
        self.assertIsNone(current["now"])  # nothing new to say about a current fact
        live = claim_detail({"text": "Gold is $2,000.", "type": "live_data", "verdict": "unverifiable",
                             "freshness": {"score": 30}, "needs_live_feed": True, "current_statement": "Gold is $2,000.",
                             "recheck": {}})
        self.assertIsNone(live["now"])
        self.assertIn("live data feed", live["why"])
        disputed = claim_detail({"text": "A", "type": "x", "verdict": "disputed", "freshness": {"score": 45},
                                 "current_statement": "B", "recheck": {"reason": "The judges disagree"}})
        self.assertIsNone(disputed["now"])  # never present one side of a dispute as the truth
        self.assertEqual(disputed["why"], "The judges disagree")
        saved = {"models": {"m": {"rows": [{"category": "live_data", "answer": "I don’t have real‑time data.",
                                            "claims": [], "honest_on_live": False}]}}}
        self.assertEqual(rescore(saved)["models"]["m"]["scorecard"]["honest_on_live"], 1)

    def test_bench_command_runs_and_refuses_replay_mode(self):
        import subprocess
        env = {**os.environ, "FRESHPIN_MODE": "replay"}
        root = Path(__file__).resolve().parent.parent
        out = subprocess.run([sys.executable, "-m", "freshpin", "bench", "--models", "a/b"], cwd=root, env=env,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 2, out.stderr)
        self.assertIn("needs FRESHPIN_MODE=live", out.stdout)

    def test_bench_questions_file_is_valid(self):
        from freshpin import bench
        qs = bench.load_questions()
        self.assertEqual(len(qs), 6)
        self.assertTrue({"live_data", "event"} <= {q["category"] for q in qs})


class McpTests(unittest.TestCase):
    """The MCP server, driven the way an AI assistant drives it: JSON lines in, JSON lines out."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = replay_app(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def talk(self, *msgs):
        import io
        from freshpin.mcp import run
        out = io.StringIO()
        run(io.StringIO("".join(json.dumps(m) + "\n" for m in msgs)), out, lambda: self.app)
        return [json.loads(line) for line in out.getvalue().splitlines()]

    def test_initialize_and_list_tools(self):
        replies = self.talk(
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual(len(replies), 2)  # the notification gets no reply
        self.assertEqual(replies[0]["result"]["serverInfo"]["name"], "freshpin")
        names = [t["name"] for t in replies[1]["result"]["tools"]]
        self.assertEqual(names, ["check_answer", "get_ledger"])
        self.assertNotIn("approve", " ".join(names))  # only people re-approve

    def test_check_answer_reports_stale_claims_and_reopens(self):
        self.app.load_sample("dubai_building_law")
        answer = self.app.samples()["dubai_building_law"]["answer"]
        (reply,) = self.talk({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                              "params": {"name": "check_answer", "arguments": {"answer": answer}}})
        res = reply["result"]
        self.assertFalse(res["isError"])
        text = res["content"][0]["text"]
        self.assertIn("2 stale or contradicted", text)
        self.assertIn("Q-112", text)
        self.assertIn("PB-LEASE-3", text)
        self.assertIn("Now: Under Dubai Law No. 3 of 2026", text)
        states = {d["id"]: d["status"] for d in self.app.ledger.state()["decisions"]}
        self.assertEqual(states["Q-112"], "reopened")

    def test_unverified_claims_are_not_called_stale(self):
        # Replay has no recording for this answer, so the AI is "down": the claim must read UNVERIFIED.
        (reply,) = self.talk({"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {
            "name": "check_answer", "arguments": {"answer": "The Acme Widget API is at version 9 and costs $5 a month."}}})
        text = reply["result"]["content"][0]["text"]
        self.assertIn("UNVERIFIED", text)
        self.assertIn("0 stale or contradicted", text)
        self.assertNotIn("Do not repeat the stale claims", text)

    def test_errors_do_not_crash_the_server(self):
        replies = self.talk(
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "check_answer", "arguments": {"answer": "  "}}},
            {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "delete_everything", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 6, "method": "no/such/method"})
        self.assertTrue(replies[0]["result"]["isError"])
        self.assertIn("empty", replies[0]["result"]["content"][0]["text"])
        self.assertEqual(replies[1]["error"]["code"], -32602)
        self.assertEqual(replies[2]["error"]["code"], -32601)


class MiscTests(unittest.TestCase):
    def test_parse_json_reply_handles_prose(self):
        self.assertEqual(parse_json_reply('Sure! {"a": 1} hope that helps'), {"a": 1})

    def test_heuristic_keeps_time_sensitive_sentences(self):
        out = heuristic_claims("Hello there friend. The API costs $5 per 1,000 calls in 2025.")
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    os.environ.setdefault("FRESHPIN_MODE", "replay")
    unittest.main()
