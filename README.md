# FreshPin

**A stale-answer detector that knows what depends on the answer.**

FreshPin checks an AI-generated answer claim by claim against the live web (Tavily), scores how fresh each claim is, and, when a fact has changed, **reopens every decision that was approved on the old version** so a person can review it. All model calls go through the Bifrost gateway, with fallbacks across providers and a spend limit on every run.

**Four ways in:** a web app for people, an **MCP tool** so AI assistants (Claude Code, Claude Desktop, Cursor) can check their own answer before they reply, a command line, and a JSON API for other apps.

Built for the Geek Room AI Global Hackathon 2026, for the **"Build a Stale-Answer Detector for AI Applications"** problem statement (Tavily + Bifrost).

![FreshPin screenshot](docs/screenshot.png)

---

## The problem

AI answers go out of date quietly. A model trained before a law, a price change or an acquisition will state the old fact with full confidence. The damage isn't the answer alone: people copy it into quotes, checklists, budgets and policies, and those decisions keep running on the old fact after it changes.

Example used in the demo: until 2026, older Dubai buildings had no general periodic safety certificate. **Dubai Law No. 3 of 2026** now requires a building quality and safety certificate once a building is 20 years past completion, and the authorities can suspend lease (Ejari) attestation for buildings that don't comply. An assistant with older training data still says "no certificate needed", and a property manager's maintenance budget and lease checklist were approved on that answer.

## What FreshPin does

1. **Splits the answer into claims.** A model (via Bifrost) extracts atomic, standalone claims, tags their type (pricing, API version, regulation, company, statistic) and decides which are time-sensitive.
2. **Searches the live web.** Tavily searches for the *current* value of each claim. Fast-changing types (prices, versions, company facts) are limited to the last year.
3. **Verifies against sources only.** A second model (via Bifrost) judges each claim using only the sources Tavily returned: `current`, `partly_current`, `stale`, `contradicted` or `unverifiable`. It also states the fact as it stands today and cites the sources it used.
4. **Scores freshness with a fixed formula** (0 to 100), so the same evidence always gives the same score and every point can be explained (see below).
5. **Reopens what depended on it.** Facts are versioned. A stale verdict creates a new version, and every decision pinned to the old version is **reopened**. Unclear evidence marks decisions **at risk** instead. Only a person can re-approve, after confirming they've read the new evidence; the approval is pinned to the new fact version and its evidence hash.

## What's new about it

Most stale-answer checkers stop at a score for each claim. FreshPin adds a **pin ledger**: a record of which approved decisions relied on which version of which fact. "This sentence is outdated" then becomes "these two approvals are now unsafe, here's what changed, and here's who has to sign again". Decisions that relied only on facts that are still current are left alone.

## How Tavily and Bifrost are used

| | Role in FreshPin |
|---|---|
| **Tavily** | Live evidence for every claim: `search_depth: advanced`, recent time range for fast-changing claim types, published dates feed the recency part of the score. |
| **Bifrost** | Every model call goes through Bifrost's OpenAI-compatible endpoint. Each request carries a `fallbacks` list, so a failing provider is replaced automatically. The extractor and verifier use different primary providers. An optional virtual key (`x-bf-vk`) adds gateway-side budgets and rate limits. The UI shows a trace of every call: model asked for, model that served it, tokens, latency, errors. |

**Resilience, in three layers:**
1. **Gateway fallback:** Bifrost retries on the next provider in `fallbacks`.
2. **Client failover:** if Bifrost returns an error for the whole request, FreshPin tries the next model in its own list.
3. **Graceful degradation:** if every model is down, FreshPin still returns the Tavily evidence, marks claims "AI verifier unavailable", and **changes nothing in the ledger**. It never reopens or approves anything on guesswork.

This is not theoretical: in a live run on Groq's free tier, one of four parallel verifier calls hit Groq's rate limit (HTTP 429) and Bifrost answered it from Gemini instead. The run finished normally and the trace showed one fallback.

The **Simulate primary model outage** switch in the UI puts a model that will fail at the front of the chain, so you can watch the fallback happen live.

**Spend control:** each run has a token budget (`FRESHPIN_RUN_TOKEN_BUDGET`). Once it is used up, remaining claims are marked unverified instead of making more calls.

## Freshness score

```
score = 0.55 × verdict points   (current 100, partly 60, unverifiable 45, stale 15, contradicted 5)
      + 0.25 × recency points   (age of the newest backing source vs. the claim type's half-life)
      + 0.20 × agreement points (share of sources that back the verdict)
```

Half-lives: pricing 45 days, API versions 60, company facts 120, statistics 180, regulations 240, other 365. Caps: stale or contradicted claims can never score above 35, and claims with no sources can never score above 50. Bands: **fresh** 70 and above, **check** 40 to 69, **stale** below 40. Click "Why N?" on any claim to see the breakdown and every source.

## Architecture

```mermaid
flowchart LR
  A[AI answer] --> B[Claim extractor<br/>model via Bifrost]
  B --> C[Tavily search<br/>per claim, in parallel]
  C --> D[Verifier<br/>second model via Bifrost<br/>sources only]
  D --> E[Freshness score<br/>fixed formula]
  E --> F[(Pin ledger<br/>SQLite)]
  F -->|fact changed| G[Reopen pinned decisions]
  G --> H[Person reviews evidence<br/>and re-approves]
  H --> F
  B -. fallbacks, budget, trace .- X[Bifrost gateway]
  D -. fallbacks, budget, trace .- X
```

## Run it

FreshPin needs **Python 3.10+** and nothing else: no `pip install`.

### 1. Start Bifrost

```bash
npx -y @maximhq/bifrost -app-dir ./bifrost-data    # or: docker run -p 8080:8080 maximhq/bifrost
```

Then open http://localhost:8080. With Bifrost v2.5:

1. **Admin account.** The first time, Bifrost asks for a setup token. Start it with `BIFROST_SETUP_TOKEN=<any long random string>` set, enter that token in the browser, then create an admin username and password.
2. **Providers.** Under **Models → Model Providers**, add a key for two providers. Two free options: **Gemini** (key from aistudio.google.com) and **Groq** (key from console.groq.com; Groq, not Grok). Any providers Bifrost supports will work.
3. **Virtual key.** Under **Governance → Virtual Keys**, create a key, allow both providers **and the models you will use**, and copy it (`sk-bf-...`). If "Enforce Virtual Keys on Inference" is on, calls without it get HTTP 401.

The model names FreshPin uses by default (`gemini/gemini-2.5-flash`, `groq/openai/gpt-oss-120b`) worked on 15 Nov 2026. Providers retire models, so check the names in Bifrost's model list. See the [Bifrost docs](https://docs.getbifrost.ai) for details.

### 2. Get a Tavily key

Sign up at [app.tavily.com](https://app.tavily.com).

### 3. Configure and start

```bash
cp .env.example .env           # add TAVILY_API_KEY, BIFROST_VIRTUAL_KEY and your model names
python -m freshpin serve       # http://127.0.0.1:8787
```

The header chips show whether Bifrost is reachable and the Tavily key is set.

### 4. Try it

- Click **Dubai building rules** to load the scenario and its approved decisions, then **Check freshness**.
- Tick **Simulate primary model outage** and check again to see Bifrost's fallback in the trace.
- Open a reopened decision, tick the confirmation, add your name and re-approve.
- Paste any AI answer of your own, then use **Save a new decision that relies on these facts** to pin it.

### Command line

```bash
python -m freshpin sample dubai_building_law      # load a scenario, check it, print a summary
python -m freshpin check "The latest Python release is 3.11."
```

### From an AI assistant (MCP)

`python -m freshpin mcp` runs FreshPin as an [MCP](https://modelcontextprotocol.io) server over stdio (standard library only). An assistant can then check an answer before giving it to a person.

| Tool | What it does |
|---|---|
| `check_answer` | `answer` (and optional `context`) → each claim with its verdict, score, today's statement and sources, plus any decisions it reopened. Unverified claims are labelled UNVERIFIED, never "stale". |
| `get_ledger` | Read-only: facts and versions, decisions and their status, recent audit events. |

There is **no approve tool, on purpose**: an assistant can check facts and reopen decisions, but only a named person can re-approve, in the web app.

Add it to Claude Code (use the full path to this folder):

```bash
claude mcp add freshpin -s user -e PYTHONPATH=/path/to/freshpin -- python3 -m freshpin mcp
```

For Claude Desktop or Cursor, add the same command to the app's MCP settings:

```json
{ "mcpServers": { "freshpin": {
    "command": "python3", "args": ["-m", "freshpin", "mcp"],
    "env": { "PYTHONPATH": "/path/to/freshpin" } } } }
```

Then ask, for example: *"Check this answer with freshpin before you use it: Dubai does not require older buildings to have a safety certificate."* The MCP tool uses the same `.env` settings and the same ledger as the web app, so a decision it reopens shows up there for a person to review.

### Offline replay mode and tests

`FRESHPIN_MODE=replay` uses recorded responses from `fixtures/replay.json` (real source URLs, short summarised snippets) so the UI and tests run without keys or internet. The UI shows a banner whenever replay mode is on. Live mode is the real product.

```bash
python -m unittest discover -s tests -v
```

The tests cover the scoring formula, the ledger's reopen and re-approve rules, the full pipeline in replay mode, outage handling and degradation, the Bifrost client against a fake gateway (fallback detection, including model names that contain a slash, failure and budget), and the MCP server (tool list, a full check that reopens decisions, unverified labels, and errors that don't crash it).

## JSON API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/status` | | mode, Bifrost reachability, Tavily key set, models, budget |
| GET | `/api/samples` | | demo scenarios |
| POST | `/api/load_sample` | `{"id"}` | resets the ledger to that scenario |
| POST | `/api/check` | `{"answer", "context", "simulate_outage"}` | claims, verdicts, scores, sources, ledger effects, gateway trace |
| GET | `/api/ledger` | | decisions, pins, facts, audit log |
| POST | `/api/approve` | `{"id", "by", "note"}` | re-approves a decision on current facts |
| POST | `/api/decisions` | `{"title", "by", "owner", "keys"}` | pins a new decision to checked facts |

## Project layout

```
freshpin/
  config.py     settings from env / .env
  llm.py        Bifrost client: fallbacks, client failover, budget, call traces
  search.py     Tavily client
  pipeline.py   extract -> search -> verify -> score -> ledger
  scoring.py    the freshness formula
  ledger.py     versioned facts, decisions, pins, audit log (SQLite)
  replay.py     offline backends for tests
  app.py        wiring for the UI and CLI
  server.py     standard-library HTTP server + JSON API
  mcp.py        MCP server (stdio) for AI assistants
web/index.html  the UI (no build step, no external assets)
samples/        demo scenarios and their seeded decisions
fixtures/       recorded responses for replay mode
tests/          unittest suite
```

## Limits

- Verdicts depend on what Tavily finds. When sources are thin or undated, FreshPin says so (the "check" band, "unverifiable") rather than guessing, and it never reopens a decision on unclear evidence.
- Claim keys are produced by the extractor model and matched to known facts; two different wordings of the same fact can occasionally get different keys.
- Free provider tiers have rate limits. Bifrost's fallbacks cover short bursts; heavy use needs paid keys or more providers.
- Bifrost runs next to FreshPin (locally or on a server you control), so there is no hosted demo link: run it locally in three steps, or use replay mode with no keys.
- The demo ledger is a local SQLite file. A production version would sit next to the systems where decisions live (CRM, ERP, document approvals).

## Built during the hackathon, and use of AI tools

The FreshPin code was written during the hackathon period. The idea of pinning approvals to fact versions comes from the author's earlier work on back-office software; no code from that project was used.

AI tools were used in development: Claude (Anthropic) helped write code, tests and documentation. At runtime FreshPin uses whichever models you enable in Bifrost, plus Tavily search.

## License

MIT. See [LICENSE](LICENSE).
