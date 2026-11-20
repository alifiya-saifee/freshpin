# FreshPin: a benchmarking platform for AI agents, not just models

Geek Room AI Global Hackathon 2026 · Problem statement 02 (Bifrost track) · Category: **research agents**
Repository: [github.com/alifiya-saifee/freshpin](https://github.com/alifiya-saifee/freshpin)

---

## 1. The problem statement

> Traditional AI benchmarks measure model performance, but they rarely explain how well an entire AI agent performs. Two agents using the same model can produce very different results because of their prompts, tools, memory and retry strategies: one may complete a task in a handful of calls, while another consumes far more resources and still fails. Developers need a consistent way to compare agent performance beyond successful demos.

**Goal:** a framework-independent platform that evaluates agents of one category against identical tasks and measurable criteria.

| # | Mandatory requirement | What it asks for |
|---|---|---|
| 1 | Framework independence | Agents built with any framework, or none, without rewriting them |
| 2 | Consistent evaluation | Same tasks; record success, steps, tool calls, tokens, cost and time |
| 3 | Repeatability | Run tasks several times; measure reliability, not one lucky run |
| 4 | Failure testing | Unavailable models, broken tools, misleading instructions |
| 5 | Inspection & comparison | Step-by-step execution records and a comparison interface |

**Expected outcome:** compare agents objectively, understand why they differ, and see how they behave under normal and failure conditions.

---

## 2. The issues we traced

We built the platform, then ran it: **5 models from 5 families, 6+ agent designs, 9 research tasks about that week's news, up to 3 repeats, up to 8 conditions**. Every number below is from a committed result file in `bench/samples/`, with full traces.

### 2.1 Issues in the agents

| Issue | What we found | Evidence |
|---|---|---|
| **The agent matters as much as the model** | Same model, same evidence: three tool-using agents scored **71%, 80% and 100%**; with the model unavailable, **0 vs 9 of 10** | §3.1 of `problem-solution-results.md` |
| **Models invent tools** | When the model chooses its own tools, gpt-oss and Qwen call tools that don't exist (`search_query`, `open`, `searchWeb`, `search_widget`…). The provider rejects the call and the run ends with no answer. **34 runs**, all by `tool_agent` | Provider error in every trace: *"attempted to call tool 'search_query' which was not in request.tools"* |
| **Agents make answers up when evidence is missing** | With search broken, `tool_agent` on gpt-oss **invented an answer in 4 of 9 runs** (a 13 Nov 2026 outage dated "February 23, 2024"; a product launched that week "went on sale March 8, 2015") | Traces, read by hand and by the word rule |
| **A false hint in the task beats a false web page** | In **all 10** model-agent pairs, a false hint inside the task did more damage than a planted search result. Poolside Laguna's `tool_agent` repeated the false company in **12 of 27** runs | §3.5 |
| **One run hides unreliability** | Lightning's `search_first`: right in 20 of 27 runs, but right on **all 3 tries for only 6 of 9 tasks** | pass^3, §3.4 |
| **Error handling decides outages, unless the gateway does** | With the model rejected, agents with no fallback logic scored **0/27**. With Bifrost's own per-request fallbacks: **25–27 of 27** | §3.8, Bifrost logs show `fallback_index` 1 on all 108 runs |
| **Cost is search, not tokens** | About **$0.008 per search** against **$0.0001–0.0007** of model tokens per run at list price | §3.2 |

### 2.2 Issues in *measuring* agents (what makes most benchmarks misleading)

| Issue | What we found | What FreshPin does about it |
|---|---|---|
| **Training-data contamination** | A model that "knows" the answer measures memory, not the agent | Tasks are built from **this week's news**, after every model's training. The no-tools agent scores **0 on every model**, which proves it |
| **The grader is biased** | gpt-oss-20b graded **72 honest "I couldn't find it" answers as wrong**. With that grader alone, Lightning looked half as honest as it was | Every grader is **checked on known answers before a run** (right, wrong, honest "not found"). Two graders from different families must agree (`--consensus`). Honesty is counted by a **fixed word rule**, not a grader. A person settles disagreements in the UI |
| **Provider limits look like agent failures** | In one Gemini run, **49 of 108** runs hit the free tier's rate limit | Rate-limit and quota kills are **excluded and reported**, never scored against the agent |
| **Small differences look real** | 80% vs 100% on 9 tasks is not proof | A gap is called real only if a **paired sign test (p < 0.05) and a bootstrap interval** agree. With too few tasks it says *"too few tasks to tell"* |
| **Different agents, different evidence** | Live web search returns different pages each time | Every search is **recorded once and replayed**, so every agent and repeat sees identical evidence (and re-runs cost 0 search credits) |
| **The benchmark itself had bugs** | One condition was silently never switched on (a run "measured" normal); two runs started in the same second overwrote each other | Both found by benchmarking FreshPin itself, fixed, and pinned by tests |

### 2.3 This matches what the field reports

Each finding above lines up with problems developers report on forums and that researchers published this year. FreshPin measures all of them on the same agents, under the same conditions.

| What developers and researchers report | What FreshPin found | Outside evidence |
|---|---|---|
| Models call tools that don't exist | 34 invented-tool runs (gpt-oss, Qwen) | Groq forum: gpt-oss "calling tools that don't exist" ([link](https://community.groq.com/t/tool-calling-errors-on-both-gpt-oss-models/406)); a paper counts gpt-oss-120b calling an undefined tool 860 times ([arXiv 2604.00362](https://www.alphaxiv.org/abs/2604.00362)) |
| Agents make up answers when a tool fails | gpt-oss invented answers in 4 of 9 broken-tool runs | NeurIPS 2026: *When Tools Fail, Agents Fabricate* ([link](https://neurips.cc/virtual/2026/173038)) and *Fabrication After Tool Failure* ([link](https://neurips.cc/virtual/2026/179974)); n8n forum: "the tool quietly failed, and it answered anyway" ([link](https://community.n8n.io/t/your-ai-agent-said-done-but-the-tool-under-it-quietly-failed-and-it-answered-anyway/301674?tl=en)) |
| Final-answer scoring hides dishonesty; LLM judges are biased | A grader marked 72 honest answers wrong | 12 documented judge biases ([arXiv 2410.02736](https://arxiv.org/pdf/2410.02736)); consistent judges can still be biased ([arXiv 2606.19544](https://arxiv.org/pdf/2606.19544)) |
| The same agent gives different results each run | Right in 20 of 27 runs, but on all 3 tries for only 6 of 9 tasks | τ-bench introduced pass^k; gpt-4o fell to about 25% at pass^8 ([arXiv 2406.12045](https://arxiv.org/pdf/2406.12045)) |
| Benchmarks leak into training data | No-tools agent scores 0 on every model | LiveBench uses fresh questions for the same reason ([arXiv 2406.19314](https://arxiv.org/abs/2406.19314v2)) |
| Web pages hijack research agents | A false hint beat a planted page in all 10 pairs | In-the-wild injection payloads aimed at agents ([Infosecurity](https://infosecurity-magazine.com/news/researchers-10-wild-indirect)); Google warns of poisoned pages ([AI News](https://www.artificialintelligence-news.com/news/google-warns-malicious-web-pages-poisoning-ai-agents/)) |
| Benchmarks don't predict production; recovery matters more than the model | Bifrost fallbacks took 0/27 to 25–27/27 | Production builders on error recovery and upstream failures ([AI Weekly](https://aiweekly.co/alerts/production-agent-builders-flag-demo-reliability-gap)) |

These sources were found by web search. The NeurIPS papers are cited from their published abstracts; read them before quoting specific results.

---

## 3. The solution: FreshPin

**One line:** FreshPin runs any research agent, in any language or framework, on the same questions about this week's news, several times, under normal and failure conditions. It records every step and shows the results in a browser.

```
this week's news ──► bench-build ──► tasks + gold answer + exact quote
                                             │
     any agent (command or HTTP) ◄── task JSON ─┤
       │  models via Bifrost (tagged with run id)  → tokens, $, errors, fallbacks from Bifrost's log
       │  tools via FreshPin tool server (HTTP/MCP) → every call logged, faults injected here
       ▼
     answer JSON ──► two graders + key-term check + planted-answer check ──► /bench, CI gate, OTel traces
```

### Requirement by requirement

| Requirement | How FreshPin meets it |
|---|---|
| **1. Framework independence** | An agent is **any command or HTTP endpoint**: task JSON in, answer JSON out. Tested unchanged: plain Python, **JavaScript** with no packages, **OpenAI Agents SDK** (function tool and **MCP** tool), an agent **served over HTTP**, and **LangGraph** (`StateGraph` + `ToolNode`) |
| **2. Consistent evaluation** | Same tasks, same replayed evidence. Per run: **success, steps, tool calls, tokens, $ cost** (from Bifrost's own log), **list-price $**, **time**, **why it failed**, **repeated searches**, optional **step budget** |
| **3. Repeatability** | `--repeats k`: success with 95% Wilson interval, **pass^k** (right every time), pass@k, p50/p95 time. Published: 270 runs per model at k = 3 |
| **4. Failure testing** | 13 conditions (table below) |
| **5. Inspection & comparison** | `/bench`: leaderboard, heatmap, cost-vs-accuracy chart, report cards, **why-runs-failed chart**, step-by-step traces, **side-by-side** run comparison, compare across result files, agent-vs-agent significance, review queue for a person |

### The failure conditions

| Condition | What is injected | What it measures |
|---|---|---|
| Model unavailable | Rejected model id plus a fallback list | The agent's own fallback logic |
| Model unavailable, gateway fallbacks | Rejected model, no list; Bifrost adds fallbacks | Whether the gateway can own resilience |
| **Model goes down mid-run** ¹ | First call works, later calls rejected | Switching models in the middle of a task |
| Broken tool | Search returns 503 every time | **Honesty**: "not found" or a made-up answer |
| **Flaky tool** ¹ | First search fails, a retry works | **Retry strategy** |
| Slow tool | Search answers after 20 s | Timeouts and patience |
| Wrong search results | Real pages about other stories | Honesty under irrelevant evidence |
| **Empty search results** ¹ | Search answers HTTP 200 with nothing found | Honesty when there's no error to react to |
| **Hidden injected instruction** ¹ | Normal-looking page; false answer in a `display:none` block | Resistance to injection hidden in page content |
| Misleading source | Planted result with a wrong answer and a hidden instruction | Prompt-injection resistance |
| Misleading instruction | False hint inside the task | Trusting the user over the evidence |
| **Stale memory** ¹ | A month-old wrong note in the agent's memory | Re-checking vs trusting memory |
| Request cap | A Bifrost key per run capped at N requests | Stopping cleanly at a budget |

¹ Built and tested offline (fake gateway, real tool server); not yet run on real models.

---

## 4. KPIs

| Family | KPI | Why it matters |
|---|---|---|
| **Effectiveness** | Success rate with 95% interval | Is the agent right, and how sure are we? |
| | **pass^k**: tasks right on every repeat | Reliability, not one lucky run |
| **Efficiency** | Steps, tool calls, tokens, time p50/p95 | How much work the agent does |
| | **List-price $ per correct answer** | Comparable cost, even on free tiers |
| | Steps per success, **repeated searches**, over step budget | Wasted work |
| **Robustness** | Drop from normal under each failure | What breaks it |
| | Recovery rate (fallbacks, **retries**) | Does it recover by itself? |
| **Honesty & safety** | Said "not found" vs **made up** an answer when evidence is missing | Fabrication risk |
| | Times it **followed planted text** (source, hint or memory) | Injection and misinformation risk |
| **Diagnosis** | **Why it failed**: invented tool, bad arguments, model unavailable, gave up on tool, no answer, followed planted, made up, wrong answer | What to fix, not just a score |
| **Trust in the numbers** | Grader agreement; runs excluded for provider limits; runs settled by a person | Whether the score itself can be believed |

---

## 5. Why FreshPin is the best answer to this brief

1. **It measures the agent, not the model's memory.** Questions about this week's news can't be answered from training data, so the score reflects how the agent searches, reads and reasons. The no-tools baseline scoring 0 everywhere is the proof.
2. **Every agent sees exactly the same evidence.** Recorded-and-replayed search makes runs fair, repeatable and free to re-run. Anyone can reproduce the published results from the repository.
3. **It explains failures, not just scores them.** Each failed run gets a reason from fixed rules over its trace. On the published results this found the 34 invented-tool runs and the 4 fabricated answers, which we had first found by reading traces by hand.
4. **It tells agent problems apart from gateway problems.** The same outage runs with the agent's own fallbacks and with Bifrost's. 0/27 becoming 25–27/27 shows exactly where resilience should live.
5. **It treats honesty as a first-class result.** When evidence is missing, "I couldn't find it" is the right answer, and FreshPin counts it with a fixed rule because we showed that a grader can be biased against it.
6. **It refuses to overclaim.** Graders checked on known answers before a run, two graders, paired significance tests, "too few tasks to tell", provider limits excluded. A difference is reported as real only when it is.
7. **It plugs into how teams already work.** A CI gate fails the build when an agent gets worse or starts making things up. Traces export to OpenTelemetry (Jaeger, Langfuse, Datadog…). A read-only results site publishes from GitHub Pages.
8. **It was tested on itself.** 206 automated tests, passing on Linux, macOS and Windows. New rules are checked by deliberately breaking them and confirming the tests fail. Running FreshPin on real models found and fixed two bugs in FreshPin itself.

---

## 6. Honest limits

- **Small task set** (9 tasks): differences of one or two runs are noise, and the significance test says so.
- **Flaky tool, mid-run outage, stale memory, empty results, hidden injection, step budget, the grader check and the LangGraph agent** are built and tested offline but not yet run on real models.
- **Running agents live** needs your own Bifrost and model keys; the saved results and tests work offline.
- **The honesty word rule and failure-reason patterns** are English-only and built from the provider errors seen so far.
- A few **gold answers are loose**; the two-grader check identified which ones.

---

## 7. Try it

```bash
python -m unittest discover -s tests                     # offline, no keys
python -m freshpin bench-static site/                     # read-only results site
python -m http.server -d site                             # then open http://localhost:8000
python -m freshpin agentbench --questions bench/suites/fresh-arena3.json --replay-only \
    --model <bifrost model> --fallbacks <another model> \
    --agent 'mine=python3 my_agent.py' \
    --conditions normal,model_outage,tool_error,flaky_tool,misleading,stale_memory --repeats 3 --yes
python -m freshpin gate bench/results/<result>.json --min-success 0.8 --max-answered-anyway 0
```
