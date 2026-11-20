# FreshPin demo video: 2-minute script

For: hackathon judges. Record the screen of the read-only results site (`https://alifiya-saifee.github.io/freshpin/`) or a local `python -m freshpin serve` → `/bench`. Each `?file=` link below opens straight onto the result being shown.

| Time | On screen | Say |
|---|---|---|
| **0:00–0:15** | README title, then the results site's leaderboard | "Benchmarks score models. But you ship agents: a model plus prompts, tools, retries and fallbacks. FreshPin benchmarks the agent." |
| **0:15–0:35** | `?file=agentbench-20261115-155950.json`: leaderboard, normal column | "Same model, same evidence, three agent designs: 71, 80 and 100 percent. With the model unavailable: zero against nine of ten. The design decided the result, not the model." |
| **0:35–0:50** | Scroll to the heatmap; point at the no-tools row | "The questions are about this week's news, after every model's training. The agent with no tools scores zero everywhere, so nothing here comes from memory." |
| **0:50–1:15** | `?file=agentbench-20261116-093348.json`: "Why runs failed" chart, click "invented a tool ×12", open one trace | "When the model chooses its own tools, Qwen calls tools that don't exist: search_query, searchWeb. The provider rejects the call and the run ends with nothing. FreshPin labels every failure from the trace, with no grader involved." |
| **1:15–1:35** | `?file=agentbench-20261117-125639.json`: Run card (two graders), then a broken-tool trace saying "I couldn't find it" | "We also caught the grader. One marked seventy-two honest 'I couldn't find it' answers as wrong. So FreshPin checks every grader on known answers before it runs, and counts honesty with a fixed rule." |
| **1:35–1:50** | Side-by-side view: `search_first` vs `tool_agent` on one task under model unavailable | "Every run has a step-by-step trace: each model call with tokens and cost from Bifrost, each tool call with the fault we injected. Pick two runs and compare them side by side." |
| **1:50–2:00** | Terminal: `python -m freshpin agentbench ... --agent 'mine=python3 my_agent.py'` then `python -m freshpin gate ...` | "Bring any agent, in any language or framework. Thirteen failure conditions, repeats, significance tests, and a CI gate that fails the build when your agent gets worse." |

**Tips**
- Keep the cursor still while talking; click only on the cue.
- Read the numbers exactly as written: each is from a committed result file.
- Don't claim results for the new conditions (flaky tool, mid-run outage, stale memory, empty results, hidden injection) until they have been run on real models.
