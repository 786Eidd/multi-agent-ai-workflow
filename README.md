# Multi-Agent AI Workflow | Autonomous Research & Decision Automation

**Role:** AI Engineer | Multi-Agent Systems Developer

## What this is

A single general-purpose prompt tends to blur together three very different jobs: going and getting information, making sense of it, and checking that the sense you made of it is actually right. This project splits those jobs across specialized agents that hand off to each other through a real state machine, instead of asking one model to silently do all three in its head. A Research Agent retrieves data, an Analysis Agent turns it into metrics and a summary, and a Verification Agent independently re-checks the Analysis Agent's work against the raw data before anything is shown to a human — and a human still has to approve it before it's final.

The concrete task is company research: given a ticker, the system pulls financials and recent news, computes a profit margin and a news-sentiment read, sanity-checks both against the source data, and produces a report that a human either approves or rejects. The architecture — specialized agents, shared state, conditional routing, a real approval gate — is the point; company research is the scenario it's demonstrated on.

## Key capabilities

- **Multi-agent orchestration** — a LangGraph state machine coordinates four agent nodes plus conditional routing, not a single prompt doing everything.
- **Specialized AI agents** — Research, Analysis, and Verification are separate modules with distinct responsibilities and no overlap: Analysis never fetches data, Verification never generates a summary, it checks one.
- **Agent-to-agent communication** — each agent reads the prior agents' output from shared graph state and writes its own back; nothing is re-derived from scratch or passed as an opaque string.
- **Tool and API usage** — the Research Agent makes real HTTP calls to two mock external APIs (company financials, news headlines).
- **Information retrieval** — structured company data and recent headlines are retrieved per request, not baked into a prompt.
- **Research and analysis workflows** — retrieval feeds computation (profit margin, revenue per employee, keyword-based news sentiment) feeds independent verification.
- **Task delegation** — the Orchestrator Agent validates the incoming request and hands off to the specialist agents; it doesn't do their jobs itself.
- **Context management** — a typed `WorkflowState` is threaded through every node; each agent only writes the fields it owns.
- **Workflow state management** — LangGraph's checkpointer persists state across the human-approval pause, so a workflow can be resumed in a completely separate request/process.
- **Human-in-the-loop approval** — a real `interrupt()` pause, not a step that always says yes. Nothing is finalized until a separate, explicit approve/reject call resumes the graph.

## Workflow

```
User / Business Request
        ↓
Orchestrator Agent
   ↙    ↓    ↘
Research Agent | Analysis Agent | Verification Agent
        ↓
Result Aggregation
        ↓
Final Response / Report
        ↓
Human Approval
```

Concretely, in `src/orchestrator.py` (a LangGraph `StateGraph`):

1. **User / Business Request** — a ticker arrives via `POST /workflow/start`.
2. **Orchestrator Agent** — validates the ticker format and initializes workflow state; a malformed request is rejected here, before any specialist agent runs.
3. **Research Agent** — calls the mock Company Data API and News API over real HTTP.
4. **Analysis Agent** — computes profit margin, revenue per employee, and a keyword-scored news-sentiment label from what Research retrieved.
5. **Verification Agent** — independently recomputes the margin from the raw figures and compares it to Analysis's output, checks for internally impossible financial data, and flags thin news coverage. If the data fetch itself failed, it recommends a retry; the graph loops back to Research (once) before giving up.
6. **Result Aggregation** — combines everything into one structured report with a recommendation (`auto_finalize` or `needs_human_decision`).
7. **Human Approval** — the graph genuinely pauses (`interrupt()`) and returns the draft report; a separate `POST /workflow/{thread_id}/decide` call is what actually finalizes it.

## Frameworks actually used

**LangGraph** — the entire orchestration layer. This is the one project in this portfolio where the routing is genuinely dynamic (the Verification Agent's result decides whether the graph loops back to Research or moves on), which is exactly what LangGraph is for — contrast with `ai-integration-hub`, which deliberately does **not** use LangGraph because its routing is fixed business logic decided at design time.

**LangChain was not used.** Nothing here needs LangChain's own abstractions (document loaders, chains, message types) — building directly on LangGraph doesn't require it, and importing it would add a dependency without adding a capability.

**CrewAI was not used.** It's the other well-known multi-agent framework, and it was seriously considered — but CrewAI's own agent abstraction is built around making real LLM calls to reason and act, and this sandbox has no outbound route to an LLM API. Wiring in a framework that can't actually be exercised or tested here, and then listing it as a working part of the deliverable, is exactly the kind of unverified claim this portfolio avoids. Every agent here is real, tested Python instead, with a genuine optional LLM call where an LLM is actually useful (see below) — not a framework integration nobody could verify.

## Why rule-based analysis by default, and the real optional LLM path

This sandbox has no outbound route to an LLM API, so `analysis_agent.py`'s narrative summary is a filled template by default — real, tested code, not a placeholder. If `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` is set, it instead makes a genuine LLM call to write the summary, falling back to the template if that call fails. The profit-margin math and the keyword-based sentiment scoring are deterministic either way — an LLM doesn't get to invent a number that contradicts the retrieved data; it only gets to phrase the summary of numbers that were already computed and already verified.

## Real measured results

Measured by `src/evaluate.py` against 11 hand-labeled scenarios (`data/test_scenarios.json`) — 8 fictional companies covering clean cases, a legitimate loss, thin news coverage, and one company with deliberately impossible financial data (net profit reported higher than revenue), plus two unknown-ticker cases and one malformed-request case — run end to end through the real LangGraph workflow, including resuming past the human-approval pause. Ground truth (profit margins, sentiment labels) was computed by hand from the raw source data, independent of the running code.

| Metric | Result |
|---|---|
| Profit-margin calculation accuracy | **100%** (8/8) |
| News-sentiment classification accuracy | **100%** (8/8) |
| Verification pass/fail decision accuracy | **100%** (10/10) |
| Sparse-coverage warning detection accuracy | **100%** (8/8) |
| Retry-behavior accuracy (research retried exactly once on failure, not zero or unbounded) | **100%** (2/2) |
| Malformed-request rejection accuracy | **100%** (1/1) |
| Human-approval round trips completed (pause → resume → final report) | **10/10** |
| Resilience check (Company Data API unreachable → graceful degradation, not a crash) | **Passed** |

Full per-case results are in `data/eval_results.json`.

### On the 100% scores

Unlike the OCR and extraction accuracy numbers elsewhere in this portfolio, these aren't measuring an inherently uncertain judgment call against messy real-world input — the underlying arithmetic and keyword-matching logic is deterministic, and the ground truth was verified by hand before the code ran, not tuned to match it afterward. What's actually being tested here, and what 100% demonstrates, is that the coordination logic works correctly: verification independently re-derives the margin rather than trusting analysis, a deliberately-impossible data point (WLTR) really does get caught rather than waved through, an unknown ticker really does get retried exactly once rather than looping forever or giving up immediately, a malformed request really does get rejected before any agent runs, and the human-approval pause really is a pause — the workflow does not silently finalize on its own. Those are the claims worth verifying for a multi-agent system, and each one has a scenario in the test set that would have caught it failing.

### Honest limitations

- **Sentiment scoring is a small hand-built keyword lexicon, not a trained model.** It's genuinely computed from the headline text (not a field the mock API hands back), but it will miss sarcasm, negation, and any vocabulary outside its ~30-word list. CDRG's case (2 positive-leaning headlines, 1 negative) scored as "positive" at a sentiment score of only +0.2 — reasonable, but a close call a more sophisticated model might read differently.
- **The Company Data and News APIs are mocks with 8 fictional companies.** There's no live market-data integration; pointing `data_client.py` at a real financial-data provider's REST API is a base-URL-and-auth change, not a rewrite, but it hasn't been done or tested here.
- **Retry logic only helps with transient/missing-data failures, not bad data.** WLTR's impossible financials aren't something a re-fetch of the same static mock record would ever fix, so verification correctly does *not* recommend a retry for that case — it goes straight to a human decision instead.

## Running it locally

```bash
pip install -r requirements.txt

# Three separate processes:
uvicorn src.company_data_service:app --port 8020
uvicorn src.news_service:app --port 8021
uvicorn src.app:app --reload --port 8019

# Start a workflow run — it pauses at the human-approval step:
curl -X POST http://localhost:8019/workflow/start -H "Content-Type: application/json" \
     -d '{"ticker": "MRDN", "thread_id": "demo-1"}'

# Inspect the paused report in the response, then approve or reject it:
curl -X POST http://localhost:8019/workflow/demo-1/decide -H "Content-Type: application/json" \
     -d '{"decision": "approve", "note": "looks good"}'
```

To reproduce the measured results above from a clean checkout:

```bash
pip install -r requirements.txt
python3 -m src.evaluate
```

`src/evaluate.py` starts its own isolated copies of the Company Data and News services on separate ports, so running it never depends on (or interferes with) a normal run of `src/app.py`.

## Project structure

```
src/
  company_data_service.py   # Mock Company Data API (financials) — FastAPI
  news_service.py             # Mock News API (headlines, no pre-computed sentiment) — FastAPI
  data_client.py                # REST client for both mock services
  research_agent.py              # Research Agent — retrieves raw data
  analysis_agent.py               # Analysis Agent — computes metrics + sentiment + summary
  verification_agent.py            # Verification Agent — independent cross-checks
  orchestrator.py                    # The LangGraph state machine + human-in-the-loop
  app.py                               # FastAPI workflow start/decide endpoints
  evaluate.py                           # Runs the labeled test set, writes real metrics
data/
  test_scenarios.json      # 11 hand-labeled test scenarios
  eval_results.json         # Real measured output of the last evaluate.py run
```

## License

MIT — see `LICENSE`.
