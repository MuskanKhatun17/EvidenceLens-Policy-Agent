# EvidenceLens AI - Deep Agents + MCP + Memory (Article E)

**Agent = Model + Harness.** Model: Gemini (3-key fallback). Harness: Deep Agents orchestrator (planning via `write_todos`,
delegation via `task`) -> 4 subagents -> tools served by a **local MCP server** -> report accepted only via validated
`submit_final_report`.

## MCP server used
`mcp_server/server.py` - our own local server **`evidencelens-tools`** (FastMCP, stdio transport). The agent starts it as a
subprocess through `langchain-mcp-adapters` (`MultiServerMCPClient`). Tools exposed: `search_evidence` (Serper),
`fetch_source`, `assess_sources`, `score_policy_options`, `validate_report`.
- Only SERPER_API_KEY is passed to the server; Gemini keys never are.
- If MCP can't start, the app falls back to identical in-process tools and says so ("Tools used via").
- Other MCP servers can be added to `mcp_config()` in `agent/orchestrator.py` (none others are tested here).

## Memory
| Layer | Where | What |
|---|---|---|
| Short-term (session) | LangGraph `InMemorySaver` per session id + file-backed source registry `outputs/sessions/<id>.json` | Conversation history, registered sources (S1, S2...) so follow-up questions reuse them. "New session" resets it. |
| Long-term | `outputs/memory/long_term.json` (`agent/memory.py`) | User priorities/weights + metadata of reports you click "Approve". Injected into the prompt as hints, marked NOT evidence. Viewable/clearable in the sidebar. |

## Subagents
evidence-researcher (search_evidence, fetch_source) | evidence-analyst (assess_sources) | policy-evaluator (score_policy_options) | source-validator (validate_report)

Guardrails: tool-call budget (30/run, enforced inside the tools), max 3 submits, recursion limit, input validation,
untrusted-web-text rules + injection stripping, citations must match registered IDs, no sources => no recommendation, keys never logged.

## Run
```
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add GEMINI_API_KEY_1..3 and SERPER_API_KEY
python -m pytest tests -v   # offline tests (spawns the MCP server)
streamlit run app.py
```
Never commit `.env`. Mark live tests as passed only after you run them.

## When Gemini rate-limits you (quota handling)
1. Every call tries each key on each model in `GEMINI_MODELS` (quota is per model; default flash then flash-lite). 429/503/504/timeouts rotate; auth errors do not.
2. If everything is limited, it waits `QUOTA_WAIT_SECONDS` (25s) and retries once.
3. Still limited -> **Quota-saver mode**: tools run in Python, Gemini makes only 1-2 calls to write the report (marked in the UI).
4. Still limited -> **evidence-only report**: retrieved sources + quality labels, no AI analysis and NO recommendations.
Tick "Quota-saver mode" in the sidebar to skip the full agent run. Note: keys from the same Google project share one quota.
