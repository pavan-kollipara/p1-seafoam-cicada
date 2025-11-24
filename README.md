# Phase 1 – LangGraph Ticket Triage Agent

This project implements a minimal multi-agent workflow using **LangGraph**, demonstrating:

- Ticket ingestion
- Issue classification
- Order lookup (via ToolNode)
- Reply drafting
- Conditional control flow (extract order_id if missing)
- FastAPI endpoint (`POST /triage/invoke`)
- LangSmith tracing support
- Passing CI tests

---

## 🧠 Architecture (LangGraph)

**State Fields**

- `messages`
- `ticket_text`
- `order_id`
- `issue_type`
- `evidence`
- `recommendation`
- `order` (internal)

**Nodes**

1. `ingest`
2. `classify_issue`
3. `extract_order_id` (conditional)
4. `fetch_order` (ToolNode)
5. `draft_reply`

---

## 🚀 Run Locally

**Setup**
```bash
# Activate venv
python -m venv .venv
source .venv/bin/activate
```

# Install dependencies
```bash
pip3 install -r requirements.txt
```

**Load FastAPI**
```bash
uvicorn app.main:app --reload
```

**Curl Example**
```bash
curl -X POST http://localhost:8000/triage/invoke \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "The smart watch I got (ORD1004) is not working."}'
```

**Tests**
```bash
python3 -m pytest tests/test_triage.py -q
```

### 🔍 Tracing

Set the following `env vars` to enable LangSmith tracing (auto-detected by the graph):

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your_langsmith_key>
```

---

## 🧰 Dev Notes (Cursor / Claude Code)

Leveraged Claude Code as a structured copilot: prompted it to outline the LangGraph state contract and node sequencing, then iteratively filled in each node in `app/main.py`.

The ingest node emits a HumanMessage, classify_issue applies keyword rules from `mock_data/issues.json` and appends an AIMessage, a conditional router chooses between extract_order_id or a direct fetch_order, a ToolNode wraps fetch_order_tool to retrieve mock orders, and draft_reply templates responses from `mock_data/replies.json`.

Claude also generated the mock datasets and sanity-checked FastAPI error handling for missing or unknown orders.

For tests, it proposed pytest coverage aligned with the graph contracts: parametrized happy-path tickets verifying issue_type, order_id, reply presence, and message history, plus a negative case for tickets without detectable order IDs. These were refined and incorporated into `tests/test_triage.py` to guard expected agent behaviors.

