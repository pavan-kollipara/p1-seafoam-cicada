
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import json, os, re
from typing import Annotated, Optional, TypedDict, List, Dict, Any

# --- LangGraph / LangChain imports ---
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

app = FastAPI(title="Phase 1 Mock API")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOCK_DIR = os.path.join(ROOT, "mock_data")

def load(name):
    with open(os.path.join(MOCK_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)

ORDERS = load("orders.json")
ISSUES = load("issues.json")
REPLIES = load("replies.json")

class TriageInput(BaseModel):
    ticket_text: str
    order_id: str | None = None

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/orders/get")
def orders_get(order_id: str = Query(...)):
    for o in ORDERS:
        if o["order_id"] == order_id: return o
    raise HTTPException(status_code=404, detail="Order not found")

@app.get("/orders/search")
def orders_search(customer_email: str | None = None, q: str | None = None):
    matches = []
    for o in ORDERS:
        if customer_email and o["email"].lower() == customer_email.lower():
            matches.append(o)
        elif q and (o["order_id"].lower() in q.lower() or o["customer_name"].lower() in q.lower()):
            matches.append(o)
    return {"results": matches}

@app.post("/classify/issue")
def classify_issue(payload: dict):
    text = payload.get("ticket_text", "").lower()
    for rule in ISSUES:
        if rule["keyword"] in text:
            return {"issue_type": rule["issue_type"], "confidence": 0.85}
    return {"issue_type": "unknown", "confidence": 0.1}

def render_reply(issue_type: str, order):
    template = next((r["template"] for r in REPLIES if r["issue_type"] == issue_type), None)
    if not template: template = "Hi {{customer_name}}, we are reviewing order {{order_id}}."
    return template.replace("{{customer_name}}", order.get("customer_name","Customer")).replace("{{order_id}}", order.get("order_id",""))

@app.post("/reply/draft")
def reply_draft(payload: dict):
    issue_type = str(payload.get("issue_type") or "unknown")
    return {"reply_text": render_reply(
        issue_type, 
        payload.get("order", {})
        )
    }

# ---------- LangGraph: State, tools, and nodes ----------


class TriageState(TypedDict):
    # required by assignment
    messages: Annotated[List[AnyMessage], add_messages]
    ticket_text: str
    order_id: Optional[str]
    issue_type: Optional[str]
    evidence: Optional[str]
    recommendation: Optional[str]

    # internal convenience fields
    order: Optional[Dict[str, Any]]


# --- Tool for fetching orders (used by ToolNode) ---


@tool
def fetch_order_tool(order_id: str) -> Dict[str, Any]:
    """Look up an order in the mock database."""
    for o in ORDERS:
        if o["order_id"] == order_id:
            return o
    # Raise ValueError so FastAPI can surface a clean 404 later
    raise ValueError(f"Order {order_id} not found")


tool_node = ToolNode([fetch_order_tool])


# --- Graph nodes ---


def ingest_node(state: TriageState) -> Dict[str, Any]:
    """
    Ingest the incoming ticket text into the messages list.
    """
    ticket_text = state["ticket_text"]
    return {"messages": [HumanMessage(content=ticket_text)]}


def classify_issue_node(state: TriageState) -> Dict[str, Any]:
    """
    Classify the issue using simple keyword rules (from issues.json).
    """
    text = state["ticket_text"].lower()
    issue_type = "unknown"
    evidence = "no matching keyword found"

    for rule in ISSUES:
        if rule["keyword"] in text:
            issue_type = rule["issue_type"]
            evidence = f"matched keyword '{rule['keyword']}'"
            break

    # we also append an AI message describing classification
    explanation = f"Detected issue_type='{issue_type}' ({evidence})."
    return {
        "issue_type": issue_type,
        "evidence": evidence,
        "messages": [AIMessage(content=explanation)],
    }


def extract_order_id_node(state: TriageState) -> Dict[str, Any]:
    """
    Extract order_id from either the provided field or the ticket text.
    This implements the 'control flow: extract order_id if missing'.
    """
    order_id = state.get("order_id")
    if not order_id:
        m = re.search(r"(ORD\d{4})", state["ticket_text"], re.IGNORECASE)
        if m:
            order_id = m.group(1).upper()
    return {"order_id": order_id}


def fetch_order_node(state: TriageState) -> Dict[str, Any]:
    """
    Use ToolNode to call fetch_order_tool and attach the order to state.
    """
    order_id = state.get("order_id")
    if not order_id:
        # let FastAPI transform this into a 400 later
        raise ValueError("order_id missing and not found in text")

    # Create an AIMessage with a tool call
    tool_calls = [
        {
            "name": "fetch_order_tool",
            "args": {"order_id": order_id},
            "id": "fetch_order_tool-1",
        }
    ]
    ai_msg = AIMessage(content="", tool_calls=tool_calls)

    # Run the tool node with existing messages + this tool-call message
    result_state = tool_node.invoke({"messages": state["messages"] + [ai_msg]})
    messages = result_state["messages"]
    last_msg = messages[-1]

    # ToolNode returns a ToolMessage as the last message
    if isinstance(last_msg, ToolMessage):
        content = last_msg.content
        if isinstance(content, str):
            try:
                order = json.loads(content)
            except Exception:
                order = {"raw": content}
        else:
            order = content
    else:
        order = None

    return {"messages": messages, "order": order}


def draft_reply_node(state: TriageState) -> Dict[str, Any]:
    """
    Draft a reply using the issue_type and order (mock template-based).
    """
    order = state.get("order") or {}
    issue_type = state.get("issue_type") or "unknown"
    reply_text = render_reply(issue_type, order)
    ai_msg = AIMessage(content=reply_text)

    return {
        "messages": [ai_msg],
        "recommendation": reply_text,
    }


# --- Graph wiring ---


def route_after_classify(state: TriageState) -> str:
    """
    Conditional edge:
    - If we already have an order_id (provided or previously extracted),
      go directly to fetch_order.
    - Otherwise, go to extract_order_id first.
    """
    if state.get("order_id"):
        return "fetch_order"
    return "extract_order_id"


graph_builder = StateGraph(TriageState)

graph_builder.add_node("ingest", ingest_node)
graph_builder.add_node("classify_issue", classify_issue_node)
graph_builder.add_node("extract_order_id", extract_order_id_node)
graph_builder.add_node("fetch_order", fetch_order_node)
graph_builder.add_node("draft_reply", draft_reply_node)

graph_builder.add_edge(START, "ingest")
graph_builder.add_edge("ingest", "classify_issue")

graph_builder.add_conditional_edges(
    "classify_issue",
    route_after_classify,
    {
        "fetch_order": "fetch_order",
        "extract_order_id": "extract_order_id",
    },
)

graph_builder.add_edge("extract_order_id", "fetch_order")
graph_builder.add_edge("fetch_order", "draft_reply")
graph_builder.add_edge("draft_reply", END)

graph = graph_builder.compile()
# If you set LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY env vars,
# this graph will automatically send traces to LangSmith.

@app.post("/triage/invoke")

def triage_invoke(body: TriageInput):
    # initial state for the graph
    initial_state: TriageState = {
        "messages": [],
        "ticket_text": body.ticket_text,
        "order_id": body.order_id,
        "issue_type": None,
        "evidence": None,
        "recommendation": None,
        "order": None,
    }

    try:
        result = graph.invoke(initial_state)
    except ValueError as e:
        # map our ValueErrors to HTTP errors similar to the original impl
        msg = str(e)
        if "not found in text" in msg:
            raise HTTPException(status_code=400, detail="order_id missing and not found in text")
        if "not found" in msg:
            raise HTTPException(status_code=404, detail="order not found")
        raise

    if not result.get("order_id"):
        raise HTTPException(status_code=400, detail="order_id missing and not found in text")
    if not result.get("order"):
        raise HTTPException(status_code=404, detail="order not found")

    # For debugging/demo, we serialize messages into strings
    serialized_messages = []
    for m in result["messages"]:
        if isinstance(m, (HumanMessage, AIMessage, ToolMessage)):
            serialized_messages.append(
                {"type": m.__class__.__name__, "content": m.content}
            )
        else:
            serialized_messages.append(str(m))

    return {
        "order_id": result["order_id"],
        "issue_type": result["issue_type"],
        "evidence": result["evidence"],
        "order": result["order"],
        "reply_text": result["recommendation"],
        "messages": serialized_messages,
    }
