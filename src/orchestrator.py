"""
The multi-agent workflow itself, built on LangGraph's `StateGraph`:

    User / Business Request
        |
        v
    Orchestrator Agent
        |
        v
    Research Agent -> Analysis Agent -> Verification Agent
        |                                     |
        |            (fails, retryable)       |
        +-------------------------------------+
        |
        v
    Result Aggregation
        |
        v
    Human Approval (interrupt — a real pause, not a simulated one)
        |
        v
    Final Response / Report

## Why LangGraph (and only LangGraph)

This is the one project in this portfolio where the orchestration is
genuinely dynamic: the Verification Agent's result decides whether the
graph loops back to Research (a transient fetch failure is retryable) or
moves on to aggregation and human approval — that's exactly the
conditional, stateful routing LangGraph is built for. Contrast this with
`ai-integration-hub`, which deliberately does NOT use LangGraph because
its routing is fixed business logic decided at design time, not by an
agent's runtime output.

LangChain and CrewAI are intentionally not used. This project doesn't
need LangChain's own abstractions (no document loaders, chains, or
LangChain message types are involved), so importing it would add a
dependency without adding a capability. CrewAI was considered — it's
the other well-known multi-agent framework — but CrewAI's own agent
abstraction requires a working LLM call to do anything, and this sandbox
has no outbound route to an LLM API; wiring in a framework that can't
actually be exercised or tested here, and then claiming it as a working
part of the deliverable, is exactly the kind of unverified claim this
portfolio avoids. Every agent here is a real, independently-tested Python
module instead (see `research_agent.py`, `analysis_agent.py`,
`verification_agent.py`), each with a genuine optional LLM call behind an
API key where an LLM is actually useful (narrative summarization).

## Human-in-the-loop

`human_approval_node` calls LangGraph's `interrupt()`, which genuinely
pauses graph execution and returns control to the caller — this isn't a
placeholder step that always says "approved". `run_workflow` returns the
paused state; a separate `resume_workflow` call (a human's real decision)
is required to continue the graph to completion. The checkpointer
(`MemorySaver`) is what makes this pause-and-resume possible across two
separate calls into the graph.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from .analysis_agent import analyze
from .research_agent import research_company
from .verification_agent import verify

MAX_RESEARCH_ATTEMPTS = 2


class WorkflowState(TypedDict, total=False):
    ticker: str
    valid_request: bool
    research_attempts: int
    findings: Dict[str, Any]
    analysis: Dict[str, Any]
    verification: Dict[str, Any]
    final_report: Dict[str, Any]
    human_decision: Optional[str]
    human_note: Optional[str]


def orchestrator_node(state: WorkflowState) -> WorkflowState:
    """Validates the incoming request and initializes workflow state
    before delegating to the specialized agents. A ticker must be a
    short all-letters symbol — malformed requests are rejected here
    rather than being sent through three agents that can't do anything
    useful with them."""
    ticker = state.get("ticker", "").strip().upper()
    valid = bool(ticker) and ticker.isalpha() and 1 <= len(ticker) <= 6
    return {"ticker": ticker, "valid_request": valid, "research_attempts": 0}


def research_node(state: WorkflowState) -> WorkflowState:
    attempts = state.get("research_attempts", 0) + 1
    findings = research_company(state["ticker"], fetch_attempts=attempts)
    return {"findings": asdict(findings), "research_attempts": attempts}


def analysis_node(state: WorkflowState) -> WorkflowState:
    from .research_agent import ResearchFindings

    findings = ResearchFindings(**state["findings"])
    result = analyze(findings)
    return {"analysis": asdict(result)}


def verification_node(state: WorkflowState) -> WorkflowState:
    from .analysis_agent import AnalysisResult
    from .research_agent import ResearchFindings

    findings = ResearchFindings(**state["findings"])
    analysis = AnalysisResult(**state["analysis"])
    result = verify(findings, analysis)
    return {"verification": asdict(result)}


def decide_after_verification(state: WorkflowState) -> str:
    verification = state["verification"]
    if verification["recommend_retry_research"] and state["research_attempts"] < MAX_RESEARCH_ATTEMPTS:
        return "retry_research"
    return "aggregate"


def aggregate_node(state: WorkflowState) -> WorkflowState:
    verification = state["verification"]
    analysis = state["analysis"]
    findings = state["findings"]

    recommendation = "auto_finalize" if verification["passed"] else "needs_human_decision"
    report = {
        "ticker": state["ticker"],
        "company_name": (findings.get("company") or {}).get("name"),
        "profit_margin_pct": analysis["profit_margin_pct"],
        "revenue_per_employee": analysis["revenue_per_employee"],
        "sentiment_label": analysis["sentiment_label"],
        "sentiment_score": analysis["sentiment_score"],
        "summary": analysis["summary"],
        "verification_passed": verification["passed"],
        "critical_issues": verification["critical_issues"],
        "warnings": verification["warnings"],
        "research_attempts": state["research_attempts"],
        "recommendation": recommendation,
    }
    return {"final_report": report}


def human_approval_node(state: WorkflowState) -> WorkflowState:
    decision = interrupt({
        "message": "Human approval requested before finalizing this report.",
        "report": state["final_report"],
    })
    # `decision` is whatever value resume_workflow() passes to Command(resume=...)
    return {"human_decision": decision.get("decision"), "human_note": decision.get("note")}


def rejected_request_node(state: WorkflowState) -> WorkflowState:
    return {
        "final_report": {
            "ticker": state.get("ticker", ""),
            "recommendation": "rejected_invalid_request",
            "critical_issues": [f"'{state.get('ticker', '')}' is not a valid ticker symbol."],
        },
        "human_decision": "not_applicable",
    }


def route_after_orchestrator(state: WorkflowState) -> str:
    return "research" if state["valid_request"] else "reject"


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("research", research_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("verification", verification_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("reject", rejected_request_node)

    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges("orchestrator", route_after_orchestrator, {"research": "research", "reject": "reject"})
    graph.add_edge("research", "analysis")
    graph.add_edge("analysis", "verification")
    graph.add_conditional_edges(
        "verification", decide_after_verification, {"retry_research": "research", "aggregate": "aggregate"}
    )
    graph.add_edge("aggregate", "human_approval")
    graph.add_edge("human_approval", END)
    graph.add_edge("reject", END)

    return graph.compile(checkpointer=MemorySaver())


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_workflow(ticker: str, thread_id: str) -> Dict[str, Any]:
    """Starts (or restarts) a workflow run for `ticker`, runs it up to
    the human-approval interrupt (or straight to END for a rejected
    request), and returns what happened. Real pause: nothing is
    'approved' here."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"ticker": ticker}, config=config)

    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        return {"status": "awaiting_approval", "thread_id": thread_id, **interrupt_payload}
    return {"status": "completed", "thread_id": thread_id, "final_report": result.get("final_report")}


def resume_workflow(thread_id: str, decision: str, note: Optional[str] = None) -> Dict[str, Any]:
    """Resumes a paused workflow with a real human decision ('approve' or
    'reject'). This is the only way a paused workflow reaches a final
    report — nothing auto-approves itself."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume={"decision": decision, "note": note}), config=config)
    return {
        "status": "completed", "thread_id": thread_id, "human_decision": result.get("human_decision"),
        "human_note": result.get("human_note"), "final_report": result.get("final_report"),
    }
