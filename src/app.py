"""
FastAPI service exposing the multi-agent workflow.

Run (three terminals):
    uvicorn src.company_data_service:app --port 8020
    uvicorn src.news_service:app --port 8021
    uvicorn src.app:app --reload --port 8019

Then:
    curl -X POST http://localhost:8019/workflow/start -H "Content-Type: application/json" \
         -d '{"ticker": "MRDN", "thread_id": "demo-1"}'

    # Inspect the paused report in the response, then approve or reject it:
    curl -X POST http://localhost:8019/workflow/demo-1/decide -H "Content-Type: application/json" \
         -d '{"decision": "approve", "note": "looks good"}'
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .orchestrator import resume_workflow, run_workflow

app = FastAPI(
    title="Multi-Agent AI Workflow",
    description=(
        "Specialized AI agents (Research, Analysis, Verification) coordinate "
        "over a LangGraph state machine to research a company, compute and "
        "sanity-check findings, and produce a report — paused at a real "
        "human-approval checkpoint before anything is finalized."
    ),
)


class StartRequest(BaseModel):
    ticker: str
    thread_id: str


class DecisionRequest(BaseModel):
    decision: str  # "approve" | "reject"
    note: Optional[str] = None


@app.post("/workflow/start")
def workflow_start(req: StartRequest) -> Dict[str, Any]:
    return run_workflow(req.ticker, req.thread_id)


@app.post("/workflow/{thread_id}/decide")
def workflow_decide(thread_id: str, req: DecisionRequest) -> Dict[str, Any]:
    return resume_workflow(thread_id, req.decision, req.note)


@app.get("/health")
def health():
    return {"status": "ok"}
