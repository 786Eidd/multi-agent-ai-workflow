"""
Mock News API — a real, separately-runnable FastAPI service standing in
for a news-headlines provider (NewsAPI/Benzinga-shaped: recent headlines
for a ticker, with a date, no pre-computed sentiment). Sentiment is
deliberately NOT included in this API's response — `analysis_agent.py`
computes it itself from the headline text, so "AI-powered analysis" means
something was actually analyzed here, not just relayed from a field this
mock happened to hardcode.

Every headline is fabricated for this project's fictional companies (see
`company_data_service.py`). One ticker (ANPT) deliberately has very sparse
coverage (a single headline) so there's a genuine "not enough information"
case for the verification agent to flag, not just a happy-path demo.

Run standalone:
    uvicorn src.news_service:app --port 8021
"""
from __future__ import annotations

from typing import Dict, List

from fastapi import FastAPI

app = FastAPI(title="Mock News API", version="1.0")

_NEWS: Dict[str, List[dict]] = {
    "BRLX": [
        {"headline": "Brightline Robotics wins record factory-automation contract with major auto manufacturer", "days_ago": 3},
        {"headline": "Brightline Robotics beats quarterly delivery targets, expands Ohio plant", "days_ago": 9},
        {"headline": "Analysts praise Brightline Robotics' strong order backlog heading into next fiscal year", "days_ago": 21},
    ],
    "CDRG": [
        {"headline": "Cedar Ridge Health Systems opens two new regional clinics", "days_ago": 5},
        {"headline": "Cedar Ridge Health Systems reports steady patient volume growth", "days_ago": 14},
        {"headline": "Cedar Ridge faces staffing shortage concerns amid regional nursing gap", "days_ago": 27},
    ],
    "MRDN": [
        {"headline": "Meridian Cloud Technologies posts record profit as enterprise demand surges", "days_ago": 2},
        {"headline": "Meridian Cloud announces strong renewal rates and expanding margins", "days_ago": 8},
        {"headline": "Meridian Cloud stock upgraded after strong earnings beat", "days_ago": 16},
    ],
    "GRVP": [
        {"headline": "Grovepoint Energy Partners posts steep quarterly loss amid falling energy prices", "days_ago": 4},
        {"headline": "Grovepoint Energy delays two solar projects, cites financing setbacks", "days_ago": 11},
        {"headline": "Grovepoint Energy warns of continued weak results into next quarter", "days_ago": 19},
    ],
    "SBFH": [
        {"headline": "Sable & Finch Holdings holiday sales roughly in line with expectations", "days_ago": 6},
        {"headline": "Sable & Finch opens 40 new stores, expands into two new states", "days_ago": 13},
        {"headline": "Sable & Finch faces rising freight costs but maintains full-year guidance", "days_ago": 22},
    ],
    "ANPT": [
        {"headline": "Anchor Point Security wins mid-size government cybersecurity contract", "days_ago": 30},
    ],
    "TDLN": [
        {"headline": "Thistledown Logistics warns of margin pressure from rising fuel costs", "days_ago": 7},
        {"headline": "Thistledown Logistics expands regional trucking fleet despite soft freight demand", "days_ago": 15},
        {"headline": "Thistledown Logistics reports flat volumes, in line with industry trends", "days_ago": 25},
    ],
    "WLTR": [
        {"headline": "Waltham River Materials announces new supply contract with regional builders", "days_ago": 5},
        {"headline": "Waltham River Materials faces scrutiny over reported earnings inconsistencies", "days_ago": 10},
        {"headline": "Waltham River Materials stock volatile after analysts question reported margins", "days_ago": 12},
    ],
}


@app.get("/news/{ticker}")
def get_news(ticker: str, limit: int = 10):
    headlines = _NEWS.get(ticker.upper(), [])
    return {"ticker": ticker.upper(), "headlines": headlines[:limit]}


@app.get("/health")
def health():
    return {"status": "ok", "tickers_covered": len(_NEWS)}
