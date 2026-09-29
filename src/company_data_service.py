"""
Mock Company Data API — a real, separately-runnable FastAPI service
standing in for a financial-data provider (Bloomberg/Yahoo Finance/
Finnhub-shaped: company profile + financial statement fields). A
different domain from every other mock service in this portfolio (order
management, CRM, accounts payable) — this one models a market-data
provider.

Why a mock, not a live data API: this sandbox has no outbound route to
arbitrary third-party hosts, so there's no way to call a real financial
data provider here. `src/data_client.py` only depends on this service's
HTTP contract (GET /companies/{ticker}), so pointing at a real provider's
REST API later is a base-URL-and-auth-header change.

Every company here is fictional (clearly named as such) — no real public
company's financials are used or implied. One record (WLTR) is
deliberately internally inconsistent (net_profit > revenue, which is
impossible) specifically so `verification_agent.py` has a real data
-integrity problem to genuinely catch, independent of any agent's own
arithmetic.

Run standalone:
    uvicorn src.company_data_service:app --port 8020
"""
from __future__ import annotations

from typing import Dict, Optional

from fastapi import FastAPI, HTTPException

app = FastAPI(title="Mock Company Data API", version="1.0")

# All companies and figures below are fictional, invented for this
# portfolio project. Any resemblance to a real ticker or company is
# coincidental and not intended.
_COMPANIES: Dict[str, dict] = {
    "BRLX": {
        "ticker": "BRLX", "name": "Brightline Robotics Inc.", "sector": "Industrial Automation",
        "fiscal_year": 2025, "revenue": 842_000_000, "net_profit": 96_000_000,
        "employees": 3100, "founded_year": 2011,
    },
    "CDRG": {
        "ticker": "CDRG", "name": "Cedar Ridge Health Systems", "sector": "Healthcare",
        "fiscal_year": 2025, "revenue": 2_150_000_000, "net_profit": 118_000_000,
        "employees": 14200, "founded_year": 1998,
    },
    "MRDN": {
        "ticker": "MRDN", "name": "Meridian Cloud Technologies", "sector": "Software",
        "fiscal_year": 2025, "revenue": 1_320_000_000, "net_profit": 214_000_000,
        "employees": 5400, "founded_year": 2009,
    },
    "GRVP": {
        "ticker": "GRVP", "name": "Grovepoint Energy Partners", "sector": "Renewable Energy",
        "fiscal_year": 2025, "revenue": 615_000_000, "net_profit": -22_000_000,
        "employees": 2200, "founded_year": 2015,
    },
    "SBFH": {
        "ticker": "SBFH", "name": "Sable & Finch Holdings", "sector": "Consumer Retail",
        "fiscal_year": 2025, "revenue": 3_480_000_000, "net_profit": 187_000_000,
        "employees": 41000, "founded_year": 1985,
    },
    "ANPT": {
        "ticker": "ANPT", "name": "Anchor Point Security Corp.", "sector": "Cybersecurity",
        "fiscal_year": 2025, "revenue": 410_000_000, "net_profit": 38_000_000,
        "employees": 1650, "founded_year": 2014,
    },
    "TDLN": {
        "ticker": "TDLN", "name": "Thistledown Logistics Group", "sector": "Transportation & Logistics",
        "fiscal_year": 2025, "revenue": 980_000_000, "net_profit": 41_000_000,
        "employees": 8700, "founded_year": 2002,
    },
    "WLTR": {
        # Deliberately inconsistent record: net_profit exceeds revenue, which
        # is impossible — a genuine data-integrity problem for the
        # verification agent to catch, not an OCR/extraction artifact.
        "ticker": "WLTR", "name": "Waltham River Materials Ltd.", "sector": "Industrial Materials",
        "fiscal_year": 2025, "revenue": 275_000_000, "net_profit": 310_000_000,
        "employees": 1900, "founded_year": 2006,
    },
}


@app.get("/companies/{ticker}")
def get_company(ticker: str):
    record = _COMPANIES.get(ticker.upper())
    if record is None:
        raise HTTPException(status_code=404, detail=f"No company found with ticker '{ticker.upper()}'")
    return record


@app.get("/health")
def health():
    return {"status": "ok", "companies_available": len(_COMPANIES)}
