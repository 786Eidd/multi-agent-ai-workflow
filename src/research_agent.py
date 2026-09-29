"""
Research Agent — retrieves raw information from external systems. This is
the only agent that talks to the outside world (the mock Company Data API
and News API, over real HTTP via `data_client.py`); every other agent
works only from what this one collected, which is what makes the
Verification Agent's cross-checks meaningful — it's checking real
retrieved data, not data another agent invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import data_client


@dataclass
class ResearchFindings:
    ticker: str
    ok: bool
    company: Optional[dict] = None
    headlines: List[dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    fetch_attempts: int = 1


def research_company(ticker: str, fetch_attempts: int = 1) -> ResearchFindings:
    errors: List[str] = []

    company_result = data_client.get_company_financials(ticker)
    if not company_result.ok:
        errors.append(f"Company data fetch failed: {company_result.error}")

    news_result = data_client.get_news(ticker)
    headlines: List[dict] = []
    if not news_result.ok:
        errors.append(f"News fetch failed: {news_result.error}")
    else:
        headlines = news_result.data.get("headlines", [])

    ok = company_result.ok  # news coverage may legitimately be sparse or empty; company data is required
    return ResearchFindings(
        ticker=ticker.upper(), ok=ok, company=company_result.data if company_result.ok else None,
        headlines=headlines, errors=errors, fetch_attempts=fetch_attempts,
    )
