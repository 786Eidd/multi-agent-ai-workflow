"""
Analysis Agent — turns the Research Agent's raw findings into derived
metrics and a narrative summary. Two real, independently-testable pieces
of analysis happen here, not just a pass-through of retrieved data:

1. Financial ratios computed directly from the raw figures (profit
   margin, revenue per employee, company age) — plain arithmetic, but
   genuinely computed here, not relayed from the API.
2. Headline sentiment scored with a small keyword lexicon over the
   Research Agent's raw headline text — real (if simple) NLP, not a
   sentiment field the mock News API happened to hardcode (it doesn't;
   see `news_service.py`).

The narrative summary is rule-based (a filled template) by default, with
a genuine optional LLM path behind an API key — the same pattern used
everywhere else in this portfolio, for the same reason: this sandbox has
no outbound route to an LLM API, so the default path has to be real,
tested code, not a placeholder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from .research_agent import ResearchFindings

_POSITIVE_WORDS = {
    "record", "beats", "beat", "growth", "strong", "surges", "surge", "upgraded", "expands",
    "expand", "wins", "win", "praise", "praises", "profit", "steady", "opens", "renewal",
}
_NEGATIVE_WORDS = {
    "loss", "delays", "delay", "warns", "warn", "shortage", "pressure", "scrutiny",
    "inconsistencies", "volatile", "question", "questions", "weak", "concern", "concerns",
    "soft", "falling",
}


@dataclass
class AnalysisResult:
    ticker: str
    profit_margin_pct: Optional[float]
    revenue_per_employee: Optional[float]
    company_age_years: Optional[int]
    sentiment_label: str  # "positive" | "neutral" | "negative"
    sentiment_score: float  # (positive_hits - negative_hits) / total_hits, -1..1
    headline_count: int
    positive_hits: int
    negative_hits: int
    matched_words: List[str] = field(default_factory=list)
    summary: str = ""
    path_used: str = "rule_based"


def _score_sentiment(headlines: List[dict]) -> tuple:
    positive_hits = 0
    negative_hits = 0
    matched: List[str] = []
    for item in headlines:
        words = item.get("headline", "").lower().replace(",", " ").replace(".", " ").split()
        for w in words:
            w_clean = w.strip("'\"")
            if w_clean in _POSITIVE_WORDS:
                positive_hits += 1
                matched.append(w_clean)
            elif w_clean in _NEGATIVE_WORDS:
                negative_hits += 1
                matched.append(w_clean)
    total = positive_hits + negative_hits
    score = (positive_hits - negative_hits) / total if total else 0.0
    if total == 0:
        label = "neutral"
    elif score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    else:
        label = "neutral"
    return label, round(score, 3), positive_hits, negative_hits, matched


def _generate_summary_template(ticker: str, company: dict, margin: Optional[float],
                                rev_per_emp: Optional[float], sentiment_label: str, headline_count: int) -> str:
    name = company.get("name", ticker)
    sector = company.get("sector", "an unspecified sector")
    margin_txt = f"a {margin:.1f}% net profit margin" if margin is not None else "an undetermined profit margin"
    parts = [
        f"{name} ({ticker}) operates in {sector} and reported {margin_txt} "
        f"for fiscal year {company.get('fiscal_year', 'N/A')}."
    ]
    if rev_per_emp is not None:
        parts.append(f"Revenue per employee is approximately ${rev_per_emp:,.0f}.")
    if headline_count == 0:
        parts.append("No recent news coverage was found for this ticker.")
    else:
        parts.append(f"Recent news sentiment across {headline_count} headline(s) reads as {sentiment_label}.")
    return " ".join(parts)


def _generate_summary_with_llm(ticker: str, company: dict, findings_summary: str) -> Optional[str]:
    prompt = (
        f"Write a concise, neutral 2-3 sentence business research summary for {ticker} "
        f"based on this data: {findings_summary}. Do not speculate beyond the given data."
    )
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic()
            resp = client.messages.create(
                model="claude-3-5-haiku-20241022", max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception:
            return None
    elif os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini", max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception:
            return None
    return None


def analyze(findings: ResearchFindings) -> AnalysisResult:
    company = findings.company or {}
    revenue = company.get("revenue")
    net_profit = company.get("net_profit")
    employees = company.get("employees")
    fiscal_year = company.get("fiscal_year")
    founded_year = company.get("founded_year")

    margin = round((net_profit / revenue) * 100, 2) if revenue else None
    rev_per_emp = round(revenue / employees, 2) if revenue and employees else None
    age = (fiscal_year - founded_year) if fiscal_year and founded_year else None

    label, score, pos_hits, neg_hits, matched = _score_sentiment(findings.headlines)

    path_used = "rule_based"
    summary = _generate_summary_template(findings.ticker, company, margin, rev_per_emp, label, len(findings.headlines))
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        llm_summary = _generate_summary_with_llm(findings.ticker, company, summary)
        if llm_summary:
            summary = llm_summary
            path_used = "llm"

    return AnalysisResult(
        ticker=findings.ticker, profit_margin_pct=margin, revenue_per_employee=rev_per_emp,
        company_age_years=age, sentiment_label=label, sentiment_score=score,
        headline_count=len(findings.headlines), positive_hits=pos_hits, negative_hits=neg_hits,
        matched_words=matched, summary=summary, path_used=path_used,
    )
