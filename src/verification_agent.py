"""
Verification Agent — cross-checks the Analysis Agent's output against the
Research Agent's raw data before anything reaches a human. This is a real
independent check, not a rubber stamp: it recomputes the profit margin
itself from the raw figures and compares it to what the Analysis Agent
reported, checks for data that's internally impossible (net profit can't
exceed revenue), and flags when there isn't enough information to have
real confidence in the result (sparse news coverage, a failed fetch).

Issues are split into two severities:
- `critical` issues block auto-finalizing: missing company data (nothing
  to analyze) or internally impossible financial data (the numbers
  themselves are wrong, not just uncertain) — data integrity errors that
  a re-fetch of the same mock data cannot fix, and that a human should
  see before the report is finalized.
- `warning` issues (e.g. thin news coverage) don't block the workflow but
  are surfaced in the final report so a human isn't left thinking the
  analysis was more complete than it was.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .analysis_agent import AnalysisResult
from .research_agent import ResearchFindings

MARGIN_TOLERANCE = 0.05
MIN_HEADLINES_FOR_CONFIDENCE = 2


@dataclass
class VerificationResult:
    ticker: str
    passed: bool  # no critical issues
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommend_retry_research: bool = False


def verify(findings: ResearchFindings, analysis: AnalysisResult) -> VerificationResult:
    critical: List[str] = []
    warnings: List[str] = []
    recommend_retry = False

    if not findings.ok or not findings.company:
        critical.append("Company financial data could not be retrieved.")
        recommend_retry = True
        return VerificationResult(ticker=findings.ticker, passed=False, critical_issues=critical,
                                   warnings=warnings, recommend_retry_research=recommend_retry)

    company = findings.company
    revenue = company.get("revenue")
    net_profit = company.get("net_profit")

    # Recompute the margin independently from the raw figures and compare
    # to what the Analysis Agent reported — a real consistency check, not
    # a re-statement of the same code path.
    if revenue:
        recomputed_margin = round((net_profit / revenue) * 100, 2)
        if analysis.profit_margin_pct is None or abs(recomputed_margin - analysis.profit_margin_pct) > MARGIN_TOLERANCE:
            critical.append(
                f"Profit margin mismatch: recomputed {recomputed_margin}% vs. Analysis Agent's "
                f"reported {analysis.profit_margin_pct}%."
            )

    # Data-integrity check: net profit cannot exceed revenue.
    if revenue is not None and net_profit is not None and net_profit > revenue:
        critical.append(
            f"Data integrity error: reported net profit (${net_profit:,}) exceeds reported "
            f"revenue (${revenue:,}), which is not possible. Source data is suspect."
        )

    # Data-completeness check: not enough news coverage to be confident in sentiment.
    if len(findings.headlines) < MIN_HEADLINES_FOR_CONFIDENCE:
        warnings.append(
            f"Only {len(findings.headlines)} headline(s) found — sentiment analysis has low confidence "
            f"with this little coverage."
        )

    if findings.errors:
        warnings.extend(findings.errors)

    passed = len(critical) == 0
    return VerificationResult(
        ticker=findings.ticker, passed=passed, critical_issues=critical, warnings=warnings,
        recommend_retry_research=recommend_retry,
    )
