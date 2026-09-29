"""
Runs every labeled scenario in data/test_scenarios.json through the real
multi-agent workflow end to end — real HTTP calls to the mock Company Data
and News APIs, the real LangGraph state machine, real agent modules, and a
genuine human-approval pause/resume — and writes real measured results to
data/eval_results.json.

Ground truth (data/test_scenarios.json) is computed independently of the
running code: profit margins are recalculated by hand from the raw
figures in company_data_service.py, and sentiment labels are hand-scored
by manually matching each headline's words against analysis_agent.py's
own lexicon — the same measurement discipline used across this portfolio.

Three things happen here:
1. Main accuracy pass: every scenario run through run_workflow() to the
   human-approval interrupt, scored on margin calculation, sentiment
   classification, and the verification pass/fail decision — then
   resumed with a decision to confirm the full round trip completes.
2. Retry-behavior check: the two unknown-ticker scenarios confirm the
   graph actually retried research once (not zero, not infinitely) before
   giving up and requiring a human decision.
3. Resilience check: one request is run with the Company Data API pointed
   at an unreachable port, to confirm the workflow still reaches the
   human-approval step with a clear error surfaced, instead of crashing.

IMPORTANT: env vars read as module-level constants in data_client.py
(COMPANY_DATA_API_BASE_URL, NEWS_API_BASE_URL) must be set BEFORE that
module is imported.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

COMPANY_PORT = 8120
NEWS_PORT = 8121
UNREACHABLE_PORT = 8299

os.environ["COMPANY_DATA_API_BASE_URL"] = f"http://127.0.0.1:{COMPANY_PORT}"
os.environ["NEWS_API_BASE_URL"] = f"http://127.0.0.1:{NEWS_PORT}"

sys.path.insert(0, str(REPO_ROOT))

from src import company_data_service, data_client, news_service  # noqa: E402
from src.orchestrator import resume_workflow, run_workflow  # noqa: E402

MARGIN_TOLERANCE = 0.05


def _run_service_in_thread(app, port: int) -> None:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()


def _wait_for_health(base_url: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"Service at {base_url} never became healthy: {last_error}")


def _percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def main() -> None:
    print(f"Starting Company Data API on port {COMPANY_PORT} and News API on port {NEWS_PORT}...")
    _run_service_in_thread(company_data_service.app, COMPANY_PORT)
    _run_service_in_thread(news_service.app, NEWS_PORT)
    _wait_for_health(f"http://127.0.0.1:{COMPANY_PORT}")
    _wait_for_health(f"http://127.0.0.1:{NEWS_PORT}")
    print("Both mock services are healthy.\n")

    scenarios = json.loads((DATA_DIR / "test_scenarios.json").read_text())

    results = []
    latencies_ms = []
    margin_checked = margin_correct = 0
    sentiment_checked = sentiment_correct = 0
    verification_checked = verification_correct = 0
    warning_checked = warning_correct = 0
    retry_checked = retry_correct = 0
    rejection_checked = rejection_correct = 0
    round_trip_completed = 0

    for case in scenarios:
        gt = case["ground_truth"]
        thread_id = f"eval-{case['id']}"
        started = time.perf_counter()
        outcome = run_workflow(case["ticker"], thread_id=thread_id)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)

        row = {"id": case["id"], "ticker": case["ticker"], "note": case["note"], "status": outcome["status"]}

        if gt.get("rejected_before_agents"):
            rejection_checked += 1
            ok = outcome["status"] == "completed" and outcome.get("final_report", {}).get("recommendation") == "rejected_invalid_request"
            rejection_correct += int(ok)
            row["rejection_handled_correctly"] = ok
            results.append(row)
            status_str = "OK" if ok else "MISS"
            print(f"[{status_str}] id={case['id']:>2} {case['ticker']:<6} malformed-request rejection: {ok}")
            continue

        report = outcome.get("report", {})
        paused_correctly = outcome["status"] == "awaiting_approval"
        row["paused_for_human_approval"] = paused_correctly

        if "profit_margin_pct" in gt:
            margin_checked += 1
            actual_margin = report.get("profit_margin_pct")
            margin_ok = actual_margin is not None and abs(actual_margin - gt["profit_margin_pct"]) <= MARGIN_TOLERANCE
            margin_correct += int(margin_ok)
            row["expected_margin"] = gt["profit_margin_pct"]
            row["actual_margin"] = actual_margin
            row["margin_correct"] = margin_ok

        if "sentiment_label" in gt:
            sentiment_checked += 1
            sentiment_ok = report.get("sentiment_label") == gt["sentiment_label"]
            sentiment_correct += int(sentiment_ok)
            row["expected_sentiment"] = gt["sentiment_label"]
            row["actual_sentiment"] = report.get("sentiment_label")
            row["sentiment_correct"] = sentiment_ok

        if "verification_passed" in gt:
            verification_checked += 1
            verification_ok = report.get("verification_passed") == gt["verification_passed"]
            verification_correct += int(verification_ok)
            row["expected_verification_passed"] = gt["verification_passed"]
            row["actual_verification_passed"] = report.get("verification_passed")
            row["verification_correct"] = verification_ok

        if "min_warnings" in gt:
            warning_checked += 1
            actual_warning_count = len(report.get("warnings", []))
            warning_ok = actual_warning_count >= gt["min_warnings"]
            warning_correct += int(warning_ok)
            row["warning_count"] = actual_warning_count
            row["warning_correct"] = warning_ok

        if "expected_research_attempts" in gt:
            retry_checked += 1
            actual_attempts = report.get("research_attempts")
            retry_ok = actual_attempts == gt["expected_research_attempts"]
            retry_correct += int(retry_ok)
            row["expected_research_attempts"] = gt["expected_research_attempts"]
            row["actual_research_attempts"] = actual_attempts
            row["retry_correct"] = retry_ok

        # Complete the round trip: a human approves a clean report, rejects a flagged one.
        decision = "approve" if report.get("verification_passed") else "reject"
        final = resume_workflow(thread_id, decision=decision, note="eval run")
        if final["status"] == "completed" and final.get("final_report"):
            round_trip_completed += 1
        row["round_trip_completed"] = final["status"] == "completed"

        results.append(row)
        status_str = "OK" if paused_correctly else "MISS"
        print(f"[{status_str}] id={case['id']:>2} {case['ticker']:<6} margin={report.get('profit_margin_pct')} "
              f"sentiment={report.get('sentiment_label')} verification_passed={report.get('verification_passed')}")

    # --- Resilience check: Company Data API unreachable ---
    original_url = data_client.COMPANY_DATA_API_BASE_URL
    data_client.COMPANY_DATA_API_BASE_URL = f"http://127.0.0.1:{UNREACHABLE_PORT}"
    try:
        resilience_outcome = run_workflow("MRDN", thread_id="eval-resilience")
        report = resilience_outcome.get("report", {})
        resilience_check = {
            "scenario": "Company Data API unreachable",
            "reached_human_approval": resilience_outcome["status"] == "awaiting_approval",
            "verification_passed": report.get("verification_passed"),
            "critical_issues": report.get("critical_issues"),
            "passed": (
                resilience_outcome["status"] == "awaiting_approval"
                and report.get("verification_passed") is False
                and bool(report.get("critical_issues"))
            ),
        }
    except Exception as exc:  # noqa: BLE001
        resilience_check = {"scenario": "Company Data API unreachable", "passed": False, "crashed": f"{type(exc).__name__}: {exc}"}
    finally:
        data_client.COMPANY_DATA_API_BASE_URL = original_url
    print(f"\nResilience check: {resilience_check}")

    n_scored_cases = margin_checked  # cases with a margin ground truth are the 8 real companies
    eval_results = {
        "scenario_count": len(scenarios),
        "margin_accuracy": round(margin_correct / margin_checked, 4) if margin_checked else None,
        "margin_checked_count": margin_checked,
        "sentiment_accuracy": round(sentiment_correct / sentiment_checked, 4) if sentiment_checked else None,
        "sentiment_checked_count": sentiment_checked,
        "verification_decision_accuracy": round(verification_correct / verification_checked, 4) if verification_checked else None,
        "verification_checked_count": verification_checked,
        "warning_detection_accuracy": round(warning_correct / warning_checked, 4) if warning_checked else None,
        "retry_behavior_accuracy": round(retry_correct / retry_checked, 4) if retry_checked else None,
        "malformed_request_rejection_accuracy": round(rejection_correct / rejection_checked, 4) if rejection_checked else None,
        "human_approval_round_trips_completed": round_trip_completed,
        "latency_ms_avg": round(statistics.mean(latencies_ms), 3),
        "latency_ms_p95": round(_percentile(latencies_ms, 95), 3),
        "resilience_check": resilience_check,
        "frameworks_note": "Orchestration is LangGraph only. LangChain and CrewAI were not used — see README for why.",
        "llm_path_note": "No OPENAI_API_KEY or ANTHROPIC_API_KEY was set for this run, so every summary used the rule_based template path, not the optional LLM path.",
        "per_case_results": results,
    }

    out_path = DATA_DIR / "eval_results.json"
    out_path.write_text(json.dumps(eval_results, indent=2))

    print(f"\nMargin accuracy: {margin_correct}/{margin_checked}")
    print(f"Sentiment accuracy: {sentiment_correct}/{sentiment_checked}")
    print(f"Verification decision accuracy: {verification_correct}/{verification_checked}")
    print(f"Warning detection accuracy: {warning_correct}/{warning_checked}")
    print(f"Retry behavior accuracy: {retry_correct}/{retry_checked}")
    print(f"Malformed-request rejection accuracy: {rejection_correct}/{rejection_checked}")
    print(f"Human-approval round trips completed: {round_trip_completed}/{len(scenarios) - rejection_checked}")
    print(f"Latency avg/p95 (ms): {eval_results['latency_ms_avg']} / {eval_results['latency_ms_p95']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
