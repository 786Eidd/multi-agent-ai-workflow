"""
REST API integration — the client side of `company_data_service.py` and
`news_service.py`. Real HTTP calls via `httpx`, uniform result shape
matching the pattern used across this portfolio's other client modules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

COMPANY_DATA_API_BASE_URL = os.environ.get("COMPANY_DATA_API_BASE_URL", "http://127.0.0.1:8020")
NEWS_API_BASE_URL = os.environ.get("NEWS_API_BASE_URL", "http://127.0.0.1:8021")
REQUEST_TIMEOUT_SECONDS = 5.0


@dataclass
class ApiResult:
    ok: bool
    data: Optional[Any] = None
    error: Optional[str] = None


def _get(base_url: str, path: str, params: Optional[dict] = None) -> ApiResult:
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            resp = client.get(f"{base_url}{path}", params=params or {})
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            return ApiResult(ok=False, error=str(detail))
        return ApiResult(ok=True, data=resp.json())
    except httpx.ConnectError:
        return ApiResult(ok=False, error=f"Could not reach {base_url}.")
    except httpx.TimeoutException:
        return ApiResult(ok=False, error=f"Request to {base_url} timed out.")
    except Exception as exc:  # noqa: BLE001
        return ApiResult(ok=False, error=f"Unexpected error calling {base_url}: {exc}")


def get_company_financials(ticker: str) -> ApiResult:
    return _get(COMPANY_DATA_API_BASE_URL, f"/companies/{ticker}")


def get_news(ticker: str, limit: int = 10) -> ApiResult:
    return _get(NEWS_API_BASE_URL, f"/news/{ticker}", params={"limit": limit})
