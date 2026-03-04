import os
import time
import random
import requests
from typing import Any, Dict, Optional, List

BASE_URL = "https://api.hh.ru"

# IMPORTANT:
# - HH blocks some User-Agent strings. Your previous DEFAULT_UA is blacklisted.
# - Provide a unique UA via env var HH_USER_AGENT in deployment if possible.
DEFAULT_UA = "HH-Job-Recommender/2.0 (support: rana.shoaib7777@gmail.com; purpose: job matching)"
HH_USER_AGENT = os.getenv("HH_USER_AGENT", DEFAULT_UA).strip()

DEFAULT_HEADERS = {
    "Accept": "application/json",
    # HH expects this header name (case-insensitive, but keep canonical)
    "HH-User-Agent": HH_USER_AGENT,
    # Also provide standard UA header
    "User-Agent": HH_USER_AGENT,
}


def _sleep_backoff(attempt: int) -> None:
    # exponential backoff + jitter, capped
    base = min(8.0, 0.6 * (2 ** attempt))
    time.sleep(base + random.uniform(0.0, 0.25))


def _get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict[str, Any]:
    last_err: Optional[Exception] = None

    for attempt in range(5):
        try:
            r = requests.get(url, params=params or {}, headers=DEFAULT_HEADERS, timeout=timeout)

            # Retry on rate limit / transient errors
            if r.status_code in (429, 500, 502, 503, 504):
                _sleep_backoff(attempt)
                continue

            if r.status_code >= 400:
                # Include HH request_id if provided
                req_id = r.headers.get("X-Request-Id") or r.headers.get("x-request-id")
                rid_txt = f" request_id={req_id}" if req_id else ""
                raise RuntimeError(f"HH API error {r.status_code}:{rid_txt} {r.text}")

            return r.json()

        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = e
            _sleep_backoff(attempt)

    # If we exhausted retries
    if last_err:
        raise RuntimeError(f"HH API request failed after retries: {last_err}") from last_err
    raise RuntimeError("HH API request failed after retries")


def search_vacancies(
    text: Optional[str] = None,
    area: Optional[int] = None,
    page: int = 0,
    per_page: int = 50,
    period_days: Optional[int] = None,
    order_by: Optional[str] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"page": int(page), "per_page": int(per_page)}
    if text:
        params["text"] = text
    if area is not None:
        params["area"] = int(area)
    if period_days is not None:
        params["period"] = int(period_days)
    if order_by:
        params["order_by"] = order_by
    return _get(f"{BASE_URL}/vacancies", params=params)


def fetch_vacancies(
    text: Optional[str] = None,
    area: Optional[int] = None,
    max_items: int = 300,
    per_page: int = 50,
    period_days: Optional[int] = None,
    order_by: Optional[str] = None,
    sleep_s: float = 0.15,
) -> List[dict]:
    out: List[dict] = []
    page = 0
    while len(out) < int(max_items):
        payload = search_vacancies(
            text=text,
            area=area,
            page=page,
            per_page=per_page,
            period_days=period_days,
            order_by=order_by,
        )
        items = payload.get("items") or []
        if not items:
            break

        out.extend(items)

        pages = payload.get("pages")
        if pages is not None and page + 1 >= int(pages):
            break

        if len(out) >= int(max_items):
            break

        page += 1
        if sleep_s:
            time.sleep(float(sleep_s))

    return out[: int(max_items)]


def vacancy_details(vacancy_id: str) -> Dict[str, Any]:
    return _get(f"{BASE_URL}/vacancies/{vacancy_id}")
