import os
import time
import requests
from typing import Any, Dict, Optional, List

BASE_URL = "https://api.hh.ru"

DEFAULT_UA = "Job-Recommendor/1.0 (contact: rana.shoaib7777@gmail.com)"
HH_USER_AGENT = os.getenv("HH_USER_AGENT", DEFAULT_UA)

DEFAULT_HEADERS = {
    "Accept": "application/json",
    "HH-User-Agent": HH_USER_AGENT,
    "User-Agent": HH_USER_AGENT,
}

def _get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict[str, Any]:
    r = requests.get(url, params=params or {}, headers=DEFAULT_HEADERS, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HH API error {r.status_code}: {r.text}")
    return r.json()

def search_vacancies(
    text: Optional[str] = None,
    area: Optional[int] = None,
    page: int = 0,
    per_page: int = 50,
    period_days: Optional[int] = None,
    order_by: Optional[str] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"page": page, "per_page": per_page}
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
    sleep_s: float = 0.1,
) -> List[dict]:
    out: List[dict] = []
    page = 0
    while len(out) < max_items:
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
        if len(out) >= max_items:
            break
        pages = payload.get("pages")
        if pages is not None and page + 1 >= int(pages):
            break
        page += 1
        if sleep_s:
            time.sleep(sleep_s)
    return out[:max_items]

def vacancy_details(vacancy_id: str) -> Dict[str, Any]:
    return _get(f"{BASE_URL}/vacancies/{vacancy_id}")
