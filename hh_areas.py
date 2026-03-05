import os
import json
import time
import requests
from typing import Any, Dict, List, Tuple

BASE_URL = "https://api.hh.ru"

ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
CACHE_PATH = os.path.join(ARTIFACT_DIR, "hh_areas_cache.json")
CACHE_TTL_SECONDS = int(os.getenv("HH_AREAS_CACHE_TTL_SECONDS", str(7 * 24 * 60 * 60)))  # 7 days


def _headers() -> Dict[str, str]:
    # keep consistent with hh_client.py env var if present
    default_ua = os.getenv("HH_USER_AGENT", "JobRecommendorHH/1.0 (contact: rana.shoaib7777@gmail.com.com)")
    return {
        # ✅ FIX: HH expects standard User-Agent header
        "User-Agent": default_ua,
        "Accept": "application/json",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    }


def _read_cache(fresh_only: bool = True) -> Any | None:
    try:
        if not os.path.exists(CACHE_PATH):
            return None
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "tree" not in data:
            return None
        if not fresh_only:
            return data["tree"]
        ts = float(data.get("_cached_at", 0))
        if ts and (time.time() - ts) <= CACHE_TTL_SECONDS:
            return data["tree"]
        return None
    except Exception:
        return None


def _write_cache(tree: Any) -> None:
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"_cached_at": time.time(), "tree": tree}, f, ensure_ascii=False)
    except Exception:
        pass


def fetch_areas_tree() -> List[Dict[str, Any]]:
    """Fetch full areas tree from HH with retries + disk cache fallback."""
    # ✅ 1) Use fresh cache if available (avoids rate limits, faster startup)
    cached = _read_cache(fresh_only=True)
    if isinstance(cached, list):
        return cached

    last_err: str | None = None

    # ✅ 2) Retries with exponential backoff (handles 429/temporary 5xx)
    for attempt in range(4):
        try:
            r = requests.get(f"{BASE_URL}/areas", headers=_headers(), timeout=30)
            if r.status_code < 400:
                tree = r.json()
                if isinstance(tree, list):
                    _write_cache(tree)
                    return tree
                last_err = f"Unexpected JSON type: {type(tree)}"
            else:
                last_err = f"HH API error {r.status_code}: {r.text}"
        except Exception as e:
            last_err = str(e)

        time.sleep(0.8 * (2 ** attempt))

    # ✅ 3) If HH fails, fall back to stale cache so app still boots
    stale = _read_cache(fresh_only=False)
    if isinstance(stale, list):
        return stale

    raise RuntimeError(last_err or "HH API error: failed to fetch /areas and no cache available")


def _find_country(tree: List[Dict[str, Any]], country_name: str = "Россия") -> Dict[str, Any]:
    for node in tree:
        if node.get("name") == country_name:
            return node
    # fallback: first item
    return tree[0] if tree else {"areas": []}


def list_regions_and_cities(
    tree: List[Dict[str, Any]], country_name: str = "Россия"
) -> Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
    """
    Returns:
      regions: list of {id,name}
      cities_by_region_id: {region_id: [{id,name},...]}
    HH structure: country -> regions -> cities (nested).
    """
    country = _find_country(tree, country_name=country_name)
    regions: List[Dict[str, str]] = []
    cities_by_region_id: Dict[str, List[Dict[str, str]]] = {}

    for region in country.get("areas", []) or []:
        rid = str(region.get("id", ""))
        rname = str(region.get("name", ""))
        if not rid or not rname:
            continue
        regions.append({"id": rid, "name": rname})

        cities: List[Dict[str, str]] = []
        for city in (region.get("areas", []) or []):
            cid = str(city.get("id", ""))
            cname = str(city.get("name", ""))
            if cid and cname:
                cities.append({"id": cid, "name": cname})

        # some regions can have deeper nesting; include deeper leaves as well
        if not cities:
            stack = list(region.get("areas", []) or [])
            while stack:
                node = stack.pop()
                children = node.get("areas") or []
                if children:
                    stack.extend(children)
                else:
                    cid = str(node.get("id", ""))
                    cname = str(node.get("name", ""))
                    if cid and cname:
                        cities.append({"id": cid, "name": cname})

        cities_by_region_id[rid] = cities

    regions.sort(key=lambda x: x["name"])
    for rid in list(cities_by_region_id.keys()):
        cities_by_region_id[rid].sort(key=lambda x: x["name"])
    return regions, cities_by_region_id
