import os
import requests
from typing import Any, Dict, List, Tuple

BASE_URL = "https://api.hh.ru"

def _headers() -> Dict[str, str]:
    # keep consistent with hh_client.py env var if present
    default_ua = os.getenv("HH_USER_AGENT", "JobRecommendorHH/1.0 (contact: example@example.com)")
    return {
        "HH-User-Agent": default_ua,
        "Accept": "application/json",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    }

def fetch_areas_tree() -> List[Dict[str, Any]]:
    """Fetch full areas tree from HH."""
    r = requests.get(f"{BASE_URL}/areas", headers=_headers(), timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"HH API error {r.status_code}: {r.text}")
    return r.json()

def _find_country(tree: List[Dict[str, Any]], country_name: str = "Россия") -> Dict[str, Any]:
    for node in tree:
        if node.get("name") == country_name:
            return node
    # fallback: first item
    return tree[0] if tree else {"areas": []}

def list_regions_and_cities(tree: List[Dict[str, Any]], country_name: str = "Россия") -> Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
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
        # some regions can have deeper nesting; include second level as well
        if not cities:
            for sub in (region.get("areas", []) or []):
                for city in (sub.get("areas", []) or []):
                    cid = str(city.get("id", ""))
                    cname = str(city.get("name", ""))
                    if cid and cname:
                        cities.append({"id": cid, "name": cname})

        cities_by_region_id[rid] = cities

    regions.sort(key=lambda x: x["name"])
    for rid in list(cities_by_region_id.keys()):
        cities_by_region_id[rid].sort(key=lambda x: x["name"])
    return regions, cities_by_region_id
