import requests

HH_AREAS_URL = "https://api.hh.ru/areas"


def fetch_areas_tree():
    r = requests.get(HH_AREAS_URL, timeout=30)
    r.raise_for_status()
    return r.json()


def list_regions_and_cities(areas_tree):
    """
    Returns:
        regions: List[str]
        cities: Dict[str, List[str]]
    """

    regions = []
    cities = {}

    for country in areas_tree:
        for region in country.get("areas", []):
            region_name = region.get("name")
            if not region_name:
                continue

            regions.append(region_name)

            city_names = []
            for city in region.get("areas", []):
                name = city.get("name")
                if name:
                    city_names.append(name)

            cities[region_name] = city_names

    regions = sorted(regions)
    return regions, cities
