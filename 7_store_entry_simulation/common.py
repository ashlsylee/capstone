"""
7주차 스크립트들이 같이 쓰는 경로·상수·데이터 로딩·거리 계산.
"""
import json
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

PERSONA_PATH = DATA_DIR / "nemotron_personas_korea_demo.parquet"
MARKET_PATH = DATA_DIR / "market_facts.json"
STAGE1_PATH = DATA_DIR / "stage1_district_facts.json"
PLACES_PATH = DATA_DIR / "places.json"
ENTRANCES_PATH = DATA_DIR / "entrances.json"
CANDIDATES_PATH = DATA_DIR / "candidate_sites.json"
ROAD_GRAPH_PATH = DATA_DIR / "road_graph.json"
OSM_STORES_PATH = RAW_DIR / "osm_supermarkets.json"
BOUNDARY_PATH = RAW_DIR / "district_boundary.geojson"

TARGET_INDUSTRY = "슈퍼마켓"   # 서울시 상권분석서비스 업종명
NEW_BRAND = "롯데마트 슈퍼"     # 신규 점포 (구 롯데슈퍼)
BASELINE = "BASE"              # 신규 점포가 없는 현재 상태

AGES = ["10대", "20대", "30대", "40대", "50대", "60대+"]
TIMES = ["00-06", "06-11", "11-14", "14-17", "17-21", "21-24"]
DAYS = ["월", "화", "수", "목", "금", "토", "일"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def haversine_m(lat1, lng1, lat2, lng2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_places() -> list:
    return load_json(PLACES_PATH)["places"]


def load_entrances() -> list:
    return load_json(ENTRANCES_PATH)["entrances"]


def load_candidates() -> list:
    return load_json(CANDIDATES_PATH)["sites"]


def load_boundary() -> list:
    return load_json(BOUNDARY_PATH)["geometry"]["coordinates"][0]


def load_existing_stores() -> list:
    """OSM 슈퍼마켓 (시장·상가 오등록 제외) → [{store_id, name, lat, lng, inside_district}]"""
    from shapely.geometry import Point, Polygon
    poly = Polygon(load_boundary())
    stores = []
    for s in load_json(OSM_STORES_PATH)["stores"]:
        if s["exclude"]:
            continue
        stores.append({
            "store_id": f"S{s['osm_id']}", "name": s["name"], "lat": s["lat"], "lng": s["lng"],
            "inside_district": poly.contains(Point(s["lng"], s["lat"])), "is_new": False,
        })
    return stores


def new_store(site: dict) -> dict:
    return {"store_id": f"NEW_{site['id']}", "name": f"{NEW_BRAND} ({site['name']})",
            "lat": site["lat"], "lng": site["lng"], "inside_district": True, "is_new": True}


def age_to_group(age: int) -> str:
    if age < 20:
        return "10대"
    if age < 30:
        return "20대"
    if age < 40:
        return "30대"
    if age < 50:
        return "40대"
    if age < 60:
        return "50대"
    return "60대+"


def is_open(place: dict, day: str, hour: float) -> bool:
    """places.json의 open_hours: [[시작, 끝], ...] (시 단위, 끝 미포함). closed_sunday_night: 일요일 20시 이후 휴장."""
    if place.get("closed_sunday_night") and day == "일" and hour >= 20:
        return False
    return any(start <= hour < end for start, end in place["open_hours"])


# ---------------------------------------------------------------- 도로 그래프
class RoadGraph:
    """data/road_graph.json → 최근접 노드 찾기, 최단거리, 경로 좌표."""

    def __init__(self):
        import networkx as nx
        g = load_json(ROAD_GRAPH_PATH)
        self.nx = nx
        self.coords = {int(k): tuple(v) for k, v in g["nodes"].items()}  # id → (lat, lng)
        self.G = nx.Graph()
        for u, v, length in g["edges"]:
            self.G.add_edge(int(u), int(v), length=length)
        self._ids = list(self.coords.keys())
        self._lat = [self.coords[i][0] for i in self._ids]
        self._lng = [self.coords[i][1] for i in self._ids]

    def nearest_node(self, lat: float, lng: float) -> int:
        kx = math.cos(math.radians(lat))
        best, bd = None, 1e18
        for i, la, ln in zip(self._ids, self._lat, self._lng):
            d = (la - lat) ** 2 + ((ln - lng) * kx) ** 2
            if d < bd:
                best, bd = i, d
        return best

    def distances_from(self, node: int) -> dict:
        return self.nx.single_source_dijkstra_path_length(self.G, node, weight="length")

    def path_coords(self, a: int, b: int) -> list:
        nodes = self.nx.shortest_path(self.G, a, b, weight="length")
        return [self.coords[n] for n in nodes]
