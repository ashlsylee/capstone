"""
OpenStreetMap에서 시뮬레이션에 쓰는 실제 지도 데이터를 받아 data/raw/ 에 저장한다.

- osm_roads.json         : 상권 주변 도로·보행로 (Overpass API, way[highway] + 좌표)
                           → build_road_graph.py 가 보행 네트워크로 만든다
- osm_supermarkets.json  : 상권 주변 슈퍼마켓 (Nominatim 검색 "supermarket")
                           → 시장·상가 건물이 슈퍼마켓으로 잘못 등록된 경우는 exclude=true 로 표시
- osm_landmarks_raw.json : 상가·역·호텔 위치 (장소 좌표 근거, 이미 저장되어 있으면 건너뜀)

실행 (인터넷 필요, 1회):
    python fetch_osm_data.py
Overpass 공용 서버는 자주 500 오류를 내서 몇 번 재시도한다. 이미 data/raw/에 파일이 있으면
다시 받을 필요 없다 (저장소에 포함).

지도 데이터 © OpenStreetMap contributors (ODbL)
"""
import json
import time
import urllib.parse
import urllib.request
from datetime import date

from common import RAW_DIR

BBOX = (37.5620, 127.0010, 37.5730, 127.0210)  # south, west, north, east (상권 경계 + 여백)
OVERPASS = "https://overpass.kumi.systems/api/interpreter"
NOMINATIM = "https://nominatim.openstreetmap.org/search"
UA = {"User-Agent": "capstone-research/1.0 (student project)"}

# 시장·상가 건물 이름이 shop=supermarket 으로 등록된 경우 (실제 슈퍼마켓 아님)
NOT_SUPERMARKET_KEYWORDS = ["시장", "상가"]


def get_json(url: str, params: dict, retries: int = 4):
    full = url + "?" + urllib.parse.urlencode(params)
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(full, headers=UA), timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # 공용 서버 일시 오류 → 재시도
            print(f"  재시도 {i + 1}/{retries}: {e}")
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"다운로드 실패: {url}")


def fetch_roads():
    s, w, n, e = BBOX
    q = f"[out:json][timeout:90];way[highway]({s},{w},{n},{e});out geom;"
    data = get_json(OVERPASS, {"data": q})
    (RAW_DIR / "osm_roads.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"도로 {len(data['elements'])}개 → osm_roads.json")


def fetch_supermarkets():
    s, w, n, e = BBOX
    rows = get_json(NOMINATIM, {"q": "supermarket", "viewbox": f"{w},{n},{e},{s}", "bounded": 1,
                                "format": "json", "limit": 50})
    out = []
    for x in rows:
        if x.get("type") != "supermarket":
            continue
        name = x.get("name") or "(이름 없음)"
        out.append({
            "osm_id": x["osm_id"], "name": name, "lat": float(x["lat"]), "lng": float(x["lon"]),
            "exclude": any(k in name for k in NOT_SUPERMARKET_KEYWORDS),
        })
    payload = {
        "source": f"OpenStreetMap (Nominatim 검색 'supermarket', {date.today()} 수집) © OpenStreetMap contributors, ODbL",
        "note": "exclude=true: 시장·상가 건물이 슈퍼마켓으로 잘못 등록된 경우. OSM에 등록된 곳만 있어 실제 점포 수보다 적다.",
        "stores": out,
    }
    (RAW_DIR / "osm_supermarkets.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"슈퍼마켓 {len(out)}곳 (제외 {sum(o['exclude'] for o in out)}) → osm_supermarkets.json")


if __name__ == "__main__":
    fetch_roads()
    time.sleep(2)
    fetch_supermarkets()
