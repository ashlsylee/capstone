"""
동대문패션타운 관광특구 배경 지도 이미지를 실제 지도 타일로 만든다.

OpenStreetMap 표준 타일을 받아 상권 경계 주변 영역만 이어 붙여
assets/background.png 한 장으로 저장하고, 위경도 ↔ 픽셀 변환에 필요한 범위 정보를
assets/background_meta.json 에 함께 저장한다. 시각화(district_map.html)는 이 이미지를
배경으로 깔고 meta 정보로 점포·후보지·에이전트 위치를 찍는다.

지도 출처: © OpenStreetMap contributors (시각화 화면에 표기)

실행 (인터넷 필요, 타일 140여 장 — OSM 타일 정책상 1회만 받고 결과 이미지를 재사용):
    python build_background.py
"""
import io
import json
import math
import time
import urllib.request
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
BOUNDARY_PATH = BASE_DIR / "data" / "raw" / "district_boundary.geojson"
OUT_IMG = BASE_DIR / "assets" / "background.png"
OUT_META = BASE_DIR / "assets" / "background_meta.json"

ZOOM = 18
MARGIN_DEG = 0.0012  # 상권 경계 바깥으로 약 100m 여백
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_PX = 256


def lnglat_to_tile(lng: float, lat: float, z: int) -> tuple:
    """위경도 → (소수점 포함) 타일 좌표 (Web Mercator)"""
    n = 2 ** z
    x = (lng + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(z: int, x: int, y: int) -> Image.Image:
    url = TILE_URL.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": "capstone-research/1.0 (student project, one-time download)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def main():
    boundary = json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))
    coords = boundary["geometry"]["coordinates"][0]
    lngs = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    west, east = min(lngs) - MARGIN_DEG, max(lngs) + MARGIN_DEG
    south, north = min(lats) - MARGIN_DEG, max(lats) + MARGIN_DEG

    fx0, fy0 = lnglat_to_tile(west, north, ZOOM)
    fx1, fy1 = lnglat_to_tile(east, south, ZOOM)
    tx0, ty0, tx1, ty1 = int(fx0), int(fy0), int(fx1), int(fy1)

    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * TILE_PX, (ty1 - ty0 + 1) * TILE_PX))
    total = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    done = 0
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            canvas.paste(fetch_tile(ZOOM, tx, ty), ((tx - tx0) * TILE_PX, (ty - ty0) * TILE_PX))
            done += 1
            print(f"\r타일 {done}/{total}", end="")
            time.sleep(0.2)
    print()

    # 여백 기준 범위로 잘라내기
    left = round((fx0 - tx0) * TILE_PX)
    top = round((fy0 - ty0) * TILE_PX)
    right = round((fx1 - tx0) * TILE_PX)
    bottom = round((fy1 - ty0) * TILE_PX)
    img = canvas.crop((left, top, right, bottom))

    OUT_IMG.parent.mkdir(exist_ok=True)
    img.save(OUT_IMG, optimize=True)
    meta = {
        "zoom": ZOOM,
        "width": img.width,
        "height": img.height,
        # Web Mercator 타일 좌표 기준 이미지 범위 (픽셀 변환은 이 값으로 선형 보간)
        "tile_x0": fx0, "tile_y0": fy0, "tile_x1": fx1, "tile_y1": fy1,
        "bounds": {"west": west, "east": east, "south": south, "north": north},
        "attribution": "© OpenStreetMap contributors",
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT_IMG} ({img.width}x{img.height}), {OUT_META}")


if __name__ == "__main__":
    main()
