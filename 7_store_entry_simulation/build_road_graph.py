"""
OSM 도로(data/raw/osm_roads.json) → 보행 네트워크 그래프(data/road_graph.json).

- 사람이 걸을 수 없는 도로(공사 중·계획 도로·버스 전용 등)는 뺀다. 차도(대로)도 보도가 있다고 보고 포함.
- 도로의 꼭짓점을 노드, 이웃한 꼭짓점 사이를 간선(길이 = 실제 거리 m)으로 만든다.
- 가장 큰 연결 덩어리만 남긴다 (끊긴 작은 조각에서 출발하면 경로가 안 나오므로).

실행:
    python build_road_graph.py
"""
import json

import networkx as nx

from common import RAW_DIR, ROAD_GRAPH_PATH, haversine_m

EXCLUDE_HIGHWAY = {"construction", "proposed", "busway", "bus_guideway", "raceway", "motorway", "motorway_link"}


def main():
    ways = json.loads((RAW_DIR / "osm_roads.json").read_text(encoding="utf-8"))["elements"]
    G = nx.Graph()
    coords = {}
    used = 0
    for w in ways:
        hw = w.get("tags", {}).get("highway")
        if hw in EXCLUDE_HIGHWAY or "geometry" not in w:
            continue
        used += 1
        for nid, pt in zip(w["nodes"], w["geometry"]):
            coords[nid] = (round(pt["lat"], 7), round(pt["lon"], 7))
        for a, b in zip(w["nodes"][:-1], w["nodes"][1:]):
            la, lb = coords[a], coords[b]
            G.add_edge(a, b, length=round(haversine_m(la[0], la[1], lb[0], lb[1]), 1))

    largest = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest).copy()
    out = {
        "source": "OpenStreetMap way[highway] (data/raw/osm_roads.json), © OpenStreetMap contributors",
        "nodes": {str(n): coords[n] for n in H.nodes},
        "edges": [[u, v, d["length"]] for u, v, d in H.edges(data=True)],
    }
    ROAD_GRAPH_PATH.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    total_km = sum(d["length"] for _, _, d in H.edges(data=True)) / 1000
    print(f"도로 {used}개 사용 → 노드 {H.number_of_nodes():,} / 간선 {H.number_of_edges():,} "
          f"(총 {total_km:.1f}km, 연결 덩어리 {nx.number_connected_components(G)}개 중 최대만 사용)")
    print(f"저장: {ROAD_GRAPH_PATH}")


if __name__ == "__main__":
    main()
