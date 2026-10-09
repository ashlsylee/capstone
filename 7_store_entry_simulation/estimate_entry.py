"""
롯데마트 슈퍼 신규 출점 평가 — 실제 도로망 위 동선 + 허프 모델 + 실제 매출 보정. API 호출 없음.

입력: outputs/agents.csv (simulate_agents.py 결과: 요일·시각·목적·장보기 여부)

1) 장소 배정 (규칙): 목적별로 갈 수 있는 장소(data/places.json의 purposes) 중, 그 요일·시각에 문을 연
   곳에서 고른다 (모두 닫혀 있으면 목적에 맞는 곳 전체에서). 장소 간 가중치는 균등(기본값).
2) 동선 (실제 도로망): 입구(지하철 출구·도로 진입점, data/entrances.json) → 장소 → 출구.
   입구·출구는 가중치 × exp(−장소까지 도보거리/L)로 고른다 (가까운 입구를 더 많이 이용).
3) 점포 선택 (허프 모델): 장 볼 일이 있는 사람(need=Y)은 들어오는 길 또는 나가는 길에 슈퍼마켓 하나를 들른다.
   점포 s를 고를 확률 ∝ 매력도_s × exp(−돌아가는 거리_s / λ)
   돌아가는 거리 = (장소→점포→출구) − (장소→출구) 도보거리 (들어오는 길도 같은 방식, 둘 중 짧은 쪽)
4) 경쟁점: OSM 슈퍼마켓(시장·상가 오등록 제외) + OSM에 없는 나머지 점포(서울시 점포 수 − OSM 상권 내 점포 수)를
   상권 안 도로 위에 무작위로 배치한 '보이지 않는 경쟁점'. 기본은 상권 전체에 고르게, 민감도 분석에서는
   OSM 점포(상권 안팎) 300m 이내에 몰린 배치도 시험.
5) 보정 (실제 매출): 시간대 t마다 BASE에서 상권 안 점포로 가는 기대 이용을 2025년 실제 슈퍼마켓 결제 건수에 맞춘다.
   k_t = 실제 결제 건수_t ÷ Σ(need=Y 에이전트의 상권 안 점포 선택 확률)
   신규점 연 결제 = Σ_t k_t × Σ(신규점 선택 확률),  연매출 = × 실제 객단가_t
   → 장보기 여부는 점포 위치와 무관하게 LLM이 한 번만 답하므로, 상권 전체 슈퍼마켓 수요는 시나리오마다 같고
     신규점은 기존 점포 몫을 가져오는 만큼만 매출이 된다 (후보지 바로 앞 사람이 '산다'로 바뀌는 효과가 없음).
6) 민감도 분석: 에이전트 복원추출 + λ·신규점 매력도·입구/장소 가중치·경쟁점 배치를 무작위로 바꿔 R번 반복.

진출 판정 기준 (실행 전에 정함):
  후보지별로 R번 반복 중 '신규점 연매출 ≥ 상권 슈퍼마켓 점포당 평균 연매출(2025)'인 비율
  - Go: 70% 이상 / 조건부: 40-70% / No-Go: 40% 미만
  상권 진출 판정 = 가장 좋은 후보지의 판정

실행:
    python estimate_entry.py           # outputs/ 결과로
    python estimate_entry.py --mock    # outputs/mock/ 결과로 (파이프라인 점검용)

결과: outputs/entry_evaluation.json, outputs/entry_evaluation.md, outputs/agent_assignments.csv (시각화용)
"""
import argparse
import json
import math

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from common import (AGES, BASELINE, DAYS, MARKET_PATH, NEW_BRAND, OUTPUT_DIR, TIMES, RoadGraph, is_open,
                    load_boundary, load_candidates, load_entrances, load_existing_stores, load_json, load_places,
                    new_store)

SEED = 42
N_RUNS = 500
BASE_PARAMS = {"lambda_m": 200.0, "alpha_new": 1.0, "entrance_decay_m": 500.0, "virtual_layout": "uniform"}
# 범위는 기본값을 중심으로 (로그) 대칭: λ 200m의 0.6-1.75배, 매력도 1.0의 1/1.5-1.5배(로그 균등), 입구 감쇠 500m의 0.6-1.6배
RANGES = {"lambda_m": (120.0, 350.0), "alpha_new": (1 / 1.5, 1.5), "entrance_decay_m": (300.0, 800.0),
          "weight_jitter": (0.5, 1.5)}
GO_SHARE, COND_SHARE = 0.7, 0.4
INF = 1e9


# ---------------------------------------------------------------- 공간 준비
class Space:
    def __init__(self, rng: np.random.Generator):
        self.rg = RoadGraph()
        self.places = load_places()
        self.entrances = load_entrances()
        self.candidates = load_candidates()
        self.existing = load_existing_stores()
        self.poly = Polygon(load_boundary())
        market = load_json(MARKET_PATH)["target_industry"]
        self.n_official = market["stores"]
        self.n_osm_inside = sum(s["inside_district"] for s in self.existing)
        self.n_virtual = max(0, round(self.n_official - self.n_osm_inside))

        snap = lambda o: self.rg.nearest_node(o["lat"], o["lng"])
        self.place_nodes = [snap(p) for p in self.places]
        self.ent_nodes = [snap(e) for e in self.entrances]
        self.real_store_nodes = [snap(s) for s in self.existing] + [snap(c) for c in self.candidates]
        # 상권 안 도로 노드 (보이지 않는 경쟁점 배치 후보)
        self.inside_nodes = np.array([n for n, (la, ln) in self.rg.coords.items() if self.poly.contains(Point(ln, la))])
        # 출발점(장소·입구)마다 모든 노드까지 도보거리
        self.dist = {n: self.rg.distances_from(n) for n in set(self.place_nodes + self.ent_nodes)}
        self.d_pe = np.array([[self.dist[p].get(e, INF) for e in self.ent_nodes] for p in self.place_nodes])
        self.virtual_cache = {}
        self.rng = rng

    def virtual_nodes(self, layout: str, key: int) -> np.ndarray:
        """보이지 않는 경쟁점 위치. uniform = 상권 안 도로 전체에 무작위, clustered = 상권 안 도로 중 OSM 점포(상권 안팎) 300m 이내에 무작위."""
        ck = (layout, key)
        if ck not in self.virtual_cache:
            r = np.random.default_rng(10_000 + key)
            pool = self.inside_nodes
            if layout == "clustered":
                near = set()
                for node in self.real_store_nodes[:len(self.existing)]:  # 상권 안팎 OSM 점포 전체 기준
                    near |= {n for n, d in self.rg.distances_from(node).items() if d <= 300}
                cand = np.array([n for n in pool if n in near])
                pool = cand if len(cand) else pool
            self.virtual_cache[ck] = r.choice(pool, size=self.n_virtual, replace=True)
        return self.virtual_cache[ck]

    def store_table(self, layout: str, key: int):
        """[노드, 상권 안 여부, 신규 후보 인덱스(-1=기존)] 배열."""
        nodes = list(self.real_store_nodes[:len(self.existing)]) + list(self.virtual_nodes(layout, key))
        inside = [s["inside_district"] for s in self.existing] + [True] * self.n_virtual
        nodes += self.real_store_nodes[len(self.existing):]
        inside += [True] * len(self.candidates)
        cand_idx = [-1] * (len(self.existing) + self.n_virtual) + list(range(len(self.candidates)))
        return np.array(nodes), np.array(inside), np.array(cand_idx)

    def detour(self, store_nodes: np.ndarray) -> np.ndarray:
        """D[p, e_in, e_out, s] = 들어오는 길/나가는 길 중 짧은 쪽의 돌아가는 거리."""
        P, E, S = len(self.place_nodes), len(self.ent_nodes), len(store_nodes)
        d_ps = np.array([[self.dist[p].get(int(s), INF) for s in store_nodes] for p in self.place_nodes])  # P×S
        d_es = np.array([[self.dist[e].get(int(s), INF) for s in store_nodes] for e in self.ent_nodes])    # E×S
        leg = d_ps[:, None, :] + d_es[None, :, :] - self.d_pe[:, :, None]                                  # P×E×S
        leg = np.clip(leg, 0, None)
        return np.minimum(leg[:, :, None, :], leg[:, None, :, :])                                          # P×E×E×S


# ---------------------------------------------------------------- 에이전트 → 장소 확률
def place_matrix(agents: pd.DataFrame, places: list, place_w: np.ndarray) -> np.ndarray:
    """에이전트 × 장소 확률 (목적에 맞고 그 시각에 문 연 곳, 장소 가중치 비례)."""
    M = np.zeros((len(agents), len(places)))
    for i, a in enumerate(agents.itertuples()):
        hour = int(a.clock[:2]) + int(a.clock[3:]) / 60
        ok = [j for j, p in enumerate(places) if a.purpose in p["purposes"]]
        open_ok = [j for j in ok if is_open(places[j], a.day, hour)]
        use = open_ok or ok
        w = place_w[use]
        M[i, use] = w / w.sum()
    return M


def entrance_probs(space: Space, ent_w: np.ndarray, decay: float) -> np.ndarray:
    """P(입구 e | 장소 p) ∝ 가중치 × exp(−도보거리/decay)."""
    W = ent_w[None, :] * np.exp(-space.d_pe / decay)
    return W / W.sum(axis=1, keepdims=True)


def store_probs(space: Space, D: np.ndarray, ent_p: np.ndarray, attract: np.ndarray, lam: float) -> np.ndarray:
    """Q[p, s] = 장소 p에 있는 장보기 고객이 점포 s를 고를 확률 (입구·출구 조합 평균)."""
    U = attract[None, None, None, :] * np.exp(-D / lam)
    Pr = U / np.maximum(U.sum(axis=3, keepdims=True), 1e-300)
    w = ent_p[:, :, None] * ent_p[:, None, :]                                     # P×E×E
    return (Pr * w[:, :, :, None]).sum(axis=(1, 2))                               # P×S


# ---------------------------------------------------------------- 한 번의 평가
def evaluate(space, agents, A, need, band_idx, real_tx, ticket, params, layout_key, w_agent=None):
    nodes, inside, cand_idx = space.store_table(params["virtual_layout"], layout_key)
    D = space.detour(nodes)
    ent_p = entrance_probs(space, params["entrance_w"], params["entrance_decay_m"])
    w = need * (w_agent if w_agent is not None else 1.0)
    n_cand = len(space.candidates)

    def agent_store_p(scn: int):  # scn = -1 → BASE
        attract = np.where(cand_idx == -1, 1.0, 0.0)
        if scn >= 0:
            attract = np.where(cand_idx == scn, params["alpha_new"], attract)
        Q = store_probs(space, D, ent_p, attract, params["lambda_m"])
        return A @ Q                                                               # N×S

    base = agent_store_p(-1)
    base_in = np.bincount(band_idx, weights=w * base[:, inside].sum(axis=1), minlength=len(TIMES))
    k = np.divide(real_tx, base_in, out=np.zeros(len(TIMES)), where=base_in > 0)
    out = {"k": k, "base": base, "nodes": nodes, "inside": inside, "cand_idx": cand_idx, "scn": {}}
    for c in range(n_cand):
        P = agent_store_p(c)
        p_new = P[:, cand_idx == c].sum(axis=1)
        tx_t = k * np.bincount(band_idx, weights=w * p_new, minlength=len(TIMES))
        out["scn"][c] = {"P": P, "p_new": p_new, "tx": float(tx_t.sum()), "sales": float((tx_t * ticket).sum())}
    return out


# ---------------------------------------------------------------- 결과 정리
def share(values: pd.Series, weights: np.ndarray, order=None) -> dict:
    s = pd.Series(weights, index=values.values).groupby(level=0).sum()
    total = s.sum()
    keys = order if order else list(s.sort_values(ascending=False).index)
    return {str(k): round(float(s.get(k, 0)) / total * 100, 1) for k in keys} if total > 0 else {}


def verdict(p: float) -> str:
    return "Go" if p >= GO_SHARE else ("조건부" if p >= COND_SHARE else "No-Go")


def fmt_won(x) -> str:
    return f"{x / 1e8:,.2f}억원"


def write_markdown(ev: dict, path):
    m = ev["market"]
    L = [f"# {NEW_BRAND} 신규 출점 평가 — 동대문패션타운 관광특구", "",
         f"- 실행: {ev['run_label']} · 에이전트 {ev['n_agents']}명 (장보기 필요 {ev['n_need']}명) · 민감도 분석 {ev['n_runs']}회",
         f"- 기준(2025년 실제, 서울시 상권분석서비스 '슈퍼마켓'): 점포 {m['stores']}개, 연매출 {fmt_won(m['sales_amount_2025'])}, "
         f"점포당 평균 연매출 **{fmt_won(m['sales_per_store_2025'])}**, 객단가 {m['avg_ticket']:,}원",
         f"- 경쟁점: OSM 슈퍼마켓 {ev['competitors']['osm_total']}곳(상권 안 {ev['competitors']['osm_inside']}곳) + "
         f"OSM에 없는 상권 안 점포 {ev['competitors']['virtual']}곳을 도로 위에 배치", "",
         "## 후보지별 결과", "",
         "| 후보지 | 연 결제건수(기본) | 연매출(기본) | 연매출 90% 구간 | 평균 이상 비율 | 1위 비율 | 판정 |",
         "|---|---|---|---|---|---|---|"]
    for c in ev["candidates"]:
        b, r = c["base"], c["robust"]
        L.append(f"| {c['id']} {c['name']} | {b['annual_transactions']:,} | {fmt_won(b['annual_sales'])} "
                 f"({b['ratio_to_avg'] * 100:.0f}%) | {fmt_won(r['p5'])}-{fmt_won(r['p95'])} | "
                 f"{r['share_above_avg'] * 100:.0f}% | {r['rank1_share'] * 100:.0f}% | **{c['verdict']}** |")
    L += ["", f"**상권 진출 판정: {ev['district_verdict']}** (최적 후보지 {ev['best_candidate']})", "",
          "## 롯데 미입점(BASE) 대비 변화", "",
          "| 후보지 | 상권 슈퍼마켓 결제 중 점유 | 기존 점포 1곳당 평균 감소 | 신규점 고객이 원래 가던 곳 (상권 안 / 밖) | 가장 크게 줄어드는 기존 점포(OSM) |",
          "|---|---|---|---|---|"]
    for c in ev["candidates"]:
        v = c["vs_base"]
        top = ", ".join(f"{x['store']} −{x['loss_pct']}%" for x in v["named_store_loss_top"])
        L.append(f"| {c['id']} {c['name']} | {v['district_tx_share_pct']}% | {v['avg_loss_per_existing_store_tx']:,}건 "
                 f"(−{v['avg_loss_per_existing_store_pct']}%) | {v['customers_from_inside_pct']}% / {v['customers_from_outside_pct']}% | {top} |")
    L += ["",
          "## BASE 검증 — 시뮬레이션 장보기 고객 구성 vs 실제 슈퍼마켓 결제 구성 (MAE, %p)", "",
          "| 항목 | MAE |", "|---|---|"]
    for key in ["age", "gender", "time", "day"]:
        L.append(f"| {key} | {ev['validation'][key]['mae']} |")
    L += ["", "## LLM 판단 요약 — 목적별 인원과 장보기 필요 비율", "", "| 목적 | 인원 | 장보기 필요 |", "|---|---|---|"]
    for k, v in ev["llm_summary"]["by_purpose"].items():
        L.append(f"| {k} | {v['n']} | {v['need_rate'] * 100:.0f}% |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def load_agents(out_dir) -> pd.DataFrame:
    agents = pd.read_csv(out_dir / "agents.csv", encoding="utf-8-sig")
    if "need_supermarket" not in agents.columns:
        raise SystemExit("outputs/agents.csv가 이전 형식입니다 (need_supermarket 열 없음). python simulate_agents.py를 먼저 성공시키세요.")
    return agents[agents["purpose"].notna()].reset_index(drop=True)


def run_evaluation(agents: pd.DataFrame, runs: int, label: str, entrance_weights=None, verbose: bool = True):
    """기본값 1회 + 민감도 분석 runs회 → (ev 결과 dict, 시각화용 내부값). entrance_weights로 입구 가중치 가정을 바꿀 수 있다."""
    need = agents["need_supermarket"].astype(str).str.lower().eq("true").to_numpy(dtype=float)
    band_idx = agents["time_band"].map({t: i for i, t in enumerate(TIMES)}).to_numpy()

    market = load_json(MARKET_PATH)["target_industry"]
    real_tx = np.array([market["time"][t]["transactions"] for t in TIMES], dtype=float)
    ticket = np.array([market["time"][t]["avg_ticket"] for t in TIMES], dtype=float)
    avg_store = market["sales_per_store_2025"]

    rng = np.random.default_rng(SEED)
    space = Space(rng)
    n_cand, P, E = len(space.candidates), len(space.places), len(space.entrances)
    base_ent_w = np.array(entrance_weights if entrance_weights is not None else [e["weight"] for e in space.entrances])

    # ---- 기본값 1회
    params = {**BASE_PARAMS, "entrance_w": base_ent_w}
    A = place_matrix(agents, space.places, np.ones(P))
    base = evaluate(space, agents, A, need, band_idx, real_tx, ticket, params, layout_key=0)
    if verbose:
        print("시간대별 보정계수 k_t:", {t: round(v) for t, v in zip(TIMES, base["k"])})

    # ---- 민감도 분석
    sales_runs = np.zeros((runs, n_cand))
    for r in range(runs):
        lo, hi = RANGES["weight_jitter"]
        prm = {
            "lambda_m": rng.uniform(*RANGES["lambda_m"]),
            "alpha_new": float(np.exp(rng.uniform(*np.log(RANGES["alpha_new"])))),
            "entrance_decay_m": rng.uniform(*RANGES["entrance_decay_m"]),
            "virtual_layout": "uniform" if rng.random() < 0.5 else "clustered",
            "entrance_w": base_ent_w * rng.uniform(lo, hi, size=E),
        }
        A_r = place_matrix(agents, space.places, rng.uniform(lo, hi, size=P))
        w_boot = rng.multinomial(len(agents), np.full(len(agents), 1 / len(agents))).astype(float)
        res = evaluate(space, agents, A_r, need, band_idx, real_tx, ticket, prm, layout_key=r % 20, w_agent=w_boot)
        sales_runs[r] = [res["scn"][c]["sales"] for c in range(n_cand)]
        if verbose and (r + 1) % 50 == 0:
            print(f"\r민감도 분석 {r + 1}/{runs}", end="")
    if verbose:
        print()
    rank1 = np.bincount(sales_runs.argmax(axis=1), minlength=n_cand) / runs

    # ---- 후보지별 정리
    results = []
    weight_need = need
    for c, site in enumerate(space.candidates):
        s = base["scn"][c]
        w_new = weight_need * s["p_new"] * base["k"][band_idx]
        share_above = float((sales_runs[:, c] >= avg_store).mean())
        # 신규점이 매출을 가져오는 곳 (기존 점포별 감소분, 상위 5곳)
        loss = (weight_need[:, None] * (base["base"] - s["P"][:, :len(base["base"][0])]) * base["k"][band_idx][:, None]).sum(axis=0)
        names = [x["name"] for x in space.existing] + ["(OSM 미등록 상권 내 점포)"] * space.n_virtual + \
                [f"후보 {x['id']}" for x in space.candidates]
        loss_by = pd.Series(loss, index=names).groupby(level=0).sum().sort_values(ascending=False)
        # ---- 롯데 미입점(BASE) 대비 변화
        kb = base["k"][band_idx]
        base_tx_store = (weight_need[:, None] * base["base"] * kb[:, None]).sum(axis=0)       # 점포별 BASE 연 결제
        scn_tx_store = (weight_need[:, None] * s["P"] * kb[:, None]).sum(axis=0)
        named = [(x["name"], j) for j, x in enumerate(space.existing)]
        named_loss = sorted(
            [{"store": n, "inside_district": space.existing[j]["inside_district"],
              "base_tx": round(float(base_tx_store[j])),
              "loss_pct": round(float((base_tx_store[j] - scn_tx_store[j]) / base_tx_store[j] * 100), 1)}
             for n, j in named if base_tx_store[j] > 0], key=lambda r: -r["loss_pct"])
        new_tx_t = np.bincount(band_idx, weights=weight_need * s["p_new"] * kb, minlength=len(TIMES))
        moved = np.clip(base_tx_store - scn_tx_store, 0, None)[:len(base["inside"])]
        is_old = base["cand_idx"] == -1
        moved_in = float(moved[is_old & base["inside"]].sum())
        moved_out = float(moved[is_old & ~base["inside"]].sum())
        vs_base = {
            "district_tx_share_pct": round(float(s["tx"] / market["transactions_2025"] * 100), 2),
            "avg_loss_per_existing_store_tx": round(float(s["tx"] / market["stores"])),
            "avg_loss_per_existing_store_pct": round(float(s["tx"] / market["stores"] / market["transactions_per_store_2025"] * 100), 2),
            "named_store_loss_top": named_loss[:3],
            "customers_from_inside_pct": round(moved_in / max(moved_in + moved_out, 1e-9) * 100, 1),
            "customers_from_outside_pct": round(moved_out / max(moved_in + moved_out, 1e-9) * 100, 1),
            "new_store_time_pct": dict(zip(TIMES, [round(float(v), 1) for v in new_tx_t / max(new_tx_t.sum(), 1e-9) * 100])),
            "district_time_pct": {t: market["time"][t]["tx_pct"] for t in TIMES},
        }
        results.append({
            **site,
            "vs_base": vs_base,
            "base": {"annual_transactions": round(s["tx"]), "annual_sales": round(s["sales"]),
                     "ratio_to_avg": round(s["sales"] / avg_store, 3),
                     "share_of_district_sales": round(s["sales"] / market["sales_amount_2025"], 4)},
            "robust": {"p5": round(float(np.percentile(sales_runs[:, c], 5))),
                       "p50": round(float(np.percentile(sales_runs[:, c], 50))),
                       "p95": round(float(np.percentile(sales_runs[:, c], 95))),
                       "share_above_avg": round(share_above, 3), "rank1_share": round(float(rank1[c]), 3)},
            "verdict": verdict(share_above),
            "sales_taken_from_tx": {k: round(float(v)) for k, v in loss_by.head(5).items()},
            "customer_profile": {
                "age_pct": share(agents["age_group"], w_new, AGES),
                "gender_pct": share(agents["sex"], w_new),
                "time_pct": share(agents["time_band"], w_new, TIMES),
                "day_pct": share(agents["day"], w_new, DAYS),
                "purpose_pct": share(agents["purpose"], w_new),
                "top_occupations": dict(list(share(agents["occupation"], w_new).items())[:6]),
                "top_items": dict(list(share(agents["items"].fillna(""), w_new).items())[:8]),
            },
        })
    best = max(results, key=lambda x: (x["robust"]["share_above_avg"], x["robust"]["p50"]))

    # ---- BASE 검증 (보정 전 구성)
    w_base = need * base["base"][:, base["inside"]].sum(axis=1)
    sim = {"age": share(agents["age_group"], w_base, AGES),
           "gender": share(agents["sex"].map({"남자": "남", "여자": "여"}), w_base, ["남", "여"]),
           "time": share(agents["time_band"], w_base, TIMES), "day": share(agents["day"], w_base, DAYS)}
    real = {"age": market["age_tx_pct"], "gender": market["gender_tx_pct"],
            "time": {t: market["time"][t]["tx_pct"] for t in TIMES}, "day": {d: market["day"][d]["tx_pct"] for d in DAYS}}
    validation = {k: {"real": real[k], "sim": sim[k],
                      "mae": round(float(np.mean([abs(sim[k].get(c, 0) - real[k][c]) for c in real[k]])), 2)} for k in sim}

    by_purpose = agents.groupby("purpose")["need_supermarket"].agg(
        n="size", need_rate=lambda s: s.astype(str).str.lower().eq("true").mean()).sort_values("n", ascending=False)
    ev = {
        "run_label": label,
        "n_agents": len(agents), "n_need": int(need.sum()), "n_runs": runs,
        "market": {k: market[k] for k in ["stores", "sales_amount_2025", "transactions_2025", "avg_ticket", "sales_per_store_2025"]},
        "competitors": {"osm_total": len(space.existing), "osm_inside": space.n_osm_inside, "virtual": space.n_virtual},
        "base_params": BASE_PARAMS, "ranges": RANGES,
        "criteria": {"go_share": GO_SHARE, "conditional_share": COND_SHARE, "benchmark": "상권 슈퍼마켓 점포당 평균 연매출(2025)"},
        "calibration_k": {t: round(float(v), 1) for t, v in zip(TIMES, base["k"])},
        "candidates": results, "best_candidate": best["id"], "district_verdict": best["verdict"],
        "validation": validation,
        "llm_summary": {"need_rate": round(float(need.mean()), 3),
                        "by_purpose": {k: {"n": int(v.n), "need_rate": round(float(v.need_rate), 3)} for k, v in by_purpose.iterrows()}},
        "places": space.places,
    }
    return ev, {"space": space, "base": base, "A": A, "need": need, "base_ent_w": base_ent_w}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--runs", type=int, default=N_RUNS)
    args = parser.parse_args()
    out_dir = OUTPUT_DIR / "mock" if args.mock else OUTPUT_DIR

    agents = load_agents(out_dir)
    label = "MOCK (파이프라인 점검용 가짜 응답 — 실험 결과 아님)" if args.mock else "LLM 시뮬레이션"
    ev, inner = run_evaluation(agents, args.runs, label)
    space, base, A, need, base_ent_w = inner["space"], inner["base"], inner["A"], inner["need"], inner["base_ent_w"]
    P, E = len(space.places), len(space.entrances)
    (out_dir / "entry_evaluation.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(ev, out_dir / "entry_evaluation.md")

    # ---- 시각화용: 기본값 기준으로 에이전트마다 장소·입구·출구·시나리오별 점포를 하나씩 뽑아 둔다
    vr = np.random.default_rng(SEED + 7)
    ent_p = entrance_probs(space, base_ent_w, BASE_PARAMS["entrance_decay_m"])
    rows = []
    for i, a in enumerate(agents.itertuples()):
        p = int(vr.choice(P, p=A[i]))
        row = {"agent_id": a.agent_id, "place_id": space.places[p]["id"],
               "entrance_id": space.entrances[int(vr.choice(E, p=ent_p[p]))]["id"],
               "exit_id": space.entrances[int(vr.choice(E, p=ent_p[p]))]["id"]}
        for sc in [BASELINE] + [c["id"] for c in space.candidates]:
            row[f"store_{sc}"] = ""
        if need[i]:
            u = vr.random()
            for c_i, sc in enumerate([BASELINE] + [c["id"] for c in space.candidates]):
                probs = base["base"][i] if sc == BASELINE else base["scn"][c_i - 1]["P"][i]
                probs = probs / probs.sum()
                j = int(min(np.searchsorted(np.cumsum(probs), u), len(probs) - 1))  # 같은 난수 → 시나리오 간 비교 가능
                node = int(base["nodes"][j])
                if base["cand_idx"][j] >= 0:
                    row[f"store_{sc}"] = f"NEW_{space.candidates[base['cand_idx'][j]]['id']}"
                elif j < len(space.existing):
                    row[f"store_{sc}"] = space.existing[j]["store_id"]
                else:
                    row[f"store_{sc}"] = f"V{node}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "agent_assignments.csv", index=False, encoding="utf-8-sig")
    (out_dir / "virtual_stores.json").write_text(json.dumps(
        [{"store_id": f"V{int(n)}", "lat": space.rg.coords[int(n)][0], "lng": space.rg.coords[int(n)][1]}
         for n in space.virtual_nodes(BASE_PARAMS["virtual_layout"], 0)]), encoding="utf-8")
    print((out_dir / "entry_evaluation.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
