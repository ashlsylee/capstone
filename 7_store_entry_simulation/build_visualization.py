"""
배경 지도 + 에이전트 이동 시각화 HTML 생성 (API 호출 없음).

assets/background.png(실제 지도) 위에 상권 경계·실제 슈퍼마켓(OSM)·OSM 미등록 경쟁점(추정 위치)·후보지를 찍고,
에이전트들이 각자의 요일·시각에 입구(지하철 출구·도로)로 들어와 → 실제 도로망을 따라 목적지 장소로 걷고 →
장 볼 일이 있으면 슈퍼마켓에 들렀다가 → 출구로 나가는 모습을 시계에 맞춰 보여준다.
시나리오(BASE / C1~C5)를 바꾸면 같은 에이전트들이 그 시나리오에서 고른 점포로 움직인다
(estimate_entry.py가 기본값 확률로 뽑아 둔 outputs/agent_assignments.csv 사용).

실행:
    python build_visualization.py          # → district_map.html (outputs/ 결과 사용)
    python build_visualization.py --mock   # → outputs/mock/district_map.html (파이프라인 점검용)
"""
import argparse
import json
import os

import pandas as pd

from common import (BASE_DIR, BASELINE, NEW_BRAND, OUTPUT_DIR, RoadGraph, haversine_m, load_boundary,
                    load_candidates, load_entrances, load_existing_stores, load_json, load_places, new_store)

META_PATH = BASE_DIR / "assets" / "background_meta.json"


class PathMaker:
    def __init__(self):
        self.rg = RoadGraph()
        self.cache = {}

    def node(self, o):
        return self.rg.nearest_node(o["lat"], o["lng"])

    def path(self, a: dict, b: dict) -> list:
        """a→b 도로 경로 [[lat,lng],...] (양 끝은 실제 좌표)."""
        na, nb = self.node(a), self.node(b)
        if (na, nb) not in self.cache:
            pts = self.rg.path_coords(na, nb) if na != nb else [self.rg.coords[na]]
            self.cache[(na, nb)] = [[round(p[0], 6), round(p[1], 6)] for p in pts]
        return [[a["lat"], a["lng"]]] + self.cache[(na, nb)] + [[b["lat"], b["lng"]]]


def path_len(pts: list) -> float:
    return sum(haversine_m(p[0], p[1], q[0], q[1]) for p, q in zip(pts[:-1], pts[1:]))


def build_payload(out_dir, label: str) -> dict:
    agents = pd.read_csv(out_dir / "agents.csv", encoding="utf-8-sig")
    agents = agents[agents["purpose"].notna()].set_index("agent_id")
    assign = pd.read_csv(out_dir / "agent_assignments.csv", encoding="utf-8-sig").fillna("")
    ev = load_json(out_dir / "entry_evaluation.json")
    virtual = load_json(out_dir / "virtual_stores.json")

    places = {p["id"]: p for p in load_places()}
    entrances = {e["id"]: e for e in load_entrances()}
    candidates = load_candidates()
    existing = load_existing_stores()
    stores = {s["store_id"]: {"id": s["store_id"], "name": s["name"], "lat": s["lat"], "lng": s["lng"],
                              "kind": "osm", "inside": s["inside_district"]} for s in existing}
    for v in virtual:
        stores[v["store_id"]] = {"id": v["store_id"], "name": "OSM 미등록 경쟁점(추정 위치)", "lat": v["lat"],
                                 "lng": v["lng"], "kind": "virtual", "inside": True}
    for c in candidates:
        ns = new_store(c)
        stores[ns["store_id"]] = {"id": ns["store_id"], "name": ns["name"], "lat": ns["lat"], "lng": ns["lng"],
                                  "kind": "new", "inside": True, "candidate": c["id"]}

    pm = PathMaker()
    scenarios = [BASELINE] + [c["id"] for c in candidates]
    out_agents = []
    for r in assign.itertuples():
        if r.agent_id not in agents.index:
            continue
        a = agents.loc[r.agent_id]
        place, ent, ext = places[r.place_id], entrances[r.entrance_id], entrances[r.exit_id]
        # 목적지 장소 대표 지점 주변 ±40m 안 위치 (에이전트끼리 겹치지 않게, 결정적)
        h = sum(map(ord, r.agent_id))
        spot = {"lat": place["lat"] + ((h % 9) - 4) * 0.00008, "lng": place["lng"] + (((h // 9) % 9) - 4) * 0.0001}
        legs = {"in": pm.path(ent, spot), "out": pm.path(spot, ext), "store": {}}
        for sc in scenarios:
            sid = getattr(r, f"store_{sc}")
            if sid and sid in stores and sid not in legs["store"]:
                st = stores[sid]
                legs["store"][sid] = [pm.path(spot, st), pm.path(st, ext)]
        out_agents.append({
            "id": r.agent_id, "age": int(a.age), "sex": a.sex, "occ": a.occupation, "day": a.day, "clock": a.clock,
            "purpose": a.purpose, "need": str(a.need_supermarket).lower() == "true",
            "items": a["items"] if isinstance(a["items"], str) else "",
            "place": r.place_id, "entrance": ent["name"], "exit": ext["name"],
            "choice": {sc: getattr(r, f"store_{sc}") for sc in scenarios},
            "legs": legs,
        })

    results = {c["id"]: {"sales": c["base"]["annual_sales"], "ratio": c["base"]["ratio_to_avg"],
                         "p5": c["robust"]["p5"], "p95": c["robust"]["p95"],
                         "above": c["robust"]["share_above_avg"], "verdict": c["verdict"]} for c in ev["candidates"]}
    return {
        "label": label, "brand": NEW_BRAND,
        "meta": load_json(META_PATH), "boundary": load_boundary(),
        "places": [{k: p[k] for k in ("id", "name", "kind", "lat", "lng")} for p in places.values()],
        "entrances": [{k: e[k] for k in ("id", "name", "lat", "lng")} for e in entrances.values()],
        "candidates": [{k: c[k] for k in ("id", "name", "zone", "lat", "lng")} for c in candidates],
        "stores": list(stores.values()), "agents": out_agents, "results": results,
        "avgStoreSales": ev["market"]["sales_per_store_2025"],
        "verdict": {"district": ev["district_verdict"], "best": ev["best_candidate"]},
    }


HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>동대문 출점 시뮬레이션</title>
<style>
:root {
  --surface-0: #f6f5f1; --surface-1: #ffffff; --border: #dcdad2;
  --text-primary: #1d1d1b; --text-secondary: #5b5a55; --text-muted: #85847d;
  --series-2: #eb6834;  /* 신규 롯데마트 슈퍼 */
  --neutral: #6f6e69;   /* 기존 슈퍼마켓 (OSM) */
  --agent: #2a78d6;     /* 에이전트 */
  --boundary: #c0392b;
  --good: #1f8a3b; --warning: #b77900; --critical: #c23934;
  --map-dim: 0.72;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --border: #33332f;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --series-2: #d95926; --neutral: #9a998f; --agent: #3987e5; --map-dim: 0.6;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --border: #33332f;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
  --series-2: #d95926; --neutral: #9a998f; --agent: #3987e5; --map-dim: 0.6;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font-family: -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", sans-serif; font-size: 14px; }
header { padding: 16px 20px 8px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0; }
.badge-mock { display: inline-block; background: var(--warning); color: #fff; border-radius: 4px; padding: 1px 6px; font-size: 12px; margin-left: 6px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 8px 20px; }
.controls label { color: var(--text-secondary); font-size: 13px; }
select, button { font: inherit; background: var(--surface-1); color: var(--text-primary); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; cursor: pointer; }
button[aria-pressed="true"] { border-color: var(--text-primary); font-weight: 600; }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 330px; gap: 12px; align-items: start; padding: 0 20px 20px; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; padding: 0 16px 16px; } .controls { padding: 8px 16px; } header { padding: 16px 16px 8px; } }
.map-wrap { position: relative; background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
canvas { display: block; width: 100%; height: auto; }
.clock { position: absolute; top: 10px; left: 10px; background: var(--surface-1); border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; font-variant-numeric: tabular-nums; font-size: 18px; font-weight: 600; }
.attr { position: absolute; right: 6px; bottom: 4px; font-size: 11px; color: #555; background: rgba(255,255,255,.75); padding: 1px 4px; border-radius: 3px; }
.tooltip { position: absolute; pointer-events: none; background: var(--surface-1); color: var(--text-primary); border: 1px solid var(--border); border-radius: 6px; padding: 8px 10px; font-size: 12px; line-height: 1.5; box-shadow: 0 2px 8px rgba(0,0,0,.15); display: none; max-width: 270px; z-index: 5; }
.panel { display: flex; flex-direction: column; gap: 12px; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; }
.card h2 { font-size: 14px; margin: 0 0 8px; color: var(--text-secondary); font-weight: 600; }
.tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.tile .v { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
.tile .k { font-size: 12px; color: var(--text-muted); }
table { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 4px 4px; border-bottom: 1px solid var(--border); }
th { color: var(--text-muted); font-weight: 500; }
td.num, th.num { text-align: right; }
tr.sel td { font-weight: 700; }
.verdict.go::before { content: "● "; color: var(--good); }
.verdict.cond::before { content: "▲ "; color: var(--warning); }
.verdict.nogo::before { content: "■ "; color: var(--critical); }
.legend { display: grid; gap: 6px; font-size: 12px; color: var(--text-secondary); }
.legend span.sw { display: inline-block; width: 12px; height: 12px; border-radius: 50%; vertical-align: -2px; margin-right: 6px; border: 2px solid var(--surface-1); box-shadow: 0 0 0 1px var(--border); }
.note { font-size: 12px; color: var(--text-muted); margin: 6px 0 0; }
</style>
</head>
<body>
<header>
  <h1><span id="brand"></span> 신규 출점 시뮬레이션 — 동대문패션타운 관광특구<span id="mockBadge"></span></h1>
  <p class="sub">실제 유동인구 분포로 뽑은 에이전트가 지하철 출구·도로로 들어와 실제 도로를 따라 목적지로 걷고, 장 볼 일이 있으면 슈퍼마켓에 들릅니다</p>
</header>
<div class="controls">
  <label for="scenario">시나리오</label>
  <select id="scenario"></select>
  <label for="day">요일</label>
  <select id="day"><option value="all">전체(7일 겹쳐 보기)</option></select>
  <button id="play" aria-pressed="true">일시정지</button>
  <button class="speed" data-s="1" aria-pressed="true">1×</button>
  <button class="speed" data-s="3">3×</button>
  <button class="speed" data-s="8">8×</button>
  <button id="shoppersOnly" aria-pressed="false">장보기 고객만</button>
  <button id="theme" title="밝게/어둡게">테마</button>
</div>
<div class="layout">
  <div class="map-wrap" id="mapWrap">
    <canvas id="map" aria-label="동대문패션타운 지도 위 에이전트 이동"></canvas>
    <div class="clock" id="clock">00:00</div>
    <div class="attr">지도·도로 © OpenStreetMap contributors · 경계: 서울시 상권분석서비스</div>
    <div class="tooltip" id="tip"></div>
  </div>
  <div class="panel">
    <div class="card">
      <h2 id="nowTitle">지금 상권 안</h2>
      <div class="tiles">
        <div class="tile"><div class="v" id="tPresent">0</div><div class="k">상권 안 에이전트</div></div>
        <div class="tile"><div class="v" id="tShop">0</div><div class="k">슈퍼마켓 이용(누적)</div></div>
        <div class="tile"><div class="v" id="tNew">–</div><div class="k">신규점 이용(누적)</div></div>
        <div class="tile"><div class="v" id="tNeed">0</div><div class="k">장 볼 일이 있는 사람(전체)</div></div>
      </div>
    </div>
    <div class="card">
      <h2>후보지별 추정 (연간, 기본값)</h2>
      <table id="resTable"><thead><tr><th>후보지</th><th class="num">연매출</th><th class="num">평균 이상</th><th>판정</th></tr></thead><tbody></tbody></table>
      <p class="note" id="resNote"></p>
    </div>
    <div class="card">
      <h2>범례</h2>
      <div class="legend">
        <div><span class="sw" style="background:var(--series-2)"></span>신규 <span class="brandName"></span> (선택한 후보지) — ★</div>
        <div><span class="sw" style="background:var(--neutral)"></span>기존 슈퍼마켓 (OSM 등록) — ●</div>
        <div>· 작은 점 = OSM에 없는 경쟁점 (서울시 점포 수에 맞춰 도로 위에 무작위 배치한 추정 위치)</div>
        <div><span class="sw" style="background:var(--agent)"></span>에이전트 · 테두리 주황 = 신규점으로 가는 중</div>
        <div>△ 다른 후보지 · ▢ 목적지 장소 P1~P8 · ◇ 입구(지하철 출구·도로) · 붉은 점선 = 공식 상권 경계</div>
      </div>
      <p class="note">점·에이전트에 마우스를 올리면 상세 정보가 보입니다.</p>
    </div>
  </div>
</div>
<script>
const DATA = __DATA__;
const BG_SRC = "__BG__";
const DAYS = ["월","화","수","목","금","토","일"];
const SEC_PER_HOUR = 6;            // 1×에서 시뮬레이션 1시간 = 6초 (하루 약 2분 24초)
const WALK_M_PER_H = 4500;         // 보행 속도 4.5km/h
const STAY_H = 0.75, SHOP_H = 0.2; // 목적지 체류 45분, 장보기 12분

const canvas = document.getElementById("map"), ctx = canvas.getContext("2d");
const tip = document.getElementById("tip"), wrap = document.getElementById("mapWrap");
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let colors = {};
function readColors(){ colors = {s2: css("--series-2"), neutral: css("--neutral"), agent: css("--agent"), boundary: css("--boundary"), surface: css("--surface-1"), dim: parseFloat(css("--map-dim")) || 1}; }

const M = DATA.meta;
function toImg(lat, lng){
  const n = Math.pow(2, M.zoom), x = (lng + 180) / 360 * n;
  const y = (1 - Math.asinh(Math.tan(lat * Math.PI / 180)) / Math.PI) / 2 * n;
  return [(x - M.tile_x0) / (M.tile_x1 - M.tile_x0) * M.width, (y - M.tile_y0) / (M.tile_y1 - M.tile_y0) * M.height];
}
let scale = 1;
function toCanvas(lat, lng){ const [x, y] = toImg(lat, lng); return [x * scale, y * scale]; }

function hav(a, b){ const R = 6371000, r = Math.PI / 180, dLa = (b[0] - a[0]) * r, dLn = (b[1] - a[1]) * r;
  const h = Math.sin(dLa / 2) ** 2 + Math.cos(a[0] * r) * Math.cos(b[0] * r) * Math.sin(dLn / 2) ** 2; return 2 * R * Math.asin(Math.sqrt(h)); }
function prep(pts){ const cum = [0]; for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + hav(pts[i - 1], pts[i])); return {pts, cum, len: cum[cum.length - 1]}; }
function at(seg, d){ const {pts, cum} = seg; if (d <= 0) return pts[0]; if (d >= seg.len) return pts[pts.length - 1];
  let i = 1; while (cum[i] < d) i++; const f = (d - cum[i - 1]) / (cum[i] - cum[i - 1] || 1);
  return [pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * f, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * f]; }

const storeById = Object.fromEntries(DATA.stores.map(s => [s.id, s]));
const placeById = Object.fromEntries(DATA.places.map(p => [p.id, p]));
const agents = DATA.agents.map(a => {
  const [hh, mm] = a.clock.split(":").map(Number);
  const legs = {in: prep(a.legs.in), out: prep(a.legs.out), store: {}};
  for (const [sid, pair] of Object.entries(a.legs.store)) legs.store[sid] = [prep(pair[0]), prep(pair[1])];
  return {...a, t0: hh + mm / 60, L: legs};
});

let scenario = "BASE", dayFilter = "all", speed = 1, playing = true, simHour = 7, shoppersOnly = false;
document.getElementById("brand").textContent = DATA.brand;
document.querySelectorAll(".brandName").forEach(e => e.textContent = DATA.brand);
const scenarioSel = document.getElementById("scenario");
["BASE", ...DATA.candidates.map(c => c.id)].forEach(id => {
  const c = DATA.candidates.find(x => x.id === id), o = document.createElement("option"); o.value = id;
  o.textContent = id === "BASE" ? "BASE — 현재(신규점 없음)" : `${id} ${c.name}`; scenarioSel.appendChild(o);
});
const daySel = document.getElementById("day");
DAYS.forEach(d => { const o = document.createElement("option"); o.value = d; o.textContent = d + "요일"; daySel.appendChild(o); });
if (DATA.label.startsWith("MOCK")) document.getElementById("mockBadge").innerHTML = '<span class="badge-mock">MOCK 데이터 — 파이프라인 점검용</span>';

// 에이전트 일정: 입구→장소(걷기) → 체류 → [장소→점포 → 장보기 → 점포→출구] 또는 [장소→출구]
function plan(a){
  const sid = a.choice[scenario]; const legs = [[a.L.in, "walk"], [null, "stay", STAY_H]];
  if (sid && a.L.store[sid]) { const [toS, fromS] = a.L.store[sid]; legs.push([toS, "toStore"], [null, "shop", SHOP_H], [fromS, "walk"]); }
  else legs.push([a.L.out, "walk"]);
  return {sid, legs};
}
function state(a, t){
  let dt = t - a.t0; if (dt < 0) dt += 24;
  const p = plan(a); let acc = 0, last = a.L.in.pts[0];
  for (const [seg, kind, dur] of p.legs) {
    const d = seg ? seg.len / WALK_M_PER_H : dur;
    if (dt <= acc + d) {
      if (seg) { const pos = at(seg, (dt - acc) * WALK_M_PER_H); return {pos, kind, sid: p.sid}; }
      return {pos: last, kind, sid: p.sid};
    }
    acc += d; if (seg) last = seg.pts[seg.pts.length - 1];
  }
  return null;
}
function shopTime(a){ const p = plan(a); if (!p.sid) return null; return a.t0 + a.L.in.len / WALK_M_PER_H + STAY_H + a.L.store[p.sid][0].len / WALK_M_PER_H; }

const bg = new Image(); bg.src = BG_SRC;
function resize(){ const w = wrap.clientWidth, dpr = window.devicePixelRatio || 1; scale = w / M.width;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(M.height * scale * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); }
function star(x, y, r){ ctx.beginPath(); for (let i = 0; i < 10; i++){ const rr = i % 2 ? r * 0.45 : r, g = Math.PI / 5 * i - Math.PI / 2; ctx.lineTo(x + rr * Math.cos(g), y + rr * Math.sin(g)); } ctx.closePath(); }
let hoverables = [];
function draw(){
  const W = canvas.width, H = canvas.height; ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = colors.surface; ctx.fillRect(0, 0, W, H);
  if (bg.complete && bg.naturalWidth) { ctx.globalAlpha = colors.dim; ctx.drawImage(bg, 0, 0, M.width * scale, M.height * scale); ctx.globalAlpha = 1; }
  const ms = Math.max(1, scale * 3.2);
  ctx.beginPath(); DATA.boundary.forEach(([lng, lat], i) => { const [x, y] = toCanvas(lat, lng); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.closePath(); ctx.setLineDash([8, 5]); ctx.strokeStyle = colors.boundary; ctx.lineWidth = 2.5; ctx.stroke(); ctx.setLineDash([]);
  hoverables = [];
  ctx.font = `600 ${Math.round(11 * ms)}px sans-serif`;
  DATA.entrances.forEach(e => { const [x, y] = toCanvas(e.lat, e.lng), r = 6 * ms;
    ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath();
    ctx.fillStyle = "#ffffff"; ctx.fill(); ctx.strokeStyle = "#3d3c39"; ctx.lineWidth = 1.5; ctx.stroke();
    hoverables.push({x, y, r: r + 3, html: `<b>입구 ${e.id}</b><br>${e.name}`}); });
  DATA.places.forEach(p => { const [x, y] = toCanvas(p.lat, p.lng), r = 7 * ms;
    ctx.strokeStyle = "#3d3c39"; ctx.lineWidth = 1.5; ctx.strokeRect(x - r, y - r, 2 * r, 2 * r);
    ctx.fillStyle = "#3d3c39"; ctx.fillText(p.id, x + r + 2, y - r);
    hoverables.push({x, y, r: r + 3, html: `<b>${p.id} ${p.name}</b><br>${p.kind}`}); });
  DATA.candidates.forEach(c => { if (c.id === scenario) return; const [x, y] = toCanvas(c.lat, c.lng), r = 7 * ms;
    ctx.beginPath(); ctx.moveTo(x, y - r); ctx.lineTo(x - r, y + r * 0.8); ctx.lineTo(x + r, y + r * 0.8); ctx.closePath();
    ctx.strokeStyle = "#3d3c39"; ctx.lineWidth = 1.5; ctx.stroke(); ctx.fillStyle = "#3d3c39"; ctx.fillText(c.id, x + r + 1, y + r);
    hoverables.push({x, y, r: r + 3, html: `<b>후보지 ${c.id}</b><br>${c.name} (${c.zone})`}); });
  const visits = {}; agents.forEach(a => { if (dayFilter !== "all" && a.day !== dayFilter) return; const s = a.choice[scenario]; if (s) visits[s] = (visits[s] || 0) + 1; });
  DATA.stores.forEach(s => {
    if (s.kind === "new" && s.candidate !== scenario) return;
    const [x, y] = toCanvas(s.lat, s.lng);
    if (s.kind === "virtual") { ctx.beginPath(); ctx.arc(x, y, 3.4 * ms, 0, Math.PI * 2); ctx.fillStyle = colors.neutral; ctx.globalAlpha = 0.75; ctx.fill(); ctx.globalAlpha = 1; ctx.strokeStyle = colors.surface; ctx.lineWidth = 1; ctx.stroke();
      hoverables.push({x, y, r: 5 * ms, html: `<b>${s.name}</b><br>이 시나리오 이용 에이전트: ${visits[s.id] || 0}명`}); return; }
    const r = (s.kind === "new" ? 12 : 6.5) * ms;
    ctx.fillStyle = s.kind === "new" ? colors.s2 : colors.neutral; ctx.strokeStyle = colors.surface; ctx.lineWidth = 2;
    if (s.kind === "new") { star(x, y, r); ctx.fill(); ctx.stroke(); } else { ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
    hoverables.push({x, y, r: r + 3, html: `<b>${s.name}</b><br>${s.kind === "new" ? "신규 후보" : "OSM 등록 슈퍼마켓"}${s.inside ? "" : " (상권 경계 밖)"}<br>이 시나리오 이용 에이전트: ${visits[s.id] || 0}명`});
  });
  let present = 0;
  agents.forEach(a => {
    if (dayFilter !== "all" && a.day !== dayFilter) return;
    if (shoppersOnly && !a.choice[scenario]) return;
    const st = state(a, simHour); if (!st) return; present++;
    const [x, y] = toCanvas(st.pos[0], st.pos[1]), r = 3.6 * ms;
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fillStyle = colors.agent; ctx.fill();
    const toNew = st.sid && storeById[st.sid] && storeById[st.sid].kind === "new" && (st.kind === "toStore" || st.kind === "shop");
    ctx.lineWidth = toNew ? 3 : 1.5; ctx.strokeStyle = toNew ? colors.s2 : colors.surface; ctx.stroke();
    const target = st.sid ? storeById[st.sid].name : "장보기 없음";
    hoverables.push({x, y, r: r + 4, html: `<b>${a.age}세 ${a.sex} · ${a.occ}</b><br>${a.day}요일 ${a.clock} · ${a.entrance}로 진입<br>목적: ${a.purpose} → ${placeById[a.place].name}<br>장보기: ${a.need ? (a.items || "있음") : "없음"}<br>점포: ${target}`});
  });
  updateStats(present);
}
function updateStats(present){
  document.getElementById("tPresent").textContent = present;
  let shop = 0, nw = 0, need = 0;
  agents.forEach(a => { if (dayFilter !== "all" && a.day !== dayFilter) return; if (a.need) need++;
    const t = shopTime(a); if (t === null || t > simHour) return; shop++; if (storeById[a.choice[scenario]].kind === "new") nw++; });
  document.getElementById("tShop").textContent = shop;
  document.getElementById("tNew").textContent = scenario === "BASE" ? "–" : nw;
  document.getElementById("tNeed").textContent = need;
  document.getElementById("nowTitle").textContent = `지금 상권 안 (${dayFilter === "all" ? "7일 겹쳐 보기" : dayFilter + "요일"})`;
}
function won(x){ return (x / 1e8).toFixed(2) + "억"; }
function renderTable(){
  const tb = document.querySelector("#resTable tbody"); tb.innerHTML = "";
  DATA.candidates.forEach(c => {
    const r = DATA.results[c.id], tr = document.createElement("tr"); if (c.id === scenario) tr.className = "sel";
    const cls = r.verdict === "Go" ? "go" : (r.verdict === "조건부" ? "cond" : "nogo");
    tr.innerHTML = `<td title="${c.name}">${c.id} ${c.zone}</td><td class="num" title="민감도 90% 구간 ${won(r.p5)}~${won(r.p95)}">${won(r.sales)}</td><td class="num">${Math.round(r.above * 100)}%</td><td class="verdict ${cls}">${r.verdict}</td>`;
    tr.style.cursor = "pointer"; tr.onclick = () => { scenarioSel.value = c.id; scenarioSel.dispatchEvent(new Event("change")); };
    tb.appendChild(tr);
  });
  document.getElementById("resNote").textContent = `상권 진출 판정: ${DATA.verdict.district} (최적 ${DATA.verdict.best}) · 평균 이상 = 민감도 분석에서 연매출이 상권 슈퍼마켓 점포당 평균(${won(DATA.avgStoreSales)}) 이상인 비율 · 행을 누르면 그 시나리오로 전환`;
}
scenarioSel.onchange = () => { scenario = scenarioSel.value; renderTable(); };
daySel.onchange = () => { dayFilter = daySel.value; };
document.getElementById("play").onclick = e => { playing = !playing; e.target.textContent = playing ? "일시정지" : "재생"; e.target.setAttribute("aria-pressed", playing); };
document.getElementById("shoppersOnly").onclick = e => { shoppersOnly = !shoppersOnly; e.target.setAttribute("aria-pressed", shoppersOnly); };
document.querySelectorAll(".speed").forEach(b => b.onclick = () => { speed = +b.dataset.s; document.querySelectorAll(".speed").forEach(x => x.setAttribute("aria-pressed", x === b)); });
document.getElementById("theme").onclick = () => { const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark"; readColors(); };
canvas.addEventListener("mousemove", e => {
  const rect = canvas.getBoundingClientRect(), mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = null, bd = 1e9; hoverables.forEach(h => { const d = Math.hypot(h.x - mx, h.y - my); if (d < h.r && d < bd) { bd = d; best = h; } });
  if (!best) { tip.style.display = "none"; return; }
  tip.innerHTML = best.html; tip.style.display = "block";
  tip.style.left = Math.min(mx + 14, wrap.clientWidth - tip.offsetWidth - 6) + "px"; tip.style.top = Math.min(my + 14, wrap.clientHeight - tip.offsetHeight - 6) + "px";
});
canvas.addEventListener("mouseleave", () => tip.style.display = "none");
new ResizeObserver(resize).observe(wrap);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", readColors);
let last = performance.now();
function frame(now){ const dt = (now - last) / 1000; last = now;
  if (playing) simHour = (simHour + dt / SEC_PER_HOUR * speed) % 24;
  const h = Math.floor(simHour), m = Math.floor((simHour - h) * 60);
  document.getElementById("clock").textContent = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
  draw(); requestAnimationFrame(frame); }
readColors(); resize(); renderTable(); bg.onload = () => draw(); requestAnimationFrame(frame);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    out_dir = OUTPUT_DIR / "mock" if args.mock else OUTPUT_DIR
    html_path = out_dir / "district_map.html" if args.mock else BASE_DIR / "district_map.html"
    label = "MOCK (파이프라인 점검용)" if args.mock else "LLM 시뮬레이션"

    payload = build_payload(out_dir, label)
    bg_rel = os.path.relpath(BASE_DIR / "assets" / "background.png", html_path.parent).replace(os.sep, "/")
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))).replace("__BG__", bg_rel)
    html_path.write_text(html, encoding="utf-8")
    print(f"저장: {html_path} (에이전트 {len(payload['agents'])}명, {html_path.stat().st_size / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
