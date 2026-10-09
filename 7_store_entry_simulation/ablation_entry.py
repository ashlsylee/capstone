"""
대조실험 — 결론(최적 후보지·판정)이 LLM 판단과 가정값에 얼마나 기대고 있는지 확인한다. API 호출 없음.

같은 outputs/agents.csv 로 estimate_entry.py 의 평가를 조건만 바꿔 다시 돌린다.

대조실험 A — LLM 판단의 기여도
  - 실제 LLM 응답 (기준)
  - 장보기 여부 섞기   : need_supermarket 을 에이전트끼리 무작위로 섞음 (장보기 비율은 그대로, '누가' 사는지만 무작위)
  - 목적 섞기          : 같은 시간대 안에서 purpose 를 무작위로 섞음 (시간대별 목적 구성은 그대로)
  - 둘 다 섞기         : 목적·장보기 여부를 모두 무작위로 섞음 (LLM이 페르소나를 보고 한 판단이 전혀 반영되지 않은 상태)
  → 섞어도 결과가 같으면 그 결론은 LLM이 아니라 공간 구조(도로·입구·경쟁점 배치)가 만든 것이다.

대조실험 B — 입구 가중치 가정
  - 가정값 (지하철 60% / 도로 40%), 입구 10곳 균등, 도로 위주 (지하철 30% / 도로 70%)

실행:
    python ablation_entry.py            # outputs/ablation_entry.md, outputs/ablation_entry.csv
    python ablation_entry.py --mock     # outputs/mock/ 기준
"""
import argparse
import contextlib
import io

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, load_entrances
from estimate_entry import load_agents, run_evaluation

RUNS = 200
SEED = 0


def llm_variants(agents: pd.DataFrame) -> dict:
    rng = np.random.default_rng(SEED)
    out = {"실제 LLM 응답": agents}
    v = agents.copy()
    v["need_supermarket"] = rng.permutation(v["need_supermarket"].to_numpy())
    out["장보기 여부 섞기"] = v
    v = agents.copy()
    v["purpose"] = v.groupby("time_band")["purpose"].transform(lambda s: rng.permutation(s.to_numpy()))
    out["목적 섞기(시간대 안)"] = v
    v = agents.copy()
    v["need_supermarket"] = rng.permutation(v["need_supermarket"].to_numpy())
    v["purpose"] = rng.permutation(v["purpose"].to_numpy())
    out["둘 다 섞기"] = v
    return out


def entrance_variants() -> dict:
    ents = load_entrances()
    n_road = sum(e["kind"] == "도로" for e in ents)
    return {
        "가정값(지하철 60%)": [e["weight"] for e in ents],
        "입구 10곳 균등": [1 / len(ents)] * len(ents),
        "도로 위주(지하철 30%)": [e["weight"] * 0.5 if e["kind"] == "지하철" else 0.7 / n_road for e in ents],
    }


def summarize(ev: dict) -> dict:
    row = {}
    for c in ev["candidates"]:
        row[f"{c['id']} 연매출(억)"] = round(c["base"]["annual_sales"] / 1e8, 2)
        row[f"{c['id']} 평균이상%"] = round(c["robust"]["share_above_avg"] * 100)
        row[f"{c['id']} 판정"] = c["verdict"]
    row["최적 후보지"] = ev["best_candidate"]
    row["성별 MAE"] = ev["validation"]["gender"]["mae"]
    row["시간대 MAE"] = ev["validation"]["time"]["mae"]
    return row


def run_quiet(agents, label, entrance_weights=None):
    with contextlib.redirect_stdout(io.StringIO()):
        ev, _ = run_evaluation(agents, RUNS, label, entrance_weights=entrance_weights, verbose=False)
    return ev


def to_markdown(df: pd.DataFrame, title: str, cand_ids: list) -> list:
    lines = [f"## {title}", "", "| 조건 | " + " | ".join(cand_ids) + " | 최적 | 성별 MAE | 시간대 MAE |",
             "|---|" + "---|" * (len(cand_ids) + 3)]
    for name, r in df.iterrows():
        cells = [f"{r[f'{c} 연매출(억)']:.2f}억 · {r[f'{c} 평균이상%']}% · {r[f'{c} 판정']}" for c in cand_ids]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {r['최적 후보지']} | {r['성별 MAE']} | {r['시간대 MAE']} |")
    return lines + [""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    out_dir = OUTPUT_DIR / "mock" if args.mock else OUTPUT_DIR
    agents = load_agents(out_dir)

    rows_a = {}
    for name, df in llm_variants(agents).items():
        print(f"[A] {name} ...")
        rows_a[name] = summarize(run_quiet(df, name))
    rows_b = {}
    for name, w in entrance_variants().items():
        print(f"[B] {name} ...")
        rows_b[name] = summarize(run_quiet(agents, name, entrance_weights=w))

    a, b = pd.DataFrame(rows_a).T, pd.DataFrame(rows_b).T
    cand_ids = sorted({k.split()[0] for k in a.columns if k[0] == "C"})
    pd.concat({"A_LLM기여도": a, "B_입구가중치": b}).to_csv(out_dir / "ablation_entry.csv", encoding="utf-8-sig")
    lines = ["# 대조실험 — 결론이 LLM 판단·가정값에 얼마나 기대는가", "",
             f"각 칸: 기본값 연매출 · 민감도 분석({RUNS}회)에서 점포당 평균 연매출 이상인 비율 · 판정", ""]
    lines += to_markdown(a, "A. LLM 판단의 기여도 (같은 에이전트, 응답만 무작위로 섞음)", cand_ids)
    lines += to_markdown(b, "B. 입구 가중치 가정 (실제 LLM 응답 고정)", cand_ids)
    (out_dir / "ablation_entry.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
