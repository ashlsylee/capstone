"""
시뮬레이션 1·2·(3)이 있는 만큼 결과와 실제 유동인구 데이터를 비교해서
① 실험 설정 표, ② 연령대/성별/시간대/요일 분포 비교 표(오차 MAE 포함),
③ 시뮬레이션 간 개선 요약, ④ 10대 제외 정규화 시 연령대 오차 변화를 마크다운으로 만든다.
API 호출 없음. outputs/에 있는 파일만큼 자동으로 비교 열이 늘어난다
(visit_simulation1_results.csv만 있으면 1열, 1·2 있으면 2열, 1·2·3 있으면 3열).

목표(target)에 도달한 시점까지 "처음 500명이 수락한 순서대로"만 잘라서 비교한다
(질의 도중 배치 단위로 target을 살짝 넘겨 저장되므로, 정확히 500번째 수락자가 나온
시점까지의 질의 수를 "몇 명을 물어봐야 500명이 채워지는지"로 사용한다).

실행 (시뮬레이션을 돌린 뒤):
    python compare_results.py --target 500
"""
import argparse
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REAL_DIST_PATH = BASE_DIR / "data" / "real_distribution.json"

AGE_ORDER = ["10대", "20대", "30대", "40대", "50대", "60대+"]
AGE_ORDER_NO_TEEN = ["20대", "30대", "40대", "50대", "60대+"]
GENDER_ORDER = ["남", "여"]
TIME_ORDER = ["00-06", "06-11", "11-14", "14-17", "17-21", "21-24"]
DAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]
SEX_MAP = {"남자": "남", "여자": "여"}

SIM_FILES = [
    ("visit_simulation1_results.csv", "시뮬1", "시뮬레이션 1 (기본 정보만)"),
    ("visit_simulation2_results.csv", "시뮬2", "시뮬레이션 2 (+ 1단계 상권 분석 결과)"),
    ("visit_simulation3_results.csv", "시뮬3", "시뮬레이션 3 (+ 요일특성 + 반편향 지시)"),
]


def normalize_excl_teen(pct: dict) -> dict:
    """'10대' 항목을 빼고 나머지 5개 구간을 다시 100%로 정규화한다."""
    remaining = {k: pct[k] for k in AGE_ORDER_NO_TEEN}
    total = sum(remaining.values())
    if total == 0:
        return {k: 0.0 for k in AGE_ORDER_NO_TEEN}
    return {k: round(v / total * 100, 2) for k, v in remaining.items()}


def load_sim(name: str, target: int):
    path = OUTPUT_DIR / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["gender"] = df["sex"].map(SEX_MAP).fillna(df["sex"])
    accepted = df[df["visit"] == True].reset_index(drop=True)

    reached_target = len(accepted) >= target
    if reached_target:
        used = accepted.iloc[:target]
        target_uuid = used.iloc[-1]["uuid"]
        queried_to_target = df.index[df["uuid"] == target_uuid][0] + 1
    else:
        used = accepted
        queried_to_target = None

    age_pct = (used["age_group"].value_counts(normalize=True) * 100).reindex(AGE_ORDER).fillna(0).round(2).to_dict()
    gender_pct = (used["gender"].value_counts(normalize=True) * 100).reindex(GENDER_ORDER).fillna(0).round(2).to_dict()
    time_pct = (used["time_slot"].value_counts(normalize=True) * 100).reindex(TIME_ORDER).fillna(0).round(2).to_dict()
    day_pct = (used["day_of_week"].value_counts(normalize=True) * 100).reindex(DAY_ORDER).fillna(0).round(2).to_dict()

    return {
        "total_queried": len(df),
        "total_accepted": len(accepted),
        "reached_target": reached_target,
        "queried_to_target": queried_to_target,
        "used_n": len(used),
        "accept_rate": len(accepted) / len(df) if len(df) else 0,
        "age_pct": age_pct,
        "age_pct_no_teen": normalize_excl_teen(age_pct),
        "gender_pct": gender_pct,
        "time_pct": time_pct,
        "day_pct": day_pct,
    }


def mae(real: dict, sim: dict, order: list) -> float:
    diffs = [abs(real[k] - sim[k]) for k in order]
    return round(sum(diffs) / len(diffs), 2)


def dist_table(title: str, key: str, real: dict, sims: list, order: list, sim_labels: list) -> str:
    header = f"| {title} | 실제(%) | " + " | ".join(f"{l}(%)" for l in sim_labels) + " |"
    sep = "|---|---|" + "---|" * len(sim_labels)
    lines = [header, sep]
    for k in order:
        row = [f"{real[k]:.2f}"]
        for s in sims:
            row.append(f"{s[key][k]:.2f}")
        lines.append(f"| {k} | " + " | ".join(row) + " |")
    lines.append("| **합계** | **100.00** | " + " | ".join("**100.00**" for _ in sims) + " |")
    maes = [mae(real, s[key], order) for s in sims]
    lines.append("| **MAE** |  | " + " | ".join(f"**{m}**" for m in maes) + " |")
    return "\n".join(lines)


def setup_table(sims: list, sim_headers: list, real_total: int, store_counts: dict, model: str, target: int) -> str:
    header = "| 항목 | " + " | ".join(sim_headers) + " |"
    sep = "|---|" + "---|" * len(sim_headers)
    lines = [
        header, sep,
        "| 대상 상권 | " + " | ".join("동대문패션타운 관광특구" for _ in sims) + " |",
        f"| 실제 총 유동인구 | " + " | ".join(f"{real_total:,}" for _ in sims) + " |",
        "| 에이전트 풀 | " + " | ".join("1,000,000" for _ in sims) + " |",
        "| 질의한 에이전트 수(target 도달 시점) | " +
        " | ".join(str(s["queried_to_target"]) if s["reached_target"] else "target 미도달" for s in sims) + " |",
        "| 채워진 방문자 수 | " +
        " | ".join(str(target) if s["reached_target"] else str(s["total_accepted"]) for s in sims) + " |",
        "| 수락률(질의 대비 방문 응답 비율) | " +
        " | ".join(f"{s['accept_rate']:.1%}" for s in sims) + " |",
    ]
    store_desc = (f"마트 {store_counts['마트']}개·백화점 {store_counts['백화점']}개·"
                  f"편의점 {store_counts['편의점']}개·카페 {store_counts['카페']}개·"
                  f"음식점 {store_counts['음식점']}개 (5종)")
    lines.append("| 매장 수 / 유형 | " + " | ".join(store_desc for _ in sims) + " |")
    lines.append("| LLM | " + " | ".join(model for _ in sims) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()

    if not REAL_DIST_PATH.exists():
        raise RuntimeError("data/real_distribution.json이 없습니다. 먼저 prepare_real_data.py를 실행하세요.")
    real = json.loads(REAL_DIST_PATH.read_text(encoding="utf-8"))
    real_fp = real["floating_population"]
    real_age_no_teen = normalize_excl_teen(real_fp["age_pct"])

    sims, sim_labels, sim_headers = [], [], []
    for fname, label, header in SIM_FILES:
        s = load_sim(fname, args.target)
        if s is not None:
            sims.append(s)
            sim_labels.append(label)
            sim_headers.append(header)

    if not sims:
        names = ", ".join(f[0] for f in SIM_FILES)
        print(f"outputs/ 안에 {names} 중 아무것도 없습니다. 먼저 visit_simulation*.py를 실행하세요.")
        return

    md = ["# 동대문패션타운 관광특구 — 시뮬레이션 비교 결과\n"]
    md.append(f"실제 데이터 기준 분기: {real_fp['quarter']} (서울열린데이터광장 서울시 상권분석서비스)\n")

    md.append("## ① 실험 설정 표\n")
    md.append(setup_table(sims, sim_headers, real_fp["total_floating_population"],
                           real["stores"]["store_counts"], args.model, args.target))
    md.append("")

    md.append("## ② 분포 비교 표 (연령대는 10대 포함 원본)\n")
    dist_specs = [("연령대", "age_pct", AGE_ORDER), ("성별", "gender_pct", GENDER_ORDER),
                  ("시간대", "time_pct", TIME_ORDER), ("요일", "day_pct", DAY_ORDER)]
    for title, key, order in dist_specs:
        md.append(f"### {title}\n")
        md.append(dist_table(title, key, real_fp[key], sims, order, sim_labels))
        md.append("")

    if len(sims) >= 2:
        md.append("## ③ 시뮬레이션 간 MAE 비교\n")
        md.append("| 항목 | " + " | ".join(f"{l} MAE" for l in sim_labels) + " |")
        md.append("|---|" + "---|" * len(sim_labels))
        for title, key, order in dist_specs:
            row_vals = [mae(real_fp[key], s[key], order) for s in sims]
            md.append(f"| {title} | " + " | ".join(str(v) for v in row_vals) + " |")
        md.append("")

    md.append("## ④ 10대 제외 정규화 — 연령대 오차 변화\n")
    md.append(
        "10대 구간을 빼고 나머지 5개 구간(20대~60대+)만 다시 100%로 정규화했을 때 "
        "연령대 오차(MAE)가 어떻게 바뀌는지 본다.\n"
    )
    md.append(dist_table("연령대(10대 제외 정규화)", "age_pct_no_teen", real_age_no_teen,
                          sims, AGE_ORDER_NO_TEEN, sim_labels))
    md.append("")
    md.append("| 조건 | 연령대 MAE |")
    md.append("|---|---|")
    for label, s in zip(sim_labels, sims):
        md.append(f"| {label}, 원본(10대 포함) | {mae(real_fp['age_pct'], s['age_pct'], AGE_ORDER)} |")
        md.append(f"| {label}, 10대 제외 정규화 | {mae(real_age_no_teen, s['age_pct_no_teen'], AGE_ORDER_NO_TEEN)} |")
    md.append("")

    out_path = OUTPUT_DIR / "comparison_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
