"""
ablation_visit_wording.py 결과를 실제 요일·시간대 분포와 비교해서 어떤 표현이
가장 편향을 줄였는지 순위를 매긴다. API 호출 없음.

실행:
    python compare_wording.py
"""
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REAL_DIST_PATH = BASE_DIR / "data" / "real_distribution.json"

TIME_ORDER = ["00-06", "06-11", "11-14", "14-17", "17-21", "21-24"]
DAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]

WORDING_LABELS = {
    "visit": "방문할지 (기존)",
    "drop_by": "들를 일이 있을지",
    "errand": "가서 물건을 사거나 볼일을 볼지",
    "use": "이용할지",
    "go": "갈지",
}


def mae(real: dict, sim: dict, order: list) -> float:
    return round(sum(abs(real[k] - sim[k]) for k in order) / len(order), 2)


def main():
    path = OUTPUT_DIR / "ablation_visit_wording_results.csv"
    if not path.exists():
        print("outputs/ablation_visit_wording_results.csv가 없습니다. 먼저 ablation_visit_wording.py를 실행하세요.")
        return
    df = pd.read_csv(path)
    real = json.loads(REAL_DIST_PATH.read_text(encoding="utf-8"))["floating_population"]

    rows = []
    md = ["# '방문' 표현 대조실험 결과\n", "| 표현 | 표본 | 수락률 | 시간대 MAE | 요일 MAE | 토요일 비중 |",
          "|---|---|---|---|---|---|"]
    for variant, label in WORDING_LABELS.items():
        sub = df[df["variant"] == variant]
        if len(sub) == 0:
            continue
        accepted = sub[sub["visit"] == True]
        n = len(sub)
        rate = len(accepted) / n * 100 if n else 0
        time_pct = (accepted["time_slot"].value_counts(normalize=True) * 100).reindex(TIME_ORDER).fillna(0).to_dict()
        day_pct = (accepted["day_of_week"].value_counts(normalize=True) * 100).reindex(DAY_ORDER).fillna(0).to_dict()
        time_mae = mae(real["time_pct"], time_pct, TIME_ORDER)
        day_mae = mae(real["day_pct"], day_pct, DAY_ORDER)
        sat_share = day_pct.get("토", 0)
        rows.append((variant, label, n, rate, time_mae, day_mae, sat_share))
        md.append(f"| {label} | {n} | {rate:.1f}% | {time_mae} | {day_mae} | {sat_share:.1f}% |")

    md.append("")
    md.append(f"(참고: 실제 토요일 비중 {real['day_pct']['토']:.1f}%, 실제 요일 분포는 "
              f"월-금 14.8-15.9%, 토·일 11.4-11.7%로 평일이 더 높음)")

    print("\n".join(md))

    if rows:
        best = min(rows, key=lambda r: r[5])  # day_mae 기준 최적
        print(f"\n요일 MAE 기준 최적 표현: {best[1]} (MAE {best[5]})")

    out_path = OUTPUT_DIR / "wording_comparison_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
