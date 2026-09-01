"""
ablation_fashion_occupation.py 결과(패션/섬유/액세서리 직업군만)를 시뮬레이션 3(일반 인구
무작위 표본) 및 실제 요일 분포와 비교한다. API 호출 없음.

실행:
    python compare_fashion_occupation.py
"""
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REAL_DIST_PATH = BASE_DIR / "data" / "real_distribution.json"

DAY_ORDER = ["월", "화", "수", "목", "금", "토", "일"]
TIME_ORDER = ["00-06", "06-11", "11-14", "14-17", "17-21", "21-24"]


def mae(real: dict, sim: dict, order: list) -> float:
    return round(sum(abs(real[k] - sim[k]) for k in order) / len(order), 2)


def dist(df: pd.DataFrame, col: str, order: list) -> dict:
    return (df[col].value_counts(normalize=True) * 100).reindex(order).fillna(0).round(2).to_dict()


def main():
    fash_path = OUTPUT_DIR / "ablation_fashion_occupation_results.csv"
    sim3_path = OUTPUT_DIR / "visit_simulation3_results.csv"
    if not fash_path.exists():
        print("outputs/ablation_fashion_occupation_results.csv가 없습니다. 먼저 ablation_fashion_occupation.py를 실행하세요.")
        return
    if not sim3_path.exists():
        print("outputs/visit_simulation3_results.csv가 없습니다.")
        return

    real = json.loads(REAL_DIST_PATH.read_text(encoding="utf-8"))["floating_population"]

    fash = pd.read_csv(fash_path)
    fash_acc = fash[fash["visit"] == True]

    sim3 = pd.read_csv(sim3_path)
    sim3_acc = sim3[sim3["visit"] == True].iloc[:500]  # compare_results.py와 동일하게 처음 500명만

    fash_day = dist(fash_acc, "day_of_week", DAY_ORDER)
    sim3_day = dist(sim3_acc, "day_of_week", DAY_ORDER)
    fash_time = dist(fash_acc, "time_slot", TIME_ORDER)
    sim3_time = dist(sim3_acc, "time_slot", TIME_ORDER)

    md = ["# 패션/섬유/액세서리 직업군 vs 일반 인구(시뮬3) vs 실제\n"]
    md.append(f"패션 직업군 표본: {len(fash)}명 질의, {len(fash_acc)}명 수락 "
              f"(수락률 {len(fash_acc)/len(fash)*100:.1f}%)\n")

    md.append("## 요일 분포\n")
    md.append("| 요일 | 실제(%) | 시뮬3-일반인구(%) | 패션직업군(%) |")
    md.append("|---|---|---|---|")
    for d in DAY_ORDER:
        md.append(f"| {d} | {real['day_pct'][d]:.2f} | {sim3_day[d]:.2f} | {fash_day[d]:.2f} |")
    mae_sim3_day = mae(real["day_pct"], sim3_day, DAY_ORDER)
    mae_fash_day = mae(real["day_pct"], fash_day, DAY_ORDER)
    md.append(f"| **MAE** |  | **{mae_sim3_day}** | **{mae_fash_day}** |")
    md.append("")

    md.append("## 시간대 분포\n")
    md.append("| 시간대 | 실제(%) | 시뮬3-일반인구(%) | 패션직업군(%) |")
    md.append("|---|---|---|---|")
    for t in TIME_ORDER:
        md.append(f"| {t} | {real['time_pct'][t]:.2f} | {sim3_time[t]:.2f} | {fash_time[t]:.2f} |")
    mae_sim3_time = mae(real["time_pct"], sim3_time, TIME_ORDER)
    mae_fash_time = mae(real["time_pct"], fash_time, TIME_ORDER)
    md.append(f"| **MAE** |  | **{mae_sim3_time}** | **{mae_fash_time}** |")
    md.append("")

    md.append(f"토요일 비중: 실제 {real['day_pct']['토']:.1f}% / 시뮬3(일반인구) {sim3_day['토']:.1f}% "
              f"/ 패션직업군 {fash_day['토']:.1f}%")
    md.append(f"\n요일 MAE 변화: {mae_sim3_day} → {mae_fash_day} "
              f"({'개선' if mae_fash_day < mae_sim3_day else '악화'} {round(abs(mae_sim3_day-mae_fash_day),2)})")

    print("\n".join(md))
    out_path = OUTPUT_DIR / "fashion_occupation_comparison_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
