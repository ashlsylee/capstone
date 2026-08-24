"""
LLM이 판단한 "방문 여부·시간대·선호 매장"(visit_simulation_v2_results.csv)을 바탕으로
30일치 매장 방문 로그를 만든다. API 호출 없음(무료), 순수 코드 시뮬레이션.

가정(실측 데이터 아님, 설계상 가정):
- 방문(Y) 판정된 사람은 30일 중 1~4일 방문 (가중치: 1일 55%, 2일 25%, 3일 12%, 4일 8%)
- 방문 요일은 30일 중 무작위 선택
- 매장·시간대는 LLM이 판단한 값을 그대로 유지 (매 방문 동일)

실행:
    python expand_30day.py --input outputs/visit_simulation_v2_results.csv
"""
import argparse
import random
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

STORES = [
    {"code": "mart", "name": "마트", "type": "마트", "lat": 37.56867, "lon": 127.01467},
    {"code": "cvs", "name": "편의점", "type": "편의점", "lat": 37.57047, "lon": 127.01014},
    {"code": "cafe", "name": "카페", "type": "카페", "lat": 37.56822, "lon": 127.01127},
    {"code": "hotel", "name": "호텔", "type": "호텔", "lat": 37.56552, "lon": 127.01410},
    {"code": "dept", "name": "백화점(던던 동대문점)", "type": "백화점", "lat": 37.56957, "lon": 127.01071},
]
STORE_BY_CODE = {s["code"]: s for s in STORES}

VISIT_COUNT_CHOICES = [1, 2, 3, 4]
VISIT_COUNT_WEIGHTS = [55, 25, 12, 8]


def expand(df: pd.DataFrame, days: int, seed: int) -> pd.DataFrame:
    random.seed(seed)
    accepted = df[df["visit"] == True].copy()
    accepted = accepted.dropna(subset=["store_code"])
    accepted = accepted[accepted["store_code"].isin(STORE_BY_CODE)]

    logs = []
    for _, row in accepted.iterrows():
        n_visits = random.choices(VISIT_COUNT_CHOICES, weights=VISIT_COUNT_WEIGHTS)[0]
        visit_days = sorted(random.sample(range(1, days + 1), k=min(n_visits, days)))
        store = STORE_BY_CODE[row["store_code"]]
        for day in visit_days:
            logs.append({
                "day": day,
                "uuid": row["uuid"],
                "sex": row["sex"],
                "age": row["age"],
                "age_group": row["age_group"],
                "store_code": store["code"],
                "store_name": store["name"],
                "store_type": store["type"],
                "time_slot": row["time_slot"],
            })
    return pd.DataFrame(logs)


def summarize(log: pd.DataFrame) -> pd.DataFrame:
    summary = (
        log.groupby(["store_code", "store_name", "store_type"])
        .agg(총_방문수=("uuid", "count"), 순방문자수=("uuid", "nunique"))
        .reset_index()
        .sort_values("총_방문수", ascending=False)
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="30일 매장 방문 로그 생성 (API 호출 없음)")
    parser.add_argument("--input", default=str(OUTPUT_DIR / "visit_simulation_v2_results.csv"))
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    log = expand(df, args.days, args.seed)

    OUTPUT_DIR.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / "daily_visit_log.csv"
    log.to_csv(log_path, index=False, encoding="utf-8-sig")

    summary = summarize(log)
    summary_path = OUTPUT_DIR / "store_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"방문(Y) 응답자: {df['visit'].sum():,}명")
    print(f"{args.days}일간 총 방문 이벤트: {len(log):,}건")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"저장: {log_path}")
    print(f"저장: {summary_path}")


if __name__ == "__main__":
    main()
