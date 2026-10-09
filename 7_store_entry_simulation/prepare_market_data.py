"""
실제 상권 데이터(2025년 1~4분기)를 정리해서 data/market_facts.json 으로 저장한다.
API 호출 없음 (재현 가능한 전처리 단계).

입력 원본 (data/raw/, 모두 상권_코드=3001493 동대문패션타운 관광특구 행만 추출해 둔 것):
- dongdaemun_floating_population_raw.csv
    서울시 상권분석서비스(추정유동인구-상권) OA-15568 — 연령·성별·시간대·요일별 유동인구
- dongdaemun_sales_2025_raw.csv
    서울시 상권분석서비스(추정매출-상권) OA-15572 2025년 — 업종별 매출 금액·건수
    (시간대·요일·연령·성별 분해 포함)
- dongdaemun_store_counts_2025_raw.csv
    서울시 상권분석서비스(점포-상권) OA-15577 2025년 — 업종별 점포 수
    (similr_induty_stor_co = 일반 점포 + 프랜차이즈 점포)

모든 값을 같은 기간(2025년 1~4분기 합계)으로 맞춘다. 시뮬레이션 에이전트의 연령·성별·
시간대·요일은 이 유동인구 분포로 뽑고, 신규 점포의 방문수·매출 환산은 대상 업종(common.TARGET_INDUSTRY, 현재 슈퍼마켓)의 매출 건수·
객단가로 한다.

실행:
    python prepare_market_data.py
"""
import json
from pathlib import Path

import pandas as pd

from common import TARGET_INDUSTRY

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_PATH = BASE_DIR / "data" / "market_facts.json"

DISTRICT_CODE = 3001493
QUARTERS = [20251, 20252, 20253, 20254]

AGES = ["10대", "20대", "30대", "40대", "50대", "60대+"]
TIMES = ["00-06", "06-11", "11-14", "14-17", "17-21", "21-24"]
DAYS = ["월", "화", "수", "목", "금", "토", "일"]
DAY_FULL = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

POP_AGE_COLS = ["연령대_10_유동인구_수", "연령대_20_유동인구_수", "연령대_30_유동인구_수",
                "연령대_40_유동인구_수", "연령대_50_유동인구_수", "연령대_60_이상_유동인구_수"]
POP_TIME_COLS = ["시간대_00_06_유동인구_수", "시간대_06_11_유동인구_수", "시간대_11_14_유동인구_수",
                 "시간대_14_17_유동인구_수", "시간대_17_21_유동인구_수", "시간대_21_24_유동인구_수"]
POP_DAY_COLS = [f"{d}_유동인구_수" for d in DAY_FULL]

# 원본 매출 CSV의 시간대 '건수' 컬럼명은 서울시 원본 표기 그대로 ("시간대_건수~06_매출_건수" 등)
SALES_TIME_AMT_COLS = ["시간대_00~06_매출_금액", "시간대_06~11_매출_금액", "시간대_11~14_매출_금액",
                       "시간대_14~17_매출_금액", "시간대_17~21_매출_금액", "시간대_21~24_매출_금액"]
SALES_TIME_CNT_COLS = ["시간대_건수~06_매출_건수", "시간대_건수~11_매출_건수", "시간대_건수~14_매출_건수",
                       "시간대_건수~17_매출_건수", "시간대_건수~21_매출_건수", "시간대_건수~24_매출_건수"]
SALES_DAY_AMT_COLS = [f"{d}_매출_금액" for d in DAY_FULL]
SALES_DAY_CNT_COLS = [f"{d}_매출_건수" for d in DAY_FULL]
SALES_AGE_CNT_COLS = ["연령대_10_매출_건수", "연령대_20_매출_건수", "연령대_30_매출_건수",
                      "연령대_40_매출_건수", "연령대_50_매출_건수", "연령대_60_이상_매출_건수"]


def pct(values: list) -> list:
    total = sum(values)
    return [round(v / total * 100, 2) for v in values]


def summarize_floating(df: pd.DataFrame) -> dict:
    d = df[(df["상권_코드"] == DISTRICT_CODE) & (df["기준_년분기_코드"].isin(QUARTERS))]
    assert sorted(d["기준_년분기_코드"].unique()) == QUARTERS, "유동인구 원본에 2025년 4개 분기가 모두 있어야 함"
    s = d.sum(numeric_only=True)
    return {
        "total_2025": int(s["총_유동인구_수"]),
        "quarterly": {str(q): int(d.loc[d["기준_년분기_코드"] == q, "총_유동인구_수"].sum()) for q in QUARTERS},
        "age_pct": dict(zip(AGES, pct([s[c] for c in POP_AGE_COLS]))),
        "gender_pct": dict(zip(["남", "여"], pct([s["남성_유동인구_수"], s["여성_유동인구_수"]]))),
        "time_pct": dict(zip(TIMES, pct([s[c] for c in POP_TIME_COLS]))),
        "day_pct": dict(zip(DAYS, pct([s[c] for c in POP_DAY_COLS]))),
    }


def summarize_industry_sales(sales: pd.DataFrame, stores: pd.DataFrame, industry: str) -> dict:
    d = sales[(sales["상권_코드"] == DISTRICT_CODE) & (sales["서비스_업종_코드_명"] == industry)
              & (sales["기준_년분기_코드"].isin(QUARTERS))]
    st = stores[(stores["trdar_cd"] == DISTRICT_CODE) & (stores["svc_induty_cd_nm"] == industry)
                & (stores["stdr_yyqu_cd"].isin(QUARTERS))]
    assert len(d) == 4 and len(st) == 4, f"{industry}: 2025년 4개 분기 매출/점포 행이 모두 있어야 함"
    s = d.sum(numeric_only=True)
    n_stores = round(st["similr_induty_stor_co"].mean(), 1)  # 분기별 점포 수 평균
    amount, count = int(s["당월_매출_금액"]), int(s["당월_매출_건수"])
    time_cnt = [s[c] for c in SALES_TIME_CNT_COLS]
    time_amt = [s[c] for c in SALES_TIME_AMT_COLS]
    day_cnt = [s[c] for c in SALES_DAY_CNT_COLS]
    day_amt = [s[c] for c in SALES_DAY_AMT_COLS]
    return {
        "industry": industry,
        "stores": n_stores,
        "stores_by_quarter": {str(r.stdr_yyqu_cd): int(r.similr_induty_stor_co) for r in st.itertuples()},
        "franchise_stores_by_quarter": {str(r.stdr_yyqu_cd): int(r.frc_stor_co) for r in st.itertuples()},
        "sales_amount_2025": amount,
        "transactions_2025": count,
        "avg_ticket": round(amount / count),
        "sales_per_store_2025": round(amount / n_stores),
        "transactions_per_store_2025": round(count / n_stores),
        "quarterly": {
            str(r.기준_년분기_코드): {"amount": int(r.당월_매출_금액), "transactions": int(r.당월_매출_건수)}
            for r in d.itertuples()
        },
        "time": {
            t: {"transactions": int(c), "amount": int(a), "tx_pct": p, "avg_ticket": round(a / c) if c else None}
            for t, c, a, p in zip(TIMES, time_cnt, time_amt, pct(time_cnt))
        },
        "day": {
            dname: {"transactions": int(c), "amount": int(a), "tx_pct": p, "avg_ticket": round(a / c) if c else None}
            for dname, c, a, p in zip(DAYS, day_cnt, day_amt, pct(day_cnt))
        },
        "age_tx_pct": dict(zip(AGES, pct([s[c] for c in SALES_AGE_CNT_COLS]))),
        "gender_tx_pct": dict(zip(["남", "여"], pct([s["남성_매출_건수"], s["여성_매출_건수"]]))),
    }


def main():
    pop = pd.read_csv(RAW_DIR / "dongdaemun_floating_population_raw.csv", encoding="utf-8-sig")
    sales = pd.read_csv(RAW_DIR / "dongdaemun_sales_2025_raw.csv", encoding="utf-8-sig")
    stores = pd.read_csv(RAW_DIR / "dongdaemun_store_counts_2025_raw.csv", encoding="utf-8-sig")

    latest_sales = sales[sales["기준_년분기_코드"] == max(QUARTERS)]
    industry_rank = (latest_sales.groupby("서비스_업종_코드_명")["당월_매출_금액"].sum()
                     .sort_values(ascending=False))

    facts = {
        "district_code": DISTRICT_CODE,
        "district_name": "동대문패션타운 관광특구",
        "period": "2025년 1~4분기 합계",
        "source": {
            "floating_population": "서울시 상권분석서비스(추정유동인구-상권) OA-15568, data.seoul.go.kr",
            "sales": "서울시 상권분석서비스(추정매출-상권) OA-15572, data.seoul.go.kr",
            "stores": "서울시 상권분석서비스(점포-상권) OA-15577, data.seoul.go.kr",
        },
        "floating_population": summarize_floating(pop),
        "target_industry": summarize_industry_sales(sales, stores, TARGET_INDUSTRY),
        "industry_sales_rank_2025q4": [
            {"industry": k, "sales_amount": int(v)} for k, v in industry_rank.head(10).items()
        ],
    }
    OUT_PATH.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")

    t = facts["target_industry"]
    print(f"저장: {OUT_PATH}")
    print(f"유동인구 2025 합계: {facts['floating_population']['total_2025']:,}명")
    print(f"{TARGET_INDUSTRY} 점포 {t['stores']}개 | 매출 {t['sales_amount_2025']:,}원 | "
          f"건수 {t['transactions_2025']:,}건 | 객단가 {t['avg_ticket']:,}원 | "
          f"점포당 연매출 {t['sales_per_store_2025']:,}원")


if __name__ == "__main__":
    main()
