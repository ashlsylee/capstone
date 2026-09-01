"""
실제 동대문패션타운 관광특구 유동인구/점포 데이터를 정리해서
data/real_distribution.json 으로 저장한다. API 호출 없음 (재현 가능한 전처리 단계).

입력 원본:
- data/raw/dongdaemun_floating_population_raw.csv
    서울열린데이터광장 "서울시 상권분석서비스(추정유동인구-상권)" 원본 CSV.
    상권_코드=3001493 (동대문패션타운 관광특구), 기준_년분기_코드별 연령대/성별/시간대 유동인구.
- data/raw/dongdaemun_store_counts_raw.xlsx
    서울시 상권분석서비스(점포-상권) 원본. 상권_코드=3001493, 서비스_업종_코드별 점포 수.

실행:
    python prepare_real_data.py
"""
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_PATH = BASE_DIR / "data" / "real_distribution.json"

DISTRICT_CODE = 3001493
DISTRICT_NAME = "동대문패션타운 관광특구"

AGE_COL_MAP = {
    "연령대_10_유동인구_수": "10대",
    "연령대_20_유동인구_수": "20대",
    "연령대_30_유동인구_수": "30대",
    "연령대_40_유동인구_수": "40대",
    "연령대_50_유동인구_수": "50대",
    "연령대_60_이상_유동인구_수": "60대+",
}
TIME_COL_MAP = {
    "시간대_00_06_유동인구_수": "00-06",
    "시간대_06_11_유동인구_수": "06-11",
    "시간대_11_14_유동인구_수": "11-14",
    "시간대_14_17_유동인구_수": "14-17",
    "시간대_17_21_유동인구_수": "17-21",
    "시간대_21_24_유동인구_수": "21-24",
}
DAY_COL_MAP = {
    "월요일_유동인구_수": "월",
    "화요일_유동인구_수": "화",
    "수요일_유동인구_수": "수",
    "목요일_유동인구_수": "목",
    "금요일_유동인구_수": "금",
    "토요일_유동인구_수": "토",
    "일요일_유동인구_수": "일",
}

# 음식점 = 한식/중식/일식/양식/패스트푸드/치킨/분식/제과점 (호프-간이주점은 주류업으로 별도 분류해 제외)
FOOD_CODES = {
    "한식음식점", "중식음식점", "일식음식점", "양식음식점",
    "패스트푸드점", "치킨전문점", "분식전문점", "제과점",
}
MART_CODES = {"슈퍼마켓"}
CVS_CODES = {"편의점"}
CAFE_CODES = {"커피-음료"}


def load_floating_population():
    df = pd.read_csv(RAW_DIR / "dongdaemun_floating_population_raw.csv", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["상권_코드"].astype(str) == str(DISTRICT_CODE)].copy()
    num_cols = [c for c in df.columns if c not in
                ("기준_년분기_코드", "상권_구분_코드", "상권_구분_코드_명", "상권_코드", "상권_코드_명")]
    for c in num_cols:
        df[c] = df[c].astype(int)
    latest_q = df["기준_년분기_코드"].astype(int).max()
    row = df[df["기준_년분기_코드"].astype(int) == latest_q].iloc[0]

    total = int(row["총_유동인구_수"])
    age_pct = {label: round(int(row[col]) / total * 100, 2) for col, label in AGE_COL_MAP.items()}
    time_pct = {label: round(int(row[col]) / total * 100, 2) for col, label in TIME_COL_MAP.items()}
    day_pct = {label: round(int(row[col]) / total * 100, 2) for col, label in DAY_COL_MAP.items()}
    gender_pct = {
        "남": round(int(row["남성_유동인구_수"]) / total * 100, 2),
        "여": round(int(row["여성_유동인구_수"]) / total * 100, 2),
    }
    return {
        "quarter": str(latest_q),
        "total_floating_population": total,
        "age_pct": age_pct,
        "gender_pct": gender_pct,
        "time_pct": time_pct,
        "day_pct": day_pct,
    }


def load_store_counts():
    df = pd.read_excel(RAW_DIR / "dongdaemun_store_counts_raw.xlsx", header=None)
    # 컬럼 순서(헤더 없는 원본): 분기,상권구분코드,상권구분코드명,상권코드,상권코드명,업종코드,업종명,점포수,...
    df.columns = ["quarter", "biz_type_code", "biz_type_name", "district_code",
                  "district_name", "service_code", "service_name", "store_count"] + \
                 [f"col{i}" for i in range(8, df.shape[1])]
    df = df[df["district_code"].astype(str) == str(DISTRICT_CODE)]
    latest_q = df["quarter"].astype(int).max()
    df = df[df["quarter"].astype(int) == latest_q]

    def sum_for(codes):
        return int(df[df["service_name"].isin(codes)]["store_count"].sum())

    counts = {
        "마트": sum_for(MART_CODES),
        "편의점": sum_for(CVS_CODES),
        "카페": sum_for(CAFE_CODES),
        "음식점": sum_for(FOOD_CODES),
        "백화점": 0,  # 아래 note 참고
    }
    note = (
        "서울시 상권분석서비스(점포-상권) 원본에는 이 상권에 '백화점' 업종 코드가 존재하지 않음 "
        "(동대문패션타운은 도소매 상가 중심 상권). 시뮬레이션 프롬프트에는 데이터에 없는 시설을 "
        "임의로 채워 넣지 않고 '백화점 0개'라고 있는 그대로 서술함."
    )
    return {"quarter": str(latest_q), "store_counts": counts, "note": note}


def main():
    fp = load_floating_population()
    stores = load_store_counts()

    result = {
        "district_code": DISTRICT_CODE,
        "district_name": DISTRICT_NAME,
        "source": {
            "floating_population": "서울열린데이터광장 - 서울시 상권분석서비스(추정유동인구-상권), data.seoul.go.kr",
            "store_counts": "서울열린데이터광장 - 서울시 상권분석서비스(점포-상권), data.seoul.go.kr",
        },
        "floating_population": fp,
        "stores": stores,
    }

    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"=== 실제 데이터 정리 완료 ({DISTRICT_NAME}, {fp['quarter']}분기 기준) ===")
    print(f"총 유동인구: {fp['total_floating_population']:,}")
    print("연령대(%):", fp["age_pct"])
    print("성별(%):", fp["gender_pct"])
    print("시간대(%):", fp["time_pct"])
    print("요일(%):", fp["day_pct"])
    print("\n점포 수 ({}분기):".format(stores["quarter"]), stores["store_counts"])
    print(stores["note"])
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
