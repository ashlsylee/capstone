"""
대조실험 — 페르소나 풀을 "패션·섬유·액세서리 도소매 관련 직업"으로만 좁혔을 때
요일 편향(토요일 쏠림)이 줄어드는지 확인.

배경: 시뮬레이션 3 결과를 직업별로 나눠보니(README 참고), 판매·상인·자영업 계열 직업을
가진 페르소나는 일반 직업군보다 토요일 비중이 낮았다(59.1% vs 77.6%, n=22 vs 478).
동대문패션타운은 의류·원단·액세서리 도소매가 핵심인 상권인데, 일반 인구 100만 명 무작위
표본에는 이런 특정 직업군이 소수(4~5%)만 있어서 전체 결과가 여전히 일반 인구의 "주말
나들이" 패턴에 묻힌 것으로 보인다. 이 대조실험은 아예 페르소나 풀을 그 직업군으로만
필터링해서, 그 안에서는 요일 패턴이 실제(평일이 더 붐빔)에 가까워지는지 방향성만 확인한다
(논문 본 실험이 아니라 진단용 실험 — target 500 채우기가 아니라 고정 표본 n명).

상권 정보·프롬프트 형식은 시뮬레이션 3과 완전히 동일(요일특성 필드 + 반편향 지시 포함),
바뀌는 건 페르소나 풀 필터링뿐이다.

직업 필터 기준: 페르소나 풀(100만 명)에서 의류·섬유·신발·가방·액세서리 관련 직업만
골랐다(현직만, "전직 ...현재 구직중"은 제외). "노점 및 이동 판매원"은 특정 품목을
명시하지 않는 범용 카테고리라 제외했다(포함 시 4,501명으로 표본을 지배해버림).

실행:
    python ablation_fashion_occupation.py --n 300 --batch-size 25
"""
import argparse
import json
import os
import random
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PERSONA_PATH = BASE_DIR / "data" / "nemotron_personas_korea_demo.parquet"
REAL_DIST_PATH = BASE_DIR / "data" / "real_distribution.json"
STAGE1_PATH = BASE_DIR / "data" / "stage1_district_facts.json"
OUTPUT_DIR = BASE_DIR / "outputs"
RESULT_NAME = "ablation_fashion_occupation_results.csv"

load_dotenv(BASE_DIR / ".env")

TIME_SLOTS = ["00-06", "06-11", "11-14", "14-17", "17-21", "21-24"]
DAYS = ["월", "화", "수", "목", "금", "토", "일"]
SEED = 42

FASHION_OCCUPATIONS = [
    "가방 및 신발 디자이너",
    "그 외 섬유 및 가죽 관련 기능 종사원",
    "그 외 직물·신발 관련 기계 조작원 및 조립원",
    "섬유 공정 개발 기술자 및 연구원",
    "섬유 및 의복 제품 생산 관리자",
    "섬유 및 펠트 관련 선별원",
    "섬유 소재 개발 기술자 및 연구원",
    "섬유 제조 기계 조작원",
    "섬유공학 시험원",
    "섬유기계 설치 및 정비원",
    "신발 및 구두 재단사",
    "신발 및 액세서리 판매원",
    "신발 재봉사",
    "신발 제조기 조작원 및 조립원",
    "액세서리 디자이너",
    "의류 판매원",
]


def build_district_context() -> str:
    """시뮬레이션 3과 동일: 기본 정보 + 1단계 결과(한줄요약/업종/방문목적) + 요일특성."""
    dist = json.loads(REAL_DIST_PATH.read_text(encoding="utf-8"))
    stores = dist["stores"]["store_counts"]
    facts = json.loads(STAGE1_PATH.read_text(encoding="utf-8"))
    biz_lines = "; ".join(f"{b['업종']}({b['운영시간']})" for b in facts["대표업종_운영시간"])
    purposes = ", ".join(facts["방문목적"])
    base = (
        f"{dist['district_name']}(상권 코드 {dist['district_code']})이라는 상권입니다. "
        f"이 상권 안의 매장 구성은 마트 {stores['마트']}개, 백화점 {stores['백화점']}개, "
        f"편의점 {stores['편의점']}개, 카페 {stores['카페']}개, 음식점 {stores['음식점']}개입니다."
    )
    extra = (
        f"{facts['한줄요약']}. "
        f"대표 업종과 운영시간은 다음과 같습니다: {biz_lines}. "
        f"주요 방문 목적은 {purposes}입니다. "
        f"요일별 특성: {facts['요일특성']}."
    )
    return base + " " + extra


SYSTEM_PROMPT_TEMPLATE = """당신은 상권 방문 시뮬레이터입니다. 아래는 대상 상권 정보입니다.

{district_context}

위 상권 정보와 아래 주어지는 각 인물의 특성을 함께 고려해, 그 인물이 실제로 이 사람이라면
향후 30일 내 이 상권을 방문할지(Y/N) 판단하세요. 방문한다면({{"v":"Y"}}) 주로 몇 시경 방문할지와
무슨 요일에 방문할지도 판단하세요:
- 시간대는 아래 6개 구간 중 하나로: {time_slots}
- 요일은 아래 7개 중 하나로: {days}
방문하지 않으면({{"v":"N"}}) 시간대·요일은 비워도 됩니다.

※ 상권 정보는 '이곳이 어떤 장소인지'에 대한 설명일 뿐입니다. 인물의 성향에 비추어 방문 여부와
시간대·요일을 스스로 추론하세요. 실제 방문객의 연령·성별·시간대·요일별 분포나 유동인구
통계를 맞추려고 하지 마세요. 각 인물은 서로 독립적으로 판단하세요.

※ "상권 방문"이라고 하면 습관적으로 "주말 나들이"를 떠올리기 쉬운데, 이런 일반적인
선입견에 의존하지 말고 위 상권 정보(특히 요일별 특성)에 근거해서 요일을 판단하세요.
업종·방문목적은 성별과 무관하니, 성별에 대한 고정관념에 의존하지 말고 인물 개개인의
특성만으로 판단하세요.

반드시 아래 형식의 JSON 배열만 출력하세요. 다른 설명·마크다운 없이 배열만:
[{{"i":0,"v":"Y","t":"17-21","d":"토"}},{{"i":1,"v":"N"}}, ...]
"""


def age_to_group(age: int) -> str:
    if age < 20:
        return "10대"
    if age < 30:
        return "20대"
    if age < 40:
        return "30대"
    if age < 50:
        return "40대"
    if age < 60:
        return "50대"
    return "60대+"


def format_persona_line(i: int, row) -> str:
    return (
        f"{i}) 나이:{int(row.age)} 성별:{row.sex} 직업:{row.occupation} "
        f"거주지:{row.district} 취미·성향:{row.hobbies_and_interests}"
    )


def safe_parse_batch(text: str, batch_size: int) -> list:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    return [{"i": i, "v": "N"} for i in range(batch_size)]


def run(n: int, batch_size: int, model: str):
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다 (.env 또는 환경변수).")
    client = OpenAI(api_key=api_key)

    personas = pd.read_parquet(PERSONA_PATH)
    pool = personas[personas["occupation"].isin(FASHION_OCCUPATIONS)].reset_index(drop=True)
    print(f"패션/섬유/액세서리 직업군 페르소나 풀: {len(pool):,}명")

    idx = list(range(len(pool)))
    random.seed(SEED)
    random.shuffle(idx)
    n = min(n, len(pool))
    sample = pool.iloc[idx[:n]].reset_index(drop=True)

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        district_context=build_district_context(),
        time_slots=", ".join(TIME_SLOTS),
        days=", ".join(DAYS),
    )

    results = []
    n_batches = (len(sample) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = sample.iloc[b * batch_size: (b + 1) * batch_size].reset_index(drop=True)
        lines = "\n".join(format_persona_line(i, row) for i, row in batch.iterrows())
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=800,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": lines},
            ],
        )
        decisions = safe_parse_batch(response.choices[0].message.content or "", len(batch))
        decision_map = {d.get("i"): d for d in decisions if isinstance(d, dict)}
        for i, row in batch.iterrows():
            d = decision_map.get(i, {"v": "N"})
            visit = str(d.get("v", "N")).upper().startswith("Y")
            results.append({
                "uuid": row.uuid,
                "sex": row.sex,
                "age": int(row.age),
                "age_group": age_to_group(int(row.age)),
                "occupation": row.occupation,
                "visit": visit,
                "time_slot": d.get("t") if visit else None,
                "day_of_week": d.get("d") if visit else None,
            })
        print(f"[batch {b + 1}/{n_batches}] 누적 {len(results)}명 처리")

    df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / RESULT_NAME
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    accepted = df[df["visit"] == True]
    rate = len(accepted) / len(df) * 100 if len(df) else 0
    sat_share = (accepted["day_of_week"] == "토").mean() * 100 if len(accepted) else 0
    print(f"\n=== 결과 (n={len(df)}, 패션/섬유/액세서리 직업군) ===")
    print(f"수락률: {rate:.1f}%")
    print(f"토요일 비중: {sat_share:.1f}%")
    print("\n요일 분포:")
    print(accepted["day_of_week"].value_counts(normalize=True).mul(100).round(1))
    print(f"\n결과 저장: {out_path}")
    print("python compare_fashion_occupation.py 로 시뮬레이션 3(일반 인구)와 비교하세요.")


def main():
    parser = argparse.ArgumentParser(description="패션/섬유/액세서리 직업군 페르소나 대조실험")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()
    run(args.n, args.batch_size, args.model)


if __name__ == "__main__":
    main()
