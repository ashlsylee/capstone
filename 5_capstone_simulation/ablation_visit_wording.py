"""
대조실험 — "방문"이라는 단어를 다른 표현으로 바꾸면 요일·시간대 편향이 줄어드는지 확인.

시뮬레이션 3 결과, "이 상권을 방문할지"라는 표현이 LLM에게 "주말 나들이"를 강하게
연상시켜서 요일특성 정보·반편향 지시문을 넣어도 토요일 쏠림이 잘 안 풀렸다(82.8%→76.8%,
거의 그대로). "방문" 대신 다른 동사/명사를 쓰면 이 연상이 줄어드는지 확인하기 위한
저비용 대조실험이다.

같은 300명(seed=42, 시뮬1~3과 동일한 셔플이지만 처음 300명만 사용 — 본실험과 표본이
겹칠 수 있음, 진단용이라 문제 없음)에게 표현만 바꿔가며 물어본다. 상권 정보(시뮬레이션 3과
동일: 기본 정보 + 1단계 결과 + 요일특성)와 반편향 지시문은 모든 변형에서 동일하게 유지하고,
"방문할지"에 해당하는 동사구만 바꾼다. target 채우기가 아니라 고정 표본이라 비용이 작다
(변형 5개 x 300명 = 1,500명 질의, batch-size 25 기준 60회 호출).

실행:
    python ablation_visit_wording.py --n 300 --batch-size 25
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

load_dotenv(BASE_DIR / ".env")

TIME_SLOTS = ["00-06", "06-11", "11-14", "14-17", "17-21", "21-24"]
DAYS = ["월", "화", "수", "목", "금", "토", "일"]
SEED = 42

# 테스트할 표현들. {verb}는 "그 인물이 실제로 이 사람이라면 향후 30일 내 이 상권{verb}"에
# 들어간다(조사 포함, 동사별로 자연스러운 조사가 다름). 키(variant id)는 결과 CSV의 variant
# 컬럼과 콘솔 출력에 쓰인다.
WORDING_VARIANTS = {
    "visit": "을 방문할지",                            # 기존(시뮬1~3) 표현 — 베이스라인
    "drop_by": "에 들를 일이 있을지",                    # 볼일 있어서 잠깐 들르는 느낌
    "errand": "에 가서 물건을 사거나 볼일을 볼지",         # 쇼핑/용무 목적 명시
    "use": "을 이용할지",                               # 중립적 "이용"
    "go": "에 갈지",                                    # 가장 단순/중립적인 "가다"
}


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


def build_system_prompt(verb: str, district_context: str) -> str:
    return f"""당신은 상권 방문 시뮬레이터입니다. 아래는 대상 상권 정보입니다.

{district_context}

위 상권 정보와 아래 주어지는 각 인물의 특성을 함께 고려해, 그 인물이 실제로 이 사람이라면
향후 30일 내 이 상권{verb}(Y/N) 판단하세요. 그렇다면({{"v":"Y"}}) 주로 몇 시경일지와
무슨 요일일지도 판단하세요:
- 시간대는 아래 6개 구간 중 하나로: {", ".join(TIME_SLOTS)}
- 요일은 아래 7개 중 하나로: {", ".join(DAYS)}
아니라면({{"v":"N"}}) 시간대·요일은 비워도 됩니다.

※ 상권 정보는 '이곳이 어떤 장소인지'에 대한 설명일 뿐입니다. 인물의 성향에 비추어 스스로
추론하세요. 실제 방문객의 연령·성별·시간대·요일별 분포나 유동인구 통계를 맞추려고 하지
마세요. 각 인물은 서로 독립적으로 판단하세요.

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


def run_variant(client, model: str, variant_id: str, verb: str, district_context: str,
                 sample: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    system_prompt = build_system_prompt(verb, district_context)
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
                "variant": variant_id,
                "uuid": row.uuid,
                "sex": row.sex,
                "age": int(row.age),
                "age_group": age_to_group(int(row.age)),
                "visit": visit,
                "time_slot": d.get("t") if visit else None,
                "day_of_week": d.get("d") if visit else None,
            })
        print(f"  [{variant_id}] batch {b + 1}/{n_batches} 완료")
    return pd.DataFrame(results)


def run(n: int, batch_size: int, model: str):
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다 (.env 또는 환경변수).")
    client = OpenAI(api_key=api_key)

    personas = pd.read_parquet(PERSONA_PATH)
    idx = list(range(len(personas)))
    random.seed(SEED)
    random.shuffle(idx)
    sample = personas.iloc[idx[:n]].reset_index(drop=True)

    district_context = build_district_context()

    out_path = OUTPUT_DIR / "ablation_visit_wording_results.csv"
    prior_df = pd.read_csv(out_path) if out_path.exists() else None
    done_variants = set(prior_df["variant"].unique()) if prior_df is not None else set()

    all_parts = [prior_df] if prior_df is not None else []
    for variant_id, verb in WORDING_VARIANTS.items():
        if variant_id in done_variants:
            print(f"[{variant_id}] 이미 완료됨, 건너뜀")
            continue
        print(f"\n=== 표현 실험: {variant_id} ('...{verb}') ===")
        df = run_variant(client, model, variant_id, verb, district_context, sample, batch_size)
        all_parts.append(df)
        combined = pd.concat(all_parts, ignore_index=True)
        OUTPUT_DIR.mkdir(exist_ok=True)
        combined.to_csv(out_path, index=False, encoding="utf-8-sig")  # 변형마다 즉시 저장(중단 대비)

    combined = pd.concat(all_parts, ignore_index=True)
    print(f"\n=== 전체 종료 ===")
    for variant_id in WORDING_VARIANTS:
        sub = combined[combined["variant"] == variant_id]
        if len(sub) == 0:
            continue
        rate = sub["visit"].mean() * 100
        sat_share = (sub.loc[sub["visit"], "day_of_week"] == "토").mean() * 100 if sub["visit"].any() else 0
        print(f"[{variant_id}] n={len(sub)} 수락률={rate:.1f}% 토요일비중={sat_share:.1f}%")
    print(f"\n결과 저장: {out_path}")
    print("python compare_wording.py 로 실제 요일 분포와 비교하세요.")


def main():
    parser = argparse.ArgumentParser(description="'방문' 표현 대조실험")
    parser.add_argument("--n", type=int, default=300, help="변형당 표본 크기")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()
    run(args.n, args.batch_size, args.model)


if __name__ == "__main__":
    main()
