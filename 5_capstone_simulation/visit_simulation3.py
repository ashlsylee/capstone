"""
시뮬레이션 3 — 시뮬레이션 2 + 요일특성 반영 + 반편향(anti-stereotype) 지시.

시뮬레이션 2를 돌려본 결과, 오히려 시뮬레이션 1보다 요일·성별 오차가 커졌다(README/
outputs/comparison_report.md 참고). 원인 분석 후 아래 두 가지만 추가해서 시뮬레이션 3을
만든다. 페르소나 추출 순서(seed=42)·매장 구성·target(500)·질문 형식·매장 정보 문장은
시뮬레이션 1·2와 동일하게 유지한다 — 비교 가능성을 위해 이 두 가지 외에는 건드리지 않는다.

추가 ① 요일특성 필드 반영:
    시뮬레이션 2는 2단계 프롬프트 스펙("한줄요약+대표업종_운영시간+방문목적"만 사용)을
    그대로 따르느라 stage1_district_facts.json의 "요일특성" 필드(도매는 평일형, 소매·관광은
    상시형이라는 이중구조를 설명)를 넣지 않았다. 이 필드가 요일 판단에 가장 직접적으로
    관련된 정보라고 보고 시뮬레이션 3에서는 추가한다.

추가 ② 반편향 지시문 추가:
    시뮬레이션 2 결과, "상권 방문"을 물으면 실제 정보와 무관하게 "주말 나들이"로 판단하는
    쏠림이 오히려 더 심해졌고(토요일 74%→84%), "도매·야간" 서술이 성별 응답을 남성 쪽으로
    쏠리게 했다(성별 MAE 0.4→4.8). 실제 방문객 통계를 알려주는 게 아니라, LLM이 흔히 갖는
    "상권=주말 나들이" 선입견과 성별 고정관념에 의존하지 말고 주어진 사실 정보만으로
    판단하라는 지시문을 추가한다(SYSTEM_PROMPT_TEMPLATE의 마지막 문단 참고).

실행:
    python visit_simulation3.py --target 500 --max-agents 5000 --batch-size 25
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
RESULT_NAME = "visit_simulation3_results.csv"

load_dotenv(BASE_DIR / ".env")

TIME_SLOTS = ["00-06", "06-11", "11-14", "14-17", "17-21", "21-24"]
DAYS = ["월", "화", "수", "목", "금", "토", "일"]
SEED = 42  # 시뮬레이션 1·2와 동일 시드 -> 세 시뮬레이션 모두 짝비교 가능


def build_extra_district_info() -> str:
    """시뮬레이션 2와 동일하게 한줄요약+대표업종_운영시간+방문목적을 쓰고,
    여기에 '요일특성' 필드를 추가한다(시뮬레이션 2와의 유일한 상권 정보 차이)."""
    facts = json.loads(STAGE1_PATH.read_text(encoding="utf-8"))
    biz_lines = "; ".join(
        f"{b['업종']}({b['운영시간']})" for b in facts["대표업종_운영시간"]
    )
    purposes = ", ".join(facts["방문목적"])
    return (
        f"{facts['한줄요약']}. "
        f"대표 업종과 운영시간은 다음과 같습니다: {biz_lines}. "
        f"주요 방문 목적은 {purposes}입니다. "
        f"요일별 특성: {facts['요일특성']}."
    )


def build_district_context() -> str:
    """기본 정보 부분은 시뮬레이션 1·2와 동일하게, 실제 데이터에 있는 내용만 사용."""
    dist = json.loads(REAL_DIST_PATH.read_text(encoding="utf-8"))
    stores = dist["stores"]["store_counts"]
    base = (
        f"{dist['district_name']}(상권 코드 {dist['district_code']})이라는 상권입니다. "
        f"이 상권 안의 매장 구성은 마트 {stores['마트']}개, 백화점 {stores['백화점']}개, "
        f"편의점 {stores['편의점']}개, 카페 {stores['카페']}개, 음식점 {stores['음식점']}개입니다."
    )
    return base + " " + build_extra_district_info()


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


def run(target: int, max_agents: int, batch_size: int, model: str):
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다 (.env 또는 환경변수).")
    client = OpenAI(api_key=api_key)

    if not REAL_DIST_PATH.exists():
        raise RuntimeError("data/real_distribution.json이 없습니다. 먼저 prepare_real_data.py를 실행하세요.")
    if not STAGE1_PATH.exists():
        raise RuntimeError("data/stage1_district_facts.json이 없습니다.")

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        district_context=build_district_context(),
        time_slots=", ".join(TIME_SLOTS),
        days=", ".join(DAYS),
    )

    personas = pd.read_parquet(PERSONA_PATH)
    idx = list(range(len(personas)))
    random.seed(SEED)
    random.shuffle(idx)

    out_path = OUTPUT_DIR / RESULT_NAME
    prior_df = None
    if out_path.exists():
        prior_df = pd.read_csv(out_path)
        print(f"기존 결과 발견: {len(prior_df):,}명 이미 질의됨 (이어서 진행)")

    results = []
    accepted = int((prior_df["visit"] == True).sum()) if prior_df is not None else 0
    queried = len(prior_df) if prior_df is not None else 0
    cursor = queried
    calls = 0

    while accepted < target and queried < max_agents and cursor < len(idx):
        batch_idx = idx[cursor: cursor + batch_size]
        cursor += batch_size
        batch = personas.iloc[batch_idx].reset_index(drop=True)

        lines = "\n".join(format_persona_line(i, row) for i, row in batch.iterrows())
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=800,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": lines},
            ],
        )
        calls += 1
        decisions = safe_parse_batch(response.choices[0].message.content or "", len(batch))
        decision_map = {d.get("i"): d for d in decisions if isinstance(d, dict)}

        for i, row in batch.iterrows():
            d = decision_map.get(i, {"v": "N"})
            visit = str(d.get("v", "N")).upper().startswith("Y")
            record = {
                "uuid": row.uuid,
                "sex": row.sex,
                "age": int(row.age),
                "age_group": age_to_group(int(row.age)),
                "occupation": row.occupation,
                "district": row.district,
                "visit": visit,
                "time_slot": d.get("t") if visit else None,
                "day_of_week": d.get("d") if visit else None,
            }
            results.append(record)
            if visit:
                accepted += 1
        queried += len(batch)
        print(f"[batch {calls}] queried={queried} accepted={accepted}/{target} "
              f"(수락률 {accepted/queried:.1%})")

    OUTPUT_DIR.mkdir(exist_ok=True)
    new_df = pd.DataFrame(results)
    combined = pd.concat([prior_df, new_df], ignore_index=True) if prior_df is not None else new_df
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n=== 시뮬레이션 3 종료 ===")
    print(f"에이전트 풀(전체): {len(personas):,}")
    print(f"이번 실행에서 새로 호출: {calls}회 ({len(new_df):,}명)")
    print(f"누적 질의한 에이전트 수: {queried:,}")
    print(f"누적 채워진 방문자 수: {accepted:,}")
    print(f"누적 수락률: {accepted/queried:.1%}")
    if accepted < target:
        print(f"※ target({target})에 도달하지 못하고 max-agents({max_agents}) 상한으로 종료됨. "
              f"--max-agents를 늘려 재실행하세요 (기존 결과에 이어서 진행됨).")
    print(f"결과 저장(누적): {out_path}")
    return combined, queried, accepted, calls


def main():
    parser = argparse.ArgumentParser(description="동대문패션타운 방문 시뮬레이션 3 (시뮬2 + 요일특성 + 반편향 지시)")
    parser.add_argument("--target", type=int, default=500, help="채우고자 하는 방문자(수락) 수")
    parser.add_argument("--max-agents", type=int, default=5000, help="안전 상한 (예산 보호용)")
    parser.add_argument("--batch-size", type=int, default=25, help="호출당 페르소나 수")
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()
    run(args.target, args.max_agents, args.batch_size, args.model)


if __name__ == "__main__":
    main()
