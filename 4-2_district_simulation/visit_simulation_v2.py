"""
동대문패션타운 관광특구 방문 시뮬레이션 v2 — 매장 선택 포함 (업종명만, 브랜드 라벨 없음).

v1(visit_simulation.py)과 차이: 방문 여부·시간대에 더해 "어느 매장에 갈지"까지
한 번의 LLM 호출로 같이 판단한다. 결과 파일 스키마가 달라져서 v1 결과와는
별도 파일(visit_simulation_v2_results.csv)에 저장한다.

이 스크립트는 "한 시점 판단"까지만 하고, 30일 반복·매장별 집계·동선은
expand_30day.py(별도, API 호출 없음)에서 이어서 처리한다.

실행:
    python visit_simulation_v2.py --target 999999 --max-agents 20000 --batch-size 25
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
OUTPUT_DIR = BASE_DIR / "outputs"

load_dotenv(BASE_DIR / ".env")

DISTRICT_CONTEXT = (
    "동대문패션타운 관광특구: 서울 중구 소재, 패션 의류 도소매 특화 상권. "
    "두타몰·밀리오레·헬로apM·평화시장 등 대형 패션몰이 밀집해 있고, "
    "심야~새벽까지 영업하는 도매 상권 특성이 있으며 외국인 관광객 비중이 높음."
)

TIME_SLOTS = ["00-06", "06-11", "11-14", "14-17", "17-21", "21-24"]

STORES = [
    {"code": "mart", "name": "마트", "type": "마트"},
    {"code": "cvs", "name": "편의점", "type": "편의점"},
    {"code": "cafe", "name": "카페", "type": "카페"},
    {"code": "hotel", "name": "호텔", "type": "호텔"},
    {"code": "dept", "name": "백화점(던던 동대문점)", "type": "백화점"},
]
STORE_LINE = ", ".join(f'{s["code"]}={s["name"]}({s["type"]})' for s in STORES)
STORE_CODES = {s["code"] for s in STORES}

SYSTEM_PROMPT = f"""당신은 상권 방문 시뮬레이터입니다. 아래는 대상 상권 설명입니다.

{DISTRICT_CONTEXT}

이 상권 안에는 아래 5개 매장이 있습니다: {STORE_LINE}

이제 페르소나 목록이 주어집니다. 각 사람이 실제로 이 사람이라면, 나이·직업·거주지 등
현실적인 생활 패턴을 근거로 이번 분기 안에 이 상권을 방문할지(Y/N) 판단하세요.
방문한다면({{"v":"Y"}}):
- 방문 시간대를 아래 6개 구간 중 하나로: {", ".join(TIME_SLOTS)}
- 위 5개 매장 코드 중 이 사람이 가장 갈 법한 매장 하나를 "s"에 고르세요.
방문하지 않으면({{"v":"N"}}) 시간대·매장은 비워도 됩니다.

반드시 아래 형식의 JSON 배열만 출력하세요. 다른 설명·마크다운 없이 배열만:
[{{"i":0,"v":"Y","t":"17-21","s":"cafe"}},{{"i":1,"v":"N"}}, ...]
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
    return f"{i}) {int(row.age)}세 {row.sex} {row.occupation} {row.district}"


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

    personas = pd.read_parquet(PERSONA_PATH)
    idx = list(range(len(personas)))
    random.seed(43)  # v1과 다른 시드 (독립적인 새 표본)
    random.shuffle(idx)

    out_path = OUTPUT_DIR / "visit_simulation_v2_results.csv"
    prior_df = None
    if out_path.exists():
        prior_df = pd.read_csv(out_path)
        print(f"기존 v2 결과 발견: {len(prior_df):,}명 이미 질의됨 (이어서 진행)")

    results = []
    accepted = len(prior_df[prior_df["visit"] == True]) if prior_df is not None else 0
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
            max_completion_tokens=1200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": lines},
            ],
        )
        calls += 1
        decisions = safe_parse_batch(response.choices[0].message.content or "", len(batch))
        decision_map = {d.get("i"): d for d in decisions if isinstance(d, dict)}

        for i, row in batch.iterrows():
            d = decision_map.get(i, {"v": "N"})
            visit = str(d.get("v", "N")).upper().startswith("Y")
            store_code = d.get("s") if visit else None
            if store_code not in STORE_CODES:
                store_code = random.choice(list(STORE_CODES)) if visit else None
            record = {
                "uuid": row.uuid,
                "sex": row.sex,
                "age": int(row.age),
                "age_group": age_to_group(int(row.age)),
                "occupation": row.occupation,
                "district": row.district,
                "visit": visit,
                "time_slot": d.get("t") if visit else None,
                "store_code": store_code,
            }
            results.append(record)
            if visit:
                accepted += 1
        queried += len(batch)
        print(f"[batch {calls}] queried={queried} accepted={accepted} "
              f"(수락률 {accepted/queried:.1%})")

    OUTPUT_DIR.mkdir(exist_ok=True)
    new_df = pd.DataFrame(results)
    combined = pd.concat([prior_df, new_df], ignore_index=True) if prior_df is not None else new_df
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n=== 종료 ===")
    print(f"에이전트 풀(전체): {len(personas):,}")
    print(f"이번 실행에서 새로 호출: {calls}회 ({len(new_df):,}명)")
    print(f"누적 질의한 에이전트 수: {queried:,}")
    print(f"누적 채워진 방문자 수: {accepted:,}")
    print(f"누적 수락률: {accepted/queried:.1%}")
    print(f"결과 저장(누적): {out_path}")
    return combined, queried, accepted, calls


def main():
    parser = argparse.ArgumentParser(description="동대문패션타운 방문 시뮬레이션 v2 (매장 선택 포함)")
    parser.add_argument("--target", type=int, default=999999, help="채우고자 하는 방문자(수락) 수")
    parser.add_argument("--max-agents", type=int, default=20000, help="질의할 에이전트 수(누적)")
    parser.add_argument("--batch-size", type=int, default=25, help="호출당 페르소나 수")
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()
    run(args.target, args.max_agents, args.batch_size, args.model)


if __name__ == "__main__":
    main()
