"""
동대문패션타운 관광특구 방문 시뮬레이션.

nvidia/Nemotron-Personas-Korea 페르소나 풀(100만 명)에서 무작위로 뽑아 25명씩
묶어 OpenAI(gpt-5.4-mini)에게 "이 상권을 방문할지"를 물어본다. 방문(Y) 응답이
target 수만큼 채워지면(또는 안전 상한에 도달하면) 종료하고, 실제 유동인구
데이터(연령대/성별/시간대 비율)와 비교한다.

실행 전 준비:
    pip install openai python-dotenv pyarrow pandas
    OPENAI_API_KEY 환경변수 설정 (또는 4_district_simulation/.env)

실행:
    python visit_simulation.py --target 1000 --max-agents 5000 --batch-size 25
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

SYSTEM_PROMPT = f"""당신은 상권 방문 시뮬레이터입니다. 아래는 대상 상권 설명입니다.

{DISTRICT_CONTEXT}

이제 페르소나 목록이 주어집니다. 각 사람이 실제로 이 사람이라면, 나이·직업·거주지 등
현실적인 생활 패턴을 근거로 이번 분기 안에 이 상권을 방문할지(Y/N) 판단하세요.
방문한다면({{"v":"Y"}}) 방문 시간대를 아래 6개 구간 중 하나로 고르세요: {", ".join(TIME_SLOTS)}
방문하지 않으면({{"v":"N"}}) 시간대는 비워도 됩니다.

반드시 아래 형식의 JSON 배열만 출력하세요. 다른 설명·마크다운 없이 배열만:
[{{"i":0,"v":"Y","t":"17-21"}},{{"i":1,"v":"N"}}, ...]
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
    # 파싱 실패 시 전부 N 처리 (해당 배치는 방문 없음으로 기록)
    return [{"i": i, "v": "N"} for i in range(batch_size)]


def run(target: int, max_agents: int, batch_size: int, model: str):
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다 (.env 또는 환경변수).")
    client = OpenAI(api_key=api_key)

    personas = pd.read_parquet(PERSONA_PATH)
    idx = list(range(len(personas)))
    random.seed(42)
    random.shuffle(idx)

    results = []
    accepted = 0
    queried = 0
    cursor = 0
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
            record = {
                "uuid": row.uuid,
                "sex": row.sex,
                "age": int(row.age),
                "age_group": age_to_group(int(row.age)),
                "occupation": row.occupation,
                "district": row.district,
                "visit": visit,
                "time_slot": d.get("t") if visit else None,
            }
            results.append(record)
            if visit:
                accepted += 1
        queried += len(batch)
        print(f"[batch {calls}] queried={queried} accepted={accepted} "
              f"(수락률 {accepted/queried:.1%})")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "visit_simulation_results.csv"
    pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n=== 종료 ===")
    print(f"에이전트 풀(전체): {len(personas):,}")
    print(f"질의한 에이전트 수: {queried:,}")
    print(f"채워진 방문자 수: {accepted:,}")
    print(f"수락률: {accepted/queried:.1%}")
    print(f"API 호출 횟수: {calls}")
    print(f"결과 저장: {out_path}")
    return pd.DataFrame(results), queried, accepted, calls


def main():
    parser = argparse.ArgumentParser(description="동대문패션타운 방문 시뮬레이션")
    parser.add_argument("--target", type=int, default=1000, help="채우고자 하는 방문자(수락) 수")
    parser.add_argument("--max-agents", type=int, default=5000, help="안전 상한 (예산 보호용)")
    parser.add_argument("--batch-size", type=int, default=25, help="호출당 페르소나 수")
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()
    run(args.target, args.max_agents, args.batch_size, args.model)


if __name__ == "__main__":
    main()
