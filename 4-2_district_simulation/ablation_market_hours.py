"""
상권의 낮시장/밤시장 영업시간 정보(공신력 있는 자료 기반)를 프롬프트에 추가했을 때
시간대 판단이 얼마나 달라지는지 확인하는 대조 실험.

ablation_store_mention.py와 동일한 500명(seed=99)을 그대로 재사용한다. 그 스크립트의
"매장 언급 있음"(visit_with_store, time_with_store) 결과를 베이스라인으로 삼고,
이 스크립트는 영업시간 정보를 추가한 조건 하나만 새로 질의한다.

DDP 행사 정보는 이번 실험에서 제외 — 상권 도소매 구조(공신력 있는 자료)만 반영.

실행:
    python ablation_market_hours.py --n 500 --batch-size 25
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

TIME_SLOTS = ["00-06", "06-11", "11-14", "14-17", "17-21", "21-24"]

DISTRICT_CONTEXT_WITH_HOURS = (
    "동대문패션타운 관광특구: 서울 중구 소재, 패션 의류 도소매 특화 상권. "
    "두타몰·밀리오레·헬로apM·평화시장 등 대형 패션몰이 밀집해 있으며 외국인 관광객 비중이 높음. "
    "이 상권은 낮시장과 밤시장으로 나뉘어 거의 24시간 운영된다: "
    "낮시장(청평화·동평화·신평화·디오트·테크노·동대문신발상가)은 자정부터 다음날 낮까지 운영되며 "
    "전국 소매상인들이 새벽에 물건을 떼러 온다. "
    "밤시장(아트프라자·디자이너클럽·apM·누죤·DDP패션몰·맥스타일·apM플레이스·광희패션몰 등 14개 상가)은 "
    "밤 8시부터 새벽 5시까지 운영된다. 평화시장은 매일 밤 10시부터 다음날 낮 6시까지 운영된다. "
    "그래서 이 상권은 일반적인 관광지와 달리 자정~오전 시간대에도 도소매 관련 방문객이 상당히 많다."
)

STORE_LINE = "mart=마트(마트), cvs=편의점(편의점), cafe=카페(카페), hotel=호텔(호텔), dept=백화점(던던 동대문점)(백화점)"

PROMPT_WITH_HOURS = f"""당신은 상권 방문 시뮬레이터입니다. 아래는 대상 상권 설명입니다.

{DISTRICT_CONTEXT_WITH_HOURS}

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


def run(n: int, batch_size: int, model: str):
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다 (.env 또는 환경변수).")
    client = OpenAI(api_key=api_key)

    personas = pd.read_parquet(PERSONA_PATH)
    idx = list(range(len(personas)))
    random.seed(99)  # ablation_store_mention.py와 동일 표본
    random.shuffle(idx)
    sample = personas.iloc[idx[:n]].reset_index(drop=True)

    results = []
    n_batches = (len(sample) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = sample.iloc[b * batch_size: (b + 1) * batch_size].reset_index(drop=True)
        lines = "\n".join(format_persona_line(i, row) for i, row in batch.iterrows())
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=1200,
            messages=[
                {"role": "system", "content": PROMPT_WITH_HOURS},
                {"role": "user", "content": lines},
            ],
        )
        decisions = safe_parse_batch(response.choices[0].message.content or "", len(batch))
        decision_map = {d.get("i"): d for d in decisions if isinstance(d, dict)}

        for i, row in batch.iterrows():
            d = decision_map.get(i, {"v": "N"})
            v = str(d.get("v", "N")).upper().startswith("Y")
            results.append({
                "uuid": row.uuid, "sex": row.sex, "age": int(row.age),
                "age_group": age_to_group(int(row.age)),
                "visit_with_hours": v,
                "time_with_hours": d.get("t") if v else None,
                "store_code_with_hours": d.get("s") if v else None,
            })
        print(f"[batch {b+1}/{n_batches}] 누적 {len(results)}명 처리")

    df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "ablation_market_hours_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    rate = df["visit_with_hours"].mean() * 100
    print(f"\n=== 결과 ({len(df)}명, 영업시간 정보 추가 조건) ===")
    print(f"수락률: {rate:.1f}%")
    print("\n시간대 분포:")
    print(df.loc[df["visit_with_hours"], "time_with_hours"].value_counts(normalize=True).mul(100).round(1))
    print(f"\n결과 저장: {out_path}")
    print("ablation_store_mention_results.csv의 visit_with_store/time_with_store와 비교하세요 (동일 500명).")


def main():
    parser = argparse.ArgumentParser(description="영업시간 정보 추가 효과 확인 (대조 실험, ablation_store_mention.py와 짝비교)")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()
    run(args.n, args.batch_size, args.model)


if __name__ == "__main__":
    main()
