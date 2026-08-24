"""
"매장 5개가 있다"는 문구가 방문 여부 판단에 영향을 주는지 확인하는 대조 실험.

v1(매장 언급 없음)과 v2(매장 언급 있음)는 서로 다른 무작위 표본을 썼기 때문에,
수락률 차이(16.8% → 24.9%)가 매장 문구 때문인지 표본 차이 때문인지 구분이 안 된다.
이 스크립트는 **같은 사람들**에게 두 프롬프트를 모두 물어봐서(짝지은 비교),
매장 문구 하나만의 순수한 효과를 확인한다.

실행:
    python ablation_store_mention.py --n 500 --batch-size 25
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

# v1과 동일한 프롬프트 (매장 언급 없음)
PROMPT_NO_STORE = f"""당신은 상권 방문 시뮬레이터입니다. 아래는 대상 상권 설명입니다.

{DISTRICT_CONTEXT}

이제 페르소나 목록이 주어집니다. 각 사람이 실제로 이 사람이라면, 나이·직업·거주지 등
현실적인 생활 패턴을 근거로 이번 분기 안에 이 상권을 방문할지(Y/N) 판단하세요.
방문한다면({{"v":"Y"}}) 방문 시간대를 아래 6개 구간 중 하나로 고르세요: {", ".join(TIME_SLOTS)}
방문하지 않으면({{"v":"N"}}) 시간대는 비워도 됩니다.

반드시 아래 형식의 JSON 배열만 출력하세요. 다른 설명·마크다운 없이 배열만:
[{{"i":0,"v":"Y","t":"17-21"}},{{"i":1,"v":"N"}}, ...]
"""

# v2와 동일한 프롬프트 (매장 5종 언급)
STORE_LINE = "mart=마트(마트), cvs=편의점(편의점), cafe=카페(카페), hotel=호텔(호텔), dept=백화점(던던 동대문점)(백화점)"
PROMPT_WITH_STORE = f"""당신은 상권 방문 시뮬레이터입니다. 아래는 대상 상권 설명입니다.

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


def call_batch(client, model, system_prompt, batch):
    lines = "\n".join(format_persona_line(i, row) for i, row in batch.iterrows())
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=1200,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": lines},
        ],
    )
    decisions = safe_parse_batch(response.choices[0].message.content or "", len(batch))
    return {d.get("i"): d for d in decisions if isinstance(d, dict)}


def run(n: int, batch_size: int, model: str):
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다 (.env 또는 환경변수).")
    client = OpenAI(api_key=api_key)

    personas = pd.read_parquet(PERSONA_PATH)
    idx = list(range(len(personas)))
    random.seed(99)  # v1(42), v2(43)와 겹치지 않는 독립 표본
    random.shuffle(idx)
    sample_idx = idx[:n]
    sample = personas.iloc[sample_idx].reset_index(drop=True)

    results = []
    n_batches = (len(sample) + batch_size - 1) // batch_size
    for b in range(n_batches):
        batch = sample.iloc[b * batch_size: (b + 1) * batch_size].reset_index(drop=True)

        no_store = call_batch(client, model, PROMPT_NO_STORE, batch)
        with_store = call_batch(client, model, PROMPT_WITH_STORE, batch)

        for i, row in batch.iterrows():
            d0 = no_store.get(i, {"v": "N"})
            d1 = with_store.get(i, {"v": "N"})
            v0 = str(d0.get("v", "N")).upper().startswith("Y")
            v1 = str(d1.get("v", "N")).upper().startswith("Y")
            results.append({
                "uuid": row.uuid, "sex": row.sex, "age": int(row.age), "age_group": age_to_group(int(row.age)),
                "visit_no_store": v0, "time_no_store": d0.get("t") if v0 else None,
                "visit_with_store": v1, "time_with_store": d1.get("t") if v1 else None,
                "store_code": d1.get("s") if v1 else None,
                "flip": ("N->Y" if (not v0 and v1) else "Y->N" if (v0 and not v1) else "동일"),
            })
        print(f"[batch {b+1}/{n_batches}] 누적 {len(results)}명 처리")

    df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "ablation_store_mention_results.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    rate_no = df["visit_no_store"].mean() * 100
    rate_with = df["visit_with_store"].mean() * 100
    print(f"\n=== 결과 (동일 {len(df)}명, 짝지은 비교) ===")
    print(f"매장 언급 없음 수락률: {rate_no:.1f}%")
    print(f"매장 언급 있음 수락률: {rate_with:.1f}%")
    print(f"차이: {rate_with - rate_no:+.1f}%p")
    print()
    print("응답 변화:")
    print(df["flip"].value_counts())
    print(f"\n결과 저장: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="매장 언급 여부가 방문 판단에 미치는 영향 확인 (대조 실험)")
    parser.add_argument("--n", type=int, default=500, help="비교할 인원 수 (같은 사람에게 2개 조건 다 질의)")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()
    run(args.n, args.batch_size, args.model)


if __name__ == "__main__":
    main()
