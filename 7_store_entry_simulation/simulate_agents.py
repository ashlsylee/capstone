"""
롯데마트 슈퍼 신규 출점 시뮬레이션 — 에이전트 생성 + LLM 판단.

LLM에게는 '사람의 판단'(왜 왔는가, 장 볼 일이 있는가)만 맡기고, 공간(어디서 어디로, 어느 점포)은
실제 지도와 규칙으로 정한다. 5주차에서 LLM이 '언제'를 한쪽으로 몰았듯, 장소·점포 선택도 LLM에게 맡기면
한 곳에 몰리고 목록 맨 위 점포를 고르는 경향이 있어서다.

  1) 누가·언제 상권에 있는가 → 실제 데이터: 2025년 유동인구의 요일·시간대·연령·성별 비율로 뽑는다.
  2) 왜 왔는가 + 장 볼 일이 있는가 → LLM: 페르소나·요일·시각을 보고 방문 목적 1개와,
     이번 방문 중 슈퍼마켓(식료품·생활용품)에서 살 것이 있는지(Y/N)·무엇을 살지를 답한다.
     점포 목록은 보여주지 않는다 (목록이 '산다' 쪽으로 유도하고 맨 위를 고르게 하는 것을 막기 위함).
  3) 어디서 어디로, 어느 점포 → estimate_entry.py: 목적·운영시간으로 장소를 정하고, 실제 도로망
     최단경로와 허프(Huff) 모델로 점포 선택 확률을 계산한다 (API 없음, 후보지 시나리오도 여기서).

실행:
    cp .env.example .env   # OPENAI_API_KEY 입력
    python simulate_agents.py              # 1,000명, gpt-5.4-mini, API 약 100회
    python simulate_agents.py --mock       # API 없이 파이프라인 점검 (outputs/mock/, 실험 결과 아님)

중간에 끊겨도 다시 실행하면 outputs/llm_cache.jsonl 의 응답을 재사용해 이어서 진행한다.

결과: outputs/agents.csv — 에이전트별 페르소나·요일·시각·목적·장보기 여부(need_supermarket)·살 품목
"""
import argparse
import hashlib
import json
import os
import random
import re

import pandas as pd
from dotenv import load_dotenv

from common import MARKET_PATH, OUTPUT_DIR, PERSONA_PATH, age_to_group, load_json, load_places

SEED = 42
DEFAULT_N = 1000
BATCH_SIZE = 10

TIME_RANGES = {"00-06": (0, 6), "06-11": (6, 11), "11-14": (11, 14),
               "14-17": (14, 17), "17-21": (17, 21), "21-24": (21, 24)}
DAY_NAMES = {"월": "월요일", "화": "화요일", "수": "수요일", "목": "목요일",
             "금": "금요일", "토": "토요일", "일": "일요일"}

# 1단계 상권 분석의 방문목적 6개 + 유동인구에 포함되는 비쇼핑 인구 4개
PURPOSES = ["의류 도매 사입", "패션 쇼핑", "원단·부자재 구매", "관광·구경", "식사·야식", "전시·행사 관람",
            "상권 내 근무(상인·직원)", "숙박", "인근 거주·생활", "통과(환승·이동)"]


# ---------------------------------------------------------------- 에이전트 생성 (실제 분포)
def weighted_choice(rng: random.Random, pct: dict) -> str:
    keys = list(pct.keys())
    return rng.choices(keys, weights=[pct[k] for k in keys], k=1)[0]


def sample_agents(n: int) -> pd.DataFrame:
    """2025년 유동인구 비율대로 요일·시간대·연령대·성별을 뽑고, 조건에 맞는 페르소나를 붙인다.
    (원본이 각 항목의 주변 분포만 제공하므로 항목 간 독립 가정)"""
    pop = load_json(MARKET_PATH)["floating_population"]
    if not PERSONA_PATH.exists():
        raise SystemExit(f"페르소나 파일이 없습니다: {PERSONA_PATH}\n"
                         "5주차 data 폴더의 nemotron_personas_korea_demo.parquet을 복사하거나 python prepare_personas.py를 실행하세요.")
    personas = pd.read_parquet(PERSONA_PATH)
    personas["age_group"] = personas["age"].map(age_to_group)
    pools = {k: g.index.to_numpy() for k, g in personas.groupby(["age_group", "sex"])}

    rng = random.Random(SEED)
    rows, used = [], set()
    for i in range(n):
        day = weighted_choice(rng, pop["day_pct"])
        band = weighted_choice(rng, pop["time_pct"])
        age_group = weighted_choice(rng, pop["age_pct"])
        sex = "남자" if weighted_choice(rng, pop["gender_pct"]) == "남" else "여자"
        start, end = TIME_RANGES[band]
        hour, minute = rng.randrange(start, end), rng.choice([0, 10, 20, 30, 40, 50])
        pool = pools[(age_group, sex)]
        while True:
            idx = int(pool[rng.randrange(len(pool))])
            if idx not in used:
                used.add(idx)
                break
        p = personas.loc[idx]
        rows.append({
            "agent_id": f"A{i:04d}", "uuid": p.uuid, "sex": p.sex, "age": int(p.age),
            "age_group": age_group, "occupation": p.occupation, "district": p.district,
            "family_type": p.family_type, "hobbies_and_interests": p.hobbies_and_interests,
            "day": day, "time_band": band, "clock": f"{hour:02d}:{minute:02d}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- LLM 호출 (캐시 포함)
class LLM:
    def __init__(self, model: str, mock: bool, cache_path):
        self.model, self.mock, self.cache_path = model, mock, cache_path
        self.cache = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec["text"]
        self.calls = 0
        if not mock:
            from openai import OpenAI
            load_dotenv()
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY가 없습니다. .env.example을 .env로 복사해 키를 넣거나 --mock으로 실행하세요.")
            self.client = OpenAI()

    def ask(self, system: str, user: str, mock_fn) -> str:
        key = hashlib.sha1(f"{self.model}|{self.mock}|{system}|{user}".encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        if self.mock:
            text = mock_fn()
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_completion_tokens=2000,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            text = resp.choices[0].message.content or ""
        self.calls += 1
        self.cache[key] = text
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")
        return text


def parse_json_list(text: str) -> list:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return []
        out = []
        for d in data if isinstance(data, list) else []:
            if isinstance(d, dict):
                try:  # 번호가 "0"처럼 문자열로 와도 정수로 맞춘다
                    d["i"] = int(d.get("i"))
                except (TypeError, ValueError):
                    continue
                out.append(d)
        return out
    return []


def persona_line(i: int, a) -> str:
    return (f"{i}) 나이:{a.age} 성별:{a.sex} 직업:{a.occupation} 거주지:{a.district} 가족:{a.family_type} "
            f"취미·성향:{a.hobbies_and_interests} | 지금: {DAY_NAMES[a.day]} {a.clock}")


def build_system_prompt(places: list) -> str:
    place_lines = "\n".join(f"- {p['name']} ({p['kind']}) — 운영: {p['hours']}" for p in places)
    return f"""너는 아래 사람들을 한 명씩 연기한다.
각 사람은 서울시 유동인구 통계에 따라, 표시된 요일·시각에 이미 '동대문패션타운 관광특구' 상권 안에 있는 사람이다.
상권에 왔는지 여부는 이미 정해져 있으니 판단하지 말고, 두 가지만 답한다.

(1) 이 사람이 지금 이 상권에 있는 목적 하나: {", ".join(PURPOSES)}
(2) 이번에 상권에 있는 동안 슈퍼마켓(식료품·생활용품·음료를 파는 동네 마트, 편의점 제외)에서 물건을 살 일이 있는가 (Y/N), 있다면 무엇을 살지 짧게.

참고 — 상권 안 주요 장소와 운영시간:
{place_lines}

규칙:
- 직업·거주지·가족·성향과 요일·시각에 맞게 현실적으로 답한다. 문을 닫은 시간의 상가에서 쇼핑하고 있을 수는 없다.
- 상권 유동인구에는 쇼핑객뿐 아니라 일하는 사람, 숙박객, 근처 주민, 지나가는 사람도 포함된다.
- 대부분의 사람은 짧은 방문 동안 슈퍼마켓에 들르지 않는다. 실제로 살 이유가 있을 때만 Y.
- 주말 나들이나 쇼핑만 떠올리지 말고, 성별·연령 고정관념에 기대지 말 것.

출력: JSON 배열만. 형식 [{{"i":번호,"g":"목적","n":"Y 또는 N","w":"살 품목(없으면 빈 문자열)"}}]"""


def run_llm(agents: pd.DataFrame, places: list, llm: LLM) -> pd.DataFrame:
    system = build_system_prompt(places)
    rng = random.Random(SEED + 1)
    purposes, needs, items = [], [], []
    for start in range(0, len(agents), BATCH_SIZE):
        batch = agents.iloc[start:start + BATCH_SIZE].reset_index(drop=True)
        user = "\n".join(persona_line(i, a) for i, a in batch.iterrows())

        def mock_fn():
            out = []
            for i, a in batch.iterrows():
                night = a.time_band in ("00-06", "21-24")
                w = ([4, 0.5, 0.5, 0.5, 3, 0, 4, 2, 2, 2] if night else [1, 3, 1, 2, 3, 1, 2, 1, 2, 3])
                g = rng.choices(PURPOSES, weights=w)[0]
                y = rng.random() < (0.3 if g in ("상권 내 근무(상인·직원)", "숙박", "인근 거주·생활") else 0.1)
                out.append({"i": i, "g": g, "n": "Y" if y else "N", "w": "음료·간식" if y else ""})
            return json.dumps(out, ensure_ascii=False)

        decisions = {d["i"]: d for d in parse_json_list(llm.ask(system, user, mock_fn))}
        for i in range(len(batch)):
            d = decisions.get(i, {})
            g = d.get("g") if d.get("g") in PURPOSES else None
            purposes.append(g)
            needs.append(str(d.get("n", "N")).upper().startswith("Y") if g else None)
            items.append(d.get("w", "") if g else None)
        print(f"\r[LLM] {min(start + BATCH_SIZE, len(agents))}/{len(agents)} (API 호출 {llm.calls}회)", end="")
    print()
    agents = agents.copy()
    agents["purpose"] = purposes
    agents["need_supermarket"] = needs
    agents["items"] = items
    return agents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--mock", action="store_true", help="API 없이 규칙 기반 가짜 응답으로 파이프라인만 점검")
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / "mock" if args.mock else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(args.model, args.mock, out_dir / "llm_cache.jsonl")

    agents = run_llm(sample_agents(args.n), load_places(), llm)
    agents.drop(columns=["hobbies_and_interests"]).to_csv(out_dir / "agents.csv", index=False, encoding="utf-8-sig")

    ok = agents["purpose"].notna()
    print(f"\n=== 완료 ({'MOCK' if args.mock else args.model}) ===")
    print(f"에이전트 {len(agents)}명 (응답 실패 {(~ok).sum()}명), 새 API 호출 {llm.calls}회")
    print(f"장보기 필요(Y) 비율: {agents.loc[ok, 'need_supermarket'].mean():.1%}")
    print(agents.loc[ok].groupby("purpose")["need_supermarket"].agg(인원="size", 장보기_비율="mean")
          .sort_values("인원", ascending=False).round(2).to_string())
    print(f"저장: {out_dir / 'agents.csv'}")


if __name__ == "__main__":
    main()
