"""
Nemotron-Personas-Korea 데이터셋에서 대화 시뮬레이션에 쓸 주민 3명을 선정하는 스크립트.

선정 기준:
    1. 최애주민 우태은씨 (1주차에서 UUID로 고정 선정, 1_dataset_analysis/최애주민_우태은.md 참고)
    2. 수다스러운 성향의 주민 1명 (talkative_score 최고점, 남자 - 성비 2:1을 맞추기 위한 선택)
    3. 사람 만나는 것보다 혼자 지내는 것을 즐기는 주민 1명 (solitary_score 최고점, 여자)
   -> 최종 성비: 여자 2명(우태은 + 혼자족) : 남자 1명(수다쟁이) = 2:1

1주차 EDA와 동일하게 전체 100만행 중 20만행을 무작위 샘플링(SAMPLE_SIZE, RANDOM_STATE=42)해서
탐색합니다. 전체 데이터를 다 훑는 것보다 훨씬 가볍고 빠르면서도, 표본오차가 작아 결과는 거의 동일합니다.
이 스크립트는 1회성 선정용이며, 결과를 personas/residents.json 에 저장해두면
이후 persona_chat.py 를 실행할 때마다 매번 데이터셋을 다시 내려받지 않아도 됩니다.

(참고) 이 리포지토리의 personas/residents.json 에는 아래 시드로 이미 한 번 실행한 결과가
저장되어 있습니다 (김혁인 / 유화선). 다른 조합을 보고 싶다면 RANDOM_STATE나 키워드 리스트를
바꿔서 다시 실행하세요.

실행:
    python select_personas.py
"""
import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset, disable_progress_bar

disable_progress_bar()

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "personas" / "residents.json"

WOO_UUID = "da6a4829f2f849f88c9bcf531634c915"
SAMPLE_SIZE = 200_000
RANDOM_STATE = 42

NAME_RE = re.compile(r"^([가-힣]{2,4})\s*(?:씨|님)")

TALKATIVE_KEYWORDS = [
    "수다", "말이 많", "말 많", "이야기하는 것을 좋아", "이야기 나누는 것을 즐",
    "왁자지껄", "사교적", "말주변", "떠들썩", "먼저 말을 걸",
]
SOLITARY_KEYWORDS = [
    "혼자 있는 것을 좋아", "혼자만의 시간", "사람 만나는 것보다", "인간관계보다",
    "타인과 어울리기보다", "혼자 지내는", "내향적", "북적이는 곳보다",
    "조용히 혼자", "혼자 보내는 시간",
]

TEXT_COLUMNS = [
    "persona", "professional_persona", "sports_persona", "arts_persona",
    "travel_persona", "culinary_persona", "family_persona",
    "cultural_background", "hobbies_and_interests",
]

KEEP_COLUMNS = TEXT_COLUMNS + [
    "uuid", "sex", "age", "occupation", "province", "district",
]

# persona_chat.py 에 넘겨줄 최애주민(우태은씨) 고정 데이터 (1_dataset_analysis/최애주민_우태은.md 와 동일)
WOO_TAEEUN = {
    "uuid": WOO_UUID,
    "name": "우태은",
    "sex": "여자",
    "age": 33,
    "occupation": "의상 디자이너",
    "province": "서울",
    "district": "서울-강서구",
    "persona": "우태은 씨는 법학 지식을 갖춘 실용주의 의상 디자이너로, 화곡동 자가 주택에서 홀로 지내며 정적인 휴식과 탄탄한 기본 아이템 만들기에 집중하는 인물입니다.",
    "professional_persona": "우태은 씨는 원단의 수축률과 내구성을 집요하게 계산해 오래 입을 수 있는 옷을 설계하며, 법학 전공 지식을 활용해 의류 제작 외주 계약서의 독소 조항을 날카롭게 잡아냅니다.",
    "sports_persona": "우태은 씨는 격렬한 운동보다는 주말마다 개화산 둘레길을 천천히 걸으며 복잡한 머릿속을 정리하는 정적인 시간을 보냅니다.",
    "arts_persona": "우태은 씨는 손으로 직접 넘기는 종이책의 질감을 즐기며, 조용한 방에서 좋아하는 작가의 문장을 곱씹는 시간을 통해 내면의 에너지를 회복합니다.",
    "travel_persona": "우태은 씨는 북적이는 관광지보다는 탁 트인 자연 풍경을 감상할 수 있는 한적한 곳으로 친구나 연인과 함께 여행을 떠납니다.",
    "culinary_persona": "우태은 씨는 주 4~6회 외식을 하며 고기류를 제외한 정갈한 한식이나 분위기 있는 양식 레스토랑에서 식사하는 시간을 즐깁니다.",
    "family_persona": "우태은 씨는 강서구 화곡동의 오래된 다세대 주택에서 홀로 거주하며, 누구의 간섭도 받지 않는 독립적인 생활 공간이 주는 안온함 속에서 심리적 안정감을 찾습니다.",
    "cultural_background": "강서구 화곡동의 오래된 다세대 주택가에서 나고 자라 동네의 골목길 구석구석에 익숙하며, 주말이면 개화산 둘레길을 천천히 걸으며 생각을 정리합니다.",
    "hobbies_and_interests": "퇴근 후 집에서 좋아하는 작가의 종이책을 읽으며 하이볼 한 잔을 곁들이거나, 마곡지구의 조용한 와인바에서 오랜 친구 한두 명과 낮은 목소리로 대화를 나눕니다.",
    "trait_tag": "최애주민 / 조용한 자기만의 시간을 중시",
}


def keyword_score(text: str, keywords: list) -> int:
    return sum(text.count(k) for k in keywords)


def compute_batch(batch: dict) -> dict:
    n = len(batch["uuid"])
    names, talk_scores, solo_scores = [], [], []
    for i in range(n):
        combined = " ".join(str(batch[c][i]) for c in TEXT_COLUMNS)
        m = None
        for col in ("persona", "professional_persona", "family_persona"):
            m = NAME_RE.match(str(batch[col][i]).strip())
            if m:
                break
        names.append(m.group(1) if m else None)
        talk_scores.append(keyword_score(combined, TALKATIVE_KEYWORDS))
        solo_scores.append(keyword_score(combined, SOLITARY_KEYWORDS))
    return {
        "uuid": batch["uuid"], "sex": batch["sex"], "age": batch["age"],
        "occupation": batch["occupation"], "province": batch["province"],
        "district": batch["district"], "name": names,
        "talkative_score": talk_scores, "solitary_score": solo_scores,
    }


def to_resident_dict(row: pd.Series, trait_tag: str) -> dict:
    return {
        "uuid": row["uuid"],
        "name": row["name"],
        "sex": row["sex"],
        "age": int(row["age"]),
        "occupation": row["occupation"],
        "province": row["province"],
        "district": row["district"],
        "persona": row["persona"],
        "professional_persona": row["professional_persona"],
        "sports_persona": row["sports_persona"],
        "arts_persona": row["arts_persona"],
        "travel_persona": row["travel_persona"],
        "culinary_persona": row["culinary_persona"],
        "family_persona": row["family_persona"],
        "cultural_background": row["cultural_background"],
        "hobbies_and_interests": row["hobbies_and_interests"],
        "trait_tag": trait_tag,
    }


def main():
    print("데이터셋 로드 및 샘플링 중...")
    ds = load_dataset("nvidia/Nemotron-Personas-Korea", split="train")
    ds = ds.select_columns(KEEP_COLUMNS)
    ds = ds.shuffle(seed=RANDOM_STATE).select(range(SAMPLE_SIZE))

    light = ds.map(compute_batch, batched=True, batch_size=2000, remove_columns=ds.column_names)
    df = light.to_pandas()
    df = df[(df["name"].notna()) & (df["uuid"] != WOO_UUID)]

    talk_pool = df[(df["talkative_score"] > 0) & (df["sex"] == "남자")].sort_values(
        "talkative_score", ascending=False
    )
    solo_pool = df[(df["solitary_score"] > 0) & (df["sex"] == "여자")].sort_values(
        "solitary_score", ascending=False
    )

    talk_uuid = talk_pool.iloc[0]["uuid"]
    solo_uuid = solo_pool.iloc[0]["uuid"]

    full = ds.filter(lambda x: x["uuid"] in {talk_uuid, solo_uuid}).to_pandas()
    talk_row = full[full["uuid"] == talk_uuid].iloc[0]
    solo_row = full[full["uuid"] == solo_uuid].iloc[0]
    talk_row["name"] = talk_pool.iloc[0]["name"]
    solo_row["name"] = solo_pool.iloc[0]["name"]

    residents = [
        WOO_TAEEUN,
        to_resident_dict(talk_row, trait_tag="수다스러움 / 사람들과 어울려 이야기하는 것을 즐김"),
        to_resident_dict(solo_row, trait_tag="혼자 지내는 것을 즐김 / 사람 만나는 것보다 홀로 시간 보내는 것을 선호"),
    ]

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(residents, f, ensure_ascii=False, indent=2)

    print(f"\n선정 완료 -> {OUT_PATH}")
    for r in residents:
        print(f"- {r['name']} ({r['sex']}, {r['age']}세, {r['occupation']}) : {r['trait_tag']}")


if __name__ == "__main__":
    main()
