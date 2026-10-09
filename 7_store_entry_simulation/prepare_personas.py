"""
페르소나 풀(100만 명)을 nvidia/Nemotron-Personas-Korea 원본에서 받아 data/ 에 저장한다.
5주차 5_capstone_simulation/prepare_personas.py 와 같은 컬럼·같은 방식 (용량 176MB라 저장소엔 올리지 않음).
5주차에서 이미 받아 둔 파일이 있으면 data/ 로 복사해도 된다.

실행 (최초 1회, 인터넷 필요, 약 5분):
    python prepare_personas.py
"""
import pandas as pd

from common import PERSONA_PATH

N_SHARDS = 9
COLUMNS = [
    "uuid", "sex", "age", "marital_status", "occupation", "district",
    "province", "housing_type", "education_level", "family_type",
    "hobbies_and_interests",
]


def main():
    parts = []
    for i in range(N_SHARDS):
        url = f"hf://datasets/nvidia/Nemotron-Personas-Korea/data/train-{i:05d}-of-{N_SHARDS:05d}.parquet"
        print(f"[{i + 1}/{N_SHARDS}] 받는 중: {url}")
        parts.append(pd.read_parquet(url, columns=COLUMNS))
    personas = pd.concat(parts, ignore_index=True)
    assert personas["uuid"].is_unique, "uuid 중복 발견"
    PERSONA_PATH.parent.mkdir(exist_ok=True)
    personas.to_parquet(PERSONA_PATH, index=False)
    print(f"총 {len(personas):,}행 저장: {PERSONA_PATH}")


if __name__ == "__main__":
    main()
