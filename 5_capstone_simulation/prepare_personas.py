"""
페르소나 풀(100만 명)을 nvidia/Nemotron-Personas-Korea 원본에서 직접 만든다.

이 스크립트가 필요한 이유: 완성된 데이터 파일(176MB)을 그대로 전달하는 대신, 어떤 컬럼을
어디서 가져왔는지 코드로 남겨서 재현 가능하게 하기 위함. 원본 데이터셋은 9개 parquet
조각(HF `hf://datasets/nvidia/Nemotron-Personas-Korea/data/train-*.parquet`)으로 나뉘어
있고, 이 스크립트는 그중 시뮬레이션에 쓰는 컬럼만 받아온다:
    uuid, sex, age, marital_status, occupation, district, province,
    housing_type, education_level, family_type, hobbies_and_interests
(전문/스포츠/예술 등 나머지 6종 페르소나 텍스트, career_goals 등은 이번 시뮬레이션에서
쓰지 않아 받지 않는다 — 받는 데이터 크기를 줄이기 위함).

실행 (최초 1회, 인터넷 필요, 9개 조각 x 약 30~40초 = 총 5분 내외):
    pip install -r requirements.txt   (datasets, huggingface_hub 포함)
    python prepare_personas.py

결과: data/nemotron_personas_korea_demo.parquet (100만 행, 약 176MB)
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "data" / "nemotron_personas_korea_demo.parquet"

N_SHARDS = 9
COLUMNS = [
    "uuid", "sex", "age", "marital_status", "occupation", "district",
    "province", "housing_type", "education_level", "family_type",
    "hobbies_and_interests",
]


def main():
    parts = []
    for i in range(N_SHARDS):
        url = (
            "hf://datasets/nvidia/Nemotron-Personas-Korea/"
            f"data/train-{i:05d}-of-{N_SHARDS:05d}.parquet"
        )
        print(f"[{i + 1}/{N_SHARDS}] 받는 중: {url}")
        df = pd.read_parquet(url, columns=COLUMNS)
        parts.append(df)
        print(f"    {len(df):,}행")

    personas = pd.concat(parts, ignore_index=True)
    assert personas["uuid"].is_unique, "uuid 중복 발견"
    print(f"\n총 {len(personas):,}행")

    OUT_PATH.parent.mkdir(exist_ok=True)
    personas.to_parquet(OUT_PATH, index=False)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
