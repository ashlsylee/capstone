# 1. Dataset Analysis — Nemotron-Personas-Korea EDA

대한민국의 실제 인구통계·지리·직업 분포를 기반으로 생성된 NVIDIA의 합성 페르소나 데이터셋
[`nvidia/Nemotron-Personas-Korea`](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea)에 대한
탐색적 데이터 분석(EDA)입니다.

## 데이터셋 개요

| 항목 | 내용 |
|---|---|
| 출처 | [Hugging Face - nvidia/Nemotron-Personas-Korea](https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea) |
| 규모 | 1,000,000행 × 26개 컬럼 |
| 대상 | 대한민국 성인(만 19세 이상) |
| 라이선스 | CC BY 4.0 |
| 특징 | 완전 합성(synthetic) 데이터. KOSIS(국가통계포털), 대법원, 국민건강보험공단 등 실제 통계 분포를 기반으로 생성됨 |

각 행은 하나의 가상 인물이며, 성별·나이·학력·직업·거주지 등 인구통계 속성과 함께 자연어로 작성된
7종의 페르소나 설명(`professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`,
`culinary_persona`, `family_persona`, `persona`), 취미·기술 리스트를 포함합니다.

## 폴더 구성

```
1_dataset_analysis/
├── README.md                 ← 이 파일
├── data_set_analysis_.ipynb  ← EDA 전체 코드 (Jupyter Notebook)
└── eda_outputs/               ← 노트북 실행 시 생성되는 요약 테이블 (CSV 18개)
    ├── sex_distribution.csv
    ├── age_group_distribution.csv
    ├── age_group_x_marital_status.csv
    ├── marital_status_distribution.csv
    ├── family_type_top10.csv
    ├── housing_type_distribution.csv
    ├── education_level_distribution.csv
    ├── bachelors_field_distribution.csv
    ├── province_distribution.csv
    ├── district_top15.csv
    ├── province_avg_age.csv
    ├── province_x_education_heatmap.csv
    ├── occupation_top15.csv
    ├── occupation_top12_x_sex.csv
    ├── persona_length_describe.csv
    ├── age_persona_length_corr.csv
    ├── top_hobby_phrases.csv
    └── top_skill_words.csv
```

## 분석 내용

`data_set_analysis_.ipynb`는 아래 순서로 구성되어 있습니다. (표 + 그래프 쌍으로 정리)

1. 환경 설정 (라이브러리 설치, 한글 폰트 설정)
2. 데이터 로드 (`datasets` 라이브러리, 샘플링 옵션 포함)
3. 데이터 기본 정보 (구조, 결측치, 고유값)
4. 인구통계 분석 — 성별, 연령(히스토그램·인구피라미드), 혼인상태, 가구유형, 주거유형
5. 학력·전공 분석
6. 지역 분석 — 시도별 인구/평균연령, 시군구 Top15, 시도×학력 히트맵
7. 직업 분석 — 직업 Top15, 직업×성별 구성비
8. 페르소나 텍스트 분석 — 유형별 길이 분포(박스플롯), 나이 대비 길이(hexbin), 상관관계 히트맵
9. 취미·기술 리스트 분석 — 상위 취미 문구/기술 단어 빈도 Top20
10. 요약 테이블 CSV 저장 (`eda_outputs/`)
11. 결론 요약

## 주요 인사이트

- **인구구조**: 60대 이상에서 여성 비중이 뚜렷이 높아지는 항아리형(고령화) 구조. 전국 평균 연령 약 50세.
- **혼인상태**: 19-29세는 미혼 91%이나 40대부터 배우자 있음이 70% 이상으로 역전. 80대 이상은 사별 비중 58%.
- **학력**: 서울(대졸 이상 약 44%)이 전국 최고 수준, 비수도권은 고졸 비중이 상대적으로 높음.
- **지역**: 경기·서울에 인구 집중, 전라남·경상북 등 비수도권은 평균 연령이 높아 지역 간 고령화 격차가 뚜렷함.
- **직업**: `무직` 비중이 가장 크며(고령 인구 반영), 청소·주방·조리 vs 경비·하역 등 직종에서 강한 성별 분리 관찰.
- **페르소나 텍스트**: 6종 세부 페르소나 길이는 서로 강한 상관관계, 나이가 많을수록 전문 페르소나가 다소 짧아지는 경향.
- **취미/기술**: 동호회 활동, 트로트·유튜브 시청, 산책·둘레길 등 실생활 밀착형 취미가 상위권.

> ⚠️ 이 데이터셋은 실제 개인정보가 아닌 **완전 합성 데이터**이며, 변수 간 독립성 가정 하에 생성되었습니다.
> 실제 사회 현상의 인과관계 분석이 아닌 LLM 학습/평가용 합성 페르소나 데이터로 활용하는 것이 목적입니다.

## 실행 방법

```bash
pip install datasets pandas matplotlib pyarrow
```

노트북을 위에서부터 순서대로 실행하면 됩니다. 전체 데이터(100만 행)는 메모리 부담이 있을 수 있어
기본값으로 20만 행을 무작위 샘플링해서 분석합니다(`SAMPLE_SIZE` 변수로 조절 가능).
