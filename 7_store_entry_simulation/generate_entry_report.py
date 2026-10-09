"""
진출 평가 결과(outputs/entry_evaluation.json)를 LLM에 주고, 진출 판단 근거·주요 타겟 고객·
맞춤 마케팅 방안 문서를 만든다.

LLM에는 아래 사실 자료만 주고, 자료에 없는 숫자는 쓰지 말라고 지시한다.
  - entry_evaluation.json  : 후보지별 결제건수·매출·민감도 분석·판정, 매출을 가져오는 경쟁점, 신규점 고객 구성,
                             BASE 검증, LLM 판단 요약(목적별 장보기 비율)
  - market_facts.json      : 2025년 실제 유동인구·슈퍼마켓 매출 (시간대·요일·연령·성별)
  - stage1_district_facts.json : 1단계 상권 분석(업종 운영시간·방문목적·시간/요일 특성)
  - ablation_entry.md      : (있으면) 대조실험 결과 — 결론이 LLM 판단·가정값에 얼마나 기대는지
진출 판정(Go/조건부/No-Go)은 estimate_entry.py가 정해진 기준으로 이미 계산한 값이며,
LLM은 판정을 바꾸지 않고 근거를 설명·정리만 한다.

실행:
    python generate_entry_report.py            # outputs/entry_report.md
    python generate_entry_report.py --mock     # outputs/mock/entry_report.md (API 없이 자료 요약 틀만)
"""
import argparse
import json
import os

from dotenv import load_dotenv

from common import MARKET_PATH, NEW_BRAND, OUTPUT_DIR, STAGE1_PATH, load_json

SYSTEM = f"""너는 롯데 그룹 {NEW_BRAND}(기업형 슈퍼마켓) 신규 출점 검토 담당 분석가다.
주어진 JSON 자료만 근거로, 동대문패션타운 관광특구 상권에 {NEW_BRAND}를 신규 출점해도 되는지에 대한
검토 문서를 한국어 마크다운으로 작성한다.

반드시 지킬 것:
- 자료에 있는 숫자만 쓴다. 숫자를 새로 만들거나 추정하지 않는다. 숫자를 쓸 때는 어느 자료의 값인지 알 수 있게 쓴다.
- 진출 판정(district_verdict, 후보지별 verdict)은 이미 정해진 기준으로 계산된 값이다. 바꾸지 말고 그대로 쓴 뒤 근거를 설명한다.
- 자료의 출처를 구분해서 쓴다: 실제 통계(서울시 상권분석서비스·OSM 지도), LLM 에이전트 판단(방문 목적·장보기 여부), 규칙·가정(장소 배정, 허프 모델, 입구 가중치, 보이지 않는 경쟁점 배치).
- 한계(OSM 미등록 경쟁점을 추정 배치한 점, 표본 크기, 민감도 분석 구간 폭, BASE 검증 오차)를 숨기지 않는다.
- 자료에 없는 비교·순위·인과 주장을 하지 않는다. 업종 순위를 말할 때는 industry_sales_rank_2025q4의 실제 순서를 그대로 쓴다.
- 숫자 표기: 금액은 억원(소수 둘째 자리, 예: 2.36억원), 비율은 %(예: 0.782 → 78%), 큰 수는 천 단위 쉼표. JSON 키 이름
  (share_above_avg 등)을 본문에 그대로 쓰지 말고 우리말로 풀어 쓴다 (예: '민감도 분석에서 평균 이상인 비율').
- 각 후보지의 vs_base(롯데 미입점 상태 대비 변화: 상권 결제 점유, 기존 점포 1곳당 평균 감소, 신규점 고객이 원래 가던 곳,
  가장 크게 줄어드는 기존 점포)를 1장과 3장에서 '입점하지 않았을 때와 비교하면'으로 설명한다.
- ablation(대조실험) 자료가 있으면, 결론이 LLM 판단에 기대는지 공간 구조·가정에 기대는지 3장과 6장에 한두 문장으로 밝힌다.

문서 구성 (이 순서, 이 제목):
# {NEW_BRAND} 동대문패션타운 신규 출점 검토
## 1. 결론 — 이 상권에 신규 진출해도 되는가
## 2. 상권 현황 (실제 데이터)
## 3. 후보지 비교와 최적 위치 선정 근거
## 4. 주요 타겟 고객
## 5. 맞춤 마케팅 방안
   (타겟별로: 누구에게 / 언제(요일·시간대) / 무엇을(상품·서비스) / 어떻게(채널·매장 운영) — 자료의 시간대·목적·직업 구성을 근거로)
## 6. 리스크와 한계"""


def build_user_message(ev: dict, market: dict, stage1: dict, ablation_md: str = "") -> str:
    # 문서에 필요 없는 장소 좌표 등은 줄여서 전달
    ev = {**ev, "places": [{k: p[k] for k in ("id", "name", "kind", "hours", "purposes")} for p in ev["places"]]}
    payload = {"entry_evaluation": ev, "market_facts": market, "stage1_district_facts": stage1}
    if ablation_md:
        payload["ablation"] = ablation_md
    return json.dumps(payload, ensure_ascii=False, indent=1)


def mock_report(ev: dict) -> str:
    lines = [f"# {NEW_BRAND} 동대문패션타운 신규 출점 검토 (MOCK — LLM 미사용, 자료 요약 틀)", "",
             f"- 실행: {ev['run_label']}", f"- 상권 진출 판정: **{ev['district_verdict']}** (최적 후보지 {ev['best_candidate']})", ""]
    for c in ev["candidates"]:
        lines.append(f"- {c['id']} {c['name']}: 기본 연매출 {c['base']['annual_sales']:,}원, "
                     f"평균 이상 비율 {c['robust']['share_above_avg'] * 100:.0f}%, 판정 {c['verdict']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model", default="gpt-5.4-mini")
    args = parser.parse_args()
    out_dir = OUTPUT_DIR / "mock" if args.mock else OUTPUT_DIR

    ev = load_json(out_dir / "entry_evaluation.json")
    ablation_path = out_dir / "ablation_entry.md"  # ablation_entry.py 를 먼저 돌렸으면 같이 전달
    ablation_md = ablation_path.read_text(encoding="utf-8") if ablation_path.exists() else ""
    if args.mock:
        text = mock_report(ev)
    else:
        from openai import OpenAI
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY가 없습니다. .env.example을 .env로 복사해 키를 넣으세요.")
        client = OpenAI()
        resp = client.chat.completions.create(
            model=args.model,
            max_completion_tokens=8000,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_message(ev, load_json(MARKET_PATH), load_json(STAGE1_PATH), ablation_md)},
            ],
        )
        text = resp.choices[0].message.content or ""
        inputs = "outputs/entry_evaluation.json, data/market_facts.json, data/stage1_district_facts.json"
        if ablation_md:
            inputs += ", outputs/ablation_entry.md"
        text += f"\n\n---\n*생성: {args.model}, 입력 자료: {inputs}*\n"
    path = out_dir / "entry_report.md"
    path.write_text(text, encoding="utf-8")
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
