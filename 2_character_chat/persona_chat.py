"""
최애주민(우태은씨) + 대비되는 주민 2명(수다스러운 사람 / 혼자를 즐기는 사람)에게
동일한 상황을 주고, LLM으로 각자의 페르소나에 맞는 대화·행동을 시뮬레이션합니다.

실행 전 준비:
    1. python -m venv .venv && source .venv/bin/activate  (Windows: .venv\Scripts\activate)
    2. pip install -r requirements.txt
    3. .env.example 을 .env 로 복사하고 ANTHROPIC_API_KEY 값 채우기

실행:
    python persona_chat.py
    python persona_chat.py --situation "화곡동에 새로운 편집숍이 오픈했다." --rounds 3
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent
RESIDENTS_PATH = BASE_DIR / "personas" / "residents.json"
OUTPUT_DIR = BASE_DIR / "outputs"

load_dotenv(BASE_DIR / ".env")


def load_residents():
    with open(RESIDENTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def persona_to_system_prompt(resident: dict) -> str:
    return f"""당신은 '{resident['name']}'({resident['age']}세, 성별: {resident['sex']}, 직업: {resident['occupation']})입니다.
아래는 당신의 상세 페르소나입니다. 반드시 이 성격과 말투, 가치관을 일관되게 유지하며 1인칭으로 대답하세요.

[한 줄 요약] {resident['persona']}
[전문 페르소나] {resident['professional_persona']}
[예술 페르소나] {resident['arts_persona']}
[음식 페르소나] {resident['culinary_persona']}
[가족 페르소나] {resident['family_persona']}
[성향 태그] {resident['trait_tag']}

지금부터 특정 상황과 다른 사람들의 대화가 주어집니다.
대화 규칙:
- 바로 직전 사람이 한 말에 직접 반응하세요 (동의, 반박, 질문, 맞장구 등). 그 사람 이름을 언급해도 좋습니다.
- 이미 나온 대사나 행동을 그대로 반복하지 마세요. 매번 새로운 내용을 말하세요.
- 대화체로, 실제 사람이 말하듯 짧고 자연스럽게 답하세요.

당신의 캐릭터라면 어떻게 반응할지, 아래 JSON 형식으로만 답하세요. 다른 설명은 절대 붙이지 마세요.
{{"dialogue": "실제로 할 법한 말 한두 문장", "action": "방문 | 구매 | 검색 | 무반응 중 하나", "action_detail": "그 행동에 대한 한 줄 설명"}}
"""


def format_history(history: list) -> str:
    if not history:
        return "(아직 아무도 말하지 않았습니다.)"
    return "\n".join(
        f"- {h['speaker']}: \"{h['dialogue']}\" (행동: {h['action']} - {h['action_detail']})"
        for h in history
    )


def safe_parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"dialogue": text.strip(), "action": "무반응", "action_detail": "JSON 파싱 실패 (원문 그대로 기록)"}


def call_llm_api(client, model: str, system_prompt: str, user_prompt: str) -> dict:
    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return safe_parse_json(response.content[0].text)


def call_llm_openai(client, model: str, system_prompt: str, user_prompt: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=300,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return safe_parse_json(response.choices[0].message.content or "")


def call_llm_gemini(client, model: str, system_prompt: str, user_prompt: str) -> dict:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=300,
            temperature=0.8,
        ),
    )
    return safe_parse_json(response.text or "")


_LOCAL_PIPELINE = None


def call_llm_local(model: str, system_prompt: str, user_prompt: str) -> dict:
    """API 키/결제 없이 컴퓨터에서 직접 돌아가는 3B 모델 호출 (느리지만 무료)."""
    global _LOCAL_PIPELINE
    if _LOCAL_PIPELINE is None:
        from transformers import pipeline
        print(f"로컬 모델 '{model}' 을 처음 불러오는 중입니다 (다운로드 포함, 몇 분 걸릴 수 있음)...")
        _LOCAL_PIPELINE = pipeline("text-generation", model=model, device_map="auto")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output = _LOCAL_PIPELINE(
        messages, max_new_tokens=300, do_sample=True, temperature=0.8, repetition_penalty=1.3
    )
    text = output[0]["generated_text"][-1]["content"]
    return safe_parse_json(text)


# 라운드가 진행될수록 대화 주제가 자연스럽게 발전하도록 라운드별로 다른 질문을 덧붙인다.
# (같은 질문만 계속 던지면 특히 3B 로컬 모델은 매 라운드 비슷한 답을 반복하는 경향이 있음)
ROUND_PROMPTS = [
    "이 소식을 듣고 당신은 어떻게 반응하나요?",
    "다른 사람들의 반응을 보고, 당신 생각은 좀 달라졌나요? 구체적으로 언제·어떻게 행동할지 말해보세요.",
    "만약 다른 사람과 같이 하게 된다면 무엇을 같이 할지, 아니면 왜 혼자 하고 싶은지 이야기해보세요.",
    "지금까지 나온 이야기를 바탕으로 최종적으로 어떻게 할지 결론을 내려보세요.",
]


def run_simulation(situation: str, rounds: int, model: str, engine: str) -> list:
    if engine == "api":
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                ".env 파일에 ANTHROPIC_API_KEY가 없습니다. .env.example을 참고해 .env를 만들어주세요."
            )
        client = Anthropic(api_key=api_key)

        def call_llm(system_prompt, user_prompt):
            return call_llm_api(client, model, system_prompt, user_prompt)
    elif engine == "openai":
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                ".env 파일에 OPENAI_API_KEY가 없습니다. .env.example을 참고해 .env를 만들어주세요."
            )
        client = OpenAI(api_key=api_key)

        def call_llm(system_prompt, user_prompt):
            return call_llm_openai(client, model, system_prompt, user_prompt)
    elif engine == "gemini":
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                ".env 파일에 GEMINI_API_KEY가 없습니다. .env.example을 참고해 .env를 만들어주세요."
            )
        client = genai.Client(api_key=api_key)

        def call_llm(system_prompt, user_prompt):
            return call_llm_gemini(client, model, system_prompt, user_prompt)
    else:
        def call_llm(system_prompt, user_prompt):
            return call_llm_local(model, system_prompt, user_prompt)

    residents = load_residents()
    history = []

    for round_num in range(1, rounds + 1):
        print(f"\n=== {round_num}라운드 ===")
        round_prompt = ROUND_PROMPTS[min(round_num - 1, len(ROUND_PROMPTS) - 1)]
        for resident in residents:
            system_prompt = persona_to_system_prompt(resident)
            last_line = (
                f"바로 직전 발언: {history[-1]['speaker']}가 \"{history[-1]['dialogue']}\" 라고 말했습니다.\n\n"
                if history
                else ""
            )
            user_prompt = (
                f"상황: {situation}\n\n"
                f"지금까지 오간 대화:\n{format_history(history)}\n\n"
                f"{last_line}"
                f"이번 라운드 질문: {round_prompt}\n\n"
                "이제 당신 차례입니다. 특히 바로 직전 발언에 직접 반응하면서, 위 질문에 답하세요."
            )
            result = call_llm(system_prompt, user_prompt)
            result["speaker"] = resident["name"]
            history.append(result)
            print(f"[{resident['name']}] {result['dialogue']}  -> ({result['action']}: {result['action_detail']})")

    return history


def main():
    parser = argparse.ArgumentParser(description="페르소나 에이전트 대화 시뮬레이션")
    parser.add_argument(
        "--situation",
        default="화곡동에 새로운 편집숍이 오픈해서 SNS에 소식이 떴다.",
        help="세 주민에게 공통으로 던져줄 상황",
    )
    parser.add_argument("--rounds", type=int, default=3, help="대화 라운드 수")
    parser.add_argument(
        "--engine",
        choices=["api", "openai", "gemini", "local"],
        default="api",
        help=(
            "api: Anthropic 유료 API / "
            "openai: OpenAI 유료 API / "
            "gemini: Google Gemini API (무료 티어 가능) / "
            "local: 컴퓨터에서 3B 모델을 무료로 실행 (느림, requirements-local.txt 필요)"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "engine=api면 Claude 모델명(기본 claude-sonnet-5), "
            "engine=openai면 OpenAI 모델명(기본 gpt-5.4-mini), "
            "engine=gemini면 Gemini 모델명(기본 gemini-2.0-flash), "
            "engine=local이면 HuggingFace 모델명(기본 Qwen/Qwen2.5-3B-Instruct)"
        ),
    )
    args = parser.parse_args()

    if args.model is None:
        args.model = {
            "api": "claude-sonnet-5",
            "openai": "gpt-5.4-mini",
            "gemini": "gemini-2.0-flash",
            "local": "Qwen/Qwen2.5-3B-Instruct",
        }[args.engine]

    history = run_simulation(args.situation, args.rounds, args.model, args.engine)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"chat_log_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"situation": args.situation, "history": history}, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
