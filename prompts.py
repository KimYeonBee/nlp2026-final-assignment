"""후보 prompt 모음.

paraphrase_detection (finetune + inference) 단계에서 collate_fn 이 두 문장 (s1, s2) 로부터
prompt string 을 만들 때 선택할 수 있는 템플릿들.

- 모든 prompt 는 마지막 토큰 위치가 "yes" / "no" 의 next-token 예측이 되도록 구성됨.
- collate_fn 내부에서 PROMPTS[variant](s1, s2) 형태로 사용.

사용 예 (datasets.py):
    from prompts import PROMPTS
    template = PROMPTS[self.p.prompt_variant]
    cloze_style_sents = [template(s1, s2) for (s1, s2) in zip(sent1, sent2)]
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 후보 prompt 정의
# ---------------------------------------------------------------------------

def _v1_original(s1: str, s2: str) -> str:
    """원본 starter prompt (test set / 평가에 사용되던 것). length-bias 큼."""
    return f'Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": '


def _v2_question_pair(s1: str, s2: str) -> str:
    """현재 학습용 prompt — Quora 도메인 (question pair) 에 맞춘 형식."""
    return (
        f'Question 1: "{s1}"\n'
        f'Question 2: "{s2}\n'
        f'Are these questions asking the same thing?\n'
    )


def _v3_separator(s1: str, s2: str) -> str:
    """Sentence A / B 분리 — length-bias 완화 + positional 신호 정돈."""
    return (
        f"Sentence A: {s1}\n"
        f"Sentence B: {s2}\n"
        f"Do A and B mean the same thing? Answer yes or no: "
    )


def _v4_minimal(s1: str, s2: str) -> str:
    """최소 토큰 prompt — long sentence 의 truncation 압박 완화 목적."""
    return f"{s1} || {s2} || same meaning? "


def _v5_nli_style(s1: str, s2: str) -> str:
    """NLI 스타일 — premise/hypothesis 표기. 학술 PLM 들이 익숙한 포맷."""
    return (
        f"Premise: {s1}\n"
        f"Hypothesis: {s2}\n"
        f"Does the premise entail the hypothesis (paraphrase)? yes or no: "
    )


def _v6_symmetric(s1: str, s2: str) -> str:
    """순서 정보를 명시적으로 제거 — '다음 두 문장' 표현으로 swap 불변성 학습 보조."""
    return (
        f"Below are two sentences.\n"
        f"- {s1}\n"
        f"- {s2}\n"
        f"Do they have the same meaning? Answer yes or no: "
    )


# variant 이름 → callable. 기본은 "question_pair" (현재 학습 prompt) 로 둬서
# 옵션 미지정 시 동작이 바뀌지 않게 한다.
PROMPTS = {
    "original": _v1_original,
    "question_pair": _v2_question_pair,
    "separator": _v3_separator,
    "minimal": _v4_minimal,
    "nli": _v5_nli_style,
    "symmetric": _v6_symmetric,
}


def list_variants() -> list[str]:
    return list(PROMPTS.keys())


if __name__ == "__main__":
    s1 = "How do I learn Python?"
    s2 = "What is the best way to start learning Python programming?"
    for name, fn in PROMPTS.items():
        print(f"=== {name} ===")
        print(fn(s1, s2))
        print()
