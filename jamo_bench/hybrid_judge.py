# -*- coding: utf-8 -*-
"""하이브리드 판정자 — 종성 유형에 따라 CLOVA / template_match를 나눠 쓴다.

## 왜 나누는가 (2026-08-10 실측 근거)

실제 Seedream 파일럿 256장을 사람이 target-blind로 전수 판독한 결과,
두 판정자의 오차가 **정반대 방향으로** 종성 축에 걸려 있었다
(사람 정확도 대비 %p, 낮을수록 사람과 가까움):

| 종성유형   | 사람  | CLOVA 오차 | template_match 오차 |
|-----------|------|-----------|--------------------|
| no_T      | 75.9% | **+6.0**  | +21.7              |
| simple_T  | 65.1% | +24.5     | **+8.5**           |
| cluster_T | 49.3% | +37.3     | **+7.5**           |

CLOVA는 무종성(실사용 단어 다수)에서 사전 지식이 도움이 되지만 종성이
붙을수록 그 사전 지식이 발목을 잡는다. template_match는 언어 지식이
없어 종성 유무와 무관하게 일정하지만, 무종성처럼 획이 적어 변별 정보가
부족한 경우에 약하다. 각자의 강점 구간만 쓰면 오차가 평평해진다.

## 이 판정자가 무엇을 할 수 있고 무엇을 못 하는가

**못 하는 것 — 절대 점수 주장.** 사람과의 일치율이 64.5%(전체) /
77.8%(사람이 한글로 읽은 212건)로, 설계서 §14.2의 공식 점수 하한
(85%)에 못 미친다. "이 모델의 정확도는 X%"라는 절대 수치를 이
판정자만으로 주장할 수 없다 — 실측 기준 6~8.5%p 낮게 나온다.

**할 수 있는 것 — 구조 축 비교(RQ2/RQ3, §8.6의 Primary 지표).**
판정자의 자격 요건은 "오차가 작을 것"이 아니라 "오차가 측정하려는 축과
상관되지 않을 것"이다. 오차 변동폭이 CLOVA 31.3%p, template_match
14.2%p인 데 반해 **하이브리드는 2.5%p**로 사실상 상수 편향이며, 그
덕분에 차이값은 정확하게 복원된다:

    겹받침 − 단순종성 정확도 차이
      사람(진실)      −15.8%p
      하이브리드      −14.8%p  (오차 +1.0%p)   ← 사용 가능
      CLOVA 단독      −28.6%p  (오차 −12.8%p)  ← 효과를 1.8배 부풀림

즉 절대 눈금은 못 믿어도 눈금 간격은 믿을 수 있는 자(ruler)다.

## 알려진 한계

1. **"한글이 아님"을 판정할 수 없다.** 실측에서 생성물의 17.2%가 아예
   한글이 아니었는데(사람 판정), 두 판정자 모두 억지로 한글 하나를
   답한다. template_match의 유사도 점수로 걸러보려 했으나 분포가 크게
   겹쳐(중앙값 0.734 vs 0.642) 임계값 분리가 불가능했다. 이 구간은
   여전히 사람 감사가 필요하다.
2. 위 사유로 소수의 위양성이 있다(사람이 "한글 아님"이라 한 44건 중
   template_match 5건·CLOVA 2건을 "정답"으로 판정). 전체 정답 대비
   3.8%/1.8% 수준.
3. **라우팅은 타깃의 종성 유형으로 한다** — 정답을 훔쳐보는 게 아니라
   "어떤 자를 쓸지"를 고르는 것이다(타깃의 구조 분류는 생성 이전에
   이미 확정된 벤치마크 설계값이다). 다만 "CLOVA가 종성을 떨어뜨려
   읽어서 무종성 타깃과 우연히 맞는" 위양성 위험이 이론적으로 있어
   실측으로 확인했다 — no_T에서 CLOVA는 사람보다 낮았고(69.9% vs
   75.9%) 부풀림은 관측되지 않았다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from .decompose import decompose

JudgeName = Literal["clova", "template_match"]

# 실측된 셀별 편향(사람 정확도 − 하이브리드 정확도, %p). 보고서에 반드시
# 병기해야 하는 값 — 하이브리드 점수는 이만큼 낮게 나온다.
MEASURED_BIAS_PP = {
    "no_T": 6.0,
    "simple_T": 8.5,
    "cluster_T": 7.5,
}
BIAS_MEASURED_N = 256
BIAS_MEASURED_DATE = "2026-08-10"


def route_judge(target: str) -> JudgeName:
    """타깃의 종성 유형으로 어느 판정자를 쓸지 고른다.

    무종성은 CLOVA(사전 지식이 도움이 되는 구간), 종성이 있으면
    template_match(사전 편향이 해가 되는 구간).
    """
    syl = decompose(target)
    if syl is None:
        raise ValueError(f"target must be a single Hangul syllable, got {target!r}")
    return "clova" if syl.coda_class_3 == "no_T" else "template_match"


@dataclass(frozen=True)
class HybridReading:
    predicted_char: Optional[str]
    judge_used: JudgeName
    coda_class: str
    clova_reading: Optional[str]
    template_match_reading: Optional[str]
    judges_agree: bool
    """두 판정자가 같게 읽었는지 — 불일치 건은 사람 감사 우선 배정
    대상이다(실측: 256건 중 160건 불일치)."""
    expected_bias_pp: float
    """이 셀에서 하이브리드가 사람 대비 몇 %p 낮게 나오는지(실측값)."""


def combine(
    target: str,
    clova_reading: Optional[str],
    template_match_reading: Optional[str],
) -> HybridReading:
    """두 판정자의 판독 결과를 받아 하이브리드 판정을 만든다(I/O 없음).

    실제 호출(OCR API, 템플릿 매칭)은 호출부가 담당한다 — 그래야 이미
    저장된 결과로 재채점하거나(비용 0) 테스트에서 모킹할 수 있다.
    """
    syl = decompose(target)
    if syl is None:
        raise ValueError(f"target must be a single Hangul syllable, got {target!r}")

    coda_class = syl.coda_class_3
    judge: JudgeName = "clova" if coda_class == "no_T" else "template_match"
    predicted = clova_reading if judge == "clova" else template_match_reading

    return HybridReading(
        predicted_char=predicted,
        judge_used=judge,
        coda_class=coda_class,
        clova_reading=clova_reading,
        template_match_reading=template_match_reading,
        judges_agree=clova_reading == template_match_reading,
        expected_bias_pp=MEASURED_BIAS_PP[coda_class],
    )


def read_hybrid(
    image,
    target: str,
    clova_fn: Optional[Callable] = None,
    template_match_fn: Optional[Callable] = None,
) -> HybridReading:
    """편의 래퍼 — 필요한 판정자만 실제로 호출한다.

    라우팅 결과 CLOVA가 필요 없으면 CLOVA를 호출하지 않는다(API 비용
    절감: 실측 파일럿 기준 종성 있는 타깃이 65%이므로 CLOVA 호출이
    그만큼 줄어든다). 두 판독을 모두 기록하고 싶으면 양쪽 fn을 다
    넘기면 된다 — 그 경우 둘 다 호출한다.
    """
    judge = route_judge(target)

    clova_reading = None
    tm_reading = None

    if clova_fn is not None and (judge == "clova" or template_match_fn is not None):
        clova_reading = clova_fn(image)
    if template_match_fn is not None and (judge == "template_match" or clova_fn is not None):
        tm_reading = template_match_fn(image)

    if judge == "clova" and clova_fn is None:
        raise ValueError("타깃이 무종성이라 CLOVA가 필요한데 clova_fn이 없습니다.")
    if judge == "template_match" and template_match_fn is None:
        raise ValueError("타깃에 종성이 있어 template_match가 필요한데 template_match_fn이 없습니다.")

    return combine(target, clova_reading, tm_reading)
