# -*- coding: utf-8 -*-
"""JAMO v1 판정 프로토콜 — CLOVA confidence 게이트 + 사람 판정.

## 규칙 (전부)

    confidence >= CONFIDENCE_THRESHOLD
      AND onset_group != "tense_O"      →  CLOVA 판독을 그대로 채택
    그 외 (confidence 없음 포함)         →  사람 판정
                                            1) 유효 완성형인가 (이진)
                                            2) 유효하면 전사

보고 시 `MEASURED_BIAS_PP`를 **항상 병기한다**(아래 "쓰면 안 되는 것" 참고).

## 왜 이 구조인가 — 앞선 판정자들이 전부 실패한 뒤 남은 것

| 시도 | 결과 |
|---|---|
| CLOVA 단독 | 실격. 사람 99.4% vs CLOVA 32.8%(폰트 골드셋 180문항), 720장 Ceiling에서 18셀 전부 미달 |
| VLM judge | 보류. 어려운 글자일수록 추론 토큰 폭증(1장 최대 5분) — 실패가 난이도 축과 상관 |
| template_match | 개선이지만 불충분. 256장에서 52.0%(CLOVA 42.6% 대비 우위)이나 정형성·다중글자 판정 불가 |
| hybrid_judge(종성별 라우팅) | 편향 변동폭 2.5%p로 쓸 만했으나 장치가 복잡하고 template_match 한계를 그대로 승계 |
| **confidence 게이트 + 사람** | **편향 변동폭 1.6%p, 겹받침 격차 복원 오차 −1.6%p, 사람 큐 42.2%** |

## 반직관적인 점 — confidence 게이트는 자기교정적이다

11단계에서 confidence를 "한글 여부" 신호로 쓰려다 실패했고("오답 중 한글
아님 0.696 vs 진짜 한글 오독 0.669, 거의 동일"), 12단계에서는 "정답 여부와
상관된 신호로 abstain하면 자동 판정 구간이 정답 쪽으로 선별되어 정확도가
부풀려진다"고 경고했다. 실측하면 **반대 방향으로 작동한다**:

| 종성유형 | 자동 채택률 | 그 구간의 사람↔CLOVA 일치율 |
|---|---|---|
| no_T | 75.9% | 93.3% |
| simple_T | 55.2% | 84.9% |
| cluster_T | **18.8%** | 81.8% |

CLOVA는 **자기가 못 믿을 곳에서 정확히 자신감을 잃는다.** 어려운 구간이
통째로 사람에게 넘어가므로 편향이 커지는 게 아니라 오히려 줄어든다.
자동 채택 구간에서 "무효 글자를 정답으로 처리"한 은폐 사례는 133장 중
0건이었다(다만 n이 작아 95% 상한은 약 2.2%로 읽어야 한다).

## 단, 쌍자음 초성(tense_O)에서는 자기교정이 작동하지 않는다

자기교정은 **종성 축에서만** 성립했다. 18셀의 나머지 두 축으로 편향을
재보니 초성군 축이 무너져 있었다(16단계):

| 초성군 | 자동 채택 구간의 CLOVA↔사람 불일치율 |
|---|---|
| simple_O(기본 9종) | 4.5% |
| aspir_O(가획 5종) | 21.7% |
| **tense_O(ㄲㄸㅃㅆㅉ)** | **47.4%** |

쌍자음 초성 음절에서는 CLOVA가 **자신 있게 틀린다.** 오독의 정체를 보면
초성 자체는 보존하고(ㅉ→ㅉ) 중성·종성을 틀리는데, 쌍자음 음절이 실사용
빈도가 낮아 8단계에서 확인한 희귀도 편향이 그대로 재현되는 것으로 보인다.

`tense_O`를 자동 채택에서 빼면 세 축이 전부 기준을 통과하고 전체 편향도
줄어든다. 사람 큐는 42.2% → 50.0%로 늘지만 그만한 값을 한다.

| 셀 축 | 규칙 적용 전 | 적용 후 |
|---|---|---|
| 종성(coda) | 1.6%p | 1.5%p |
| 모음(vowel) | 3.1%p | **0.5%p** |
| 초성군(onset) | **10.4%p** | **1.9%p** |
| 전체 편향 | −3.5%p | **−0.8%p** |

**주의 — 이 규칙은 in-sample 적합이다.** n=63(tense_O)에서 도출해 같은
데이터로 평가했다. 기전(희귀도 편향)이 8단계와 일관되어 채택했지만
표본 보강 시 반드시 재검증한다.

## 쓰면 안 되는 것 / 써도 되는 것

**안 되는 것 — 절대 점수 주장.** 실측 편향이 −0.8%p(축별 0.0 ~ −1.5%p)로
항상 사람보다 낮거나 같게 나온다. "이 모델의 정확도는 X%"는 **사람 전수
판정(Gold split)으로만** 주장한다.

**되는 것 — 셀 축 간 비교(RQ2/RQ3).** 편향이 세 축 전반에 거의 일정해
차이값이 복원된다(변동폭 0.5~1.9%p).

**조건부로 되는 것 — 자모 위치별 오류율(H4).** 초성>중성>종성 **순서는
보존**하지만 기울기를 과장한다 — 초성−종성 격차를 사람 기준 대비
+3.3~3.9%p 부풀린다. 설계서 §6.6이 명시한 검출 하한(8%p)보다는 작지만,
**절대 자모 오류율은 Gold split에서만** 인용한다.

**해결하지 못한 것 — 정형성은 사람만 판정한다.** 파일럿 300장 기준 21.7%가
유효 완성형이 아니었고(malformed 14.0%가 지배적), 자동 채택 구간에도 6.8%가
섞여 있다. CLOVA는 이를 가장 가까운 유효 음절로 스냅해 은폐한다. 사람의
이진 유효성 판정은 Krippendorff α=0.942(2감사자 87문항), 그중 유효 판정
건의 전사는 100% 일치(23/23)로 매우 신뢰 가능한 반면, 어떤 자동 판정자도
이 능력이 없다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# scripts/eval_confidence_gate.py 스윕으로 확정(2026-08-11, n=256).
# 제약(겹받침 격차 복원 ±5%p, 편향 변동폭 ≤2.5%p)을 만족하는 것 중
# 사람 큐가 가장 작은 값. 0.70은 격차 오차 −5.2%p로 제약을 벗어났고,
# 0.90은 같은 품질에 사람 큐만 5.8%p 더 든다.
CONFIDENCE_THRESHOLD = 0.80

# 자동 채택에서 제외하는 초성군. CLOVA가 이 구간에서는 "자신 있게 틀린다"
# — 자동 채택 구간 불일치율 47.4%(simple_O 4.5%, aspir_O 21.7% 대비).
NO_AUTO_ONSET_GROUPS = frozenset({"tense_O"})

# 실측 편향(%p). 리포트에 항상 병기한다 — 이 값을 숨기고 절대 정확도를
# 주장하는 것이 이 프로토콜의 유일한 오용 경로다.
MEASURED_BIAS_PP = {"no_T": 0.0, "simple_T": -0.9, "cluster_T": -1.5, "overall": -0.8}

# 이 조건에서 측정된 값임을 함께 기록한다(다른 조건에 그대로 옮기지 말 것).
CALIBRATION = {
    "n": 256,
    "model": "dola-seedream-5-0-pro-260628",
    "occupancy_recipe": "v2 (target_max_occupancy=0.10)",
    "human_queue_frac": 0.500,
    "bias_spread_pp": {"coda": 1.5, "vowel": 0.5, "onset": 1.9},
    # 자모 위치별 오류율은 순서만 보존하고 기울기를 과장한다(아래 docstring).
    "jamo_position_gradient_inflation_pp": {"unconditional": 3.9, "conditional": 3.3},
}


@dataclass(frozen=True)
class JudgingDecision:
    """한 이미지에 대한 판정 라우팅 결과.

    `needs_human`이 True면 `reading`은 아직 미정이며, 사람 판정(이진 유효성
    + 유효 시 전사)을 받아 `resolve_human()`으로 확정해야 한다.
    """

    needs_human: bool
    reading: Optional[str]
    confidence: Optional[float]
    source: str  # "clova" | "human_pending" | "human"
    expected_bias_pp: float


def route(
    clova_reading: Optional[str],
    confidence: Optional[float],
    coda_class_3: Optional[str] = None,
    onset_group: Optional[str] = None,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> JudgingDecision:
    """판정 라우팅.

    Args:
        clova_reading: CLOVA 판독 결과. None이면 검출 실패 → 사람.
        confidence: CLOVA `inferConfidence`. None이면 사람.
        coda_class_3: 병기할 편향값 선택에만 쓴다(판정에는 개입하지 않음).
        onset_group: `tense_O`면 confidence와 무관하게 사람에게 넘긴다 —
            이 구간에서 CLOVA는 자신 있게 틀린다(불일치율 47.4%).

    타깃의 구조 분류(coda_class_3/onset_group)를 참조하는 것은 정답을
    훔쳐보는 게 아니다 — 벤치마크 설계 시점에 이미 확정된 값이며 "어떤 자를
    쓸지"를 고르는 데만 쓴다. 판독 내용 자체와는 무관하다.
    """
    bias = MEASURED_BIAS_PP.get(coda_class_3 or "", MEASURED_BIAS_PP["overall"])
    auto = (
        confidence is not None
        and confidence >= threshold
        and bool(clova_reading)
        and onset_group not in NO_AUTO_ONSET_GROUPS
    )
    if auto:
        return JudgingDecision(False, clova_reading, confidence, "clova", bias)
    return JudgingDecision(True, None, confidence, "human_pending", bias)


def resolve_human(
    decision: JudgingDecision,
    human_valid: bool,
    human_transcription: Optional[str],
) -> JudgingDecision:
    """사람 판정 결과를 채워 확정한다.

    `human_valid`가 False면(유효 완성형이 아님) 전사는 없으며 `reading`은
    None으로 남는다 — 이는 "판독 실패"가 아니라 **"모델이 존재하지 않는
    글자를 그렸다"**는 측정값이다. 어떤 타깃과도 일치하지 않는다.
    """
    if not decision.needs_human:
        return decision
    reading = human_transcription if human_valid else None
    return JudgingDecision(False, reading, decision.confidence, "human", decision.expected_bias_pp)
