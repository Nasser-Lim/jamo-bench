# -*- coding: utf-8 -*-
"""음절 단위 채점 규칙 (JAMO_benchmark_design.md §8.5, JAMO_v51_patch.md §6).

jamo_scorer_probe.py의 13종 엣지케이스 판정을 그대로 보존하고
(VALID/OVERGEN/EMPTY/NON_HANGUL), v5.1의 환각(hallucination) 분리를 추가한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .decompose import decompose, is_hangul_syllable

Verdict = Literal["VALID", "OVERGEN", "EMPTY", "NON_HANGUL", "HALLUCINATED"]


@dataclass(frozen=True)
class ScoreResult:
    verdict: Verdict
    target: str
    pred: str
    onset_ok: Optional[bool] = None
    nucleus_ok: Optional[bool] = None
    coda_ok: Optional[bool] = None

    @property
    def is_valid(self) -> bool:
        return self.verdict == "VALID"


def score(
    target: str,
    pred: Optional[str],
    ocr_confidence: Optional[float] = None,
) -> ScoreResult:
    """단일 목표 음절(target) 대 OCR primary candidate(pred) 채점.

    ocr_confidence를 주면 v5.1 §6의 환각 분리를 적용한다: confidence≥0.95이고
    초/중/종 3자리가 전부 불일치하면(=편집거리가 타깃의 자모 슬롯 수와 같음)
    "완벽하게 그렸지만 완전히 다른 글자"로 보아 HALLUCINATED로 분리한다.
    자모가 일부라도 겹치면(예: 읽→익, ㅇㅣ 보존) 통상적인 substitution이지
    환각이 아니다.
    """
    t = decompose(target)
    if t is None:
        raise ValueError(f"target must be a single Hangul syllable, got {target!r}")

    if pred is None or pred == "":
        return ScoreResult("EMPTY", target, pred or "")

    valid_chars = [c for c in pred if is_hangul_syllable(c)]
    if not valid_chars:
        return ScoreResult("NON_HANGUL", target, pred)

    if len(pred) > 1:
        return ScoreResult("OVERGEN", target, pred)

    p = decompose(pred)
    onset_ok = t.onset == p.onset
    nucleus_ok = t.nucleus == p.nucleus
    coda_ok = t.coda == p.coda

    if ocr_confidence is not None and ocr_confidence >= 0.95:
        if not onset_ok and not nucleus_ok and not coda_ok:
            return ScoreResult("HALLUCINATED", target, pred)

    return ScoreResult("VALID", target, pred, onset_ok, nucleus_ok, coda_ok)
