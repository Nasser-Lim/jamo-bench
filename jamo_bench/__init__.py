# -*- coding: utf-8 -*-
"""JAMO — Judging Accuracy of Machine-rendered Orthography.

OCR-mediated 한글 렌더링 진단 벤치마크의 코어 라이브러리. 이미지 생성이나
OCR 엔진 연동 없이도 동작하는 부분(자모 분해·채점·Route 분류·chance 보정·
표본 파티셔닝)만 이 패키지에 담는다.
"""

from .decompose import Syllable, decompose, is_hangul_syllable, normalize
from .score import ScoreResult, score

__all__ = [
    "Syllable",
    "decompose",
    "is_hangul_syllable",
    "normalize",
    "ScoreResult",
    "score",
]
