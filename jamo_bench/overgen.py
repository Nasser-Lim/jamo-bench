# -*- coding: utf-8 -*-
"""OVERGEN(과생성) 자체 감지 — "몇 글자로 보이는가"를 사람 판단이 아니라
잉크의 공간적 분리 여부로 기계적으로 정의한다.

**왜 사람의 "multi_syllable" 라벨을 정답으로 못 쓰는가.** 2번째 감사자
투입 후 Krippendorff's α를 재보니(`docs/PROGRESS.md` 14단계), 이진(유효
완성형 한 글자인가)은 α=0.942로 매우 신뢰 가능했지만, `malformed`와
`multi_syllable`을 가르는 4분류는 α=0.576으로 신뢰 불가였다. 불일치
21건 중 17건(81%)이 정확히 이 경계에 몰렸다 — 여분의 획이 많으면
"한 글자에 획이 더 있다"와 "두 글자가 붙었다"가 같은 그림에 대한 서로
다른 서술일 수 있어, 사람에게도 원리적으로 모호하다.

**그래서 다른 질문으로 바꾼다:** "사람이 몇 글자라고 느끼는가"가 아니라
**"잉크가 공간적으로 몇 개의 덩어리로 분리되는가, 그리고 그 덩어리들을
각각 독립된 음절로 봤을 때 전체를 한 음절로 보는 것보다 유효 음절
템플릿에 더 잘 들어맞는가"**. 이 정의는 사람이 몇 명이든 답이 바뀌지
않는 결정론적 신호다(원 아이디어는 2026-08-10 검토안의 "2-split
template match" 제안, 여기서는 스코어 우위를 강제하는 형태로 구현).

절차:
  1. 원본 좌표계 잉크 마스크(`template_match.raw_ink_mask`)에 약한
     팽창을 씌워 같은 글자 안의 자모 조각(초성·중성·종성이 살짝 떨어져
     있는 경우)을 하나로 합친다.
  2. 연결요소 분석. 잉크 총량 대비 너무 작은 성분(노이즈 점)은 버린다.
  3. 성분이 1개면 즉시 종료 — OVERGEN 아님(비용 0).
  4. 성분이 2개 이상이면 각 성분의 bbox를 원본 이미지에서 잘라
     `template_match.read_by_template_match()`로 독립 판독하고, 잉크
     면적 가중 평균 점수(score_split)를 이미지 전체를 한 글자로 봤을
     때의 점수(score_whole)와 비교한다.
  5. `score_split - score_whole > OVERGEN_MARGIN` 이면 OVERGEN.

margin 조건이 핵심이다 — 정상 한 음절도 초성/중성/종성 배치 때문에
연결요소가 2개 이상으로 갈리는 경우가 있는데(예: "이"의 ㅇ와 ㅣ가 살짝
떨어짐), 그런 경우 조각을 억지로 독립 음절로 읽으면 오히려 점수가
낮아진다(온전한 음절 템플릿에 진 조각들이 안 맞으므로). 진짜 다중
글자만 "쪼개서 보는 게 더 잘 맞는다."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label

from .template_match import raw_ink_mask, read_by_template_match

# 원본 이미지가 1024px 정사각형이라는 파이프라인 관례(judge_preprocess와
# 동일 전제) 기준으로 고른 값 — 자모 조각 사이 자연스러운 틈(수 px~십수
# px)은 메우되, 진짜 다른 음절 사이의 간격(보통 전체 폭의 5% 이상)은
# 남겨야 한다.
STROKE_MERGE_ITERATIONS = 12
MIN_COMPONENT_AREA_FRAC = 0.03  # 전체 잉크 대비 이보다 작은 성분은 노이즈로 버림
CROP_PADDING_PX = 8
OVERGEN_MARGIN = 0.05  # scripts/eval_overgen.py로 캘리브레이션한 값


@dataclass(frozen=True)
class OvergenResult:
    is_overgen: bool
    n_components: int
    score_whole: float
    score_split: float
    margin: float
    component_readings: Tuple[Optional[str], ...]


def _components(mask: np.ndarray, merge_iterations: int, min_area_frac: float) -> list:
    """(bbox, area) 목록. 왼쪽→오른쪽 정렬, 노이즈 성분은 제외."""
    if not mask.any():
        return []
    merged = binary_dilation(mask, iterations=merge_iterations)
    labeled, n = label(merged)
    total_area = int(mask.sum())
    out = []
    for i in range(1, n + 1):
        comp_mask = mask & (labeled == i)
        area = int(comp_mask.sum())
        if area < min_area_frac * total_area:
            continue
        ys, xs = np.where(comp_mask)
        out.append(((ys.min(), ys.max() + 1, xs.min(), xs.max() + 1), area))
    out.sort(key=lambda c: c[0][2])  # 왼쪽에서 오른쪽 순
    return out


def detect_overgen(
    image: Image.Image,
    margin: float = OVERGEN_MARGIN,
    merge_iterations: int = STROKE_MERGE_ITERATIONS,
    min_component_area_frac: float = MIN_COMPONENT_AREA_FRAC,
) -> OvergenResult:
    mask = raw_ink_mask(image)
    if not mask.any():
        return OvergenResult(False, 0, 0.0, 0.0, 0.0, ())

    comps = _components(mask, merge_iterations, min_component_area_frac)
    whole = read_by_template_match(image)
    if len(comps) <= 1:
        return OvergenResult(False, len(comps), whole.score, whole.score, 0.0, (whole.predicted_char,))

    w, h = image.size
    readings = []
    weighted_sum = 0.0
    total_area = 0
    for (y0, y1, x0, x1), area in comps:
        crop_box = (
            max(0, x0 - CROP_PADDING_PX),
            max(0, y0 - CROP_PADDING_PX),
            min(w, x1 + CROP_PADDING_PX),
            min(h, y1 + CROP_PADDING_PX),
        )
        reading = read_by_template_match(image.crop(crop_box))
        readings.append(reading.predicted_char)
        weighted_sum += reading.score * area
        total_area += area

    score_split = weighted_sum / total_area if total_area else 0.0
    gap = score_split - whole.score
    return OvergenResult(
        is_overgen=gap > margin,
        n_components=len(comps),
        score_whole=whole.score,
        score_split=score_split,
        margin=gap,
        component_readings=tuple(readings),
    )
