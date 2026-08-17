# -*- coding: utf-8 -*-
"""영역 매칭과 폴백 사다리 (JAMO_benchmark_design.md §8.3).

CLOVA/OSS OCR이 뽑아낸 필드 목록에서 "이 이미지의 대표 후보가 무엇인가"를
결정한다. 후보 선택 순서는 설계서가 명시한 그대로다:

    ① 중심 근접도 → ② 면적 → ③ 편집거리 → confidence 하한 필터

편집거리를 1차 기준으로 쓰면 "정답과 비슷한 것을 찾는" 편향이 생긴다는
경고(§8.3)에 따라 3번째 tie-break로만 쓴다.

폴백 사다리:
    F0  정상 — primary region(중앙 60% 박스)에 후보가 정확히 1개
    F1  중앙 무텍스트 → 최대면적 후보 사용, position_miss=True
    F2  중앙 침범(겹치는 후보 다수) → 감사 확대, Official 제외(needs_audit=True)
    F3  Multi(겹치지 않는 후보 다수) → 집합 기반 채점, 최근접 후보를 대표로
    F4  후보 없음 → 드롭
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .align import align_syllables
from .clova_ocr import BBox, OcrField, bbox_center
from .decompose import is_hangul_syllable

MatchingRule = str  # "F0" | "F1" | "F2" | "F3" | "F4"


def primary_region_box(width: float, height: float, frac: float = 0.6) -> BBox:
    """이미지 중앙 frac 비율(기본 60%) 박스."""
    dx = width * (1 - frac) / 2
    dy = height * (1 - frac) / 2
    return (dx, dy, width - dx, height - dy)


def bbox_area(bbox: BBox) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _point_in_box(point: Tuple[float, float], box: BBox) -> bool:
    x, y = point
    x0, y0, x1, y1 = box
    return x0 <= x <= x1 and y0 <= y <= y1


def _overlaps(a: BBox, b: BBox, min_frac: float = 0.3) -> bool:
    """두 bbox가 상당 부분 겹치는지("중앙 침범"). 더 작은 쪽 면적의
    min_frac 이상을 교집합이 차지하면 겹침으로 본다."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    smaller = min(bbox_area(a), bbox_area(b))
    if smaller <= 0:
        return False
    return inter / smaller >= min_frac


def _edit_distance_to(text: str, target: Optional[str]) -> int:
    if target is None:
        return 0
    _, dist = align_syllables(target, text)
    return dist


@dataclass(frozen=True)
class _Candidate:
    field: OcrField
    hangul_text: str
    center: Tuple[float, float]
    distance_to_image_center: float
    area: float
    edit_distance: int
    in_primary_region: bool


@dataclass(frozen=True)
class MatchResult:
    matching_rule: MatchingRule
    candidate_text: Optional[str]
    candidate_bbox: Optional[BBox]
    candidate_confidence: Optional[float]
    """채택된 후보의 OCR confidence(원본 OcrField.confidence 그대로).

    실측(2026-08-10, 256장 사람 감사 대조): confidence는 "정답 여부"의
    유효한 신호다(정답 평균 0.966 vs 오답 평균 0.677 — 0.9 이상만
    채택하면 정밀도 43.3%→75.9%). 반면 "애초에 한글이 아닌 출력"과
    "진짜 한글을 잘못 읽은 것"은 confidence로 구분되지 않는다(오답 내에서
    각각 평균 0.696 / 0.669로 거의 동일) — 이 필드를 신뢰 게이트로는
    쓰되, 한글 여부 판정기로 쓰면 안 된다."""
    position_miss: bool
    needs_audit: bool
    dropped: bool
    all_hangul_texts: Tuple[str, ...]
    spurious_count: int
    spurious_area_frac: float


def match_target_region(
    fields: Sequence[OcrField],
    image_width: float,
    image_height: float,
    target: Optional[str] = None,
    primary_region_frac: float = 0.6,
    confidence_floor: Optional[float] = None,
) -> MatchResult:
    """OCR 필드 목록에서 대표 후보 1개를 고르고 matching_rule(F0~F4)을
    함께 반환한다. target은 편집거리 tie-break에만 쓰이며(3번째 기준),
    후보 채택 여부 자체를 좌우하지 않는다 — 편집거리를 1차 필터로 쓰면
    "정답과 비슷한 것만 찾는" 편향이 생긴다(§8.3)."""
    if confidence_floor is not None:
        fields = [f for f in fields if f.confidence is None or f.confidence >= confidence_floor]

    img_center = (image_width / 2, image_height / 2)
    pbox = primary_region_box(image_width, image_height, primary_region_frac)

    candidates = []
    for f in fields:
        if f.bbox is None:
            continue
        hangul_text = "".join(c for c in f.text if is_hangul_syllable(c))
        if not hangul_text:
            continue
        center = bbox_center(f.bbox)
        candidates.append(
            _Candidate(
                field=f,
                hangul_text=hangul_text,
                center=center,
                distance_to_image_center=math.hypot(
                    center[0] - img_center[0], center[1] - img_center[1]
                ),
                area=bbox_area(f.bbox),
                edit_distance=_edit_distance_to(hangul_text, target),
                in_primary_region=_point_in_box(center, pbox),
            )
        )

    all_texts = tuple(c.hangul_text for c in candidates)
    total_area = image_width * image_height

    def _result(rule, best, others, position_miss=False, needs_audit=False, dropped=False):
        spurious_area = sum(c.area for c in others) / total_area if total_area else 0.0
        return MatchResult(
            matching_rule=rule,
            candidate_text=best.hangul_text if best else None,
            candidate_bbox=best.field.bbox if best else None,
            candidate_confidence=best.field.confidence if best else None,
            position_miss=position_miss,
            needs_audit=needs_audit,
            dropped=dropped,
            all_hangul_texts=all_texts,
            spurious_count=len(others),
            spurious_area_frac=spurious_area,
        )

    if not candidates:
        return _result("F4", None, [], dropped=True)

    primary_candidates = [c for c in candidates if c.in_primary_region]

    if not primary_candidates:
        # F1: 중앙에 텍스트가 없다 → 최대면적 후보를 대신 쓰고 position_miss 기록
        best = max(candidates, key=lambda c: c.area)
        others = [c for c in candidates if c is not best]
        return _result("F1", best, others, position_miss=True)

    if len(primary_candidates) == 1:
        best = primary_candidates[0]
        others = [c for c in candidates if c is not best]
        return _result("F0", best, others)

    ranked = sorted(
        primary_candidates,
        key=lambda c: (c.distance_to_image_center, -c.area, c.edit_distance),
    )
    best = ranked[0]
    others = [c for c in candidates if c is not best]

    overlapping = any(
        _overlaps(a.field.bbox, b.field.bbox)
        for i, a in enumerate(primary_candidates)
        for b in primary_candidates[i + 1 :]
    )
    if overlapping:
        # F2: 중앙 영역을 여러 후보가 침범 → 사람 감사로, Official 제외
        return _result("F2", best, others, needs_audit=True)

    # F3: 중앙에 겹치지 않는 후보가 여럿(예: 다중 문자 렌더링) → 최근접 후보를
    # 대표로 쓰되, all_hangul_texts로 집합 전체를 남겨 후속 분석이 재구성할
    # 수 있게 한다.
    return _result("F3", best, others)
