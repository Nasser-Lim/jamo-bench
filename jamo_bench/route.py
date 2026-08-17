# -*- coding: utf-8 -*-
"""채점 전 라우터 — Route A0/A1/B/C 분류 (JAMO_v51_patch.md §5).

v5의 3분류(A/B/C)는 "프롬프트 실패 ∪ 렌더 실패 ∪ OCR 실패"를 한 버킷에 묶어
스크리닝·Go 조건을 흔들었다. v5.1은 이를 4분류로 쪼갠다:

    A0  텍스트 유사 표식 자체가 없음(풍경/인물만)        → 프롬프트 미이행
    A1  문자 유사 표식은 있으나 한글 완성형 0개           → 문자 체계 실패
    B   primary region(중앙 60% 박스)에서 한글 검출       → 렌더링 시도
    C   한자·로마자 혼입, 자모 분리, 모호                 → 사람 감사 우선

이 모듈은 실제 OCR·비전 휴리스틱을 호출하지 않는다 — 그 결과를
`DetectionInput`으로 미리 뽑아 넣으면 판정 로직만 수행한다(Route judge는
OSS OCR로 고정하되, OCR 연동 자체는 다음 스텝 범위).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from .clova_ocr import OcrField, bbox_center
from .decompose import is_hangul_syllable
from .match_region import primary_region_box

Route = Literal["A0", "A1", "B", "C"]

# 한글 호환 자모 낱자 블록(자모 분리 렌더링 검출용) — ㄱ~ㅎ, ㅏ~ㅣ.
_COMPAT_JAMO_RANGES = ((0x3131, 0x314E), (0x314F, 0x3163))


def _is_isolated_jamo(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _COMPAT_JAMO_RANGES)


@dataclass(frozen=True)
class DetectionInput:
    has_text_like_region: bool
    """엣지밀도/획 유사 패턴 휴리스틱 또는 사람 판정 — "글자를 그리려는
    시도 자체가 있었는가"."""
    hangul_completions: list
    """OCR이 검출한 완성형 한글 후보 문자열 리스트 (없으면 빈 리스트)."""
    in_primary_region: list
    """hangul_completions와 같은 길이 — 각 후보가 중앙 60% 박스 안인지."""
    has_non_hangul_mixed: bool = False
    """한자·로마자 혼입 또는 자모 분리 렌더링(예: 'ㅇㅣㄺ') 검출 여부."""


def classify_route(det: DetectionInput) -> Route:
    """v5.1 §5 판정 순서: 글자 시도 유무 → 한글 검출 → primary region.

    판정 트리에 명시되지 않은 두 경계 케이스를 다음과 같이 해석한다
    (문서가 원칙만 제시하고 우선순위를 규정하지 않은 지점):
      - 한글 완성형이 1개 이상 검출됐더라도 다른 문자체계가 섞여 있으면
        (has_non_hangul_mixed) 렌더링 성공 여부가 모호하므로 B보다 C를
        우선한다.
      - 한글 완성형은 있으나 전부 primary region 밖(유령 텍스트 등)이면
        "렌더링 시도"로 단정하기 어려우므로 A1이 아니라 C(사람 감사)로
        보낸다.
    """
    if not det.has_text_like_region:
        return "A0"
    if det.has_non_hangul_mixed:
        return "C"
    if not det.hangul_completions:
        return "A1"
    if any(det.in_primary_region):
        return "B"
    return "C"


def build_detection_input(
    fields: Sequence[OcrField],
    image_width: float,
    image_height: float,
    has_text_like_region: bool,
    primary_region_frac: float = 0.6,
) -> DetectionInput:
    """실제 OCR 필드 목록에서 DetectionInput을 조립한다 — route.py의 분류
    로직과 실제 clova_ocr 출력을 연결하는 접착 지점.

    has_text_like_region은 **호출부가 OCR과 무관하게 판정해서 넘겨야 한다**
    (예: `jamo_bench.vision_heuristics.has_ink_marks(image)`). 예전에는
    `len(fields) > 0`로 근사했는데, 이는 "OCR이 뭐라도 검출했는가"를
    "글자를 그리려는 시도가 있었는가"와 동일시하는 것이라 v5.1 §5가 경고한
    바로 그 실패를 재현한다 — OCR이 멀쩡한 글자를 못 읽으면 그게 그대로
    프롬프트 미이행(A0)으로 둔갑한다. 실측 확인(2026-08-09): 캔버스의
    56%를 차지한 완벽한 "와" 렌더링을 CLOVA가 검출하지 못해
    `len(fields) > 0` 기준으로는 A0가 됐지만, 같은 이미지는 명백히
    "글자 시도"가 있었다(여백만 추가하자 OCR도 정상 검출했다)."""
    pbox = primary_region_box(image_width, image_height, primary_region_frac)

    hangul_completions: list = []
    in_primary_region: list = []
    has_non_hangul_mixed = False

    for f in fields:
        hangul_chars = [c for c in f.text if is_hangul_syllable(c)]
        non_hangul_chars = [c for c in f.text if not c.isspace() and not is_hangul_syllable(c)]

        if non_hangul_chars and (hangul_chars or any(_is_isolated_jamo(c) for c in non_hangul_chars)):
            has_non_hangul_mixed = True

        if not hangul_chars:
            continue
        hangul_completions.append("".join(hangul_chars))
        if f.bbox is None:
            in_primary_region.append(False)
            continue
        cx, cy = bbox_center(f.bbox)
        x0, y0, x1, y1 = pbox
        in_primary_region.append(x0 <= cx <= x1 and y0 <= cy <= y1)

    return DetectionInput(
        has_text_like_region=has_text_like_region,
        hangul_completions=hangul_completions,
        in_primary_region=in_primary_region,
        has_non_hangul_mixed=has_non_hangul_mixed,
    )
