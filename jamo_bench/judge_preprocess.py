# -*- coding: utf-8 -*-
"""OCR judge에 넣기 전 텍스트 점유율 정규화 (JAMO_v51_patch.md §7 "해상도·
텍스트 점유율 동기화").

실측(2026-08-09, CLOVA General OCR): 잉크 bbox가 캔버스 한 변의 약 40~50%를
넘게 차지하면 CLOVA가 "NO_TEXT"로 검출 자체를 실패한다(완벽하게 선명한
클린 렌더링에서도 재현됨). 텍스트 생성 모델마다 글자를 그리는 기본 크기가
달라서, 이 스케일 종속적 맹점을 그대로 두면 "어느 모델이 우연히 글자를
더 크게 그렸는가"가 "그 모델이 한글을 더 못 그린다"로 둔갑한다.

이 전처리는 **무엇을 그렸는지는 바꾸지 않고 얼마나 크게 그렸는지만** judge의
안전 동작 범위로 맞춘다. Core/Cross/Ceiling 등 모든 서브셋·모든 모델에
동일하게 적용해야 공정하다 — 특정 모델만 봐주는 보정이 아니라 judge의
알려진 맹점을 전부에게 동일하게 상쇄하는 것이다.

recipe 버전 고정: v1 (2026-08-09)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

PREPROCESS_RECIPE_VERSION = "v2"
# 실측(2026-08-10): 0.28에서도 CLOVA는 "흔한" 완성형 글자(가/나/다, 값/읽
# 등 실사용 음절)조차 25~50%밖에 못 읽었다 — 겹받침이라서가 아니라
# **글자가 여전히 CLOVA의 기대 스케일보다 컸기 때문**이었다(문서 OCR은
# 원래 페이지 안의 작은 활자를 읽도록 학습된다). 점유율을 0.28→0.03까지
# 훑어 clean 조건에서는 0.03~0.05까지 작을수록 좋았지만, **degraded
# 조건(블러+노이즈+JPEG)에서는 너무 작으면 겹받침의 미세 획이 뭉개져
# 오히려 정확도가 떨어졌다**(0.05에서 겹받침 100%→75%). clean/degraded
# 양쪽에서 가장 안정적인 지점은 0.10 — no_T 75%/75%, simple_T 100%/100%,
# cluster_T 100%/100%(흔한 음절 기준). "작을수록 좋다"가 아니라 이
# 절충점이 실측 최적값이다.
TARGET_MAX_OCCUPANCY = 0.10
INK_DIFF_THRESHOLD = 40


def _ink_bbox(image: Image.Image, diff_threshold: int = INK_DIFF_THRESHOLD) -> Optional[Tuple[int, int, int, int]]:
    gray = np.asarray(image.convert("L")).astype(int)
    bg = int(np.median(gray))
    mask = np.abs(gray - bg) > diff_threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@dataclass(frozen=True)
class PreprocessResult:
    image: Image.Image
    occupancy_before: Optional[float]
    occupancy_after: Optional[float]
    rescaled: bool
    recipe_version: str = PREPROCESS_RECIPE_VERSION


def normalize_occupancy(
    image: Image.Image,
    target_max_occupancy: float = TARGET_MAX_OCCUPANCY,
) -> PreprocessResult:
    """잉크 bbox의 긴 변이 캔버스 대비 target_max_occupancy를 넘으면
    다운스케일해서 배경색 중앙에 다시 배치한다. 넘지 않으면 원본을 그대로
    돌려준다 — 업스케일은 하지 않는다(실측상 작은 글자는 이미 안전
    범위였고, 불필요한 보간이 오히려 화질을 해칠 수 있다).

    잉크가 전혀 없는 이미지(완전 공백)는 occupancy를 계산할 수 없으므로
    None으로 표시하고 원본을 그대로 반환한다 — 이 경우는 애초에 Route A0
    대상이지 스케일 문제가 아니다.
    """
    image = image.convert("RGB")
    w, h = image.size
    bbox = _ink_bbox(image)
    if bbox is None:
        return PreprocessResult(image=image, occupancy_before=None, occupancy_after=None, rescaled=False)

    x0, y0, x1, y1 = bbox
    occupancy_before = max((x1 - x0) / w, (y1 - y0) / h)

    if occupancy_before <= target_max_occupancy:
        return PreprocessResult(
            image=image, occupancy_before=occupancy_before, occupancy_after=occupancy_before, rescaled=False
        )

    scale = target_max_occupancy / occupancy_before
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = image.resize(new_size, Image.LANCZOS)

    bg_val = int(np.median(np.asarray(image.convert("L"))))
    canvas = Image.new("RGB", (w, h), (bg_val, bg_val, bg_val))
    paste_x = (w - new_size[0]) // 2
    paste_y = (h - new_size[1]) // 2
    canvas.paste(resized, (paste_x, paste_y))

    return PreprocessResult(
        image=canvas,
        occupancy_before=occupancy_before,
        occupancy_after=target_max_occupancy,
        rescaled=True,
    )
