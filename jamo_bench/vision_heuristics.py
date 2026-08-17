# -*- coding: utf-8 -*-
"""OCR과 독립적인 최소 비전 휴리스틱 (JAMO_v51_patch.md §5).

Route 판정의 1단계("글자 시도가 있었는가")는 OCR 성공 여부와 **독립적**이어야
한다. `has_text_like_region`을 "OCR이 뭐라도 검출했는가"로 근사하면, OCR이
멀쩡한 글자를 못 읽었을 때 그게 그대로 "프롬프트 미이행(A0)"으로 둔갑한다
— v5.1 §5가 정확히 경고한 실패 모드다.

실측 확인(2026-08-09): T1 프롬프트로 만든 완벽한 "와" 렌더링이 글자가
캔버스의 56%를 차지한다는 이유만으로 CLOVA가 검출을 못 했고, 그 결과
`len(fields) > 0` 기반 근사가 이 샘플을 A0로 잘못 분류했다. 같은 이미지에
여백만 추가(잉크 픽셀은 그대로)하자 OCR이 정상 검출했다 — 이미지 자체에는
"글자 시도"가 명백히 있었다는 뜻이다.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

DEFAULT_DIFF_THRESHOLD = 40
DEFAULT_MIN_BACKGROUND_FRAC = 0.3
DEFAULT_INK_FRAC_RANGE = (0.003, 0.5)


def has_ink_marks(
    image: Image.Image,
    diff_threshold: int = DEFAULT_DIFF_THRESHOLD,
    min_background_frac: float = DEFAULT_MIN_BACKGROUND_FRAC,
    ink_frac_range: Tuple[float, float] = DEFAULT_INK_FRAC_RANGE,
) -> bool:
    """배경과 뚜렷이 다른 잉크성 픽셀이 합리적인 비율로 있는지를 OCR 없이
    판정한다.

    가정: 프롬프트가 요구하는 배경(흰 바탕 / 스튜디오 사진)은 화면 대부분을
    차지하는 단일 색조에 가깝다. 배경은 중앙값으로 추정한다(대부분의
    픽셀이 배경이라는 전제 — 우리 프롬프트가 항상 "단일 글자 + 배경"
    구도를 요구하므로 타당하다). 그와 충분히 다른 픽셀 비율이
      - 너무 적으면(거의 전부 배경) → 아무것도 안 그려짐 → False
      - 너무 많으면(화면 대부분이 배경과 다름, 즉 복잡한 사진/장면) → False
    인 경우를 걸러내고, 그 사이 구간만 "글자를 그리려는 시도가 있었다"로
    본다.

    이 함수는 "정말 글자인지"는 판정하지 않는다 — 그건 OCR과 사람 감사의
    몫이다. 여기서는 딱 하나, "OCR이 아무것도 못 찾았다는 사실이 곧 아무것도
    없었다는 뜻은 아니다"라는 최소한의 반증만 제공한다.
    """
    gray = np.asarray(image.convert("L")).astype(int)
    bg = int(np.median(gray))
    diff = np.abs(gray - bg)

    background_like_frac = float((diff <= diff_threshold).mean())
    if background_like_frac < min_background_frac:
        return False

    ink_frac = float((diff > diff_threshold).mean())
    return ink_frac_range[0] <= ink_frac <= ink_frac_range[1]
