# -*- coding: utf-8 -*-
"""폐쇄형 검증 — 형태(모양) 유사도 매칭 판정자.

CLOVA(사전/언어모델 보정)와 VLM(추론 비용·동일한 사전 편향 위험) 둘 다
"이게 무슨 글자냐"는 개방형 인식 문제를 푼다 — 그래서 실사용에 없는
조합을 만나면 "그럴듯한 실제 단어"로 흔들린다(2026-08-10, `docs/PROGRESS.md`
8단계: 겹받침 셀 60문항 중 실사용 음절 2개는 크기 보정만으로 100%가 됐지만
나머지 28개 희귀 조합은 크기·판정자 종류와 무관하게 전부 0%).

이 모듈은 다른 질문을 푼다: **"이 모양이 11,172개 완성형 음절 중 어느
것과 가장 닮았는가"** — 순수 이미지 유사도 비교이며 언어 지식이 전혀
개입하지 않는다. 사전 편향이 구조적으로 불가능하다.

방법:
  1. 11,172자 전부를 OFL 폰트로 렌더링해 정규화된 흑백 마스크 템플릿을
     1회 만들어 캐싱한다(로컬, 무료, API 호출 0건).
  2. 판독 대상 이미지에서 잉크 영역을 잘라 같은 방식으로 정규화한다.
  3. **soft-IoU**(가우시안 블러 후 min/max 중첩 비율)가 가장 높은
     템플릿을 판독 결과로 채택한다.

실측 발견(2026-08-10): 초기 버전은 순수 이진 IoU를 썼는데, 실제 Seedream
생성물(JPEG 압축·리샘플링으로 획 경계가 원본 폰트와 1~2px 어긋남)에서
"읽"이 훨씬 더 닮은 "엵"에 밀리는 사례가 나왔다 — 가는 획은 몇 픽셀만
어긋나도 겹치는 면적이 급격히 줄어들기 때문. 두 마스크에 동일한
가우시안 블러(σ=3)를 씌워 "부드러운" 중첩 비율로 비교하니 이 민감도가
사라졌다(육안상 명백한 정답 "읽"이 실제로 1위로 올라옴).

이 결과(predicted_char)는 OCR의 candidate_text와 같은 자리에 꽂을 수
있다 — score.py/route.py/match_region.py는 그대로 재사용한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter

from .decompose import all_syllables
from .forge_render import render_clean

CANVAS = 96
DIFF_THRESHOLD = 40
TEMPLATE_RENDER_SIZE = 256
TEMPLATE_TEXT_AREA_FRAC = 0.6  # OCR 안전범위와 무관 — 우리끼리 비교만 하므로 크게 그려 화질 확보
SOFT_BLUR_SIGMA = 3.0  # 실측(2026-08-10)으로 정한 값 — §본문 docstring 참고
CACHE_DIR = Path(__file__).resolve().parent.parent / "results" / "template_cache"

_bool_bank_cache: Dict[Tuple[str, int], Tuple[list, np.ndarray]] = {}
_soft_bank_cache: Dict[Tuple[str, int, float], Tuple[list, np.ndarray]] = {}


def raw_ink_mask(image: Image.Image, diff_threshold: int = DIFF_THRESHOLD) -> np.ndarray:
    """크롭·리센터 없이 원본 좌표계 그대로의 이진 잉크 마스크.

    `_ink_mask_normalized`는 "이미지 전체가 글자 하나"를 전제로 bbox를 잘라
    정사각형에 재배치하는데, 그 전제 자체를 검사(연결요소로 몇 덩어리인지)
    하려면 원본 좌표계가 필요하다 — `overgen.py`가 이 함수를 쓴다.
    """
    gray = np.asarray(image.convert("L")).astype(int)
    bg = int(np.median(gray))
    return np.abs(gray - bg) > diff_threshold


def _ink_mask_normalized(
    image: Image.Image,
    canvas_size: int = CANVAS,
    diff_threshold: int = DIFF_THRESHOLD,
) -> Optional[np.ndarray]:
    """잉크 bbox를 잘라 종횡비를 유지한 채 canvas_size 정사각형 중앙에
    배치한 이진 마스크. 폰트/생성 이미지 양쪽에 동일하게 적용해야
    비교가 공정하다."""
    mask = raw_ink_mask(image, diff_threshold)
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    cropped = mask[y0:y1, x0:x1]

    h, w = cropped.shape
    scale = canvas_size / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    mask_img = Image.fromarray((cropped * 255).astype("uint8"))
    resized = mask_img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    px, py = (canvas_size - new_w) // 2, (canvas_size - new_h) // 2
    canvas.paste(resized, (px, py))
    return np.asarray(canvas) > 127


def _soft(mask_or_stack: np.ndarray, sigma: float = SOFT_BLUR_SIGMA) -> np.ndarray:
    """이진 마스크(2D) 또는 마스크 묶음(3D, 첫 축이 후보 인덱스)에
    가우시안 블러를 씌워 float 배열로 변환한다. 3D는 각 후보를
    독립적으로(첫 축은 블러하지 않고) 블러한다."""
    arr = mask_or_stack.astype(np.float32)
    if arr.ndim == 2:
        return gaussian_filter(arr, sigma=sigma)
    return gaussian_filter(arr, sigma=(0, sigma, sigma))


def _render_template_mask(char: str, font_name: str, canvas_size: int) -> Optional[np.ndarray]:
    img = render_clean(
        char, font_name=font_name, canvas_size=TEMPLATE_RENDER_SIZE, text_area_frac=TEMPLATE_TEXT_AREA_FRAC
    )
    return _ink_mask_normalized(img, canvas_size=canvas_size)


def _build_bool_bank(font_name: str, candidates: Sequence[str], canvas_size: int) -> np.ndarray:
    bank = np.zeros((len(candidates), canvas_size, canvas_size), dtype=bool)
    for i, ch in enumerate(candidates):
        mask = _render_template_mask(ch, font_name, canvas_size)
        if mask is not None:
            bank[i] = mask
    return bank


def _full_syllable_list() -> list:
    return [s.char for s in all_syllables()]


def _full_bool_bank(font_name: str, canvas_size: int = CANVAS) -> Tuple[list, np.ndarray]:
    """11,172자 전체 이진 템플릿 뱅크. 디스크(.npy) 캐싱 — 같은
    폰트·캔버스 조합은 프로세스 재시작 후에도 다시 렌더링하지 않는다."""
    key = (font_name, canvas_size)
    if key in _bool_bank_cache:
        return _bool_bank_cache[key]

    cache_path = CACHE_DIR / f"{font_name}_{canvas_size}.npy"
    syllables = _full_syllable_list()
    if cache_path.is_file():
        arr = np.load(cache_path)
        if arr.shape[0] == len(syllables):
            _bool_bank_cache[key] = (syllables, arr)
            return syllables, arr

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bank = _build_bool_bank(font_name, syllables, canvas_size)
    np.save(cache_path, bank)
    _bool_bank_cache[key] = (syllables, bank)
    return syllables, bank


def _soft_bank(
    font_name: str, candidates: Optional[Sequence[str]], canvas_size: int, sigma: float
) -> Tuple[list, np.ndarray]:
    """블러 적용된(soft) 템플릿 뱅크. candidates=None이면 전체 11,172자
    이진 뱅크(디스크 캐시)를 블러해 메모리에 캐싱한다 — 블러 자체는
    디스크에 저장하지 않는다(용량 4배, σ를 바꾸면 무효화되므로)."""
    if candidates is None:
        key = (font_name, canvas_size, sigma)
        if key in _soft_bank_cache:
            return _soft_bank_cache[key]
        syllables, bool_bank = _full_bool_bank(font_name, canvas_size)
        soft = _soft(bool_bank, sigma)
        _soft_bank_cache[key] = (syllables, soft)
        return syllables, soft

    syllables = list(candidates)
    bool_bank = _build_bool_bank(font_name, syllables, canvas_size)
    return syllables, _soft(bool_bank, sigma)


def _best_match(query_soft: np.ndarray, bank_soft: np.ndarray) -> np.ndarray:
    inter = np.minimum(bank_soft, query_soft).sum(axis=(1, 2))
    union = np.maximum(bank_soft, query_soft).sum(axis=(1, 2))
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


RESIDUAL_TOLERANCE_PX = 2


def _ink_residuals(
    query_mask: np.ndarray,
    best_char: Optional[str],
    font_name: Optional[str],
    canvas_size: int,
    tolerance_px: Optional[int] = None,
) -> Tuple[float, float]:
    """최적 템플릿 대비 **방향을 준** 잉크 잔차.

    soft-IoU는 대칭이라 "여분의 획"이 분모(합집합)에 희석된다 — 받침 ㅁ이
    안에 십자획이 들어간 田으로 그려져도 점수가 크게 안 떨어져 결국
    "감"으로 스냅된다. 여분 획만 따로 재면 그 순간이 드러난다.

    양쪽 마스크를 tolerance_px만큼 팽창시킨 뒤 비교해, JPEG 압축·리샘플링
    으로 획 경계가 1~2px 어긋나는 것(soft-IoU를 도입한 바로 그 이유)을
    잔차로 오계상하지 않게 한다.

    Returns:
        (unexplained_ink, missing_ink) — 각각 0.0~1.0.
    """
    if best_char is None or font_name is None:
        return 0.0, 0.0
    if tolerance_px is None:
        tolerance_px = RESIDUAL_TOLERANCE_PX
    tmpl = _render_template_mask(best_char, font_name, canvas_size)
    if tmpl is None:
        return 0.0, 0.0

    q_sum, t_sum = int(query_mask.sum()), int(tmpl.sum())
    if q_sum == 0 or t_sum == 0:
        return 0.0, 0.0

    tmpl_wide = binary_dilation(tmpl, iterations=tolerance_px)
    query_wide = binary_dilation(query_mask, iterations=tolerance_px)
    unexplained = float((query_mask & ~tmpl_wide).sum()) / q_sum
    missing = float((tmpl & ~query_wide).sum()) / t_sum
    return unexplained, missing


@dataclass(frozen=True)
class TemplateReading:
    predicted_char: Optional[str]
    score: float
    top5: Tuple[Tuple[str, float], ...]
    valid: bool  # False면 이미지에 잉크 자체가 없었다는 뜻(A0에 해당)

    # --- 정형성(well-formedness) 신호 ---
    # 목적: "그려진 모양이 11,172 유효 음절 중 **어느 것도 아니다**"를
    # 판정하는 것. predicted_char는 항상 무언가를 답하므로(폐쇄형 강제
    # 선택) 그 자체로는 이걸 말할 수 없다 — 사람 감사(2026-08-10, 파일럿
    # 256장)에서 17.2%가 "윈도우 키보드로 입력 불가능한 형태"였는데
    # 판정자는 전부 멀쩡한 음절로 스냅했다.
    #
    # 주의: 아래 신호는 전부 **target-independent**여야 한다. 타깃을
    # 참조하는 신호(타깃 순위, 타깃과의 편집거리 등)를 섞으면 "정답에
    # 가까우면 통과"가 되어 자동 판정 구간이 정답 쪽으로 선별되고
    # 정확도가 인위적으로 부풀려진다.
    per_font: Tuple[Tuple[str, str, float], ...] = ()  # (폰트명, 그 폰트의 1위, 점수)
    font_agreement: float = 0.0   # 1위가 최빈 문자와 일치하는 폰트 비율 (1.0 = 전 폰트 합의)
    margin: float = 0.0           # 채택 폰트 내 1위 − 2위 점수차
    unexplained_ink: float = 0.0  # 최적 템플릿으로 설명 안 되는 잉크 비율 (여분 획)
    missing_ink: float = 0.0      # 템플릿에는 있는데 그려지지 않은 비율 (누락 획)


def read_by_template_match(
    image: Image.Image,
    fonts: Sequence[str] = ("noto_sans_kr", "pretendard", "noto_serif_kr"),
    candidates: Optional[Sequence[str]] = None,
    canvas_size: int = CANVAS,
    sigma: float = SOFT_BLUR_SIGMA,
) -> TemplateReading:
    """candidates가 None이면 11,172자 전체(캐싱된 뱅크)와 비교하는
    "완전 폐쇄형 인식"을 수행한다. candidates를 주면 그 후보들끼리만
    비교한다(테스트, 또는 특정 셀 내 최소쌍 진단용으로 유용)."""
    query_mask = _ink_mask_normalized(image, canvas_size=canvas_size)
    if query_mask is None:
        return TemplateReading(predicted_char=None, score=0.0, top5=(), valid=False)
    query_soft = _soft(query_mask, sigma)

    best_char = None
    best_font = None
    best_score = -1.0
    best_margin = 0.0
    best_top5: Tuple[Tuple[str, float], ...] = ()
    per_font: list = []

    for font_name in fonts:
        syllables, bank_soft = _soft_bank(font_name, candidates, canvas_size, sigma)
        scores = _best_match(query_soft, bank_soft)
        top_idx = np.argsort(-scores)[:5]
        idx = int(top_idx[0])
        score = float(scores[idx])
        per_font.append((font_name, syllables[idx], score))

        if score > best_score:
            best_char = syllables[idx]
            best_font = font_name
            best_score = score
            best_top5 = tuple((syllables[i], float(scores[i])) for i in top_idx)
            best_margin = score - float(scores[top_idx[1]]) if len(top_idx) > 1 else 0.0

    # 폰트 간 합의도 — 진짜 유효 음절이면 서로 다른 폰트 뱅크가 독립적으로
    # 같은 음절을 1위로 뽑는다. 존재하지 않는 글자면 각자 다른 데로 흩어진다.
    picks = [p[1] for p in per_font]
    agreement = max(picks.count(c) for c in set(picks)) / len(picks) if picks else 0.0

    unexplained, missing = _ink_residuals(query_mask, best_char, best_font, canvas_size)

    return TemplateReading(
        predicted_char=best_char,
        score=best_score,
        top5=best_top5,
        valid=True,
        per_font=tuple(per_font),
        font_agreement=agreement,
        margin=best_margin,
        unexplained_ink=unexplained,
        missing_ink=missing,
    )


def prebuild_full_bank(fonts: Sequence[str] = ("noto_sans_kr", "pretendard"), canvas_size: int = CANVAS) -> None:
    """11,172자 전체 뱅크를 미리 렌더링해 디스크 캐시를 만들어둔다
    (첫 read_by_template_match 호출 시 자동으로도 만들어지지만,
    폰트당 수만 건 렌더링이라 미리 돌려두면 이후 호출이 즉시 응답한다)."""
    for font_name in fonts:
        _full_bool_bank(font_name, canvas_size)
