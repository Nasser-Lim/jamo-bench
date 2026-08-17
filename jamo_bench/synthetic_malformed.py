# -*- coding: utf-8 -*-
"""합성 malformed 글자 생성 — "존재하지 않는 완성형"을 정답이 100%
보장된 상태로 만든다.

**왜 필요한가.** 지금까지 "판정자가 존재하지 않는 글자를 유효 음절로
스냅한다"는 증거는 전부 Seedream 실제 생성물 + 사람 감사에 의존했다
(12~14·17단계). 이건 세 가지 반박에 취약하다: (1) "Seedream 특유의 문제
아니냐" — 생성 모델이 1종뿐, (2) 표본이 300장뿐, (3) 정답이 사람 라벨
(주관)에 의존한다. 이 모듈은 세 가지를 전부 우회한다 — **생성 모델을
아예 안 쓰고, 정답을 구성적으로(by construction) 보장하며, 무료로 표본을
수천 장까지 늘릴 수 있다.**

**방법 — 실제 관찰된 실패 유형을 그대로 재현.** 감사자 메모(13단계)에
기록된 실제 malformed 사례를 프로그램으로 합성한다:

  add_stroke     "받침 ㅁ이 田으로"     — 없던 획을 더한다
  remove_stroke  "'ㅂ' 종성 획 하나 생략" — 있던 획 일부를 지운다

두 방법 모두 **깨끗한 유효 음절 렌더링에서 시작해 국소적으로만 바꾼다**
— 따라서 통제군(control, 손대지 않은 렌더링)과 조작 정도만 다르고 나머지
조건(폰트·해상도·잉크 총량)은 거의 동일하다. 이게 "가짜 이미지를
넣었으니 당연히 틀린다"는 반박을 막는다: 통제군에서 판정자가 잘 맞히면
이미지 품질 문제가 아니고, 그런데도 malformed에서만 자신 있게 유효 음절로
답한다면 그게 측정하려는 현상이다.

**정답 보장의 방식.** "이 이미지가 진짜 11,172개 완성형 중 하나가 아님"을
수학적으로 증명하진 않는다(임의의 화소 패턴이 어떤 폰트의 어떤 글자와도
다르다는 것을 엄밀히 증명하기는 어렵다). 대신 **구성적 사후 검증**을 쓴다
— `template_match`(11,172자 전체 뱅크)로 합성 결과를 채점해, 최고 유사도가
매우 높으면(우연히 다른 진짜 글자를 재구성한 경우) 그 표본은 버린다.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .forge_render import render_clean
from .template_match import raw_ink_mask, read_by_template_match

PerturbKind = Literal["control", "add_stroke", "remove_stroke"]
Severity = Literal["mild", "moderate", "severe"]

# severity -> (스트로크 개수, 스트로크 크기를 잉크 bbox 대비 비율로)
_ADD_PARAMS = {"mild": (1, 0.10), "moderate": (2, 0.16), "severe": (3, 0.24)}
_REMOVE_PARAMS = {"mild": (1, 0.10), "moderate": (2, 0.16), "severe": (3, 0.24)}

# 사후 검증 임계값 — 합성 결과가 뱅크 내 "다른" 진짜 글자와 이 이상
# 닮으면(우연히 다른 유효 음절을 재구성) 표본에서 제외한다. 정상 렌더링이
# 자기 자신과 매칭될 때 점수가 ~0.95~1.0인 것과 대비되는 값이다(9단계).
RECONSTRUCTION_REJECT_THRESHOLD = 0.92


@dataclass(frozen=True)
class SyntheticItem:
    char: str  # 원본(통제군 기준) 문자 — malformed의 "정답"은 없음(구성상 무효)
    font_name: str
    kind: PerturbKind
    severity: str  # control이면 "none"
    seed: int
    image: Image.Image
    verified_novel: bool  # True면 사후 검증 통과(뱅크 내 어떤 글자와도 안 겹침)
    best_bank_score: float  # 검증에 쓴 template_match 최고 점수(진단용)


def _ink_bbox(img: Image.Image) -> Tuple[int, int, int, int]:
    mask = raw_ink_mask(img)
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _add_stroke(img: Image.Image, rng: random.Random, severity: Severity) -> Image.Image:
    """잉크 bbox 안의 무작위 위치에 존재하지 않는 획을 그어 넣는다 —
    실제 사례("받침 ㅁ이 田으로")의 십자·사선 획 추가를 일반화한 것."""
    n_strokes, size_frac = _ADD_PARAMS[severity]
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = _ink_bbox(img)
    w, h = x1 - x0, y1 - y0
    stroke_len = max(4, int(max(w, h) * size_frac))
    stroke_width = max(2, int(min(w, h) * 0.03))

    for _ in range(n_strokes):
        cx = rng.uniform(x0 + w * 0.2, x1 - w * 0.2)
        cy = rng.uniform(y0 + h * 0.2, y1 - h * 0.2)
        angle = rng.uniform(0, 3.14159)
        dx, dy = stroke_len / 2 * np.cos(angle), stroke_len / 2 * np.sin(angle)
        draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill="black", width=stroke_width)
    return out


def _remove_stroke(img: Image.Image, rng: random.Random, severity: Severity) -> Image.Image:
    """잉크 bbox 안의 실제 획 위에 배경색 사각형을 덮어 국소적으로 지운다 —
    실제 사례("'ㅂ' 종성 획 하나 생략")를 일반화한 것. 잉크가 있는 위치만
    골라 지우므로(허공에 지우면 아무 효과 없음) 반드시 눈에 띄는 변화가
    생긴다."""
    n_patches, size_frac = _REMOVE_PARAMS[severity]
    mask = raw_ink_mask(img)
    ys, xs = np.where(mask)
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = _ink_bbox(img)
    w, h = x1 - x0, y1 - y0
    patch = max(4, int(max(w, h) * size_frac))

    ink_points = list(zip(xs.tolist(), ys.tolist()))
    if not ink_points:
        return out
    for _ in range(n_patches):
        px, py = ink_points[rng.randrange(len(ink_points))]
        draw.rectangle([px - patch // 2, py - patch // 2, px + patch // 2, py + patch // 2], fill="white")
    return out


def make_item(
    char: str,
    font_name: str,
    kind: PerturbKind,
    severity: Severity,
    seed: int,
    canvas_size: int = 1024,
    text_area_frac: float = 0.3,
    verify: bool = True,
) -> SyntheticItem:
    base = render_clean(char, font_name=font_name, canvas_size=canvas_size, text_area_frac=text_area_frac)
    rng = random.Random(seed)

    if kind == "control":
        img = base
    elif kind == "add_stroke":
        img = _add_stroke(base, rng, severity)
    elif kind == "remove_stroke":
        img = _remove_stroke(base, rng, severity)
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    verified_novel, best_score = True, 0.0
    if verify and kind != "control":
        reading = read_by_template_match(img)
        best_score = reading.score
        # 자기 자신(char)과 매칭되는 것은 "재구성 실패"가 아니다 — 원본과
        # 다른 진짜 글자로 우연히 재구성됐을 때만 검증 실패로 본다.
        verified_novel = not (reading.predicted_char != char and best_score >= RECONSTRUCTION_REJECT_THRESHOLD)

    return SyntheticItem(char, font_name, kind, severity if kind != "control" else "none", seed, img, verified_novel, best_score)


def build_dataset(
    syllables: Sequence[str],
    fonts: Sequence[str] = ("noto_sans_kr", "pretendard", "noto_serif_kr"),
    severities: Sequence[Severity] = ("mild", "moderate", "severe"),
    seed: int = 0,
    verify: bool = True,
) -> List[SyntheticItem]:
    """음절마다: control 1개 + (add_stroke, remove_stroke) × severities.

    같은 seed는 항상 같은 결과를 낸다(재현성). 검증 실패(우연히 다른 진짜
    글자를 재구성) 표본은 호출자가 `item.verified_novel`로 걸러낸다 —
    자동으로 빼지 않는 것은 거부율 자체도 보고 가치가 있기 때문이다.
    """
    rng = random.Random(seed)
    items: List[SyntheticItem] = []
    for char in syllables:
        for font_name in fonts:
            items.append(make_item(char, font_name, "control", "none", rng.randrange(1 << 30), verify=False))
            for kind in ("add_stroke", "remove_stroke"):
                for severity in severities:
                    items.append(
                        make_item(char, font_name, kind, severity, rng.randrange(1 << 30), verify=verify)
                    )
    return items
