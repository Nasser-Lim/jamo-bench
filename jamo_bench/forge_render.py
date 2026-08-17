# -*- coding: utf-8 -*-
"""OFL 폰트 렌더러 — JAMO-Forge / Judge Ceiling 대조군 (§7.1, §7.2, §8.4,
JAMO_v51_patch.md §7 degraded ceiling recipe).

두 가지 용도를 하나의 렌더러로 공유한다:
  - clean_ceiling  : Forge와 동일한 클린 렌더링 대조군
  - degraded_ceiling: 폰트 다양화 + 열화 증강(블러/노이즈/JPEG압축/원근/종이질감)

recipe는 버전 고정한다(v5.1 §7.1) — seed와 파라미터 범위를 코드에 명시해
"어떤 recipe로 만든 ceiling인지"가 항상 재현 가능하게 한다.

폰트는 OFL만 쓴다: Noto Sans KR(notofonts/noto-cjk, SIL OFL), Pretendard
(orioncactus/pretendard, SIL OFL). fonts/ 디렉터리에 고지 파일(LICENSE_*.txt)을
함께 커밋한다.
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# 실측(2026-08-09, CLOVA General OCR): text_area_frac=0.5(캔버스 높이의
# 절반을 글자가 차지)에서는 완벽하게 선명한 클린 렌더링조차 CLOVA가
# "NO_TEXT"로 아예 못 읽었다(ENGN-001). 0.3 이하로 낮추면 정상 인식된다 —
# OCR 엔진이 가정하는 "문서 텍스트" 스케일 범위를 벗어난 초대형 글자는
# judge ceiling 자체를 무너뜨린다는 뜻이므로, Ceiling 측정용 캔버스는
# 반드시 대상 T2I 모델의 실측 텍스트 점유 비율에 맞춰야 한다(§7 v5.1).
# 기본값 0.3은 이 실측에서 나온 안전값이며, Phase 0 파일럿에서 T2I 모델별
# 중앙값으로 갱신해야 한다.
FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"

# OFL 폰트만 — 목록을 코드에 고정해 임의 폰트가 섞이는 것을 막는다.
OFL_FONTS = {
    "noto_sans_kr": FONTS_DIR / "NotoSansKR-Regular.otf",
    "pretendard": FONTS_DIR / "Pretendard-Regular.ttf",
    "noto_serif_kr": FONTS_DIR / "NotoSerifKR-Regular.otf",
}

# v5.1 §7.1 degraded_ceiling recipe (버전 고정)
DEGRADE_RECIPE_VERSION = "v1"
BLUR_SIGMA_RANGE = (0.5, 1.5)
JPEG_QUALITY_RANGE = (40, 85)
NOISE_SIGMA_RANGE = (3, 12)  # 8bit 픽셀값 기준 가우시안 노이즈 표준편차
PERSPECTIVE_MAX_SHIFT_FRAC = 0.03  # 캔버스 크기 대비 코너 이동 최대 비율


class ForgeRenderError(RuntimeError):
    pass


def available_fonts() -> Tuple[str, ...]:
    return tuple(name for name, path in OFL_FONTS.items() if path.is_file())


def _load_font(font_name: str, font_size: int) -> ImageFont.FreeTypeFont:
    path = OFL_FONTS.get(font_name)
    if path is None:
        raise ForgeRenderError(f"unknown font_name: {font_name!r} (known: {sorted(OFL_FONTS)})")
    if not path.is_file():
        raise ForgeRenderError(
            f"폰트 파일이 없습니다: {path}. fonts/ 디렉터리에 OFL 폰트를 받아두세요."
        )
    return ImageFont.truetype(str(path), font_size)


def render_clean(
    char: str,
    font_name: str = "noto_sans_kr",
    canvas_size: int = 1024,
    text_area_frac: float = 0.3,
) -> Image.Image:
    """흰 배경에 글자 하나를 중앙에 그린 클린 렌더링(Ceiling 대조군 /
    Forge-plain 산출물의 기반).

    text_area_frac은 v5.1 §7 "Ceiling 측정용 이미지는 대상 T2I 모델의
    평균 해상도·텍스트 점유 면적에 맞춰 렌더링"을 만족하기 위한 손잡이다
    — 실제 파일럿에서 측정한 T2I 출력의 텍스트 바운딩박스 면적 비율(중앙값)로
    맞춰야 하며, 기본값 0.5는 초기값일 뿐 파일럿 측정 후 갱신해야 한다.
    """
    img = Image.new("RGB", (canvas_size, canvas_size), "white")
    draw = ImageDraw.Draw(img)

    target_h = int(canvas_size * text_area_frac)
    font_size = target_h
    font = _load_font(font_name, font_size)
    bbox = font.getbbox(char)
    glyph_w, glyph_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 목표 높이에 맞춰 폰트 크기를 한 번 보정한다 (getbbox는 폰트마다
    # ascent/descent 여백이 달라 font_size == glyph_h가 아니다).
    if glyph_h > 0:
        font_size = max(1, int(font_size * target_h / glyph_h))
        font = _load_font(font_name, font_size)
        bbox = font.getbbox(char)
        glyph_w, glyph_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x = (canvas_size - glyph_w) / 2 - bbox[0]
    y = (canvas_size - glyph_h) / 2 - bbox[1]
    draw.text((x, y), char, font=font, fill="black")
    return img


def _gaussian_noise(img: Image.Image, sigma: float, rng: random.Random) -> Image.Image:
    import numpy as np

    arr = np.asarray(img).astype(np.float32)
    noise = np.array(
        [rng.gauss(0, sigma) for _ in range(arr.size)], dtype=np.float32
    ).reshape(arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(noisy)


def _slight_perspective(img: Image.Image, max_shift_frac: float, rng: random.Random) -> Image.Image:
    w, h = img.size
    max_dx, max_dy = w * max_shift_frac, h * max_shift_frac
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy)),
        (w + rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy)),
        (w + rng.uniform(-max_dx, max_dx), h + rng.uniform(-max_dy, max_dy)),
        (rng.uniform(-max_dx, max_dx), h + rng.uniform(-max_dy, max_dy)),
    ]
    coeffs = _perspective_coeffs(dst, src)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor="white")


def _perspective_coeffs(src_pts, dst_pts):
    """PIL Image.transform(PERSPECTIVE)이 요구하는 8계수를 4쌍의 대응점에서
    푼다 (표준 역변환 계수 계산 — dst 좌표에서 src 좌표를 구하는 방향)."""
    import numpy as np

    matrix = []
    for (x, y), (X, Y) in zip(dst_pts, src_pts):
        matrix.append([x, y, 1, 0, 0, 0, -X * x, -X * y])
        matrix.append([0, 0, 0, x, y, 1, -Y * x, -Y * y])
    A = np.array(matrix, dtype=np.float64)
    B = np.array(src_pts, dtype=np.float64).reshape(8)
    res = np.linalg.solve(A, B)
    return res.tolist()


def _paper_texture_overlay(img: Image.Image, rng: random.Random, strength: float = 0.06) -> Image.Image:
    """가벼운 저주파 얼룩으로 종이 질감을 흉내낸다 — 실제 스캔 텍스처
    라이브러리 없이도 "완전히 균일한 흰 배경이 아님"을 만드는 최소 구현."""
    import numpy as np

    w, h = img.size
    # 저해상도 노이즈를 업샘플링해 저주파 얼룩을 만든다.
    small = np.array(
        [[rng.uniform(-1, 1) for _ in range(w // 32 + 2)] for _ in range(h // 32 + 2)],
        dtype=np.float32,
    )
    texture = Image.fromarray(((small + 1) * 127.5).astype("uint8")).resize((w, h), Image.BICUBIC)
    tex_arr = (np.asarray(texture).astype(np.float32) - 127.5) * strength
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        tex_arr = tex_arr[:, :, None]
    out = np.clip(arr + tex_arr, 0, 255).astype("uint8")
    return Image.fromarray(out)


@dataclass(frozen=True)
class DegradeParams:
    blur_sigma: float
    jpeg_quality: int
    noise_sigma: float
    perspective_applied: bool
    recipe_version: str = DEGRADE_RECIPE_VERSION


def degrade(
    img: Image.Image,
    seed: int,
    apply_perspective: bool = True,
) -> Tuple[Image.Image, DegradeParams]:
    """v5.1 §7.1 degraded_ceiling recipe. seed가 같으면 항상 같은 파라미터·
    같은 결과를 낸다(재현성 요구)."""
    rng = random.Random(seed)

    blur_sigma = rng.uniform(*BLUR_SIGMA_RANGE)
    jpeg_quality = rng.randint(*JPEG_QUALITY_RANGE)
    noise_sigma = rng.uniform(*NOISE_SIGMA_RANGE)

    out = img.convert("RGB")
    if apply_perspective:
        out = _slight_perspective(out, PERSPECTIVE_MAX_SHIFT_FRAC, rng)
    out = _paper_texture_overlay(out, rng)
    out = out.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    out = _gaussian_noise(out, noise_sigma, rng)

    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    out.load()

    return out, DegradeParams(
        blur_sigma=blur_sigma,
        jpeg_quality=jpeg_quality,
        noise_sigma=noise_sigma,
        perspective_applied=apply_perspective,
    )


def render_degraded(
    char: str,
    seed: int,
    font_name: str = "noto_sans_kr",
    canvas_size: int = 1024,
    text_area_frac: float = 0.3,
) -> Tuple[Image.Image, DegradeParams]:
    clean = render_clean(char, font_name=font_name, canvas_size=canvas_size, text_area_frac=text_area_frac)
    return degrade(clean, seed=seed)


def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()
