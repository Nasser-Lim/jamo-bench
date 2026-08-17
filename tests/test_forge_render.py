# -*- coding: utf-8 -*-
import numpy as np
import pytest
from PIL import Image

from jamo_bench.forge_render import (
    available_fonts,
    degrade,
    image_to_bytes,
    render_clean,
    render_degraded,
)


def test_fonts_are_available():
    fonts = available_fonts()
    assert "noto_sans_kr" in fonts
    assert "pretendard" in fonts


@pytest.mark.parametrize("font_name", ["noto_sans_kr", "pretendard"])
def test_render_clean_draws_black_glyph_on_white(font_name):
    img = render_clean("읽", font_name=font_name, canvas_size=256)
    assert img.size == (256, 256)
    arr = np.asarray(img.convert("L"))
    assert arr.max() >= 250  # 흰 배경
    assert arr.min() <= 30  # 검은 글자 획 존재
    # 글자가 중앙 부근에 있어야 한다 — 가장자리 10px 테두리는 전부 흰색
    border = np.concatenate([arr[:10, :].ravel(), arr[-10:, :].ravel(), arr[:, :10].ravel(), arr[:, -10:].ravel()])
    assert border.min() >= 250


def test_render_clean_rejects_unknown_font():
    from jamo_bench.forge_render import ForgeRenderError

    with pytest.raises(ForgeRenderError):
        render_clean("가", font_name="not_a_real_font")


def test_degrade_is_reproducible_with_same_seed():
    clean = render_clean("가", canvas_size=128)
    out1, params1 = degrade(clean, seed=42)
    out2, params2 = degrade(clean, seed=42)
    assert params1 == params2
    assert list(out1.getdata()) == list(out2.getdata())


def test_degrade_differs_across_seeds():
    clean = render_clean("가", canvas_size=128)
    _, params1 = degrade(clean, seed=1)
    _, params2 = degrade(clean, seed=2)
    assert params1 != params2


def test_degrade_output_is_jpeg_compressed_rgb():
    clean = render_clean("가", canvas_size=128)
    out, params = degrade(clean, seed=7)
    assert out.mode == "RGB"
    assert 40 <= params.jpeg_quality <= 85
    assert 0.5 <= params.blur_sigma <= 1.5


def test_render_degraded_end_to_end():
    img, params = render_degraded("값", seed=1, canvas_size=128)
    assert isinstance(img, Image.Image)
    assert params.recipe_version == "v1"


def test_image_to_bytes_roundtrip():
    img = render_clean("나", canvas_size=64)
    data = image_to_bytes(img, format="PNG")
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    reloaded = Image.open(__import__("io").BytesIO(data))
    assert reloaded.size == (64, 64)
