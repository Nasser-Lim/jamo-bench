# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image

from jamo_bench.forge_render import render_clean
from jamo_bench.judge_preprocess import normalize_occupancy
from jamo_bench.vision_heuristics import has_ink_marks


def test_oversized_glyph_gets_rescaled_below_target():
    img = render_clean("가", canvas_size=512, text_area_frac=0.56)
    result = normalize_occupancy(img)
    assert result.rescaled is True
    assert result.occupancy_before > 0.10
    assert result.occupancy_after <= 0.10 + 1e-6
    assert result.image.size == img.size  # 캔버스 크기는 유지, 내용만 축소


def test_already_small_glyph_is_untouched():
    img = render_clean("가", canvas_size=512, text_area_frac=0.08)
    result = normalize_occupancy(img)
    assert result.rescaled is False
    assert result.occupancy_before == result.occupancy_after
    assert list(result.image.getdata()) == list(img.convert("RGB").getdata())


def test_blank_image_returns_none_occupancy_untouched():
    img = Image.new("RGB", (256, 256), "white")
    result = normalize_occupancy(img)
    assert result.rescaled is False
    assert result.occupancy_before is None
    assert result.occupancy_after is None


def test_rescaled_glyph_still_has_ink_and_is_smaller_fraction():
    img = render_clean("값", canvas_size=512, text_area_frac=0.56)
    before_ink = np.asarray(img.convert("L"))
    result = normalize_occupancy(img)
    assert has_ink_marks(result.image) is True
    # 실제 잉크 픽셀 수(면적)가 줄어들어야 한다
    after_ink = np.asarray(result.image.convert("L"))
    bg_before = int(np.median(before_ink))
    bg_after = int(np.median(after_ink))
    ink_px_before = (np.abs(before_ink.astype(int) - bg_before) > 40).sum()
    ink_px_after = (np.abs(after_ink.astype(int) - bg_after) > 40).sum()
    assert ink_px_after < ink_px_before
