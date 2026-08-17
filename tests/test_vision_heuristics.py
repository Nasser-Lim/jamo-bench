# -*- coding: utf-8 -*-
from PIL import Image, ImageDraw

from jamo_bench.vision_heuristics import has_ink_marks
from jamo_bench.forge_render import render_clean


def test_blank_white_canvas_has_no_ink():
    img = Image.new("RGB", (256, 256), "white")
    assert has_ink_marks(img) is False


def test_normal_glyph_has_ink():
    img = render_clean("가", canvas_size=512, text_area_frac=0.3)
    assert has_ink_marks(img) is True


def test_oversized_glyph_still_has_ink_even_though_ocr_would_fail():
    # 실측 실패 사례 재현: 글자가 캔버스 대부분을 차지해도(56%대)
    # "잉크 자체는 있다"는 판정은 OCR 성공 여부와 무관하게 True여야 한다.
    img = render_clean("가", canvas_size=512, text_area_frac=0.56)
    assert has_ink_marks(img) is True


def test_image_with_no_plain_background_is_rejected():
    # 화면이 뚜렷이 다른 세 색조로 균등하게 삼분할된 경우(하늘/산/땅처럼)
    # — 어느 한 색조도 "압도적 배경"이 아니므로 다수인 두 영역이 서로를
    # ink로 만들어 ink_frac(2/3)이 상한(0.5)을 넘는다.
    img = Image.new("RGB", (300, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 300, 100), fill=(255, 255, 255))
    draw.rectangle((0, 100, 300, 200), fill=(128, 128, 128))
    draw.rectangle((0, 200, 300, 300), fill=(0, 0, 0))
    assert has_ink_marks(img) is False


def test_tiny_speck_is_rejected_as_too_little_ink():
    img = Image.new("RGB", (512, 512), "white")
    draw = ImageDraw.Draw(img)
    draw.point((256, 256), fill="black")
    assert has_ink_marks(img) is False
