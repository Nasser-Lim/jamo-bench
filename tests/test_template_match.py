# -*- coding: utf-8 -*-
"""template_match.py 단위 테스트 — 작은 candidates 목록으로 속도 확보
(전체 11,172자 뱅크는 별도 검증 스크립트에서 다룸)."""
from PIL import Image

from jamo_bench.forge_render import render_clean
from jamo_bench.template_match import read_by_template_match


def test_self_match_high_score():
    img = render_clean("값", font_name="noto_sans_kr", canvas_size=512, text_area_frac=0.4)
    result = read_by_template_match(
        img, fonts=("noto_sans_kr",), candidates=["값", "갑", "갔", "각", "가"]
    )
    assert result.predicted_char == "값"
    assert result.valid is True
    assert result.score > 0.5


def test_distinguishes_rare_confusable_pair():
    # 콃(ㅋ+ㅖ+ㄼ) vs 켸(ㅋ+ㅖ, 무종성) vs 래(ㄹ+ㅐ) — CLOVA가 "켸래"로
    # 쪼개 읽었던 바로 그 사례. 모양 비교는 사전 지식이 없으니 그대로
    # 콃을 골라야 한다.
    img = render_clean("콃", font_name="noto_sans_kr", canvas_size=512, text_area_frac=0.4)
    result = read_by_template_match(
        img, fonts=("noto_sans_kr",), candidates=["콃", "켸", "래", "콰", "콱"]
    )
    assert result.predicted_char == "콃"


def test_blank_image_is_invalid():
    img = Image.new("RGB", (256, 256), "white")
    result = read_by_template_match(img, fonts=("noto_sans_kr",), candidates=["가", "나"])
    assert result.valid is False
    assert result.predicted_char is None


def test_cross_font_robustness():
    # pretendard로 그린 걸 noto_sans_kr 후보군과 비교해도(폰트 스타일이
    # 다름) 정답을 골라야 한다 — 완전히 같은 렌더러가 아니어도 견뎌야
    # 실제 T2I 생성 이미지(폰트도 아닌 스타일)에 적용할 근거가 생긴다.
    img = render_clean("읽", font_name="pretendard", canvas_size=512, text_area_frac=0.4)
    result = read_by_template_match(
        img, fonts=("noto_sans_kr",), candidates=["읽", "익", "일", "엮", "읊"]
    )
    assert result.predicted_char == "읽"


def test_top5_contains_target_even_when_not_top1_for_hard_case():
    img = render_clean("퐋", font_name="noto_sans_kr", canvas_size=512, text_area_frac=0.4)
    result = read_by_template_match(
        img, fonts=("noto_sans_kr",), candidates=["퐋", "퉑", "폴", "퐝", "폭"]
    )
    top5_chars = [c for c, _ in result.top5]
    assert "퐋" in top5_chars


def test_score_is_between_0_and_1():
    img = render_clean("가", font_name="noto_sans_kr", canvas_size=512, text_area_frac=0.4)
    result = read_by_template_match(img, fonts=("noto_sans_kr",), candidates=["가", "나", "다"])
    assert 0.0 <= result.score <= 1.0
