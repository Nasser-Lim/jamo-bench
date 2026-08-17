# -*- coding: utf-8 -*-
import pytest

from jamo_bench.prompts import PromptSpecError, load_prompt_specs, render_prompt


def test_load_prompt_specs_has_three_templates():
    specs = load_prompt_specs()
    assert set(specs["templates"]) == {"T1", "T2", "T3"}


def test_render_t1_interpolates_target_once():
    text = render_prompt("T1", "읽")
    assert '"읽"' in text
    assert text.count("읽") == 1


def test_render_t3_korean_template():
    text = render_prompt("T3", "값")
    assert '흰 배경 정중앙에 한글 "값" 한 글자만 크게' in text


def test_render_unknown_template_raises():
    with pytest.raises(PromptSpecError):
        render_prompt("T9", "가")
