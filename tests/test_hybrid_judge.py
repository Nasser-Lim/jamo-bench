# -*- coding: utf-8 -*-
import pytest

from jamo_bench.hybrid_judge import combine, read_hybrid, route_judge


def test_route_no_final_to_clova():
    assert route_judge("가") == "clova"
    assert route_judge("이") == "clova"


def test_route_simple_final_to_template_match():
    assert route_judge("각") == "template_match"


def test_route_cluster_final_to_template_match():
    assert route_judge("값") == "template_match"
    assert route_judge("읽") == "template_match"


def test_route_rejects_non_syllable():
    with pytest.raises(ValueError):
        route_judge("A")


def test_combine_uses_clova_reading_for_no_final_target():
    result = combine("가", clova_reading="가", template_match_reading="다")
    assert result.judge_used == "clova"
    assert result.predicted_char == "가"
    assert result.coda_class == "no_T"
    assert result.expected_bias_pp == 6.0


def test_combine_uses_template_match_reading_for_final_target():
    result = combine("값", clova_reading="가", template_match_reading="값")
    assert result.judge_used == "template_match"
    assert result.predicted_char == "값"
    assert result.coda_class == "cluster_T"
    assert result.expected_bias_pp == 7.5


def test_combine_flags_disagreement():
    result = combine("각", clova_reading="감", template_match_reading="각")
    assert result.judges_agree is False
    result2 = combine("각", clova_reading="각", template_match_reading="각")
    assert result2.judges_agree is True


def test_read_hybrid_only_calls_required_judge():
    calls = {"clova": 0, "tm": 0}

    def clova_fn(img):
        calls["clova"] += 1
        return "가"

    def tm_fn(img):
        calls["tm"] += 1
        return "값"

    result = read_hybrid(object(), "가", clova_fn=clova_fn, template_match_fn=None)
    assert result.predicted_char == "가"
    assert calls == {"clova": 1, "tm": 0}


def test_read_hybrid_calls_both_when_both_provided():
    def clova_fn(img):
        return "가"

    def tm_fn(img):
        return "값"

    result = read_hybrid(object(), "값", clova_fn=clova_fn, template_match_fn=tm_fn)
    assert result.judge_used == "template_match"
    assert result.predicted_char == "값"
    assert result.clova_reading == "가"  # 기록은 되지만 채택은 안 됨
    assert result.judges_agree is False


def test_read_hybrid_raises_when_required_fn_missing():
    with pytest.raises(ValueError):
        read_hybrid(object(), "값", clova_fn=None, template_match_fn=None)
    with pytest.raises(ValueError):
        read_hybrid(object(), "가", clova_fn=None, template_match_fn=lambda img: "가")
