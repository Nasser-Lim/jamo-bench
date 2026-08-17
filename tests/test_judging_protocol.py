# -*- coding: utf-8 -*-
from jamo_bench.judging_protocol import (
    CONFIDENCE_THRESHOLD,
    MEASURED_BIAS_PP,
    NO_AUTO_ONSET_GROUPS,
    resolve_human,
    route,
)


def test_high_confidence_is_auto_accepted():
    d = route("값", 0.97, "cluster_T")
    assert not d.needs_human
    assert d.reading == "값"
    assert d.source == "clova"


def test_tense_onset_always_goes_to_human_regardless_of_confidence():
    """쌍자음 초성에서 CLOVA는 자신 있게 틀린다 — 자동 채택 구간 불일치율
    47.4%(simple_O 4.5% 대비)."""
    assert route("쌈", 0.99, "simple_T", onset_group="tense_O").needs_human
    assert not route("삼", 0.99, "simple_T", onset_group="simple_O").needs_human
    assert not route("참", 0.99, "simple_T", onset_group="aspir_O").needs_human


def test_onset_group_omitted_does_not_block_auto_accept():
    """초성군을 모르면 confidence만으로 판단한다(하위 호환)."""
    assert not route("값", 0.97, "cluster_T").needs_human


def test_no_auto_onset_groups_is_not_empty():
    """이 집합이 비면 초성군 축 편향이 10.4%p로 되돌아간다 — 회귀 방지."""
    assert "tense_O" in NO_AUTO_ONSET_GROUPS


def test_low_confidence_goes_to_human():
    d = route("값", 0.42, "cluster_T")
    assert d.needs_human
    assert d.reading is None
    assert d.source == "human_pending"


def test_threshold_is_inclusive():
    assert not route("값", CONFIDENCE_THRESHOLD, "no_T").needs_human
    assert route("값", CONFIDENCE_THRESHOLD - 0.001, "no_T").needs_human


def test_missing_confidence_always_goes_to_human():
    """CLOVA가 아무것도 검출하지 못한 경우(파일럿 256장 중 4건)."""
    assert route("값", None, "no_T").needs_human
    assert route(None, 0.99, "no_T").needs_human


def test_expected_bias_is_attached_per_coda_class():
    for coda in ("no_T", "simple_T", "cluster_T"):
        assert route("값", 0.99, coda).expected_bias_pp == MEASURED_BIAS_PP[coda]
    # 알 수 없는 축이면 전체 편향으로 폴백
    assert route("값", 0.99, None).expected_bias_pp == MEASURED_BIAS_PP["overall"]


def test_measured_bias_is_never_positive():
    """프로토콜은 사람보다 낮거나 같게 나온다 — 부호가 뒤집히면 보정 방향이
    반대가 되므로 회귀로 잡는다."""
    assert all(v <= 0 for v in MEASURED_BIAS_PP.values())


def test_resolve_human_with_valid_syllable():
    d = resolve_human(route("갑", 0.3, "simple_T"), human_valid=True, human_transcription="값")
    assert not d.needs_human
    assert d.reading == "값"
    assert d.source == "human"


def test_resolve_human_with_invalid_glyph_yields_no_reading():
    """사람이 '유효 완성형 아님'이라 하면 전사가 없다 — 판독 실패가 아니라
    '모델이 존재하지 않는 글자를 그렸다'는 측정값이다."""
    d = resolve_human(route("갑", 0.3, "simple_T"), human_valid=False, human_transcription=None)
    assert not d.needs_human
    assert d.reading is None
    assert d.reading != "갑"


def test_resolve_human_is_noop_on_auto_accepted():
    auto = route("값", 0.99, "cluster_T")
    assert resolve_human(auto, human_valid=True, human_transcription="갑") == auto
