# -*- coding: utf-8 -*-
"""decompose.py 검증 — jamo_18cell_design.py / jamo_v51_verification.py
실행 결과와 수치가 일치하는지 대조하는 regression test."""
import pytest

from jamo_bench.decompose import (
    ALL_18_CELLS,
    ALL_6_CELLS,
    Syllable,
    all_syllables,
    cell_index_18,
    cell_index_6,
    decompose,
    is_hangul_syllable,
    normalize,
)


def test_decompose_known_syllables():
    s = decompose("읽")
    assert (s.onset, s.nucleus, s.coda) == ("ㅇ", "ㅣ", "ㄺ")
    assert s.coda_class_3 == "cluster_T"
    assert s.coda_class_4 == "cluster_mixed"

    s = decompose("값")
    assert (s.onset, s.nucleus, s.coda) == ("ㄱ", "ㅏ", "ㅄ")
    assert s.coda_class_4 == "cluster_mixed"

    s = decompose("갔")
    assert s.coda == "ㅆ"
    assert s.coda_class_3 == "simple_T"  # 3-way에서는 tensed가 simple_T에 섞인다
    assert s.coda_class_4 == "tensed_double"

    s = decompose("가")
    assert s.coda == "" and s.coda_class_3 == "no_T" and s.coda_class_4 == "none"

    s = decompose("의")
    assert s.nucleus == "ㅢ"
    assert s.vowel_class_2 == "complex_V"
    assert s.vowel_shape == "complex_block"

    s = decompose("책")
    assert s.nucleus == "ㅐ"
    assert s.vowel_class_2 == "simple_V"  # 18셀 축에서는 병합
    assert s.vowel_shape == "vertical_derived"  # 메타데이터 축은 보존
    assert s.is_ae_e_pair_member is True


def test_decompose_rejects_non_syllables():
    assert decompose("A") is None
    assert decompose("ㄱ") is None
    assert decompose("") is None
    assert decompose("가나") is None


def test_is_hangul_syllable():
    assert is_hangul_syllable("가") is True
    assert is_hangul_syllable("a") is False


@pytest.mark.parametrize(
    "onset,group",
    [
        ("ㅇ", "simple_O"),
        ("ㄱ", "simple_O"),
        ("ㅎ", "aspir_O"),
        ("ㅋ", "aspir_O"),
        ("ㄲ", "tense_O"),
        ("ㅆ", "tense_O"),
    ],
)
def test_onset_group(onset, group):
    ch = chr(0xAC00 + [
        "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ",
        "ㅋ", "ㅌ", "ㅍ", "ㅎ",
    ].index(onset) * 588)
    s = decompose(ch)
    assert s.onset == onset
    assert s.onset_group == group


def test_ieung_onset_flag():
    assert decompose("아").is_ieung_onset is True
    assert decompose("가").is_ieung_onset is False


def test_total_syllable_count_is_11172():
    assert len(all_syllables()) == 11172
    assert sum(1 for s in all_syllables() if s is not None) == 11172


def test_18_cells_all_at_least_35():
    idx = cell_index_18()
    assert set(idx.keys()) == set(ALL_18_CELLS)
    counts = {cell: len(chars) for cell, chars in idx.items()}
    assert sum(counts.values()) == 11172
    assert min(counts.values()) == 35  # jamo_18cell_design.py 실측과 일치
    assert all(n >= 30 for n in counts.values())


def test_6_cells_cover_18_cells():
    idx6 = cell_index_6()
    idx18 = cell_index_18()
    assert set(idx6.keys()) == set(ALL_6_CELLS)
    for (v, t) in ALL_6_CELLS:
        expected = sum(
            len(idx18[(v, t, o)]) for o in ("simple_O", "aspir_O", "tense_O")
        )
        assert len(idx6[(v, t)]) == expected


def test_tensed_double_count_matches_verification_script():
    # jamo_v51_verification.py 실측: 798
    n = sum(1 for s in all_syllables() if s.coda_class_4 == "tensed_double")
    assert n == 798


def test_vertical_derived_count_matches_verification_script():
    # jamo_v51_verification.py 실측: 2128 (19.0%)
    n = sum(1 for s in all_syllables() if s.vowel_shape == "vertical_derived")
    assert n == 2128
    assert round(n / 11172 * 100, 1) == 19.0


def test_normalize_strips_space_and_punctuation_after_nfc():
    assert normalize("책 읽기") == "책읽기"
    assert normalize("책읽기") == "책읽기"
    assert normalize("책  읽 기") == "책읽기"
    assert normalize("책읽기!") == "책읽기"


def test_compose_is_inverse_of_decompose():
    from jamo_bench.decompose import compose

    for ch in ["가", "값", "읽", "콃", "아"]:
        s = decompose(ch)
        assert compose(s.onset, s.nucleus, s.coda) == ch


def test_compose_rejects_invalid_jamo():
    from jamo_bench.decompose import compose

    assert compose("A", "ㅏ", "") is None
    assert compose("ㄱ", "X", "") is None
    assert compose("ㄱ", "ㅏ", "X") is None


def test_compose_default_coda_is_none():
    from jamo_bench.decompose import compose

    assert compose("ㄱ", "ㅏ") == "가"
