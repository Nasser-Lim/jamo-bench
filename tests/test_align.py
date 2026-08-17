# -*- coding: utf-8 -*-
from jamo_bench.align import align_syllables, score_word


def test_one_to_one_substitution_pilot_case():
    # 파일럿 실측: "책 읽기" -> "챽 엮기" (공백은 normalize()가 이미 제거했다고 가정)
    result = score_word("책읽기", "챽엮기")
    assert result.edit_distance == 2  # 책->챽, 읽->엮만 다름; 기==기는 match
    assert result.syllable_insert_count == 0
    assert result.syllable_delete_count == 0
    assert not result.overgen
    assert len(result.substitution_scores) == 3
    verdicts = [s.verdict for s in result.substitution_scores]
    assert verdicts == ["VALID", "VALID", "VALID"]


def test_overgen_prefix_scored_tail_as_insert():
    result = score_word("읽", "읽기")
    assert result.overgen is True
    assert result.syllable_insert_count == 1
    assert result.syllable_delete_count == 0
    assert result.edit_distance == 1
    # prefix(읽)는 match로 자모 분해 대상에 포함된다
    assert len(result.substitution_scores) == 1
    assert result.substitution_scores[0].verdict == "VALID"
    assert result.substitution_scores[0].onset_ok is True


def test_deletion_counts_as_syllable_delete_only():
    result = score_word("책읽기", "책읽")
    assert result.syllable_delete_count == 1
    assert result.syllable_insert_count == 0
    assert result.edit_distance == 1
    assert len(result.substitution_scores) == 2  # 책, 읽만 자모 분해


def test_route_hint_c_when_alignment_infeasible():
    # edit_distance(2) > len(target)(1) → 정렬 신뢰 불가
    result = score_word("가", "나다")
    assert result.edit_distance == 2
    assert result.route_hint == "C"


def test_route_hint_c_on_length_guard():
    result = score_word("가", "나" * 10)  # 타깃 길이의 3배 초과
    assert result.route_hint == "C"


def test_align_ops_are_ordered_left_to_right():
    ops, dist = align_syllables("책읽기", "챽엮기")
    assert [op.target_char for op in ops] == ["책", "읽", "기"]
    assert [op.pred_char for op in ops] == ["챽", "엮", "기"]
    assert dist == 2  # 책->챽, 읽->엮 두 자리만 다름 (기==기는 match)
