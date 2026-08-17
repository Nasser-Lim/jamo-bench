# -*- coding: utf-8 -*-
from jamo_bench.clova_ocr import OcrField
from jamo_bench.match_region import match_target_region, primary_region_box, bbox_area


def _f(text, bbox, confidence=0.9, line_break=False):
    return OcrField(text=text, confidence=confidence, bbox=bbox, line_break=line_break)


def test_primary_region_box_60_percent_centered():
    box = primary_region_box(1000, 1000, frac=0.6)
    assert box == (200, 200, 800, 800)


def test_bbox_area():
    assert bbox_area((0, 0, 100, 50)) == 5000


def test_f0_single_primary_candidate():
    fields = [_f("읽", (450, 450, 550, 550))]
    result = match_target_region(fields, 1000, 1000)
    assert result.matching_rule == "F0"
    assert result.candidate_text == "읽"
    assert result.position_miss is False
    assert result.needs_audit is False
    assert result.dropped is False
    assert result.spurious_count == 0


def test_f1_no_primary_text_falls_back_to_largest_area():
    fields = [
        _f("책", (0, 0, 50, 50)),      # 바깥, 작음
        _f("가", (850, 850, 950, 950)),  # 바깥, 더 큼
    ]
    result = match_target_region(fields, 1000, 1000)
    assert result.matching_rule == "F1"
    assert result.position_miss is True
    assert result.candidate_text == "가"  # 더 큰 면적
    assert result.spurious_count == 1


def test_f2_overlapping_primary_candidates_needs_audit():
    fields = [
        _f("읽", (400, 400, 600, 600)),
        _f("익", (420, 420, 620, 620)),  # 상당 부분 겹침
    ]
    result = match_target_region(fields, 1000, 1000)
    assert result.matching_rule == "F2"
    assert result.needs_audit is True
    assert result.dropped is False


def test_f3_multiple_distinct_primary_candidates():
    fields = [
        _f("가", (300, 300, 400, 400)),
        _f("나", (600, 600, 700, 700)),
    ]
    result = match_target_region(fields, 1000, 1000)
    assert result.matching_rule == "F3"
    assert set(result.all_hangul_texts) == {"가", "나"}
    assert result.candidate_text in ("가", "나")


def test_f4_no_candidates_is_dropped():
    result = match_target_region([], 1000, 1000)
    assert result.matching_rule == "F4"
    assert result.dropped is True
    assert result.candidate_text is None


def test_f4_when_no_field_has_hangul_text():
    fields = [_f("ABC123", (450, 450, 550, 550))]
    result = match_target_region(fields, 1000, 1000)
    assert result.matching_rule == "F4"


def test_confidence_floor_filters_low_confidence_fields():
    fields = [_f("가", (450, 450, 550, 550), confidence=0.1)]
    result = match_target_region(fields, 1000, 1000, confidence_floor=0.5)
    assert result.matching_rule == "F4"


def test_edit_distance_tie_break_among_equal_distance_and_area():
    # 두 후보가 중심거리·면적 동일 — target과의 편집거리로만 갈린다.
    fields = [
        _f("나", (450, 350, 550, 450)),  # 중심 (500,400)
        _f("읽", (450, 550, 550, 650)),  # 중심 (500,600), 이미지 중심(500,500)과 대칭 거리
    ]
    result = match_target_region(fields, 1000, 1000, target="읽")
    assert result.matching_rule == "F3"
    assert result.candidate_text == "읽"  # 편집거리 0인 쪽이 선택됨


def test_spurious_area_frac_computed_over_total_image_area():
    fields = [
        _f("읽", (450, 450, 550, 550)),  # 1% area primary candidate
        _f("가", (0, 0, 100, 100)),      # 바깥, spurious
    ]
    result = match_target_region(fields, 1000, 1000)
    assert result.matching_rule == "F0"
    assert result.spurious_count == 1
    assert result.spurious_area_frac == 100 * 100 / (1000 * 1000)


def test_candidate_confidence_is_surfaced():
    fields = [_f("읽", (450, 450, 550, 550), confidence=0.87)]
    result = match_target_region(fields, 1000, 1000)
    assert result.candidate_confidence == 0.87


def test_candidate_confidence_none_when_dropped():
    result = match_target_region([], 1000, 1000)
    assert result.candidate_confidence is None
