# -*- coding: utf-8 -*-
from jamo_bench.clova_ocr import OcrField
from jamo_bench.route import DetectionInput, build_detection_input, classify_route


def test_route_a0_no_text_like_region():
    det = DetectionInput(has_text_like_region=False, hangul_completions=[], in_primary_region=[])
    assert classify_route(det) == "A0"


def test_route_a1_text_attempt_but_zero_hangul():
    det = DetectionInput(has_text_like_region=True, hangul_completions=[], in_primary_region=[])
    assert classify_route(det) == "A1"


def test_route_b_hangul_in_primary_region():
    det = DetectionInput(
        has_text_like_region=True,
        hangul_completions=["읽"],
        in_primary_region=[True],
    )
    assert classify_route(det) == "B"


def test_route_c_mixed_script_even_with_hangul_hit():
    det = DetectionInput(
        has_text_like_region=True,
        hangul_completions=["읽"],
        in_primary_region=[True],
        has_non_hangul_mixed=True,
    )
    assert classify_route(det) == "C"


def test_route_c_hangul_detected_but_outside_primary_region():
    det = DetectionInput(
        has_text_like_region=True,
        hangul_completions=["읽"],
        in_primary_region=[False],
    )
    assert classify_route(det) == "C"


def _f(text, bbox, confidence=0.9):
    return OcrField(text=text, confidence=confidence, bbox=bbox, line_break=False)


class TestBuildDetectionInput:
    """has_text_like_region은 이제 호출부가 명시적으로 넘긴다(비전
    휴리스틱 결과) — OCR 필드 유무로부터 추론하지 않는다."""

    def test_no_fields_and_no_ink_is_a0_shaped(self):
        det = build_detection_input([], 1000, 1000, has_text_like_region=False)
        assert det.has_text_like_region is False
        assert classify_route(det) == "A0"

    def test_no_fields_but_ink_present_is_not_a0(self):
        # OCR이 완전히 실패해도(fields=[]) 비전 휴리스틱이 잉크를
        # 검출했다면 A0로 떨어지면 안 된다 — 실측 버그 재현 방지 테스트.
        det = build_detection_input([], 1000, 1000, has_text_like_region=True)
        assert classify_route(det) == "A1"

    def test_non_hangul_only_is_a1_shaped(self):
        det = build_detection_input(
            [_f("ABC", (450, 450, 550, 550))], 1000, 1000, has_text_like_region=True
        )
        assert det.has_text_like_region is True
        assert det.hangul_completions == []
        assert classify_route(det) == "A1"

    def test_hangul_in_primary_region_is_b_shaped(self):
        det = build_detection_input(
            [_f("읽", (450, 450, 550, 550))], 1000, 1000, has_text_like_region=True
        )
        assert det.hangul_completions == ["읽"]
        assert det.in_primary_region == [True]
        assert classify_route(det) == "B"

    def test_hangul_outside_primary_region_is_c_shaped(self):
        det = build_detection_input(
            [_f("읽", (0, 0, 50, 50))], 1000, 1000, has_text_like_region=True
        )
        assert det.in_primary_region == [False]
        assert classify_route(det) == "C"

    def test_isolated_jamo_flags_mixed(self):
        det = build_detection_input(
            [_f("ㅇㅣㄺ", (450, 450, 550, 550))], 1000, 1000, has_text_like_region=True
        )
        assert det.has_non_hangul_mixed is True
        assert classify_route(det) == "C"

    def test_hangul_with_latin_mixed_in_same_field(self):
        det = build_detection_input(
            [_f("읽ABC", (450, 450, 550, 550))], 1000, 1000, has_text_like_region=True
        )
        assert det.has_non_hangul_mixed is True
        assert "읽" in det.hangul_completions[0]
        assert classify_route(det) == "C"
