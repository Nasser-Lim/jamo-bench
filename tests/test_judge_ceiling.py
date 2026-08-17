# -*- coding: utf-8 -*-
"""judge_ceiling.py 단위 테스트 — 실제 OCR 호출 없이 가짜 ocr_fn으로 집계
로직만 검증한다(폰트 렌더링은 fonts/ 아래 실제 OFL 폰트를 그대로 쓴다)."""
from jamo_bench.clova_ocr import OcrField, OcrResult
from jamo_bench.judge_ceiling import measure_ceiling


def _fake_ocr_queue(readings):
    """readings를 호출 순서대로 하나씩 소비하는 ocr_fn. 큰 중앙 bbox에
    읽은 텍스트를 얹어 match_target_region이 F0으로 채택하게 만든다."""
    queue = list(readings)

    def ocr_fn(image_bytes: bytes, mime_type: str) -> OcrResult:
        text = queue.pop(0)
        field = OcrField(text=text, confidence=0.99, bbox=(400, 400, 600, 600), line_break=False)
        return OcrResult(fields=(field,), lines=(text,), converted_width=1024, converted_height=1024, raw_response={})

    return ocr_fn


def test_measure_ceiling_aggregates_clean_and_degraded_separately():
    # 순서: (가,clean) (가,degraded) (나,clean) (나,degraded)
    ocr_fn = _fake_ocr_queue(["가", "X", "나", "X"])
    report = measure_ceiling(
        ["가", "나"], ocr_fn, judge_name="fake_judge",
        fonts=("noto_sans_kr",), n_degraded_per_target=1,
    )
    assert report.judge_name == "fake_judge"
    assert report.n_clean == 2
    assert report.n_degraded == 2
    assert report.clean_accuracy == 1.0
    assert report.degraded_accuracy == 0.0


def test_measure_ceiling_multiple_fonts_multiplies_samples():
    ocr_fn = _fake_ocr_queue(["가", "가", "가", "가"])  # 2 fonts × (clean+degraded)
    report = measure_ceiling(
        ["가"], ocr_fn, judge_name="fake_judge",
        fonts=("noto_sans_kr", "pretendard"), n_degraded_per_target=1,
    )
    assert report.n_clean == 2
    assert report.n_degraded == 2
    assert report.clean_accuracy == 1.0
    assert report.degraded_accuracy == 1.0


def test_official_validity_thresholds():
    ocr_fn = _fake_ocr_queue(["가", "가"])
    report = measure_ceiling(["가"], ocr_fn, "j", fonts=("noto_sans_kr",), n_degraded_per_target=1)
    # degraded_accuracy는 1.0 → official_valid
    assert report.official_validity() == "official_valid"


def test_to_json_schema():
    ocr_fn = _fake_ocr_queue(["가", "가"])
    report = measure_ceiling(["가"], ocr_fn, "paddleocr_v2.7", fonts=("noto_sans_kr",), n_degraded_per_target=1)
    j = report.to_json()
    assert j["judge_ceiling"]["paddleocr_v2.7"]["clean"] == 1.0
    assert j["judge_ceiling"]["paddleocr_v2.7"]["degraded"] == 1.0
