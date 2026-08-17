# -*- coding: utf-8 -*-
"""jamo_bench.clova_ocr 단위 테스트 — §8.2의 3가지 파싱 함정을 합성 응답으로
재현해 검증한다(vertices 순서, lineBreak 재조립, convertedImageInfo)."""
import json

import pytest

from jamo_bench import clova_ocr


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._data


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.setenv("CLOVA_API_URL", "https://example.invalid/ocr")
    monkeypatch.setenv("CLOVA_SECRET_KEY", "test-secret")


def test_is_configured():
    assert clova_ocr.is_configured() is True


def test_bbox_from_vertices_handles_out_of_order_points():
    # 설계서 §8.2 실측: V2 예제에서 y가 1277→977→977→1277 순서로 나온다.
    vertices = [
        {"x": 100, "y": 1277},
        {"x": 300, "y": 977},
        {"x": 300, "y": 977},
        {"x": 100, "y": 1277},
    ]
    bbox = clova_ocr.bbox_from_vertices(vertices)
    assert bbox == (100, 977, 300, 1277)


def test_bbox_center():
    assert clova_ocr.bbox_center((0, 0, 100, 50)) == (50, 25)


def test_bbox_from_empty_vertices_is_none():
    assert clova_ocr.bbox_from_vertices([]) is None


def _field(text, line_break=False, x=0, y=0):
    return {
        "inferText": text,
        "inferConfidence": 0.99,
        "lineBreak": line_break,
        "boundingPoly": {"vertices": [{"x": x, "y": y}, {"x": x + 10, "y": y + 10}]},
    }


def test_line_reassembly_from_word_level_fields():
    # §8.2 함정 2: "아름다운"/"이"/"세상"이 어절 단위로 나뉘어 온다.
    payload = {
        "images": [
            {
                "convertedImageInfo": {"width": 1024, "height": 1024},
                "fields": [
                    _field("아름다운"),
                    _field("세상", line_break=True),
                    _field("두번째"),
                    _field("줄", line_break=True),
                ],
            }
        ]
    }
    result = clova_ocr._parse_general_ocr(payload)
    assert result.lines == ("아름다운 세상", "두번째 줄")
    assert result.full_text == "아름다운 세상\n두번째 줄"
    assert result.converted_width == 1024
    assert result.converted_height == 1024
    assert len(result.fields) == 4


def test_trailing_words_without_final_line_break_still_flushed():
    payload = {"images": [{"fields": [_field("가"), _field("나")]}]}
    result = clova_ocr._parse_general_ocr(payload)
    assert result.lines == ("가 나",)


def test_empty_images_returns_empty_result():
    result = clova_ocr._parse_general_ocr({"images": []})
    assert result.fields == ()
    assert result.lines == ()
    assert result.converted_width is None


def test_run_general_ocr_end_to_end(monkeypatch):
    payload = {
        "images": [
            {
                "convertedImageInfo": {"width": 512, "height": 512},
                "fields": [_field("읽", x=200, y=200)],
            }
        ]
    }

    def fake_urlopen(req, timeout=None):
        assert req.get_header("X-ocr-secret") == "test-secret"
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = clova_ocr.run_general_ocr(b"fake-image-bytes")
    assert result.fields[0].text == "읽"
    assert result.fields[0].bbox == (200, 200, 210, 210)


def test_missing_config_raises(monkeypatch):
    monkeypatch.delenv("CLOVA_API_URL", raising=False)
    with pytest.raises(clova_ocr.ClovaOcrError, match="CLOVA_API_URL"):
        clova_ocr.run_general_ocr(b"x")
