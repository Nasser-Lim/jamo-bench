# -*- coding: utf-8 -*-
"""jamo_bench.vlm_judge 단위 테스트 — urllib.request.urlopen을 모킹해
네트워크 없이 파싱/조립 로직만 검증한다."""
import json

import pytest

from jamo_bench import vlm_judge


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
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    monkeypatch.setenv("ARK_MODEL_VLM", "test-vlm-model")


def _chat_payload(content_obj: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(content_obj, ensure_ascii=False)}}]}


def test_is_configured():
    assert vlm_judge.is_configured() is True


def test_read_character_composes_from_jamo_parts(monkeypatch):
    payload = _chat_payload({"onset": "ㄱ", "nucleus": "ㅏ", "coda": "", "confidence": 0.95, "note": "simple circle+line"})

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    reading = vlm_judge.read_character(b"fake-image-bytes")
    assert reading.full_char == "가"
    assert reading.valid is True
    assert reading.confidence == 0.95


def test_read_character_with_complex_coda(monkeypatch):
    payload = _chat_payload({"onset": "ㄱ", "nucleus": "ㅏ", "coda": "ㅄ", "confidence": 0.8, "note": "cluster final"})

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    reading = vlm_judge.read_character(b"fake-image-bytes")
    assert reading.full_char == "값"


def test_invalid_jamo_marks_reading_invalid(monkeypatch):
    payload = _chat_payload({"onset": "Z", "nucleus": "ㅏ", "coda": "", "confidence": 0.5, "note": "unsure"})

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    reading = vlm_judge.read_character(b"fake-image-bytes")
    assert reading.valid is False
    assert reading.full_char is None


def test_markdown_fenced_response_is_stripped(monkeypatch):
    inner = json.dumps({"onset": "ㅇ", "nucleus": "ㅣ", "coda": "ㄺ", "confidence": 0.7, "note": "x"}, ensure_ascii=False)
    payload = {"choices": [{"message": {"content": f"```json\n{inner}\n```"}}]}

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    reading = vlm_judge.read_character(b"fake-image-bytes")
    assert reading.full_char == "읽"


def test_missing_config_raises_immediately(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    def fake_urlopen(req, timeout=None):
        raise AssertionError("should not reach network call")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(vlm_judge.VlmJudgeError, match="ARK_API_KEY"):
        vlm_judge.read_character(b"x")


def test_malformed_json_raises(monkeypatch):
    payload = {"choices": [{"message": {"content": "not json at all"}}]}

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(vlm_judge.VlmJudgeError, match="파싱"):
        vlm_judge.read_character(b"x", max_retries=0)
