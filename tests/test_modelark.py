# -*- coding: utf-8 -*-
"""jamo_bench.modelark 단위 테스트 — 실제 네트워크 호출 없이 urllib.request.urlopen을
모킹해 요청/응답 파싱 로직만 검증한다."""
import base64
import io
import json
import urllib.error

import pytest

from jamo_bench import modelark


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
    monkeypatch.setenv("ARK_MODEL_SEEDREAM", "test-model")


def test_is_configured_true_when_env_set():
    assert modelark.is_configured() is True


def test_is_configured_false_when_missing(monkeypatch):
    monkeypatch.delenv("ARK_MODEL_SEEDREAM", raising=False)
    assert modelark.is_configured() is False


def test_generate_image_b64_json(monkeypatch):
    fake_bytes = b"PNGDATA"
    fake_payload = {
        "created": 1,
        "data": [{"b64_json": base64.b64encode(fake_bytes).decode()}],
    }

    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse(json.dumps(fake_payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = modelark.generate_image("plain white background, Hangul syllable")

    assert result.image_bytes == fake_bytes
    assert result.image_url is None
    assert captured["url"] == f"{modelark.ARK_BASE_URL}/images/generations"
    assert captured["body"]["model"] == "test-model"
    assert captured["body"]["watermark"] is False
    assert captured["headers"]["Authorization"] == "Bearer test-key"


def test_generate_image_url_format(monkeypatch):
    fake_payload = {"data": [{"url": "https://example.invalid/img.png"}]}

    def fake_urlopen(req, timeout=None):
        return _FakeResponse(json.dumps(fake_payload).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = modelark.generate_image("test", response_format="url")
    assert result.image_url == "https://example.invalid/img.png"
    assert result.image_bytes is None


def test_missing_api_key_raises_immediately(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    def fake_urlopen(req, timeout=None):
        raise AssertionError("should not reach network call")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(modelark.ModelArkError, match="ARK_API_KEY"):
        modelark.generate_image("test")


def test_missing_model_raises(monkeypatch):
    monkeypatch.delenv("ARK_MODEL_SEEDREAM", raising=False)
    with pytest.raises(modelark.ModelArkError, match="model"):
        modelark.generate_image("test")


def test_retries_then_raises_on_persistent_http_error(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            "http://x", 500, "boom", {}, io.BytesIO(b'{"error":"boom"}')
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(modelark.ModelArkError, match="500"):
        modelark.generate_image("test", max_retries=2)

    assert calls["n"] == 3  # 최초 시도 + 재시도 2회


def test_malformed_response_raises_without_retry(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return _FakeResponse(json.dumps({"unexpected": True}).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(modelark.ModelArkError, match="예상치 못한 응답"):
        modelark.generate_image("test", max_retries=2)

    assert calls["n"] == 1  # 스키마 불일치는 재시도하지 않는다
