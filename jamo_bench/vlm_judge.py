# -*- coding: utf-8 -*-
"""BytePlus ModelArk VLM judge 후보 (JAMO_benchmark_design.md §8.1 폴백:
"오픈 가중치 VLM judge 후보 평가").

CLOVA General OCR이 사람 감사 대비 32.8% 일치율로 실격 판정을 받았다
(2026-08-10, `docs/PROGRESS.md` 7단계). 원인은 CLOVA가 인식 결과를
"그럴듯한 실제 한국어 단어" 쪽으로 보정하는 언어모델 편향 — JAMO의 Core는
의도적으로 그런 실제 단어가 아닌 조합을 다수 포함하므로 이 편향이
정확히 측정하려는 축(종성 복잡도)에서 재앙적으로 작동했다.

이 모듈은 같은 함정을 피하도록 설계한다: VLM에게 "무슨 글자로 보이는가"를
통짜로 묻지 않고, **초성/중성/종성 각각을 보이는 형태 그대로 답하게** 한
뒤 `decompose.compose()`로 우리가 직접 조립한다. "이 조합은 실제 단어가
아니니 그럴듯한 다른 글자로 바꿔 답하라"는 유혹 자체를 프롬프트 설계로
차단하는 것이 핵심 — 모델이 통짜 인식을 하고 싶어도 우리가 요구하는
출력 형식이 그걸 허용하지 않는다.

SPARK/pipeline/egocentric_vlm.py와 같은 계정(ARK_API_KEY)·같은 엔드포인트
구조(콘솔에서 만든 ep-... ID를 model 필드에 그대로 사용)를 재사용한다.
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Optional

from ._env import ensure_env_loaded
from .decompose import FINALS, ONSETS, VOWELS, compose

ARK_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
API_TIMEOUT_SEC = 90
MAX_RETRIES = 2

ensure_env_loaded()

SYSTEM_PROMPT = (
    "You are a meticulous visual glyph analyst. You describe the literal strokes "
    "you see in a single Hangul syllable-block image. You are NOT a Korean "
    "dictionary lookup — many test images intentionally show syllables that are "
    "NOT real Korean words. Reporting a 'corrected' real-word reading instead of "
    "the literal visible shape is a failure, even if the literal shape looks unusual."
)

USER_PROMPT = f"""\
This image shows exactly one Hangul syllable block on a plain background. A \
Hangul syllable block is built from up to three components arranged in a fixed \
layout: an initial consonant (초성, top or top-left), a vowel (중성, to the \
right of or below the initial), and an optional final consonant/cluster \
(종성, at the bottom — may be absent).

Identify each component **purely by its visual shape**, choosing only from the \
exact reference lists below. Do not substitute a different component just \
because the resulting combination would "make more sense" as a real Korean \
word — many of these images deliberately show syllables that are not real \
words, and your job is to report the literal shape, not to auto-correct it.

초성 (initial, pick exactly one): {' '.join(ONSETS)}
중성 (vowel, pick exactly one): {' '.join(VOWELS)}
종성 (final, pick exactly one, or empty string if there is clearly no final \
consonant beneath the vowel): {' '.join(c for c in FINALS if c)}

Respond with ONLY a compact JSON object, no markdown fences, no extra text:
{{"onset": "<one of the 초성 list>", "nucleus": "<one of the 중성 list>", \
"coda": "<one of the 종성 list, or empty string>", "confidence": 0.0-1.0, \
"note": "one short phrase describing the literal shape you saw, in English"}}
"""


class VlmJudgeError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(os.environ.get("ARK_API_KEY") and os.environ.get("ARK_MODEL_VLM"))


@dataclass(frozen=True)
class VlmReading:
    onset: Optional[str]
    nucleus: Optional[str]
    coda: Optional[str]
    full_char: Optional[str]  # compose()로 조립한 결과, 조립 실패 시 None
    confidence: Optional[float]
    note: str
    raw_response: dict
    valid: bool  # onset/nucleus/coda가 전부 참조 목록 안에 있어 조립에 성공했는가


def _image_data_uri(image_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def _call_ark_vlm(data_uri: str, timeout_sec: float) -> dict:
    import urllib.error
    import urllib.request

    api_key = os.environ.get("ARK_API_KEY")
    model = os.environ.get("ARK_MODEL_VLM")
    if not api_key or not model:
        raise VlmJudgeError(
            "ARK_API_KEY 또는 ARK_MODEL_VLM이 설정되지 않았습니다. JAMO/.env를 확인하세요."
        )

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
    }

    req = urllib.request.Request(
        f"{ARK_BASE_URL}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise VlmJudgeError(f"ModelArk VLM API 오류({e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise VlmJudgeError(f"ModelArk VLM API 연결 실패: {e.reason}") from e


def _parse_reading(payload: dict) -> VlmReading:
    try:
        raw_text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise VlmJudgeError(f"예상치 못한 응답 형식: {payload}") from e

    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise VlmJudgeError(f"VLM 응답을 JSON으로 파싱하지 못했습니다: {raw_text!r}") from e

    onset = parsed.get("onset")
    nucleus = parsed.get("nucleus")
    coda = parsed.get("coda", "")
    if coda is None:
        coda = ""

    full_char = compose(onset, nucleus, coda) if onset and nucleus is not None else None
    valid = full_char is not None

    return VlmReading(
        onset=onset,
        nucleus=nucleus,
        coda=coda,
        full_char=full_char,
        confidence=parsed.get("confidence"),
        note=str(parsed.get("note", "")),
        raw_response=payload,
        valid=valid,
    )


def read_character(
    image_bytes: bytes,
    mime_type: str = "image/png",
    timeout_sec: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> VlmReading:
    """이미지 속 한글 음절 1개를 자모 단위로 판독한다.

    반환된 VlmReading.full_char는 모델이 통짜로 답한 게 아니라 우리가
    onset/nucleus/coda를 조립해서 만든 값이다 — valid=False면 모델이
    참조 목록 밖의 값을 답했다는 뜻(그 자체로 판정자 품질 신호).
    """
    if timeout_sec is None:
        timeout_sec = API_TIMEOUT_SEC
    if max_retries is None:
        max_retries = MAX_RETRIES

    data_uri = _image_data_uri(image_bytes, mime_type)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 2):
        try:
            payload = _call_ark_vlm(data_uri, timeout_sec)
            return _parse_reading(payload)
        except VlmJudgeError as e:
            last_err = e
            msg = str(e)
            if "설정되지" in msg:
                raise
        except (TimeoutError, OSError) as e:
            # clova_ocr.py에서 실측한 것과 같은 함정 — 소켓 읽기 단계의
            # TimeoutError는 urllib이 URLError로 감싸지 않고 그대로 올려보낸다.
            last_err = VlmJudgeError(f"ModelArk VLM 네트워크 오류(시도 {attempt}): {e}")
        if attempt <= max_retries:
            import time as _time

            _time.sleep(2.0 * attempt)

    raise last_err or VlmJudgeError("VLM 판독에 실패했습니다.")
