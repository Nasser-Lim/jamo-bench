# -*- coding: utf-8 -*-
"""BytePlus ModelArk 이미지 생성 클라이언트 (Seedream).

SPARK/pipeline/egocentric_vlm.py와 같은 계정(ARK_API_KEY 재사용)·같은
표준 라이브러리 전용 원칙(urllib, .env는 python-dotenv 없이 직접 파싱)을
따른다. base URL도 SPARK가 실측 확인한 ARK_BASE_URL과 동일하다 —
/chat/completions와 /images/generations가 같은 v3 API 아래 있다.

요청 스키마는 사용자가 실제로 API를 활성화하고 확인한 curl 예시를 그대로
반영한다(2026-08-09 실측):
    POST {ARK_BASE_URL}/images/generations
    {"model": "dola-seedream-5-0-pro-260628", "prompt": ..., "response_format": ...,
     "size": "2K"|"1024x1024", "stream": false, "watermark": bool}
"""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Optional

from ._env import ensure_env_loaded

ARK_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
API_TIMEOUT_SEC = 120
MAX_RETRIES = 2

ensure_env_loaded()


class ModelArkError(RuntimeError):
    pass


def is_configured() -> bool:
    """.env에 키/엔드포인트가 없으면 호출부가 조용히 건너뛸 수 있게 한다
    (SPARK의 is_configured()와 같은 원칙)."""
    return bool(os.environ.get("ARK_API_KEY") and os.environ.get("ARK_MODEL_SEEDREAM"))


@dataclass(frozen=True)
class GenerationResult:
    model: str
    prompt: str
    image_bytes: Optional[bytes]
    image_url: Optional[str]
    response_format: str
    size: str
    seed: Optional[int]
    raw_response: dict
    attempts: int = 1


def _call_ark_images(
    prompt: str,
    model: str,
    size: str,
    response_format: str,
    watermark: bool,
    seed: Optional[int],
    timeout_sec: float,
) -> dict:
    import urllib.error
    import urllib.request

    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        raise ModelArkError("ARK_API_KEY가 설정되지 않았습니다. JAMO/.env를 확인하세요.")

    body = {
        "model": model,
        "prompt": prompt,
        "response_format": response_format,
        "size": size,
        "stream": False,
        "watermark": watermark,
    }
    if seed is not None:
        body["seed"] = seed

    req = urllib.request.Request(
        f"{ARK_BASE_URL}/images/generations",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise ModelArkError(f"ModelArk 이미지 생성 API 오류({e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise ModelArkError(f"ModelArk 이미지 생성 API 연결 실패: {e.reason}") from e


def _parse_generation_response(
    payload: dict,
    model: str,
    prompt: str,
    size: str,
    response_format: str,
    seed: Optional[int],
    attempts: int,
) -> GenerationResult:
    data = payload.get("data")
    if not data:
        raise ModelArkError(f"예상치 못한 응답 형식(data 없음): {payload}")
    item = data[0]
    image_url = item.get("url")
    b64 = item.get("b64_json")
    image_bytes = base64.b64decode(b64) if b64 else None

    return GenerationResult(
        model=model,
        prompt=prompt,
        image_bytes=image_bytes,
        image_url=image_url,
        response_format=response_format,
        size=size,
        seed=seed,
        raw_response=payload,
        attempts=attempts,
    )


def generate_image(
    prompt: str,
    model: Optional[str] = None,
    size: str = "1024x1024",
    response_format: str = "b64_json",
    watermark: bool = False,
    seed: Optional[int] = None,
    timeout_sec: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> GenerationResult:
    """Seedream으로 프롬프트 1개에서 이미지 1장을 생성한다.

    watermark 기본값을 False로 둔다 — JAMO 프롬프트 템플릿(T1/T2/T3, §6.9)이
    전부 "no watermark, no decoration"을 명시하므로 API 레벨에서도 일관되게
    끈다. response_format 기본값은 b64_json — url 방식은 서명 URL이 곧
    만료되므로, 배치 파이프라인에서 즉시 디스크에 저장하려면 바이트를 직접
    받는 편이 재시도·저장 로직을 단순하게 만든다.
    """
    model = model or os.environ.get("ARK_MODEL_SEEDREAM")
    if not model:
        raise ModelArkError(
            "model이 지정되지 않았고 ARK_MODEL_SEEDREAM도 설정되지 않았습니다."
        )
    if timeout_sec is None:
        timeout_sec = API_TIMEOUT_SEC
    if max_retries is None:
        max_retries = MAX_RETRIES

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 2):
        try:
            payload = _call_ark_images(
                prompt, model, size, response_format, watermark, seed, timeout_sec
            )
            return _parse_generation_response(
                payload, model, prompt, size, response_format, seed, attempt
            )
        except ModelArkError as e:
            last_err = e
            msg = str(e)
            # 설정 누락·응답 스키마 불일치는 재시도해도 같은 결과이므로 즉시 포기한다.
            if "설정되지" in msg or "예상치 못한 응답" in msg:
                raise

    raise last_err or ModelArkError("이미지 생성에 실패했습니다.")
