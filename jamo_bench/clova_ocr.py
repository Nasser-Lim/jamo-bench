# -*- coding: utf-8 -*-
"""Naver CLOVA General OCR 클라이언트 (JAMO_benchmark_design.md §8.1 Reference
judge, §8.2 CLOVA API 파싱 함정).

easyscan6/apps/api/app/services/clova.py와 같은 계정의 CLOVA_API_URL·
CLOVA_SECRET_KEY(General OCR 도메인)를 재사용한다. easyscan6는 좌표를 버리고
텍스트만 쓰지만(같은 프로젝트 solar.py의 "좌표는 전부 제외" 주석 참고),
JAMO는 §8.3 중심 근접도 매칭에 bbox가 반드시 필요하므로 이 클라이언트는
vertices를 보존하고, 설계서가 실측으로 확인한 3가지 함정을 여기서 고정한다:

  1) vertices 순서가 일정하지 않다 → 4점 min/max로 bbox 재계산
  2) 어절 단위 분할 → lineBreak로 라인 재조립
  3) convertedImageInfo 크기가 원본과 다를 수 있음 → converted 기준 좌표로 보고
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ._env import ensure_env_loaded

API_TIMEOUT_SEC = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0
_EXT_MAP = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}

ensure_env_loaded()


class ClovaOcrError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(os.environ.get("CLOVA_API_URL") and os.environ.get("CLOVA_SECRET_KEY"))


BBox = Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


def bbox_from_vertices(vertices: List[dict]) -> Optional[BBox]:
    """§8.2 함정 1: vertices 순서가 일정하지 않다(V2 예제에서 y가
    1277→977→977→1277 순서로 나온 사례 실측). 순서에 의존하지 않고
    4점의 min/max로 bbox를 재계산한다."""
    if not vertices:
        return None
    xs = [v["x"] for v in vertices]
    ys = [v["y"] for v in vertices]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


@dataclass(frozen=True)
class OcrField:
    text: str
    confidence: Optional[float]
    bbox: Optional[BBox]
    line_break: bool


@dataclass(frozen=True)
class OcrResult:
    fields: Tuple[OcrField, ...]
    lines: Tuple[str, ...]
    converted_width: Optional[int]
    converted_height: Optional[int]
    raw_response: dict

    @property
    def full_text(self) -> str:
        return "\n".join(self.lines)


def _parse_general_ocr(payload: dict) -> OcrResult:
    images = payload.get("images", [])
    if not images:
        return OcrResult(fields=(), lines=(), converted_width=None, converted_height=None, raw_response=payload)

    img = images[0]
    # §8.2 함정 3: convertedImageInfo 크기가 원본과 다를 수 있다 — bbox는
    # 이 좌표계 기준이므로 폭/높이를 함께 보고해 하위 코드가 원본과 섞어
    # 쓰지 않게 한다.
    converted = img.get("convertedImageInfo") or {}
    converted_width = converted.get("width")
    converted_height = converted.get("height")

    fields: List[OcrField] = []
    lines: List[str] = []
    current_words: List[str] = []
    # §8.2 함정 2: 어절 단위로 분할되어 나온다("아름다운"/"이"/"세상" 별도
    # field) — lineBreak가 true인 field에서 줄을 닫아 재조립한다.
    for f in img.get("fields", []):
        text = f.get("inferText", "")
        confidence = f.get("inferConfidence")
        vertices = f.get("boundingPoly", {}).get("vertices", [])
        bbox = bbox_from_vertices(vertices)
        line_break = bool(f.get("lineBreak", False))
        fields.append(OcrField(text=text, confidence=confidence, bbox=bbox, line_break=line_break))

        current_words.append(text)
        if line_break:
            lines.append(" ".join(current_words))
            current_words = []
    if current_words:
        lines.append(" ".join(current_words))

    return OcrResult(
        fields=tuple(fields),
        lines=tuple(lines),
        converted_width=converted_width,
        converted_height=converted_height,
        raw_response=payload,
    )


def _call_clova_general(
    image_bytes: bytes,
    mime_type: str,
    lang: str,
    timeout_sec: float,
) -> dict:
    import urllib.error
    import urllib.request

    api_url = os.environ.get("CLOVA_API_URL")
    secret_key = os.environ.get("CLOVA_SECRET_KEY")
    if not api_url or not secret_key:
        raise ClovaOcrError(
            "CLOVA_API_URL 또는 CLOVA_SECRET_KEY가 설정되지 않았습니다. JAMO/.env를 확인하세요."
        )

    ext = _EXT_MAP.get(mime_type, "jpg")
    body = {
        "images": [{"format": ext, "name": "jamo", "data": base64.b64encode(image_bytes).decode("ascii")}],
        "requestId": uuid.uuid4().hex,
        "version": "V2",
        "timestamp": int(time.time() * 1000),
        "lang": lang,
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"X-OCR-SECRET": secret_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise ClovaOcrError(f"CLOVA OCR API 오류({e.code}): {detail}") from e
    except urllib.error.URLError as e:
        raise ClovaOcrError(f"CLOVA OCR API 연결 실패: {e.reason}") from e


def run_general_ocr(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    lang: str = "ko",
    timeout_sec: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> OcrResult:
    """CLOVA General OCR(도메인 기반) 호출. 영수증 특화 도메인이 아니라
    일반 텍스트 인식 도메인을 쓴다 — JAMO는 임의의 배경 위 한글 글자를
    읽어야 하므로 영수증 구조화 파서가 필요 없다.

    네트워크 타임아웃(대량 배치 실행 중 실측 확인 — 소켓 읽기 단계에서
    발생하는 TimeoutError는 urllib이 URLError로 감싸지 않고 그대로
    올려보낸다)에 대비해 재시도한다. 설정 누락은 재시도해도 같은 결과이므로
    즉시 포기한다."""
    if timeout_sec is None:
        timeout_sec = API_TIMEOUT_SEC
    if max_retries is None:
        max_retries = MAX_RETRIES

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 2):
        try:
            payload = _call_clova_general(image_bytes, mime_type, lang, timeout_sec)
            return _parse_general_ocr(payload)
        except ClovaOcrError as e:
            last_err = e
            if "설정되지" in str(e):
                raise
        except (TimeoutError, OSError) as e:
            last_err = ClovaOcrError(f"CLOVA OCR 네트워크 오류(시도 {attempt}): {e}")
        if attempt <= max_retries:
            time.sleep(RETRY_BACKOFF_SEC * attempt)

    raise last_err or ClovaOcrError("CLOVA OCR 호출에 실패했습니다.")
