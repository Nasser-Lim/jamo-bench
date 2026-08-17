# -*- coding: utf-8 -*-
"""공용 .env 로더 — modelark.py / clova_ocr.py가 같은 리포지토리 루트
.env를 공유한다(SPARK/pipeline의 egocentric_vlm.py와 같은 방식: python-dotenv
없이 표준 라이브러리만으로 읽는다)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

_loaded = False


def ensure_env_loaded() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not ENV_PATH.is_file():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
