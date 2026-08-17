# -*- coding: utf-8 -*-
"""PROMPT_SPECS.yaml 로더 + {target} interpolation (JAMO_v51_patch.md §11).

interpolation 방식(따옴표 유무·반복 횟수)에 모델이 민감하다는 실측 지적 때문에
템플릿을 코드에 흩어 두지 않고 YAML 한 곳에 얼려 둔다. 이 모듈은 그 파일을
읽고 `{target}`을 정확히 명시된 횟수만큼만 치환하는 것 외에 아무 것도
하지 않는다 — 임의 치환을 허용하면 프롬프트 오염(§11의 우려)이 재발한다.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_SPECS_PATH = Path(__file__).resolve().parent.parent / "PROMPT_SPECS.yaml"


class PromptSpecError(ValueError):
    pass


@lru_cache(maxsize=4)
def _load_cached(path_str: str) -> dict:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt_specs(path: Optional[Path] = None) -> dict:
    resolved = Path(path) if path is not None else DEFAULT_SPECS_PATH
    return _load_cached(str(resolved))


def render_prompt(template_id: str, target: str, specs: Optional[dict] = None) -> str:
    """template_id(T1/T2/T3)의 {target} 자리에 target을 정확히
    target_repeat번 치환한 프롬프트 문자열을 반환한다."""
    specs = specs if specs is not None else load_prompt_specs()
    templates = specs.get("templates", {})
    if template_id not in templates:
        raise PromptSpecError(f"unknown template_id: {template_id!r} (known: {sorted(templates)})")

    tpl = templates[template_id]
    text = tpl["text"]
    expected_repeat = tpl.get("target_repeat", 1)
    actual_repeat = text.count("{target}")
    if actual_repeat != expected_repeat:
        raise PromptSpecError(
            f"template {template_id!r} declares target_repeat={expected_repeat} "
            f"but contains {{target}} {actual_repeat} times — spec is inconsistent"
        )
    return text.replace("{target}", target)
