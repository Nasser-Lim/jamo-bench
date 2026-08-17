# -*- coding: utf-8 -*-
"""Judge Ceiling 측정 (JAMO_benchmark_design.md §8.4, JAMO_v51_patch.md §7.1).

Forge 렌더러(clean/degraded)로 만든 대조군을 OCR judge에 흘려서 "이 judge가
글자를 잘 그린 이미지조차 얼마나 놓치는가"를 실측한다. degraded ceiling이
90% 미만인 셀은 Official 채점에서 제외된다(§8.1 판정 규칙) — 그 판정의
근거가 되는 숫자가 여기서 나온다.

이 모듈은 특정 OCR 엔진에 결합되지 않는다. `ocr_fn(image_bytes, mime_type)
-> OcrResult` 형태의 콜러블을 받아 CLOVA든 다른 OSS judge든 그대로 넣을 수
있다(§8.4의 "판사별 딕셔너리" 요구사항).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from . import forge_render
from .clova_ocr import OcrResult
from .match_region import match_target_region

OcrFn = Callable[[bytes, str], OcrResult]


@dataclass(frozen=True)
class CeilingSample:
    char: str
    condition: str  # "clean" | "degraded"
    font_name: str
    candidate_text: Optional[str]
    matching_rule: str
    correct: bool


@dataclass(frozen=True)
class CeilingReport:
    judge_name: str
    clean_accuracy: float
    degraded_accuracy: float
    n_clean: int
    n_degraded: int
    samples: Tuple[CeilingSample, ...]

    def to_json(self) -> dict:
        """§8.4 스키마: judge_ceiling.{judge_name}.{clean,degraded}."""
        return {
            "judge_ceiling": {
                self.judge_name: {
                    "clean": self.clean_accuracy,
                    "degraded": self.degraded_accuracy,
                }
            },
            "n_clean": self.n_clean,
            "n_degraded": self.n_degraded,
        }

    def official_validity(self) -> str:
        """§8.1 Phase 0 판정 규칙 — degraded ceiling 기준."""
        if self.degraded_accuracy >= 0.98:
            return "official_valid"
        if self.degraded_accuracy >= 0.90:
            return "official_valid_with_ceiling_badge"
        return "official_excluded"


def measure_ceiling(
    chars: Sequence[str],
    ocr_fn: OcrFn,
    judge_name: str,
    fonts: Sequence[str] = ("noto_sans_kr", "pretendard"),
    n_degraded_per_target: int = 3,
    seed: int = 0,
    canvas_size: int = 1024,
    text_area_frac: float = 0.5,
) -> CeilingReport:
    """chars 각각에 대해 clean 1장 + degraded n_degraded_per_target장을
    폰트별로 만들어 ocr_fn에 흘리고 정확히 일치하는지 집계한다."""
    samples = []
    seed_counter = seed

    for char in chars:
        for font_name in fonts:
            clean_img = forge_render.render_clean(
                char, font_name=font_name, canvas_size=canvas_size, text_area_frac=text_area_frac
            )
            samples.append(
                _run_one(char, "clean", font_name, clean_img, "image/png", ocr_fn, canvas_size)
            )

            for _ in range(n_degraded_per_target):
                deg_img, _params = forge_render.render_degraded(
                    char,
                    seed=seed_counter,
                    font_name=font_name,
                    canvas_size=canvas_size,
                    text_area_frac=text_area_frac,
                )
                seed_counter += 1
                samples.append(
                    _run_one(char, "degraded", font_name, deg_img, "image/jpeg", ocr_fn, canvas_size)
                )

    return _aggregate(judge_name, samples)


def _run_one(char, condition, font_name, img, mime_type, ocr_fn, canvas_size) -> CeilingSample:
    fmt = "PNG" if mime_type == "image/png" else "JPEG"
    data = forge_render.image_to_bytes(img, format=fmt)
    ocr_result = ocr_fn(data, mime_type)
    match = match_target_region(ocr_result.fields, canvas_size, canvas_size, target=char)
    return CeilingSample(
        char=char,
        condition=condition,
        font_name=font_name,
        candidate_text=match.candidate_text,
        matching_rule=match.matching_rule,
        correct=(match.candidate_text == char),
    )


def _aggregate(judge_name: str, samples: Sequence[CeilingSample]) -> CeilingReport:
    clean = [s for s in samples if s.condition == "clean"]
    degraded = [s for s in samples if s.condition == "degraded"]
    clean_acc = sum(s.correct for s in clean) / len(clean) if clean else float("nan")
    degraded_acc = sum(s.correct for s in degraded) / len(degraded) if degraded else float("nan")
    return CeilingReport(
        judge_name=judge_name,
        clean_accuracy=clean_acc,
        degraded_accuracy=degraded_acc,
        n_clean=len(clean),
        n_degraded=len(degraded),
        samples=tuple(samples),
    )
