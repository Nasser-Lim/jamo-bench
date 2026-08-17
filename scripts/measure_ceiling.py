# -*- coding: utf-8 -*-
"""18셀 전체 Judge Ceiling 실측 (JAMO_benchmark_design.md §8.4, §14.2 Go/No-Go).

각 셀에서 대표 음절 여러 개(기본 5개, Core 540 시드=0 기준 — 실제 본
배치에서 쓰일 음절과 정렬)를 뽑아, 폰트 2종 × (clean 1장 + degraded n장)을
CLOVA General OCR에 흘려 clean/degraded ceiling을 잰다. 셀당 표본을
늘린 이유: 최초 실측(음절 1개/셀, 셀당 6장)은 표본이 너무 작아
"가"(clean 100%)와 "값"(clean 0%)처럼 한 장 차이로 극단값이 갈렸다 —
CLOVA 크레딧이 여유로우므로 표본을 키워 신뢰할 수 있는 셀별 ceiling을
얻는다.

OCR에 넣기 전 judge_preprocess.normalize_occupancy를 거친다 — 실제 채점
파이프라인과 동일한 경로로 측정해야 이 ceiling 수치를 그대로 §8.1 판정
규칙에 쓸 수 있다.

셀 단위로 results/judge_ceiling_cells.jsonl에 즉시 저장하고 재실행 시
이미 끝난 셀은 건너뛴다 — 720장 규모의 배치는 네트워크 타임아웃 등으로
중간에 끊길 수 있는데(실측: 6/18셀 진행 중 TimeoutError로 중단), 매번
처음부터 다시 돌리면 이미 쓴 API 호출이 낭비된다.

기본값(5음절 × 2폰트 × (1 clean + 3 degraded) = 셀당 40장, 18셀 합계
720장)은 CLOVA API를 그만큼 호출한다. 폰트 렌더링 자체는 무료.

결과: results/judge_ceiling.json (전 셀 완료 시 최종 집계본)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jamo_bench import clova_ocr  # noqa: E402
from jamo_bench.decompose import ALL_18_CELLS  # noqa: E402
from jamo_bench.judge_ceiling import measure_ceiling  # noqa: E402
from jamo_bench.judge_preprocess import normalize_occupancy  # noqa: E402
from jamo_bench.partitioning import partition  # noqa: E402

from run_pilot import _acquire_lock, _release_lock  # noqa: E402

OUT_PATH = REPO_ROOT / "results" / "judge_ceiling.json"
CELLS_PATH = REPO_ROOT / "results" / "judge_ceiling_cells.jsonl"
LOCK_DIR = REPO_ROOT / "results"


def clova_ocr_fn(image_bytes: bytes, mime_type: str):
    """실제 파일럿/본 배치와 동일한 전처리(occupancy 정규화)를 거쳐 CLOVA를
    호출한다 — ceiling 수치가 실제 채점 경로를 대표하게 하기 위함."""
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    pre = normalize_occupancy(img)
    fmt = "JPEG" if mime_type == "image/jpeg" else "PNG"
    buf = io.BytesIO()
    pre.image.save(buf, format=fmt)
    return clova_ocr.run_general_ocr(buf.getvalue(), mime_type=mime_type)


def representative_syllables_per_cell(n_per_cell: int, seed: int = 0) -> dict:
    result = partition(seed=seed)
    return {cell: list(result.core_by_cell[cell][:n_per_cell]) for cell in ALL_18_CELLS}


def _load_done_cells() -> Set[str]:
    if not CELLS_PATH.is_file():
        return set()
    done = set()
    with CELLS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(rec["cell_key"])
    return done


def _load_all_cells() -> dict:
    per_cell = {}
    if not CELLS_PATH.is_file():
        return per_cell
    with CELLS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            per_cell[rec["cell_key"]] = rec["data"]
    return per_cell


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-syllables-per-cell", type=int, default=5)
    parser.add_argument("--n-degraded-per-target", type=int, default=3)
    parser.add_argument("--fonts", type=str, default="noto_sans_kr,pretendard")
    args = parser.parse_args()

    if not clova_ocr.is_configured():
        raise SystemExit("CLOVA_API_URL/CLOVA_SECRET_KEY가 설정되지 않았습니다.")

    fonts = tuple(args.fonts.split(","))
    reps = representative_syllables_per_cell(args.n_syllables_per_cell, seed=0)
    per_cell_expected = args.n_syllables_per_cell * len(fonts) * (1 + args.n_degraded_per_target)
    print(f"셀당 예상 표본 {per_cell_expected}장 × 18셀 = {per_cell_expected*18}장")

    done_cells = _load_done_cells()
    remaining = [c for c in ALL_18_CELLS if "|".join(c) not in done_cells]
    print(f"완료된 셀 {len(done_cells)}/18, 남은 셀 {len(remaining)}")

    lock_path = _acquire_lock(LOCK_DIR)
    try:
        with CELLS_PATH.open("a", encoding="utf-8") as cell_f:
            for cell in remaining:
                chars = reps[cell]
                cell_key = "|".join(cell)
                print(f"cell={cell_key}  chars={chars}")
                report = measure_ceiling(
                    chars,
                    clova_ocr_fn,
                    judge_name="clova_general",
                    fonts=fonts,
                    n_degraded_per_target=args.n_degraded_per_target,
                    seed=hash(cell_key) % (2**31),
                    canvas_size=1024,
                    text_area_frac=0.3,
                )
                cell_data = {
                    "chars": chars,
                    "clean_accuracy": report.clean_accuracy,
                    "degraded_accuracy": report.degraded_accuracy,
                    "official_validity": report.official_validity(),
                    "n_clean": report.n_clean,
                    "n_degraded": report.n_degraded,
                    "samples": [
                        {
                            "char": s.char,
                            "condition": s.condition,
                            "font_name": s.font_name,
                            "candidate_text": s.candidate_text,
                            "matching_rule": s.matching_rule,
                            "correct": s.correct,
                        }
                        for s in report.samples
                    ],
                }
                cell_f.write(json.dumps({"cell_key": cell_key, "data": cell_data}, ensure_ascii=False) + "\n")
                cell_f.flush()
                print(
                    f"    clean={report.clean_accuracy:.2f}(n={report.n_clean})  "
                    f"degraded={report.degraded_accuracy:.2f}(n={report.n_degraded})  "
                    f"-> {report.official_validity()}"
                )
    finally:
        _release_lock(lock_path)

    per_cell = _load_all_cells()
    if len(per_cell) < len(ALL_18_CELLS):
        print(f"\n{len(per_cell)}/18셀만 완료됨 — 재실행하면 남은 셀부터 이어서 진행합니다.")
        return

    overall_clean = sum(v["clean_accuracy"] * v["n_clean"] for v in per_cell.values()) / sum(
        v["n_clean"] for v in per_cell.values()
    )
    overall_degraded = sum(v["degraded_accuracy"] * v["n_degraded"] for v in per_cell.values()) / sum(
        v["n_degraded"] for v in per_cell.values()
    )
    excluded_cells = [k for k, v in per_cell.items() if v["official_validity"] == "official_excluded"]

    output = {
        "judge_name": "clova_general",
        "preprocess_recipe_version": "v1",
        "n_syllables_per_cell": args.n_syllables_per_cell,
        "n_degraded_per_target": args.n_degraded_per_target,
        "fonts": list(fonts),
        "overall_clean_accuracy": overall_clean,
        "overall_degraded_accuracy": overall_degraded,
        "excluded_cell_count": len(excluded_cells),
        "excluded_cells": excluded_cells,
        "per_cell": per_cell,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"전체 clean={overall_clean:.3f}  degraded={overall_degraded:.3f}")
    print(f"Official 제외 셀: {len(excluded_cells)}/18  {excluded_cells}")
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
