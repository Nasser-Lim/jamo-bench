# -*- coding: utf-8 -*-
"""OVERGEN 자체 감지 캘리브레이션 — `jamo_bench/overgen.py`의
연결요소 병합 거리(merge_iterations)와 점수 마진(OVERGEN_MARGIN)을 정한다.

**정답이 없다는 것부터 인정한다.** 2번째 감사자 투입 후 Krippendorff's α로
확인한 바(`docs/PROGRESS.md` 14단계), 사람의 `multi_syllable` 판정 자체가
malformed와 20%가량 혼동되는 불안정한 축이다(4분류 α=0.576). 그래서 여기서
쓰는 "근거"는 확정 라벨이 아니라 **다중 출처 약한 신호의 합집합/교집합**이다:

  loose positive (있으면 의심)  = CLOVA가 2글자 이상으로 읽음(jamo_valid==
                                  OVERGEN) 또는 두 감사자 중 한 명이라도
                                  multi_syllable로 표기
  strict positive (거의 확실)   = 두 감사자 모두 multi_syllable로 일치

recall/precision을 이 두 기준 모두에 대해 보고하고, strict 쪽이 더
신뢰할 수 있는 신호다.

API 호출 0건 — 전부 로컬(template_match 재사용, 이미 캐시된 템플릿 뱅크).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jamo_bench.overgen import _components, raw_ink_mask  # noqa: E402
from jamo_bench.template_match import read_by_template_match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "overgen_eval.json"


def build_ground_truth():
    """이미지 경로 -> {"loose": bool, "strict": bool, "valid": bool}."""
    pilot = {
        r["image_path"]: r
        for r in (
            json.loads(l)
            for l in (ROOT / "results" / "pilot" / "pilot_results_v2.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        )
    }
    phase1 = {
        r["image_path"]: r
        for r in (
            json.loads(l)
            for l in (ROOT / "results" / "audit" / "human_audit_pilot.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        )
    }
    phase2 = [
        json.loads(l)
        for l in (ROOT / "results" / "audit" / "human_audit_pilot_phase2.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    by_annotator: dict = {}
    for r in phase2:
        by_annotator.setdefault(r["annotator"], {})[r["image_path"]] = r["label"]
    annotators = list(by_annotator)

    gt = {}
    for path in pilot:
        clova_multi = pilot[path].get("jamo_valid") == "OVERGEN"
        p1_multi = phase1.get(path, {}).get("label") == "extra_text"
        p1_valid = phase1.get(path, {}).get("label") == "text"

        p2_labels = [by_annotator[a].get(path) for a in annotators if path in by_annotator[a]]
        p2_multi_votes = sum(1 for l in p2_labels if l == "multi_syllable")
        p2_valid_votes = sum(1 for l in p2_labels if l == "valid_syllable")

        loose = clova_multi or p1_multi or p2_multi_votes >= 1
        strict = (clova_multi and p1_multi) or p2_multi_votes >= 2 or (clova_multi and p2_multi_votes >= 1)
        # "유효" 참조(오탐률 계산용) — 어느 출처든 유효라고 본 적이 없고 무효 근거도 없으면 미상으로 스킵
        valid = p1_valid or p2_valid_votes >= 1
        if not (loose or valid):
            continue  # 판정 근거가 전혀 없는 이미지(미감사·미채점)는 캘리브레이션에서 제외
        gt[path] = {"loose": loose, "strict": strict, "valid": valid and not loose}
    return gt


def main():
    gt = build_ground_truth()
    print(f"캘리브레이션 대상 {len(gt)}장 "
          f"(loose positive {sum(v['loose'] for v in gt.values())}, "
          f"strict positive {sum(v['strict'] for v in gt.values())}, "
          f"valid 대조군 {sum(v['valid'] for v in gt.values())})")

    rows = []
    for i, (path, labels) in enumerate(gt.items(), 1):
        img = Image.open(ROOT / path)
        mask = raw_ink_mask(img)
        whole = read_by_template_match(img)
        # 여러 merge_iterations 후보에서 성분 수/분할 점수를 한 번에 계산해둔다 —
        # 재렌더링 비용(soft-IoU 매칭)이 지배적이라 성분 crop 매칭만 후보별로 다시 돈다.
        candidates = {}
        for merge_it in (6, 12, 20, 30):
            comps = _components(mask, merge_it, 0.03)
            if len(comps) <= 1:
                candidates[merge_it] = {"n_components": len(comps), "score_split": whole.score}
                continue
            w, h = img.size
            weighted, total = 0.0, 0
            for (y0, y1, x0, x1), area in comps:
                box = (max(0, x0 - 8), max(0, y0 - 8), min(w, x1 + 8), min(h, y1 + 8))
                r = read_by_template_match(img.crop(box))
                weighted += r.score * area
                total += area
            candidates[merge_it] = {"n_components": len(comps), "score_split": weighted / total if total else 0.0}
        rows.append({"path": path, **labels, "score_whole": whole.score, "candidates": candidates})
        if i % 20 == 0:
            print(f"  {i}/{len(gt)}")

    print()
    print(f"{'merge_it':10}{'margin':8}{'strict recall':>15}{'loose recall':>14}{'valid 오탐':>11}")
    print("-" * 60)
    best = {"score": -1.0}
    for merge_it in (6, 12, 20, 30):
        for margin in (0.0, 0.02, 0.05, 0.08, 0.12):
            flags = []
            for r in rows:
                c = r["candidates"][merge_it]
                gap = c["score_split"] - r["score_whole"]
                flags.append((c["n_components"] >= 2) and (gap > margin))
            flags = np.array(flags)
            strict = np.array([r["strict"] for r in rows])
            loose = np.array([r["loose"] for r in rows])
            valid = np.array([r["valid"] for r in rows])
            strict_recall = flags[strict].mean() if strict.any() else float("nan")
            loose_recall = flags[loose].mean() if loose.any() else float("nan")
            valid_fp = flags[valid].mean() if valid.any() else float("nan")
            score = strict_recall - valid_fp  # Youden 스타일, strict 신호 우선
            print(f"{merge_it:10}{margin:8.2f}{strict_recall:15.1%}{loose_recall:14.1%}{valid_fp:11.1%}")
            if score > best["score"]:
                best = {"score": score, "merge_iterations": merge_it, "margin": margin,
                        "strict_recall": strict_recall, "loose_recall": loose_recall, "valid_fp": valid_fp}

    print()
    print(f"채택: merge_iterations={best['merge_iterations']}, margin={best['margin']}")
    print(f"  strict recall {best['strict_recall']:.1%} / loose recall {best['loose_recall']:.1%} "
          f"/ valid 오탐 {best['valid_fp']:.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"best": best, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
