# -*- coding: utf-8 -*-
"""저장된 파일럿 300장 이미지를 재생성 없이 재채점한다.

수정된 파이프라인(has_ink_marks 기반 Route 판정, occupancy 정규화 전처리)을
디스크에 이미 있는 이미지에 다시 적용한다 — Seedream 재호출 없음(비용 0),
CLOVA OCR만 300건 다시 호출한다(재생성 대비 훨씬 저렴).

원본 results/pilot/pilot_results.jsonl은 그대로 두고
results/pilot/pilot_results_v2.jsonl에 새로 쓴다 — 두 파이프라인 버전의
결과를 나란히 비교할 수 있어야 "무엇이 왜 바뀌었는지" 감사 가능하다.

사용법:
    python scripts/rescore_pilot.py --dry-run
    python scripts/rescore_pilot.py --limit 10
    python scripts/rescore_pilot.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jamo_bench import clova_ocr  # noqa: E402
from jamo_bench.judge_preprocess import normalize_occupancy  # noqa: E402
from jamo_bench.match_region import match_target_region  # noqa: E402
from jamo_bench.route import build_detection_input, classify_route  # noqa: E402
from jamo_bench.score import score  # noqa: E402
from jamo_bench.vision_heuristics import has_ink_marks  # noqa: E402

from run_pilot import _acquire_lock, _release_lock  # noqa: E402

SRC_PATH = REPO_ROOT / "results" / "pilot" / "pilot_results.jsonl"
DST_PATH = REPO_ROOT / "results" / "pilot" / "pilot_results_v2.jsonl"
LOCK_DIR = REPO_ROOT / "results" / "pilot"


def _key(rec: dict):
    return (rec["target"], rec["template_id"], rec["n_sample_idx"])


def _load_done_keys(path: Path) -> Set[tuple]:
    done = set()
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(_key(rec))
    return done


def rescore_one(src_rec: dict) -> dict:
    if "image_path" not in src_rec or src_rec.get("selected_candidate") is None and "error" in src_rec:
        # 원본 생성 자체가 실패했던 건(error 필드 존재, 이미지 없음)은 그대로 이월
        pass

    if "error" in src_rec and "image_path" not in src_rec:
        rec = dict(src_rec)
        rec["rescored_at"] = time.time()
        return rec

    from PIL import Image

    image_path = REPO_ROOT / src_rec["image_path"]
    image_bytes = image_path.read_bytes()
    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size
    target = src_rec["target"]

    has_text_like_region = has_ink_marks(img)
    pre = normalize_occupancy(img)
    fmt = "JPEG" if image_path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    buf = io.BytesIO()
    pre.image.save(buf, format=fmt)

    ocr = clova_ocr.run_general_ocr(buf.getvalue(), mime_type=mime)
    match = match_target_region(ocr.fields, width, height, target=target)
    det = build_detection_input(ocr.fields, width, height, has_text_like_region=has_text_like_region)
    route = classify_route(det)

    score_result = None
    if match.candidate_text:
        score_result = score(target, match.candidate_text)

    rec = dict(src_rec)
    rec.update({
        "has_ink_marks": has_text_like_region,
        "occupancy_before": pre.occupancy_before,
        "occupancy_after": pre.occupancy_after,
        "occupancy_rescaled": pre.rescaled,
        "preprocess_recipe_version": pre.recipe_version,
        "route": route,
        "matching_rule": match.matching_rule,
        "selected_candidate": match.candidate_text,
        "position_miss": match.position_miss,
        "needs_audit": match.needs_audit,
        "spurious_count": match.spurious_count,
        "spurious_area_frac": match.spurious_area_frac,
        "jamo_valid": score_result.verdict if score_result else None,
        "onset_ok": score_result.onset_ok if score_result else None,
        "nucleus_ok": score_result.nucleus_ok if score_result else None,
        "coda_ok": score_result.coda_ok if score_result else None,
        "rescored_at": time.time(),
    })
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep-sec", type=float, default=0.3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    src_recs = [json.loads(l) for l in SRC_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    done = _load_done_keys(DST_PATH)
    todo = [r for r in src_recs if _key(r) not in done]

    print(f"전체 {len(src_recs)}건 / 완료 {len(done)}건 / 남음 {len(todo)}건")

    if args.dry_run:
        for r in todo[: args.limit or len(todo)]:
            print(f"  [dry-run] {r['target']} {r['template_id']} sample={r['n_sample_idx']}")
        return

    if args.limit is not None:
        todo = todo[: args.limit]

    if not clova_ocr.is_configured():
        raise SystemExit("CLOVA_API_URL/CLOVA_SECRET_KEY가 설정되지 않았습니다.")

    lock_path = _acquire_lock(LOCK_DIR)
    try:
        with DST_PATH.open("a", encoding="utf-8") as f:
            for i, src_rec in enumerate(todo, 1):
                try:
                    rec = rescore_one(src_rec)
                except Exception as e:  # noqa: BLE001
                    rec = dict(src_rec)
                    rec["rescore_error"] = str(e)
                    rec["rescored_at"] = time.time()
                    print(f"  [{i}/{len(todo)}] ERROR {src_rec['target']}: {e}")
                else:
                    print(
                        f"  [{i}/{len(todo)}] {rec['target']} {rec['template_id']}#{rec['n_sample_idx']} "
                        f"-> route={rec.get('route')} pred={rec.get('selected_candidate')!r} "
                        f"verdict={rec.get('jamo_valid')}"
                    )
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if i < len(todo):
                    time.sleep(args.sleep_sec)
    finally:
        _release_lock(lock_path)


if __name__ == "__main__":
    main()
