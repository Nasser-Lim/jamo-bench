# -*- coding: utf-8 -*-
"""Phase 1 D+1 파일럿 배치 러너 (JAMO_benchmark_design.md 로드맵).

기본값(50 syllables × 2 templates × 3 samples = 300장)은 설계서 로드맵의
"D+1: 파일럿 300장"과 정확히 맞춘 것이다. 실행하면 실제 ModelArk 이미지
생성 + CLOVA OCR API를 호출한다(과금 발생) — --dry-run으로 먼저 계획만
확인할 것을 강력히 권장한다.

재실행 시 이미 완료된 (target, template_id, n_sample_idx) 조합은
건너뛴다(결과 JSONL을 읽어 판단) — 중간에 끊겨도 이어서 돌릴 수 있다.

사용법:
    python scripts/run_pilot.py --dry-run
    python scripts/run_pilot.py --limit 5          # 실제 호출 5건만
    python scripts/run_pilot.py                     # 전체 300장
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from jamo_bench import clova_ocr, modelark  # noqa: E402
from jamo_bench.decompose import decompose  # noqa: E402
from jamo_bench.judge_preprocess import normalize_occupancy  # noqa: E402
from jamo_bench.match_region import match_target_region  # noqa: E402
from jamo_bench.partitioning import partition  # noqa: E402
from jamo_bench.prompts import render_prompt  # noqa: E402
from jamo_bench.route import build_detection_input, classify_route  # noqa: E402
from jamo_bench.score import score  # noqa: E402
from jamo_bench.vision_heuristics import has_ink_marks  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "results" / "pilot"
LOCK_FILENAME = ".pilot.lock"


def _pid_is_running(pid: int) -> bool:
    """Windows tasklist로 PID 생존 여부를 확인한다. 판단이 안 서면(오류 등)
    "실행 중"으로 간주해 중복 실행 쪽보다 안전한 쪽으로 fail한다 — 중복
    실행은 API를 두 번 과금시키는 실제 사고를 낸 적이 있다(2026-08-09)."""
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True, text=True, timeout=5,
        )
        return str(pid) in out.stdout
    except Exception:
        return True


def _acquire_lock(out_dir: Path) -> Path:
    lock_path = out_dir / LOCK_FILENAME
    if lock_path.is_file():
        try:
            old_pid = int(lock_path.read_text().strip())
        except ValueError:
            old_pid = None
        if old_pid is not None and _pid_is_running(old_pid):
            raise SystemExit(
                f"다른 run_pilot.py 프로세스(PID {old_pid})가 이미 실행 중인 것으로 보입니다. "
                f"동시에 두 개를 돌리면 같은 이미지 생성·OCR 호출이 중복 과금됩니다. "
                f"그 프로세스를 종료했는데도 이 메시지가 뜨면 {lock_path}를 지우고 재시도하세요."
            )
        lock_path.unlink()  # stale lock (프로세스는 이미 죽음)
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    return lock_path


def _release_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


@dataclass(frozen=True)
class PilotTask:
    target: str
    template_id: str
    n_sample_idx: int


def select_pilot_syllables(n_syllables: int, seed: int) -> List[str]:
    """partitioning.partition()의 18셀 층화 결과에서 셀마다 고르게 뽑아
    n_syllables개를 만든다 — 파일럿 300장도 구조 균형을 최대한 지켜야
    Ceiling 1차 측정이 특정 셀에 쏠리지 않는다."""
    result = partition(seed=seed)
    cells = sorted(result.core_by_cell.keys())
    per_cell = max(1, n_syllables // len(cells))

    picked: List[str] = []
    for cell in cells:
        picked.extend(result.core_by_cell[cell][:per_cell])
    # 부족분은 남은 Core 540에서 결정론적으로(정렬 순서) 채운다.
    if len(picked) < n_syllables:
        remaining = [c for c in result.core_540 if c not in picked]
        picked.extend(remaining[: n_syllables - len(picked)])
    return picked[:n_syllables]


def build_pilot_tasks(
    n_syllables: int = 50,
    samples_per_syllable: int = 3,
    templates: Tuple[str, ...] = ("T1", "T2"),
    seed: int = 0,
) -> List[PilotTask]:
    syllables = select_pilot_syllables(n_syllables, seed)
    tasks = []
    for target in syllables:
        for template_id in templates:
            for i in range(samples_per_syllable):
                tasks.append(PilotTask(target=target, template_id=template_id, n_sample_idx=i))
    return tasks


def _load_done_keys(results_path: Path) -> Set[Tuple[str, str, int]]:
    done: Set[Tuple[str, str, int]] = set()
    if not results_path.is_file():
        return done
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((rec["target"], rec["template_id"], rec["n_sample_idx"]))
    return done


def _detect_mime(image_bytes: bytes) -> Tuple[str, str]:
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    fmt = img.format or "PNG"
    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}.get(fmt, "image/png")
    return mime, fmt


def run_one(task: PilotTask, images_dir: Path) -> dict:
    prompt = render_prompt(task.template_id, task.target)
    gen = modelark.generate_image(prompt, response_format="b64_json", watermark=False)
    image_bytes = gen.image_bytes
    mime, fmt = _detect_mime(image_bytes)

    image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    image_name = f"{task.target}_{task.template_id}_{task.n_sample_idx}_{image_hash}.{fmt.lower()}"
    # 원본(전처리 전) 이미지를 그대로 보관한다 — 판정용 전처리는 OCR
    # 입력에만 적용하고, 감사·재현용 아카이브는 실제 생성물이어야 한다.
    (images_dir / image_name).write_bytes(image_bytes)

    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    width, height = img.size

    has_text_like_region = has_ink_marks(img)
    preprocessed = normalize_occupancy(img)
    ocr_buf = io.BytesIO()
    ocr_buf_format = "JPEG" if fmt.upper() == "JPEG" else "PNG"
    preprocessed.image.save(ocr_buf, format=ocr_buf_format)
    ocr_mime = "image/jpeg" if ocr_buf_format == "JPEG" else "image/png"

    ocr = clova_ocr.run_general_ocr(ocr_buf.getvalue(), mime_type=ocr_mime)
    match = match_target_region(ocr.fields, width, height, target=task.target)
    det = build_detection_input(ocr.fields, width, height, has_text_like_region=has_text_like_region)
    route = classify_route(det)

    score_result = None
    if match.candidate_text:
        score_result = score(task.target, match.candidate_text)

    syl = decompose(task.target)

    return {
        "target": task.target,
        "template_id": task.template_id,
        "n_sample_idx": task.n_sample_idx,
        "vowel_class": syl.vowel_class_2,
        "vowel_shape": syl.vowel_shape,
        "final_class": syl.coda_class_3,
        "coda_class_4": syl.coda_class_4,
        "onset_group": syl.onset_group,
        "is_ieung_onset": syl.is_ieung_onset,
        "model": gen.model,
        "prompt": prompt,
        "watermark_masked": False,  # watermark=False 요청 — 실제 등장 여부는 감사 대상
        "image_path": str((images_dir / image_name).relative_to(REPO_ROOT)),
        "image_hash": image_hash,
        "image_width": width,
        "image_height": height,
        "ocr_engine": "clova_general",
        "has_ink_marks": has_text_like_region,
        "occupancy_before": preprocessed.occupancy_before,
        "occupancy_after": preprocessed.occupancy_after,
        "occupancy_rescaled": preprocessed.rescaled,
        "preprocess_recipe_version": preprocessed.recipe_version,
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
        "timestamp": time.time(),
    }


def run_pilot(
    tasks: List[PilotTask],
    out_dir: Path,
    sleep_sec: float = 0.5,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "pilot_results.jsonl"

    done = _load_done_keys(results_path)
    todo = [t for t in tasks if (t.target, t.template_id, t.n_sample_idx) not in done]

    print(f"전체 {len(tasks)}건 / 완료 {len(done)}건 / 남음 {len(todo)}건")

    if dry_run:
        for t in todo[: limit or len(todo)]:
            print(f"  [dry-run] {t.target}  {t.template_id}  sample={t.n_sample_idx}")
        return

    if limit is not None:
        todo = todo[:limit]

    if not modelark.is_configured():
        raise SystemExit("ARK_API_KEY/ARK_MODEL_SEEDREAM이 설정되지 않았습니다.")
    if not clova_ocr.is_configured():
        raise SystemExit("CLOVA_API_URL/CLOVA_SECRET_KEY가 설정되지 않았습니다.")

    lock_path = _acquire_lock(out_dir)
    try:
        _run_loop(todo, images_dir, results_path, sleep_sec)
    finally:
        _release_lock(lock_path)


def _run_loop(todo: List[PilotTask], images_dir: Path, results_path: Path, sleep_sec: float) -> None:
    with results_path.open("a", encoding="utf-8") as f:
        for i, task in enumerate(todo, 1):
            try:
                record = run_one(task, images_dir)
            except Exception as e:  # noqa: BLE001 - 파일럿은 한 건 실패로 전체가 죽으면 안 된다
                record = {
                    "target": task.target,
                    "template_id": task.template_id,
                    "n_sample_idx": task.n_sample_idx,
                    "error": str(e),
                    "timestamp": time.time(),
                }
                print(f"  [{i}/{len(todo)}] ERROR {task.target} {task.template_id}#{task.n_sample_idx}: {e}")
            else:
                print(
                    f"  [{i}/{len(todo)}] {task.target} {task.template_id}#{task.n_sample_idx} "
                    f"-> route={record['route']} pred={record['selected_candidate']!r} "
                    f"verdict={record['jamo_valid']}"
                )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if i < len(todo):
                time.sleep(sleep_sec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-syllables", type=int, default=50)
    parser.add_argument("--samples-per-syllable", type=int, default=3)
    parser.add_argument("--templates", type=str, default="T1,T2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="실제 호출 개수 상한(테스트용)")
    args = parser.parse_args()

    templates = tuple(args.templates.split(","))
    tasks = build_pilot_tasks(
        n_syllables=args.n_syllables,
        samples_per_syllable=args.samples_per_syllable,
        templates=templates,
        seed=args.seed,
    )
    run_pilot(tasks, args.out_dir, sleep_sec=args.sleep_sec, dry_run=args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
