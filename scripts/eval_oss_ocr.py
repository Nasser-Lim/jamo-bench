# -*- coding: utf-8 -*-
"""OSS OCR 교차검증 — 판정자 은폐 현상이 CLOVA 특유인지, 인식 기반
판정자 일반의 문제인지 확인한다.

**왜 이게 우선순위 1번인가.** 지금까지의 증거는 CLOVA(한국 상용 OCR) 1종 +
자체 template_match뿐이다. 그런데 실제 텍스트 렌더링 논문(Qwen-Image,
AnyText 등)은 PaddleOCR/EasyOCR 계열을 쓴다. "OCR 채점은 렌더링 실패를
은폐한다"는 주장을 방법론 비판으로 내려면, 그 OCR이 CLOVA만이 아니라는
증거가 있어야 한다(`docs/SCOPE.md` "공개 프레이밍 재정의" 참고).

**엔진 2개(EasyOCR, PaddleOCR)를 지원한다** — `--engine` 플래그. 아키텍처가
서로 다른(CRNN vs PP-OCR) 두 오픈소스 엔진에서 같은 현상이 재현되는지가
"CLOVA 특유의 문제"라는 반박을 막는 핵심 증거다(17단계).

**측정 대상.** 사람이 "유효 완성형 아님"(malformed/non_hangul/multi)이라
판정한 이미지에서, 엔진이 그럼에도 어떤 유효 완성형 하나를 답하며
confidence까지 높게 주는 비율.

API 호출 0건, Seedream 재생성 0건 — 저장된 파일럿 이미지 재사용.
Windows에서 파일명이 한글이면 `cv2.imread`가 실패하므로(OpenCV의 로컬
코드페이지 이슈) PIL로 읽어 numpy 배열로 넘긴다. PaddleOCR 3.x는 이 환경
(Windows CPU)에서 기본 oneDNN 가속이 깨져 있어 `enable_mkldnn=False`가
필수다(실측 확인, 2026-08-11).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jamo_bench.decompose import decompose  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
_HANGUL_SYLLABLE = re.compile(r"^[가-힣]$")


def load_human_labels() -> dict:
    """이미지 경로 -> {"valid": bool, "target": str, "human_reading": str|None}.

    Phase 2 재라벨링(이진, 신뢰 가능)이 있으면 그걸 쓰고 없으면 1차 라벨로
    폴백한다 — `judging_protocol.py`가 쓰는 것과 같은 우선순위.
    """
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
    relabel = {r["image_path"]: r for r in phase2 if r["annotator"] == "Nasser Lim"}
    pilot = [
        json.loads(l)
        for l in (ROOT / "results" / "pilot" / "pilot_results_v2.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    out = {}
    for r in pilot:
        p = r["image_path"]
        if p in relabel:
            valid = relabel[p]["label"] == "valid_syllable"
            # Phase 2 UI도 valid_syllable일 때는 전사를 받는다
            # (audit_queue.PHASE2_LABELS의 needs_transcription=True).
            reading = relabel[p].get("transcription") or None
        elif p in phase1:
            valid = phase1[p]["label"] == "text"
            reading = phase1[p]["transcription"] if valid else None
        else:
            continue  # 아직 사람이 안 본 이미지(미감사 44장 중 재라벨링 안 된 것) — 스킵
        out[p] = {"valid": valid, "target": r["target"], "human_reading": reading}
    return out


def best_single_syllable(candidates) -> tuple:
    """(텍스트, confidence) 후보들 중 완성형 한글 한 글자인 것에서 confidence
    최고를 고른다. template_match/CLOVA와 같은 자리에 꽂을 수 있는 형태로
    맞춘다 — "단일 후보를 고른다"는 §8.5.1 분모 규칙과 동일 원칙."""
    cands = [(t.strip(), c) for t, c in candidates if _HANGUL_SYLLABLE.match(t.strip())]
    if not cands:
        return None, 0.0
    return max(cands, key=lambda tc: tc[1])


def make_easyocr_reader():
    import easyocr

    print("EasyOCR(ko+en) 모델 로딩 중...")
    reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)
    print("로딩 완료.")

    def read(arr):
        result = reader.readtext(arr)
        return best_single_syllable([(t, c) for _, t, c in result])

    return read


def make_paddleocr_reader():
    from paddleocr import PaddleOCR

    print("PaddleOCR(lang=korean) 모델 로딩 중...")
    # enable_mkldnn=False 필수 — 이 Windows CPU 환경에서 기본값(True)은
    # 'ConvertPirAttribute2RuntimeAttribute not support' 오류로 크래시한다.
    ocr = PaddleOCR(
        lang="korean",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    print("로딩 완료.")

    def read(arr):
        candidates = []
        for res in ocr.predict(arr):
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            candidates.extend(zip(texts, scores))
        return best_single_syllable(candidates)

    return read


ENGINES = {"easyocr": make_easyocr_reader, "paddleocr": make_paddleocr_reader}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=sorted(ENGINES), default="easyocr")
    args = ap.parse_args()
    engine = args.engine

    # EasyOCR은 최초 실행(17단계) 파일명을 그대로 유지한다 —
    # docs/TECHNICAL_NOTE.md, DATASET_CARD.md, verify_claims.py가 이미
    # `oss_ocr_eval.json`이라는 이름으로 참조하고 있어 이름이 바뀌면 깨진다.
    out_path = ROOT / "results" / ("oss_ocr_eval.json" if engine == "easyocr" else f"oss_ocr_eval_{engine}.json")

    labels = load_human_labels()
    print(f"대상 {len(labels)}장 (사람 라벨 있는 것만) — 엔진: {engine}")

    read = ENGINES[engine]()
    print("인식 시작(로컬, API 호출 0건)...")

    rows = []
    for i, (path, info) in enumerate(labels.items(), 1):
        arr = np.array(Image.open(ROOT / path).convert("RGB"))
        reading, conf = read(arr)
        rows.append({
            "path": path,
            "target": info["target"],
            "human_valid": info["valid"],
            "human_reading": info["human_reading"],
            "engine_reading": reading,
            "engine_confidence": conf,
        })
        if i % 30 == 0:
            print(f"  {i}/{len(labels)}")

    valid_rows = [r for r in rows if r["human_valid"]]
    invalid_rows = [r for r in rows if not r["human_valid"]]

    print(f"\n=== 정확도 (사람 유효 판정 {len(valid_rows)}건 기준) ===")
    engine_correct = sum(1 for r in valid_rows if r["engine_reading"] == r["target"])
    human_correct = sum(1 for r in valid_rows if r["human_reading"] == r["target"])
    print(f"  사람(진실)  {human_correct}/{len(valid_rows)} = {human_correct/len(valid_rows):.1%}")
    print(f"  {engine:10}  {engine_correct}/{len(valid_rows)} = {engine_correct/len(valid_rows):.1%}")

    print(f"\n=== 은폐율 — 무효 판정 {len(invalid_rows)}건 중 ===")
    silent_any = [r for r in invalid_rows if r["engine_reading"] is not None]
    silent_high_conf = [r for r in silent_any if r["engine_confidence"] >= 0.80]
    print(f"  어떤 유효 완성형이든 답함: {len(silent_any)}/{len(invalid_rows)} = "
          f"{len(silent_any)/len(invalid_rows):.1%}")
    print(f"  게다가 confidence>=0.80: {len(silent_high_conf)}/{len(invalid_rows)} = "
          f"{len(silent_high_conf)/len(invalid_rows):.1%}  ← CLOVA confidence 게이트와 동일 기준")

    print(f"\n  사례 (무효 판정인데 {engine}이 자신 있게 답한 것):")
    for r in sorted(silent_high_conf, key=lambda r: -r["engine_confidence"])[:8]:
        print(f"    target={r['target']}  {engine}={r['engine_reading']}  "
              f"conf={r['engine_confidence']:.3f}")

    def jamo_bias(get_pred):
        n = {"초성": 0, "중성": 0, "종성": 0}
        ok = dict(n)
        for r in rows:
            pred = get_pred(r)
            t = decompose(r["target"])
            if pred is None or decompose(pred) is None:
                for k in n:
                    n[k] += 1
                continue
            d = decompose(pred)
            for k, a, b in [("초성", t.onset, d.onset), ("중성", t.nucleus, d.nucleus), ("종성", t.coda, d.coda)]:
                n[k] += 1
                ok[k] += a == b
        return {k: ok[k] / n[k] for k in n}

    truth = lambda r: r["human_reading"] if r["human_valid"] else None  # noqa: E731
    engine_pred = lambda r: r["engine_reading"]  # noqa: E731
    t_acc, e_acc = jamo_bias(truth), jamo_bias(engine_pred)
    print(f"\n=== 자모 위치별 오류율 (Unconditional) ===")
    print(f"{'위치':6}{'사람(진실)':>12}{engine:>12}{'오차':>9}")
    for k in ["초성", "중성", "종성"]:
        print(f"{k:6}{t_acc[k]:12.1%}{e_acc[k]:12.1%}{e_acc[k]-t_acc[k]:+9.1%}")

    out_path.write_text(
        json.dumps({"engine": engine, "n": len(rows), "n_valid": len(valid_rows), "n_invalid": len(invalid_rows),
                    "silent_any_rate": len(silent_any) / len(invalid_rows),
                    "silent_high_conf_rate": len(silent_high_conf) / len(invalid_rows),
                    "rows": rows}, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\n저장: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
