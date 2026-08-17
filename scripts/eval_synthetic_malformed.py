# -*- coding: utf-8 -*-
"""합성 malformed 글자 실험 — 판정자 은폐 현상을 생성 모델 없이,
정답이 구성상 보장된 상태로 검증한다.

**이 실험이 왜 필요한가.** 지금까지의 증거(12~14·17단계)는 전부 Seedream
생성물 + 사람 감사에 의존했다. 이 실험은 그 세 가지 취약점을 우회한다:

  1. "Seedream 특유의 문제 아니냐"  → 생성 모델을 아예 안 쓴다
  2. "표본이 300장뿐"              → 무료로 수천 장까지 늘릴 수 있다
  3. "정답이 사람 라벨(주관)"       → 정답을 구성적으로 보장한다
     (`jamo_bench.synthetic_malformed` — 깨끗한 렌더링에 없던 획을
     더하거나(add_stroke) 있던 획을 지워서(remove_stroke) 만든다.
     실제 감사자 메모에 기록된 실패 유형의 일반화다)

**측정하는 것.** 통제군(control, 손대지 않은 렌더링)에서 판정자가 잘
맞히는지 먼저 확인한 뒤(이미지 품질 문제가 아님을 배제), malformed
표본에서 판정자가 "존재하지 않는 글자"에 얼마나 자주 유효 음절 하나를
자신 있게 답하는지(은폐율) 잰다.

기본 판정자는 EasyOCR + template_match(둘 다 무료). CLOVA는 유료라
기본에서 뺐다 — `--clova` 플래그로 켤 수 있다(비용 발생 주의).

**`--engine paddleocr`**: 3번째(아키텍처가 또 다른, PP-OCR 계열) 엔진 추가
검증. 기존 `synthetic_malformed_eval.json`(EasyOCR+template_match, 이미
노트·검증 스크립트가 참조하는 확정본)은 건드리지 않고, **같은 seed로
생성한 동일한 표본**에 PaddleOCR만 추가로 돌려
`synthetic_malformed_paddleocr_eval.json`에 별도 저장한다.

API 호출 0건(기본) — 전부 로컬.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jamo_bench.decompose import all_syllables  # noqa: E402
from jamo_bench.synthetic_malformed import build_dataset  # noqa: E402
from jamo_bench.template_match import read_by_template_match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "synthetic_malformed_eval.json"

_HANGUL_RE = None  # eval_oss_ocr.py와 동일 패턴, 지연 임포트로 재사용


def stratified_syllables(n_per_coda: int, seed: int) -> list:
    """coda_class_3(무/단순/겹받침) 균등 표본 — 실제 malformed가 겹받침에
    쏠려 있던 것(13단계, 실패 중 무효 비율 no_T 44.4% → cluster_T 58.6%)이
    합성 실험에서도 재현되는지 보려면 세 축을 고르게 담아야 한다."""
    import random

    rng = random.Random(seed)
    by_coda: dict = {}
    for s in all_syllables():
        by_coda.setdefault(s.coda_class_3, []).append(s.char)
    out = []
    for coda in ("no_T", "simple_T", "cluster_T"):
        pool = by_coda[coda]
        out.extend(rng.sample(pool, min(n_per_coda, len(pool))))
    return out


OUT_PADDLEOCR = ROOT / "results" / "synthetic_malformed_paddleocr_eval.json"


def run_paddleocr(args) -> None:
    """`synthetic_malformed_eval.json`(EasyOCR+template_match, 확정본)과
    독립된 경로 — 같은 seed·같은 조건으로 재생성한 동일한 표본에 PaddleOCR만
    추가로 돌린다. 정답이 구성상 보장된 표본이므로 `build_dataset`이
    결정론적이면 두 실행의 이미지 집합은 완전히 같다(직접 이미지를 비교
    저장하진 않지만 char/font_name/kind/severity/seed가 전부 일치하면 같은
    이미지임이 보장된다)."""
    import re

    from paddleocr import PaddleOCR

    hangul_re = re.compile(r"^[가-힣]$")
    syllables = stratified_syllables(args.n_per_coda, args.seed)
    print(f"음절 {len(syllables)}개 (종성유형당 {args.n_per_coda}) × 폰트 {args.fonts} "
          f"× severity {args.severities}  [PaddleOCR 추가 검증]")

    items = build_dataset(syllables, fonts=tuple(args.fonts), severities=tuple(args.severities), seed=args.seed)
    n_rejected = sum(1 for it in items if not it.verified_novel)
    print(f"생성 {len(items)}건 (사후검증 거부 {n_rejected}건)")
    items = [it for it in items if it.verified_novel]

    print("PaddleOCR(lang=korean) 모델 로딩 중...")
    # enable_mkldnn=False 필수 — 이 Windows CPU 환경 기본값(True)은
    # 'ConvertPirAttribute2RuntimeAttribute not support' 오류로 크래시한다
    # (실측 확인, 2026-08-11 — scripts/eval_oss_ocr.py와 동일 이슈).
    ocr = PaddleOCR(
        lang="korean", use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False, enable_mkldnn=False,
    )
    print("로딩 완료.")

    def paddle_reading(img):
        arr = np.array(img.convert("RGB"))
        candidates = []
        for res in ocr.predict(arr):
            candidates.extend(zip(res.get("rec_texts", []), res.get("rec_scores", [])))
        cands = [(t.strip(), c) for t, c in candidates if hangul_re.match(t.strip())]
        if not cands:
            return None, 0.0
        return max(cands, key=lambda tc: tc[1])

    rows = []
    for i, it in enumerate(items, 1):
        reading, conf = paddle_reading(it.image)
        rows.append({
            "char": it.char, "font_name": it.font_name, "kind": it.kind,
            "severity": it.severity, "is_control": it.kind == "control",
            "paddleocr_reading": reading, "paddleocr_confidence": conf,
            "paddleocr_matches_original": reading == it.char,
        })
        if i % 40 == 0:
            print(f"  {i}/{len(items)}")

    control = [r for r in rows if r["is_control"]]
    malformed = [r for r in rows if not r["is_control"]]
    c_ans = sum(1 for r in control if r["paddleocr_reading"]) / len(control)
    m_ans = sum(1 for r in malformed if r["paddleocr_reading"]) / len(malformed)
    print(f"\n=== PaddleOCR (개방형) ===")
    print(f"통제군 답변율: {c_ans:.1%} (n={len(control)})")
    print(f"실험군 은폐율: {m_ans:.1%} (n={len(malformed)})")

    from jamo_bench.decompose import decompose
    by_coda: dict = {}
    for r in malformed:
        by_coda.setdefault(decompose(r["char"]).coda_class_3, []).append(r)
    print("\n종성유형별 은폐율:")
    for coda in ("no_T", "simple_T", "cluster_T"):
        sub = by_coda.get(coda, [])
        if sub:
            snap = sum(1 for r in sub if r["paddleocr_reading"]) / len(sub)
            print(f"  {coda:10} n={len(sub):3}  {snap:.1%}")

    OUT_PADDLEOCR.parent.mkdir(parents=True, exist_ok=True)
    OUT_PADDLEOCR.write_text(
        json.dumps({"config": vars(args), "n_items": len(rows), "n_rejected": n_rejected, "rows": rows},
                   ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\n저장: {OUT_PADDLEOCR.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-coda", type=int, default=14, help="종성유형당 음절 수(기본 14 -> 총 ~42음절)")
    ap.add_argument("--fonts", nargs="+", default=["noto_sans_kr"])
    ap.add_argument("--severities", nargs="+", default=["mild", "moderate", "severe"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", choices=["easyocr", "template_match", "both", "paddleocr"], default="both")
    args = ap.parse_args()

    if args.engine == "paddleocr":
        run_paddleocr(args)
        return

    syllables = stratified_syllables(args.n_per_coda, args.seed)
    print(f"음절 {len(syllables)}개 (종성유형당 {args.n_per_coda}) × 폰트 {args.fonts} "
          f"× severity {args.severities}")

    items = build_dataset(syllables, fonts=tuple(args.fonts), severities=tuple(args.severities), seed=args.seed)
    n_rejected = sum(1 for it in items if not it.verified_novel)
    print(f"생성 {len(items)}건 (사후검증 거부 {n_rejected}건 — 우연히 다른 진짜 글자로 재구성됨)")
    items = [it for it in items if it.verified_novel]

    reader = None
    if args.engine in ("easyocr", "both"):
        import easyocr

        print("EasyOCR(ko+en) 모델 로딩 중...")
        reader = easyocr.Reader(["ko", "en"], gpu=False, verbose=False)

    import re

    hangul_re = re.compile(r"^[가-힣]$")

    def easyocr_reading(img):
        arr = np.array(img.convert("RGB"))
        result = reader.readtext(arr)
        cands = [(t.strip(), c) for _, t, c in result if hangul_re.match(t.strip())]
        if not cands:
            return None, 0.0
        return max(cands, key=lambda tc: tc[1])

    rows = []
    for i, it in enumerate(items, 1):
        row = {
            "char": it.char, "font_name": it.font_name, "kind": it.kind,
            "severity": it.severity, "is_control": it.kind == "control",
        }
        if args.engine in ("template_match", "both"):
            tm = read_by_template_match(it.image)
            row["tm_reading"] = tm.predicted_char
            row["tm_score"] = tm.score
            row["tm_matches_original"] = tm.predicted_char == it.char
        if args.engine in ("easyocr", "both"):
            reading, conf = easyocr_reading(it.image)
            row["easyocr_reading"] = reading
            row["easyocr_confidence"] = conf
            row["easyocr_matches_original"] = reading == it.char
        rows.append(row)
        if i % 40 == 0:
            print(f"  {i}/{len(items)}")

    def snap_rate(sub, reading_key):
        """어떤 유효 음절이든 답한 비율. 개방형(EasyOCR: '답 안 함'이 가능)
        에서만 의미가 있다 — 폐쇄형(template_match)은 구조상 항상 무언가를
        답하므로 이 지표가 자명하게 100%다(별도 표시)."""
        return sum(1 for r in sub if r.get(reading_key)) / len(sub)

    def summarize_closed_form(reading_key, match_key, score_key):
        """template_match — 폐쇄형(항상 답함). 핵심 지표는 '원본으로
        스냅했는가'와 그때의 유사도 점수다."""
        control = [r for r in rows if r["is_control"]]
        malformed = [r for r in rows if not r["is_control"]]
        print(f"\n=== template_match (폐쇄형 — 항상 유효 음절 중 하나를 답함) ===")
        c_acc = sum(1 for r in control if r[match_key]) / len(control) if control else float("nan")
        print(f"통제군 정확도: {c_acc:.1%} (n={len(control)}) — 이미지 품질 배제선")

        for kind in ("add_stroke", "remove_stroke"):
            print(f"\n  [{kind}]")
            for sev in args.severities:
                sub = [r for r in malformed if r["kind"] == kind and r["severity"] == sev]
                if not sub:
                    continue
                snap_orig = sum(1 for r in sub if r[match_key]) / len(sub)
                mean_score = np.mean([r[score_key] for r in sub])
                print(f"    {sev:10} n={len(sub):3}  원본으로 스냅 {snap_orig:.1%}  "
                      f"평균 유사도 {mean_score:.3f}")

        by_coda = {}
        for r in malformed:
            from jamo_bench.decompose import decompose
            coda = decompose(r["char"]).coda_class_3
            by_coda.setdefault(coda, []).append(r)
        print(f"\n  종성유형별 '원본으로 스냅' 비율:")
        for coda in ("no_T", "simple_T", "cluster_T"):
            sub = by_coda.get(coda, [])
            if not sub:
                continue
            snap_orig = sum(1 for r in sub if r[match_key]) / len(sub)
            print(f"    {coda:10} n={len(sub):3}  {snap_orig:.1%}")

    def summarize_open_form(reading_key, match_key, conf_key):
        """EasyOCR — 개방형(아무것도 못 읽으면 답을 안 낼 수 있음). 핵심
        지표는 '무효 글자인데도 어떤 유효 음절이든 답하는가'(은폐율)다."""
        control = [r for r in rows if r["is_control"]]
        malformed = [r for r in rows if not r["is_control"]]
        print(f"\n=== EasyOCR (개방형 — '답 없음'이 가능) ===")
        c_acc = sum(1 for r in control if r[match_key]) / len(control) if control else float("nan")
        print(f"통제군 정확도: {c_acc:.1%} (n={len(control)}) — 이미지 품질 배제선")

        for kind in ("add_stroke", "remove_stroke"):
            print(f"\n  [{kind}]")
            for sev in args.severities:
                sub = [r for r in malformed if r["kind"] == kind and r["severity"] == sev]
                if not sub:
                    continue
                snap = snap_rate(sub, reading_key)
                hc = sum(1 for r in sub if (r.get(conf_key) or 0) >= 0.80) / len(sub)
                print(f"    {sev:10} n={len(sub):3}  은폐(유효음절로 답함) {snap:.1%}  "
                      f"confidence>=0.80 {hc:.1%}")

        by_coda = {}
        for r in malformed:
            from jamo_bench.decompose import decompose
            coda = decompose(r["char"]).coda_class_3
            by_coda.setdefault(coda, []).append(r)
        print(f"\n  종성유형별 은폐율:")
        for coda in ("no_T", "simple_T", "cluster_T"):
            sub = by_coda.get(coda, [])
            if not sub:
                continue
            print(f"    {coda:10} n={len(sub):3}  {snap_rate(sub, reading_key):.1%}")

    if args.engine in ("template_match", "both"):
        summarize_closed_form("tm_reading", "tm_matches_original", "tm_score")
    if args.engine in ("easyocr", "both"):
        summarize_open_form("easyocr_reading", "easyocr_matches_original", "easyocr_confidence")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"config": vars(args), "n_items": len(rows), "n_rejected": n_rejected, "rows": rows},
                   ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\n저장: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
